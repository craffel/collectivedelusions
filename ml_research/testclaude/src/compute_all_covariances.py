"""Compute input-activation covariance C_t = (1/N) Σ z z^T for every matrix
parameter in the visual encoder, per task.

Saves one .pt per task with a dict: param_name -> C_t [in, in] (CPU float32).
Covers: in_proj (MHA input), out_proj, mlp.c_fc, mlp.c_proj in every resblock.
"""
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(__file__))
from datasets_utils import get_dataset
from clip_utils import load_clip, load_visual_state_dict

TASKS = ["MNIST", "CIFAR10", "CIFAR100", "SVHN", "FashionMNIST", "EuroSAT", "GTSRB", "DTD"]


def build_tf():
    mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def register_hooks(clip_model):
    """For every linear and MultiheadAttention in the visual encoder, register a
    pre-forward hook that accumulates input second moments.

    Returns (hooks, param_name_to_accumulator).
    """
    covs = {}   # param_name -> torch.Tensor [in, in] on GPU (accumulated scaled)
    counts = {} # param_name -> int

    hooks = []
    for name, mod in clip_model.visual.named_modules():
        full = f"visual.{name}"
        # Linear weights
        if isinstance(mod, nn.Linear):
            pname = f"{full}.weight"
            def make_linear_hook(pname):
                def hook(m, inputs, out):
                    x = inputs[0]
                    x = x.reshape(-1, x.shape[-1]).float()
                    c = covs.get(pname)
                    if c is None:
                        c = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                        covs[pname] = c; counts[pname] = 0
                    covs[pname] = c + x.T @ x
                    counts[pname] += x.shape[0]
                return hook
            hooks.append(mod.register_forward_hook(make_linear_hook(pname)))
        # MultiheadAttention: in_proj_weight
        if isinstance(mod, nn.MultiheadAttention):
            pname = f"{full}.in_proj_weight"
            def make_mha_hook(pname):
                def hook(m, inputs, out):
                    x = inputs[0]
                    x = x.reshape(-1, x.shape[-1]).float()
                    c = covs.get(pname)
                    if c is None:
                        c = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                        covs[pname] = c; counts[pname] = 0
                    covs[pname] = c + x.T @ x
                    counts[pname] += x.shape[0]
                return hook
            hooks.append(mod.register_forward_hook(make_mha_hook(pname)))
        # Conv2d (patch embedding)
        if isinstance(mod, nn.Conv2d):
            pname = f"{full}.weight"
            # For Conv2d, equivalent "input activation covariance" for merging is the unfolded
            # patch covariance. But the conv weight dimensionality is (out, in, kH, kW), so
            # when we reshape to (out, in*kH*kW) the covariance is over unfolded patches.
            def make_conv_hook(pname, mod=mod):
                def hook(m, inputs, out):
                    x = inputs[0]
                    # Unfold patches: (B, C*kH*kW, L)
                    x_u = torch.nn.functional.unfold(x, kernel_size=mod.kernel_size,
                                                     stride=mod.stride, padding=mod.padding)
                    # Transpose to (B*L, C*kH*kW)
                    x_u = x_u.transpose(1, 2).reshape(-1, x_u.shape[1]).float()
                    c = covs.get(pname)
                    if c is None:
                        c = torch.zeros(x_u.shape[1], x_u.shape[1], device=x_u.device)
                        covs[pname] = c; counts[pname] = 0
                    covs[pname] = c + x_u.T @ x_u
                    counts[pname] += x_u.shape[0]
                return hook
            hooks.append(mod.register_forward_hook(make_conv_hook(pname)))
    return hooks, covs, counts


@torch.no_grad()
def compute_for_task(clip_model, task, device, max_samples=256, batch_size=64):
    tf = build_tf()
    train_ds, _, _ = get_dataset(task, tf, tf)
    if len(train_ds) > max_samples:
        train_ds = Subset(train_ds, list(range(max_samples)))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    hooks, covs, counts = register_hooks(clip_model)
    clip_model.eval()

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = clip_model.encode_image(x)

    for h in hooks:
        h.remove()

    out = {}
    for name, c in covs.items():
        out[name] = (c / max(counts[name], 1)).detach().cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints")
    ap.add_argument("--out_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/covariances_full")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--max_samples", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tasks = args.tasks.split(",")

    # Load pretrained and task sds
    pre = torch.load(os.path.join(args.ckpt_dir, "pretrained.pt"), map_location="cpu", weights_only=False)
    task_sds = {}
    for t in tasks:
        d = torch.load(os.path.join(args.ckpt_dir, f"{t}.pt"), map_location="cpu", weights_only=False)
        task_sds[t] = {k: v.float() for k, v in d["state_dict"].items()}

    clip_model, _, _ = load_clip(device=args.device)

    t0 = time.time()
    for t in tasks:
        print(f"[cov] task={t}  (elapsed {time.time()-t0:.1f}s)")
        load_visual_state_dict(clip_model, task_sds[t], strict=False)
        clip_model.to(args.device)
        covs = compute_for_task(clip_model, t, args.device, max_samples=args.max_samples)
        torch.save(covs, os.path.join(args.out_dir, f"{t}.pt"))
        print(f"  saved {len(covs)} layers")

    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
