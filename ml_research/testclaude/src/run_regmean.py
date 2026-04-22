"""Run data-based RegMean as a strong baseline using the true per-task input-activation covariances
computed by angular_validation.py --save_covariances.

For layers not covered by the covariances (non-matrix params, or layers we didn't profile), falls
back to simple averaging (same convention as the rest of the repo).
"""
import argparse
import json
import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(__file__))
from datasets_utils import get_dataset
from clip_utils import load_clip, load_visual_state_dict

TASKS = ["MNIST", "CIFAR10", "CIFAR100", "SVHN", "FashionMNIST", "EuroSAT", "GTSRB", "DTD"]


def build_eval_tf():
    mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


@torch.no_grad()
def evaluate_on_task(clip_model, task, text_classifier, device, batch_size=512):
    tf = build_eval_tf()
    _, test_ds, _ = get_dataset(task, tf, tf)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    clip_model.eval()
    logit_scale = float(clip_model.logit_scale.detach().exp().item())
    tc = text_classifier.to(device)
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        feats = clip_model.encode_image(x)
        feats = F.normalize(feats, dim=-1)
        logits = logit_scale * feats @ tc.T
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def regmean_from_covs(pretrained_sd, task_sds, task_covs, ridge=1e-4):
    """RegMean: W* = (Σ W_t C_t) (Σ C_t + ρ λ I)^{-1} per matrix layer where C_t is provided.

    Falls back to simple average for layers without C_t entries or non-matrix params.
    """
    out = {}
    for k in pretrained_sd:
        theta0 = pretrained_sd[k]
        W_t_list = [sd[k] for sd in task_sds if k in sd]
        if not W_t_list:
            out[k] = theta0.clone(); continue
        has_cov = any(k in tc for tc in task_covs)
        if theta0.ndim < 2 or not has_cov:
            acc = torch.stack([w.float() for w in W_t_list], dim=0).mean(dim=0)
            out[k] = acc.to(theta0.dtype); continue
        shape = theta0.shape
        W2_list = []
        for W in W_t_list:
            W2 = W.float().reshape(W.shape[0], -1) if W.ndim != 2 else W.float()
            W2_list.append(W2)
        din = W2_list[0].shape[1]
        C_sum = torch.zeros(din, din, device=W2_list[0].device)
        WC_sum = torch.zeros_like(W2_list[0])
        for W2, tc in zip(W2_list, task_covs):
            if k in tc:
                C_t = tc[k].to(W2.device).float()
                C_sum = C_sum + C_t
                WC_sum = WC_sum + W2 @ C_t
        lam = C_sum.diagonal().mean().item() + 1e-12
        C_sum = C_sum + ridge * torch.eye(din, device=C_sum.device) * lam
        W_star = torch.linalg.solve(C_sum.T, WC_sum.T).T
        out[k] = W_star.reshape(shape).to(theta0.dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints")
    ap.add_argument("--cov_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/covariances")
    ap.add_argument("--out_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/results")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--ridge", type=float, default=1e-4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device
    tasks = args.tasks.split(",")

    # Load pretrained + task state dicts
    pre = torch.load(os.path.join(args.ckpt_dir, "pretrained.pt"), map_location="cpu", weights_only=False)
    pre_f = {k: v.float() for k, v in pre.items()}
    task_sds = []
    text_classifiers = {}
    for t in tasks:
        d = torch.load(os.path.join(args.ckpt_dir, f"{t}.pt"), map_location="cpu", weights_only=False)
        task_sds.append({k: v.float() for k, v in d["state_dict"].items()})
        text_classifiers[t] = d["text_classifier"].to(device)

    # Load true per-task covariances (only partial — just the profiled layers)
    task_covs = []
    for t in tasks:
        task_covs.append(torch.load(os.path.join(args.cov_dir, f"{t}.pt"), map_location="cpu", weights_only=True))

    # Move sds + covs to GPU
    pre_gpu = {k: v.to(device) for k, v in pre_f.items()}
    tsds_gpu = [{k: v.to(device) for k, v in sd.items()} for sd in task_sds]
    tcov_gpu = [{k: v.to(device) for k, v in tc.items()} for tc in task_covs]

    print("Running RegMean (data-based covariances, true C_t)...")
    t0 = time.time()
    merged = regmean_from_covs(pre_gpu, tsds_gpu, tcov_gpu, ridge=args.ridge)
    merged_cpu = {k: v.cpu() for k, v in merged.items()}
    print(f"  merged in {time.time()-t0:.2f}s")

    # Evaluate
    clip_model, _, _ = load_clip(device=device)
    load_visual_state_dict(clip_model, merged_cpu, strict=False)
    clip_model.to(device)
    per_task = {}
    for t in tasks:
        acc = evaluate_on_task(clip_model, t, text_classifiers[t], device)
        per_task[t] = acc
        print(f"  [RegMean] {t}: {acc*100:.2f}")
    avg = sum(per_task.values()) / len(per_task)
    print(f"  [RegMean] AVG: {avg*100:.2f}")

    out = {"method": "RegMean (true C_t, partial-layer)", "per_task": per_task, "avg": avg,
           "num_covariance_layers": len(task_covs[0])}
    with open(os.path.join(args.out_dir, "regmean_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved", os.path.join(args.out_dir, "regmean_results.json"))


if __name__ == "__main__":
    main()
