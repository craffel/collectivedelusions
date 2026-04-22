"""Per-layer/per-block-type ablation for SCALE-Merge.

Applies SCALE to a subset of linear layers and ACTMat to the rest.
Tests which parts of the visual encoder benefit most from trim-and-elect.
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
from merging import (
    actmat, scale_merge, task_vectors, _is_matrix_param, _reshape_to_matrix,
    _unreshape, topk_mask,
)


def build_eval_transform():
    mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


@torch.no_grad()
def evaluate_on_task(clip_model, task, text_classifier, device, batch_size=512, max_samples=None):
    tf = build_eval_transform()
    _, test_ds, _ = get_dataset(task, tf, tf)
    if max_samples is not None and len(test_ds) > max_samples:
        test_ds = torch.utils.data.Subset(test_ds, list(range(max_samples)))
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    clip_model.eval()
    logit_scale = float(clip_model.logit_scale.detach().exp().item())
    text_cls = text_classifier.to(device)
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        feats = clip_model.encode_image(x)
        feats = F.normalize(feats, dim=-1)
        logits = logit_scale * feats @ text_cls.T
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def classify_layer(name):
    """Classify a parameter name into a block type."""
    if 'attn.in_proj' in name:
        return 'attn_qkv'
    if 'attn.out_proj' in name:
        return 'attn_out'
    if 'mlp.c_fc' in name:
        return 'mlp_in'
    if 'mlp.c_proj' in name:
        return 'mlp_out'
    if name.endswith('.conv1.weight') and 'resblocks' not in name:
        return 'conv1'
    if name == 'visual.proj':
        return 'proj_head'
    return 'other'


def layer_block_index(name):
    """Return block index (0-11) or None if not a transformer layer."""
    import re
    m = re.search(r'resblocks\.(\d+)\.', name)
    if m:
        return int(m.group(1))
    return None


def selective_merge(pre_sd, task_sds, keep_frac, ridge, layer_filter_fn,
                    use_sign=False, device="cuda:0"):
    """Merge: apply SCALE to layers where layer_filter_fn(name)==True, ACTMat elsewhere.

    layer_filter_fn: callable(name) -> bool
    """
    # Compute both SCALE and ACTMat once, then pick per layer.
    pre_gpu = {k: v.to(device) for k, v in pre_sd.items()}
    tsds_gpu = [{k: v.to(device) for k, v in sd.items()} for sd in task_sds]
    scale_sd = scale_merge(pre_gpu, tsds_gpu, keep_frac=keep_frac, ridge=ridge,
                           use_sign_election=use_sign)
    actmat_sd = actmat(pre_gpu, tsds_gpu, ridge=ridge)
    out = {}
    for k in pre_sd:
        if layer_filter_fn(k):
            out[k] = scale_sd[k].cpu()
        else:
            out[k] = actmat_sd[k].cpu()
    del scale_sd, actmat_sd, pre_gpu, tsds_gpu
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints")
    ap.add_argument("--out", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/results/layer_ablation.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", default="MNIST,CIFAR10,CIFAR100,SVHN,FashionMNIST,EuroSAT,GTSRB,DTD")
    ap.add_argument("--keep_frac", type=float, default=0.3)
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    device = args.device

    pre = torch.load(os.path.join(args.ckpt_dir, "pretrained.pt"), map_location="cpu", weights_only=False)
    ckpts = {}
    for t in tasks:
        c = torch.load(os.path.join(args.ckpt_dir, f"{t}.pt"), map_location="cpu", weights_only=False)
        ckpts[t] = c
    task_sds = [ckpts[t]["state_dict"] for t in tasks]
    text_classifiers = {t: ckpts[t]["text_classifier"].to(device) for t in tasks}
    pre_cpu = {k: v.float() for k, v in pre.items()}
    task_sds_cpu = [{k: v.float() for k, v in sd.items()} for sd in task_sds]

    clip_model, _, _ = load_clip(device=device)

    def eval_sd(sd, name):
        load_visual_state_dict(clip_model, sd, strict=False)
        clip_model.to(device)
        per_task = {}
        for t in tasks:
            acc = evaluate_on_task(clip_model, t, text_classifiers[t], device)
            per_task[t] = acc
            print(f"  [{name}] {t}: {acc:.4f}")
        avg = sum(per_task.values()) / len(per_task)
        print(f"  [{name}] AVG: {avg:.4f}")
        return per_task, avg

    results = {}

    # Define filter configs
    def all_layers(_): return True
    def only_block_type(btype):
        return lambda k: classify_layer(k) == btype
    def only_block_range(lo, hi):
        # lo <= block idx < hi
        def f(k):
            b = layer_block_index(k)
            return b is not None and lo <= b < hi
        return f
    def not_block_type(btype):
        return lambda k: classify_layer(k) != btype

    ablations = [
        ("all_layers", all_layers, args.keep_frac),
        ("attn_qkv_only", only_block_type("attn_qkv"), args.keep_frac),
        ("attn_out_only", only_block_type("attn_out"), args.keep_frac),
        ("mlp_in_only", only_block_type("mlp_in"), args.keep_frac),
        ("mlp_out_only", only_block_type("mlp_out"), args.keep_frac),
        ("attn_only", lambda k: classify_layer(k) in ("attn_qkv", "attn_out"), args.keep_frac),
        ("mlp_only", lambda k: classify_layer(k) in ("mlp_in", "mlp_out"), args.keep_frac),
        ("early_blocks_0_4", only_block_range(0, 4), args.keep_frac),
        ("middle_blocks_4_8", only_block_range(4, 8), args.keep_frac),
        ("late_blocks_8_12", only_block_range(8, 12), args.keep_frac),
    ]

    for name, filt, k in ablations:
        t0 = time.time()
        print(f"\n== {name} (k={k}) ==")
        merged = selective_merge(pre_cpu, task_sds_cpu, keep_frac=k, ridge=1e-4,
                                 layer_filter_fn=filt, use_sign=False,
                                 device=device)
        per_task, avg = eval_sd(merged, name)
        results[name] = {"keep_frac": k, "per_task": per_task, "avg": avg}
        print(f"  ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")

    # Print summary
    print("\n=== Summary (avg acc across 8 tasks) ===")
    for name, d in results.items():
        print(f"  {name:25s} avg={d['avg']*100:.2f}%")


if __name__ == "__main__":
    main()
