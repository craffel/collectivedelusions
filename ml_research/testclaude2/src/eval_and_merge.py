"""Run merging methods and evaluate them on all tasks.

Loads expert checkpoints from ``checkpoints/<task>.pt``, fetches the pretrained
CLIP-ViT-B/32 vision encoder as the base, then for each merging method:
  1. Produces a merged vision state dict.
  2. For every task: loads the task's classification head, plugs in the merged
     vision encoder, and evaluates top-1 accuracy on the held-out set.

Optionally also computes the empirical activation covariance C_t per linear
layer for RegMean (data oracle). To save compute we run this on a small subset
of each task's train set.

Writes ``results.json`` and ``results.csv`` with one row per (method, task).
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_task
from src.models import (
    ClipClassifier,
    build_processor,
    build_vision_encoder,
    vision_state_dict,
)
from src.merge import (
    weight_average,
    task_arithmetic,
    ties,
    iso_c,
    actmat,
    trim_actmat,
    regmean,
)
from src.train_expert import make_collate, evaluate


# ----------------- helpers ------------------------------------------------

def vision_pretrained_state_dict() -> Dict[str, torch.Tensor]:
    enc = build_vision_encoder()
    return {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}


def load_experts(ckpt_dir: str, tasks: List[str]) -> Dict[str, dict]:
    out = {}
    for t in tasks:
        p = os.path.join(ckpt_dir, f"{t}.pt")
        out[t] = torch.load(p, map_location="cpu", weights_only=False)
    return out


def linear_keys(sd: Dict[str, torch.Tensor]) -> List[str]:
    return [k for k, v in sd.items() if v.ndim == 2 and k.endswith(".weight")]


# ----------------- empirical activation covariance ------------------------

def compute_activation_covariances(
    vision_state_dict_dict: Dict[str, torch.Tensor],
    train_subset: torch.utils.data.Dataset,
    processor,
    device: torch.device,
    batch_size: int = 64,
    max_samples: int = 1024,
) -> Dict[str, torch.Tensor]:
    """For every nn.Linear module inside the vision encoder, compute
    C = (1/N) Σ z z^T where z is the *input* to that linear layer."""
    enc = build_vision_encoder().to(device)
    enc.load_state_dict(vision_state_dict_dict)
    enc.eval()

    # Find every nn.Linear; map module -> its weight key in state_dict (e.g. 'vision_model.encoder.layers.0.self_attn.q_proj.weight')
    # We use named_modules to record weight_name -> module ref.
    mod_to_key = {}
    for name, m in enc.named_modules():
        if isinstance(m, nn.Linear):
            mod_to_key[m] = f"{name}.weight"

    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = defaultdict(int)

    hooks = []

    def make_hook(key):
        def hook(_mod, inputs, _output):
            x = inputs[0]
            # x has shape (..., Di). Flatten leading dims.
            xf = x.reshape(-1, x.shape[-1]).to(torch.float32)
            cov = xf.transpose(0, 1) @ xf  # (Di, Di)
            if key not in sums:
                sums[key] = cov.detach()
            else:
                sums[key] += cov.detach()
            counts[key] += xf.shape[0]
        return hook

    for m, key in mod_to_key.items():
        hooks.append(m.register_forward_hook(make_hook(key)))

    collate = make_collate(processor)
    if len(train_subset) > max_samples:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(train_subset), generator=g)[:max_samples].tolist()
        train_subset = Subset(train_subset, idx)
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate, pin_memory=True)
    with torch.no_grad():
        for pv, _ in loader:
            pv = pv.to(device, non_blocking=True)
            # use float32 to avoid Inf/NaN from float16 overflow when squaring activations
            enc(pixel_values=pv)
    for h in hooks:
        h.remove()

    out = {}
    for key, s in sums.items():
        n = counts[key]
        c = (s / max(n, 1)).cpu().to(torch.float32)
        # sanitize any remaining inf / nan (shouldn't be any with fp32)
        c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        out[key] = c
    return out


# ----------------- evaluation harness -------------------------------------

def evaluate_merged_on_all_tasks(
    merged_vision_sd: Dict[str, torch.Tensor],
    experts: Dict[str, dict],
    processor,
    device: torch.device,
    bs: int = 256,
    num_workers: int = 4,
    max_eval_samples: int = 4000,
    data_root: str = "./data",
) -> Dict[str, float]:
    out = {}
    for task, ckpt in experts.items():
        num_classes = ckpt["num_classes"]
        model = ClipClassifier(num_classes).to(device)
        model.vision.load_state_dict({k: v.to(device) for k, v in merged_vision_sd.items()})
        model.head.load_state_dict({k: v.to(device) for k, v in ckpt["head_state_dict"].items()})
        _, val_ds, _, _ = load_task(task, data_root)
        if len(val_ds) > max_eval_samples:
            g = torch.Generator().manual_seed(1)
            idx = torch.randperm(len(val_ds), generator=g)[:max_eval_samples].tolist()
            val_ds = Subset(val_ds, idx)
        loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=num_workers, collate_fn=make_collate(processor),
                            pin_memory=True)
        acc = evaluate(model, loader, device)
        out[task] = acc
        del model
        torch.cuda.empty_cache()
    return out


# ----------------- main ---------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="./checkpoints")
    ap.add_argument("--tasks", nargs="+", required=False)
    ap.add_argument("--out", default="./results")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max_eval_samples", type=int, default=4000)
    ap.add_argument("--cov_samples", type=int, default=1024)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--skip_regmean", action="store_true",
                    help="Skip the data-oracle RegMean baseline (saves time).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    # Discover tasks
    if not args.tasks:
        args.tasks = sorted([f[:-3] for f in os.listdir(args.ckpt_dir) if f.endswith(".pt")])
    print("tasks:", args.tasks)

    experts = load_experts(args.ckpt_dir, args.tasks)
    base = vision_pretrained_state_dict()
    expert_sds = [experts[t]["vision_state_dict"] for t in args.tasks]
    expert_accs = {t: experts[t]["val_acc"] for t in args.tasks}
    print("expert_accs:", expert_accs)

    # Hyperparameters for each method's scaling coefficient.
    # We tune α via a small sweep over {0.1, 0.2, 0.3, 0.4, 0.5} per method below.
    alpha_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    ties_alpha_grid = [0.5, 0.8, 1.0, 1.2]
    ties_density_grid = [0.2]
    trim_density_grid = [0.1, 0.2, 0.3, 0.5]

    method_results: Dict[str, Dict[str, float]] = {}
    method_meta: Dict[str, Dict] = {}
    processor = build_processor()

    # 1. Pretrained zero-shot baseline (vision encoder = W_0, task heads attached)
    print("\n=== pretrained (zero-shot, with task heads) ===")
    res = evaluate_merged_on_all_tasks(base, experts, processor, device,
                                       bs=args.bs, max_eval_samples=args.max_eval_samples,
                                       data_root=args.data_root)
    method_results["pretrained"] = res
    print("pretrained:", res)

    # 2. Weight averaging
    print("\n=== weight_average ===")
    merged = weight_average(base, expert_sds)
    res = evaluate_merged_on_all_tasks(merged, experts, processor, device,
                                       bs=args.bs, max_eval_samples=args.max_eval_samples,
                                       data_root=args.data_root)
    method_results["weight_average"] = res
    print("weight_average:", res)

    # 3. Task arithmetic: sweep α
    print("\n=== task_arithmetic (alpha sweep) ===")
    best = (-1.0, None, None, None)
    for a in alpha_grid:
        merged = task_arithmetic(base, expert_sds, alpha=a)
        res = evaluate_merged_on_all_tasks(merged, experts, processor, device,
                                           bs=args.bs, max_eval_samples=args.max_eval_samples,
                                           data_root=args.data_root)
        mean = sum(res.values()) / len(res)
        print(f"  alpha={a} mean={mean:.4f} res={res}")
        if mean > best[0]:
            best = (mean, a, res, None)
    method_results["task_arithmetic"] = best[2]
    method_meta["task_arithmetic"] = {"alpha": best[1]}

    # 4. TIES: sweep α and density
    print("\n=== ties (alpha, density sweep) ===")
    best = (-1.0, None, None, None, None)
    for a in ties_alpha_grid:
        for d in ties_density_grid:
            merged = ties(base, expert_sds, alpha=a, density=d)
            res = evaluate_merged_on_all_tasks(merged, experts, processor, device,
                                               bs=args.bs, max_eval_samples=args.max_eval_samples,
                                               data_root=args.data_root)
            mean = sum(res.values()) / len(res)
            print(f"  alpha={a} density={d} mean={mean:.4f}")
            if mean > best[0]:
                best = (mean, a, d, res, None)
    method_results["ties"] = best[3]
    method_meta["ties"] = {"alpha": best[1], "density": best[2]}

    # 5. Iso-C
    print("\n=== iso_c (alpha sweep) ===")
    best = (-1.0, None, None)
    for a in alpha_grid:
        merged = iso_c(base, expert_sds, alpha=a)
        res = evaluate_merged_on_all_tasks(merged, experts, processor, device,
                                           bs=args.bs, max_eval_samples=args.max_eval_samples,
                                           data_root=args.data_root)
        mean = sum(res.values()) / len(res)
        print(f"  alpha={a} mean={mean:.4f}")
        if mean > best[0]:
            best = (mean, a, res)
    method_results["iso_c"] = best[2]
    method_meta["iso_c"] = {"alpha": best[1]}

    # 6. ACTMat (data-free covariance)
    print("\n=== actmat ===")
    merged = actmat(base, expert_sds)
    res = evaluate_merged_on_all_tasks(merged, experts, processor, device,
                                       bs=args.bs, max_eval_samples=args.max_eval_samples,
                                       data_root=args.data_root)
    method_results["actmat"] = res
    print("actmat:", res)

    # 7. TRIM (ours): density sweep
    print("\n=== TRIM-ACTMat (ours, density sweep) ===")
    best = (-1.0, None, None)
    for d in trim_density_grid:
        merged = trim_actmat(base, expert_sds, density=d, sign_resolve=True, trim=True)
        res = evaluate_merged_on_all_tasks(merged, experts, processor, device,
                                           bs=args.bs, max_eval_samples=args.max_eval_samples,
                                           data_root=args.data_root)
        mean = sum(res.values()) / len(res)
        print(f"  density={d} mean={mean:.4f}")
        if mean > best[0]:
            best = (mean, d, res)
    method_results["trim_actmat"] = best[2]
    method_meta["trim_actmat"] = {"density": best[1]}

    # 7b. Ablations: trim-only and sign-only with TRIM's best density
    best_d = method_meta["trim_actmat"]["density"]
    print(f"\n=== TRIM ablations at density={best_d} ===")
    merged = trim_actmat(base, expert_sds, density=best_d, sign_resolve=False, trim=True)
    method_results["trim_actmat__trim_only"] = evaluate_merged_on_all_tasks(
        merged, experts, processor, device,
        bs=args.bs, max_eval_samples=args.max_eval_samples, data_root=args.data_root)
    merged = trim_actmat(base, expert_sds, density=best_d, sign_resolve=True, trim=False)
    method_results["trim_actmat__sign_only"] = evaluate_merged_on_all_tasks(
        merged, experts, processor, device,
        bs=args.bs, max_eval_samples=args.max_eval_samples, data_root=args.data_root)
    print("trim_only:", method_results["trim_actmat__trim_only"])
    print("sign_only:", method_results["trim_actmat__sign_only"])

    # 8. RegMean (data oracle) -- compute activation covariances per task
    if not args.skip_regmean:
        print("\n=== RegMean (data oracle) ===")
        cov_per_task_per_layer: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for t in args.tasks:
            print(f"  computing C_t for {t} ...")
            train_ds, _, _, _ = load_task(t, args.data_root)
            ct = compute_activation_covariances(
                experts[t]["vision_state_dict"], train_ds, processor, device,
                batch_size=64, max_samples=args.cov_samples,
            )
            for k, v in ct.items():
                cov_per_task_per_layer[k].append(v)
        # Sanity: same key set as linear keys in base.
        lk = set(linear_keys(base))
        cov_keys = set(cov_per_task_per_layer.keys())
        print(f"  linear keys in base: {len(lk)}, cov keys: {len(cov_keys)}")
        merged = regmean(base, expert_sds, dict(cov_per_task_per_layer))
        res = evaluate_merged_on_all_tasks(merged, experts, processor, device,
                                           bs=args.bs, max_eval_samples=args.max_eval_samples,
                                           data_root=args.data_root)
        method_results["regmean"] = res

    # ----------------- write out --------------------------------------------
    print("\n=== summary ===")
    method_means = {}
    for m, res in method_results.items():
        mean = sum(res.values()) / len(res)
        method_means[m] = mean
        print(f"{m:25s} mean={mean:.4f}  {res}")

    payload = {
        "tasks": args.tasks,
        "expert_accs": expert_accs,
        "method_results": method_results,
        "method_meta": method_meta,
        "method_means": method_means,
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    # CSV: rows are methods, cols are tasks + mean
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + args.tasks + ["mean"])
        for m, res in method_results.items():
            row = [m] + [f"{res[t]:.4f}" for t in args.tasks] + [f"{method_means[m]:.4f}"]
            w.writerow(row)
    print(f"wrote {args.out}/results.json and results.csv")


if __name__ == "__main__":
    main()
