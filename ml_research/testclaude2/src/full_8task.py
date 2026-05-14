"""Full 8-task evaluation: all baselines and the headline TRIM configurations.

Designed to be fast on GPU: each method runs once at its best hyperparameter,
no sweeps (since we already know the best values from 7-task runs).
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_task
from src.models import ClipClassifier, build_processor, build_vision_encoder
from src.merge import (weight_average, task_arithmetic, ties, actmat,
                       trim_actmat, regmean, iso_c)
from src.train_expert import make_collate, evaluate
from src.eval_and_merge import compute_activation_covariances


def vision_pretrained():
    enc = build_vision_encoder()
    return {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}


def evaluate_merged(merged, experts, processor, device, bs=256, max_eval=4000):
    out = {}
    for task, ckpt in experts.items():
        m = ClipClassifier(ckpt["num_classes"]).to(device)
        m.vision.load_state_dict({k: v.to(device) for k, v in merged.items()})
        m.head.load_state_dict({k: v.to(device) for k, v in ckpt["head_state_dict"].items()})
        _, val_ds, _, _ = load_task(task, "./data")
        if len(val_ds) > max_eval:
            g = torch.Generator().manual_seed(1)
            idx = torch.randperm(len(val_ds), generator=g)[:max_eval].tolist()
            val_ds = Subset(val_ds, idx)
        loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2,
                            collate_fn=make_collate(processor), pin_memory=True)
        out[task] = evaluate(m, loader, device)
        del m
        torch.cuda.empty_cache()
    return out


def main():
    tasks = sorted([f[:-3] for f in os.listdir("./checkpoints") if f.endswith(".pt")])
    print("tasks:", tasks, flush=True)
    experts = {t: torch.load(f"./checkpoints/{t}.pt", map_location="cpu", weights_only=False)
               for t in tasks}
    expert_sds = [experts[t]["vision_state_dict"] for t in tasks]
    expert_accs = {t: experts[t]["val_acc"] for t in tasks}
    base = vision_pretrained()
    processor = build_processor()
    dev = torch.device("cuda:0")

    method_results = {}
    method_meta = {}

    print("=== pretrained ===", flush=True)
    method_results["pretrained"] = evaluate_merged(base, experts, processor, dev)
    print("=== weight_average ===", flush=True)
    method_results["weight_average"] = evaluate_merged(
        weight_average(base, expert_sds), experts, processor, dev)
    print("=== task_arithmetic (alpha=0.3) ===", flush=True)
    method_results["task_arithmetic"] = evaluate_merged(
        task_arithmetic(base, expert_sds, alpha=0.3), experts, processor, dev)
    method_meta["task_arithmetic"] = {"alpha": 0.3}
    print("=== TIES (alpha=0.8 density=0.2) ===", flush=True)
    method_results["ties"] = evaluate_merged(
        ties(base, expert_sds, alpha=0.8, density=0.2), experts, processor, dev)
    method_meta["ties"] = {"alpha": 0.8, "density": 0.2}
    print("=== ACTMat ===", flush=True)
    method_results["actmat"] = evaluate_merged(
        actmat(base, expert_sds, device=dev), experts, processor, dev)
    print("=== Trim-Mat density=0.5 ===", flush=True)
    method_results["trim_actmat"] = evaluate_merged(
        trim_actmat(base, expert_sds, density=0.5, sign_resolve=False, trim=True, device=dev),
        experts, processor, dev)
    method_meta["trim_actmat"] = {"density": 0.5}
    print("=== ACTMat + sign-elect ===", flush=True)
    method_results["trim_actmat__sign_only"] = evaluate_merged(
        trim_actmat(base, expert_sds, density=1.0, sign_resolve=True, trim=False, device=dev),
        experts, processor, dev)
    print("=== ACTMat + trim + sign ===", flush=True)
    method_results["trim_actmat__full_d0.5"] = evaluate_merged(
        trim_actmat(base, expert_sds, density=0.5, sign_resolve=True, trim=True, device=dev),
        experts, processor, dev)
    print("=== ACTMat + random-trim ===", flush=True)
    method_results["random_trim_d0.5_s0"] = evaluate_merged(
        trim_actmat(base, expert_sds, density=0.5, sign_resolve=False, trim=True,
                    random_trim=True, seed=0, device=dev),
        experts, processor, dev)

    # RegMean
    print("=== RegMean (data oracle) ===", flush=True)
    cov_per_task = defaultdict(list)
    for t in tasks:
        print(f"  covariance for {t} ...", flush=True)
        tr, _, _, _ = load_task(t, "./data")
        ct = compute_activation_covariances(
            experts[t]["vision_state_dict"], tr, processor, dev,
            batch_size=64, max_samples=1024)
        for k, v in ct.items():
            cov_per_task[k].append(v)
    method_results["regmean"] = evaluate_merged(
        regmean(base, expert_sds, dict(cov_per_task), device=dev),
        experts, processor, dev)

    # means
    method_means = {m: sum(r.values()) / len(r) for m, r in method_results.items()}

    # density sweep for Trim-Mat
    print("=== Trim-Mat density sweep ===", flush=True)
    for d in [0.2, 0.3, 0.7, 0.9]:
        merged = trim_actmat(base, expert_sds, density=d, sign_resolve=False, trim=True, device=dev)
        res = evaluate_merged(merged, experts, processor, dev)
        m = sum(res.values()) / len(res)
        key = f"trim_actmat__trim_only_d{d}"
        method_results[key] = res
        method_means[key] = m
        method_meta[key] = {"density": d}
        print(f"  d={d}: {m:.4f}", flush=True)

    payload = {
        "tasks": tasks,
        "expert_accs": expert_accs,
        "method_results": method_results,
        "method_meta": method_meta,
        "method_means": method_means,
    }
    with open("results/results_8task.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("=== SUMMARY ===", flush=True)
    for m, mean in sorted(method_means.items(), key=lambda x: -x[1]):
        print(f"{m:35s} {mean*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
