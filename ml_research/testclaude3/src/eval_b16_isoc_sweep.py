"""B/16-tuned alpha sweep for Iso-C and TSV (the two SVD-based baselines that
were sub-optimal in our B/16 architecture-transfer comparison with B/32-best
hyperparameters). Reports best alpha per method per seed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
import statistics

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate_merge import build_eval_loaders, eval_state_on_all, load_artifacts
from src.model import load_clip
from src.merging import iso_c, tsv


def run_seed(ckpt_dir, gpu, tasks):
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    print(f"\n=== {ckpt_dir} on {device} ===", flush=True)
    theta0, thetas, arts = load_artifacts(tasks, device, ckpt_subdir=ckpt_dir)
    clip_model, _, _ = load_clip(device=device)
    eval_loaders = build_eval_loaders(tasks, batch_size=128, num_workers=2)

    out = {}
    print("--- iso_c sweep ---", flush=True)
    best_iso = None
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        merged = iso_c(theta0, thetas, alpha=alpha)
        accs = eval_state_on_all(merged, arts, eval_loaders, clip_model, device)
        m = sum(accs.values()) / len(accs)
        print(f"  iso_c alpha={alpha} mean={m*100:6.2f}", flush=True)
        if best_iso is None or m > best_iso[1]:
            best_iso = (alpha, m, accs)
    out["iso_c_best"] = {"alpha": best_iso[0], "mean": best_iso[1], "per_task": best_iso[2]}

    print("--- tsv sweep ---", flush=True)
    best_tsv = None
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        for r in [0.2, 0.5, 0.8]:
            merged = tsv(theta0, thetas, alpha=alpha, rank_keep=r)
            accs = eval_state_on_all(merged, arts, eval_loaders, clip_model, device)
            m = sum(accs.values()) / len(accs)
            print(f"  tsv alpha={alpha} rank={r} mean={m*100:6.2f}", flush=True)
            if best_tsv is None or m > best_tsv[1]:
                best_tsv = ((alpha, r), m, accs)
    out["tsv_best"] = {"alpha_rank": best_tsv[0], "mean": best_tsv[1], "per_task": best_tsv[2]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tasks = ["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"]
    os.makedirs(Path(args.out).parent, exist_ok=True)
    res = run_seed(args.ckpt_dir, args.gpu, tasks)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {args.out}")
    print(f"\n=== Best ===")
    for k, v in res.items():
        print(f"  {k}: alpha={v.get('alpha', v.get('alpha_rank'))} mean={v['mean']*100:.2f}%")


if __name__ == "__main__":
    main()
