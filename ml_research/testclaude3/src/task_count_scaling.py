"""Evaluate merging methods as a function of number of merged tasks T.

For each T ∈ {2, 3, 4, 5, 6, 7}, merge the first T task-specific encoders
(in a fixed order) and report mean test accuracy across those T tasks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets_setup import build_task
from src.evaluate_merge import build_eval_loaders, eval_state_on_all, load_artifacts
from src.merging import (
    actmat, simple_average, tact, tact_cov_only, task_arithmetic, ties_merge,
)
from src.model import load_clip


TASK_ORDER = ["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--out", default=str(ROOT / "results/task_count_scaling.json"))
    ap.add_argument("--ckpt-dir", default="checkpoints")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f"Loading on {device} from {args.ckpt_dir} ...", flush=True)
    theta0_all, thetas_all, arts_all = load_artifacts(TASK_ORDER, device, ckpt_subdir=args.ckpt_dir)
    clip_model, _, _ = load_clip(device=device)
    print("Loaded.")

    all_loaders = build_eval_loaders(TASK_ORDER, batch_size=args.eval_batch_size, num_workers=2)

    results = {}
    for T in [2, 3, 4, 5, 6, 7]:
        tasks_T = TASK_ORDER[:T]
        print(f"\n=== T = {T}: tasks = {tasks_T} ===", flush=True)
        # Pick the first T thetas
        thetas_T = [thetas_all[i] for i in range(T)]
        arts_T = {t: arts_all[t] for t in tasks_T}
        loaders_T = {t: all_loaders[t] for t in tasks_T}
        theta0 = theta0_all

        # Use a moderate Tikhonov coefficient (1e-4) for ACTMat / TACT in the
        # task-count scaling experiment.  At low T (e.g. T=2), Σ_t Ĉ_t is
        # rank-deficient on the layers where d_i > d_o (e.g. the MLP
        # down-projections of ViT-B/32 where d_i = 3072, d_o = 768): only T = 4
        # tasks are sufficient to make Σ Ĉ_t full-rank. The headline 7-task
        # reg_eps = 1e-8 then collapses the inverse on small T. Using 1e-4
        # (matching the default in `_solve_regmean`) gives a well-defined
        # regularized solve at every T while reducing to the 7-task headline
        # in the full-rank regime.
        REG_EPS = 1e-4

        methods = {}
        # Simple average
        merged = simple_average(theta0, thetas_T)
        accs = eval_state_on_all(merged, arts_T, loaders_T, clip_model, device)
        methods["average"] = sum(accs.values()) / len(accs)
        # Task arithmetic at alpha=0.3
        merged = task_arithmetic(theta0, thetas_T, alpha=0.3)
        accs = eval_state_on_all(merged, arts_T, loaders_T, clip_model, device)
        methods["task_arithmetic"] = sum(accs.values()) / len(accs)
        # TIES at k=0.2, alpha=1.0
        merged = ties_merge(theta0, thetas_T, keep_frac=0.2, alpha=1.0)
        accs = eval_state_on_all(merged, arts_T, loaders_T, clip_model, device)
        methods["ties"] = sum(accs.values()) / len(accs)
        # ACTMat
        merged = actmat(theta0, thetas_T, reg_eps=REG_EPS)
        accs = eval_state_on_all(merged, arts_T, loaders_T, clip_model, device)
        methods["actmat"] = sum(accs.values()) / len(accs)
        # TACT-cov: best (keep=0.5, sign=False)  -- adjust based on ablations
        merged = tact_cov_only(theta0, thetas_T, keep_frac=0.5, use_sign=False,
                               reg_eps=REG_EPS)
        accs = eval_state_on_all(merged, arts_T, loaders_T, clip_model, device)
        methods["tact_cov_k0.5"] = sum(accs.values()) / len(accs)

        for name, m in methods.items():
            print(f"  {name:20s}  {m*100:.2f}%", flush=True)

        results[T] = methods

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
