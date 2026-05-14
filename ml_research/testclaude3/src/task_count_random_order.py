"""Task-count scaling with random task orderings.

For each (T, ordering), merge the first T tasks of the given ordering, and
record mean accuracy across those T tasks. This sweep complements
`task_count_scaling.py` (which uses a single fixed ordering) by checking
whether the gap-grows-with-T finding is robust to the task subset chosen.

Strategy: 5 task orderings × 3 fine-tuning seeds × T ∈ {2..7}.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import permutations
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate_merge import build_eval_loaders, eval_state_on_all, load_artifacts
from src.merging import actmat, simple_average, task_arithmetic, tact_cov_only, ties_merge
from src.model import load_clip


TASKS = ["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out", default=str(ROOT / "results/task_count_random.json"))
    ap.add_argument("--n-orderings", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for random orderings (independent of fine-tuning seed)")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    rng = random.Random(args.seed)
    orderings = []
    orderings.append(("fixed", list(TASKS)))
    for i in range(args.n_orderings - 1):
        order = list(TASKS)
        rng.shuffle(order)
        orderings.append((f"random{i}", order))

    print(f"Loading on {device} from {args.ckpt_dir} ...", flush=True)
    theta0_all, _, _ = load_artifacts(TASKS, device, ckpt_subdir=args.ckpt_dir)
    clip_model, _, _ = load_clip(device=device)
    all_loaders = build_eval_loaders(TASKS, batch_size=args.eval_batch_size, num_workers=2)

    # Pre-load checkpoints keyed by task name (rather than position) so we can
    # reorder them.
    thetas_by_task = {}
    arts_by_task = {}
    theta0, thetas, arts = load_artifacts(TASKS, device, ckpt_subdir=args.ckpt_dir)
    for i, t in enumerate(TASKS):
        thetas_by_task[t] = thetas[i]
        arts_by_task[t] = arts[t]
    print("Loaded.", flush=True)

    REG_EPS = 1e-4
    results = {}
    for ord_name, order in orderings:
        print(f"\n### Ordering = {ord_name}: {order}", flush=True)
        results[ord_name] = {}
        for T in [2, 3, 4, 5, 6, 7]:
            tasks_T = order[:T]
            thetas_T = [thetas_by_task[t] for t in tasks_T]
            arts_T = {t: arts_by_task[t] for t in tasks_T}
            loaders_T = {t: all_loaders[t] for t in tasks_T}

            methods = {}
            merged = simple_average(theta0, thetas_T)
            methods["average"] = sum(eval_state_on_all(
                merged, arts_T, loaders_T, clip_model, device).values()) / T
            merged = task_arithmetic(theta0, thetas_T, alpha=0.3)
            methods["task_arithmetic"] = sum(eval_state_on_all(
                merged, arts_T, loaders_T, clip_model, device).values()) / T
            merged = ties_merge(theta0, thetas_T, keep_frac=0.2, alpha=1.0)
            methods["ties"] = sum(eval_state_on_all(
                merged, arts_T, loaders_T, clip_model, device).values()) / T
            merged = actmat(theta0, thetas_T, reg_eps=REG_EPS)
            methods["actmat"] = sum(eval_state_on_all(
                merged, arts_T, loaders_T, clip_model, device).values()) / T
            merged = tact_cov_only(theta0, thetas_T, keep_frac=0.5,
                                   use_sign=False, reg_eps=REG_EPS)
            methods["tact_cov_k0.5"] = sum(eval_state_on_all(
                merged, arts_T, loaders_T, clip_model, device).values()) / T

            print(f"  T={T} (tasks={tasks_T})", flush=True)
            for name, v in methods.items():
                print(f"    {name:20s}  {v*100:.2f}%", flush=True)
            results[ord_name][T] = {"tasks": tasks_T, "metrics": methods}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
