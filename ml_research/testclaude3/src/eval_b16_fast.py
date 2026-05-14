"""Fast B/16 eval: only the headline methods at the best hyperparameters
already identified on B/32, avoiding the full hyperparameter sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate_merge import build_eval_loaders, eval_state_on_all, load_artifacts
from src.model import CLIPVisualClassifier, load_clip
from src.merging import (
    actmat, iso_c, simple_average, tact, task_arithmetic, ties_merge, tsv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints_b16")
    ap.add_argument("--out", default="results/b16_fast.json")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--tasks", nargs="+",
                    default=["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"])
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    os.makedirs(Path(args.out).parent, exist_ok=True)

    print("Loading B/16 artifacts...", flush=True)
    theta0, thetas, arts = load_artifacts(args.tasks, device, ckpt_subdir=args.ckpt_dir)
    print("Loading CLIP B/16 architecture...", flush=True)
    clip_model, _, _ = load_clip(device=device)
    print("Building eval loaders...", flush=True)
    eval_loaders = build_eval_loaders(args.tasks, batch_size=128, num_workers=2)

    results = {
        "tasks": args.tasks,
        "individual": {t: {"zero_shot": float(arts[t]["zero_shot_acc"]),
                           "finetuned": float(arts[t]["finetuned_acc"])}
                       for t in args.tasks},
        "merged": {},
    }

    def run(name, theta_merged):
        t0 = time.time()
        accs = eval_state_on_all(theta_merged, arts, eval_loaders, clip_model, device)
        m = sum(accs.values()) / len(accs)
        results["merged"][name] = {"per_task": accs, "mean": m,
                                   "time_s": time.time() - t0}
        print(f"  {name:20s}  mean={m*100:6.2f}%  ({time.time()-t0:.1f}s)", flush=True)

    print("\n=== Running headline methods on B/16 ===\n")
    run("average", simple_average(theta0, thetas))
    run("task_arithmetic_a0.3", task_arithmetic(theta0, thetas, alpha=0.3))
    run("ties_k0.2_a1.0", ties_merge(theta0, thetas, keep_frac=0.2, alpha=1.0))
    run("iso_c_a0.3", iso_c(theta0, thetas, alpha=0.3))
    run("tsv_a0.3_r0.5", tsv(theta0, thetas, alpha=0.3, rank_keep=0.5))
    run("actmat", actmat(theta0, thetas))
    run("tact_k0.5_signoff", tact(theta0, thetas, keep_frac=0.5, use_sign=False, alpha=1.0))
    run("tact_k0.3_signoff", tact(theta0, thetas, keep_frac=0.3, use_sign=False, alpha=1.0))
    run("tact_k0.7_signoff", tact(theta0, thetas, keep_frac=0.7, use_sign=False, alpha=1.0))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
