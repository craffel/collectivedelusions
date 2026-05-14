"""Generality of magnitude-trim across data-free merge methods.

Tests whether TIES-style magnitude trimming, applied to each Delta_t before
*any* downstream data-free merge operation, is generally helpful, or whether
its benefit is specific to ACTMat's covariance-based solve.

Methods tested (all data-free):
  - Task Arithmetic
  - Iso-C
  - TSV
  - ACTMat

Each method is run twice:
  (a) baseline: original Delta_t = theta_t - theta_0
  (b) trimmed: Delta_t' = trim(Delta_t, keep=k) for k in {0.3, 0.5}

Result is the mean test accuracy across all 7 tasks.
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
from src.evaluate_merge import (
    build_eval_loaders, eval_state_on_all, load_artifacts,
)
from src.finetune import evaluate
from src.model import (
    CLIPVisualClassifier, get_visual_state_dict, load_clip, set_visual_state_dict,
)
from src.merging import (
    _trim_topk, actmat, iso_c, task_arithmetic, tsv, task_vectors,
)


TASKS = ["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"]


def apply_trim_to_thetas(theta0, thetas, keep_frac):
    """Build new 'thetas' whose Delta_t is the trimmed task vector.

    Returns thetas_trimmed s.t. theta_t' = theta_0 + trim(theta_t - theta_0, keep_frac).
    Downstream methods that compute Delta_t = theta_t - theta_0 internally will
    then see the trimmed Delta_t.
    """
    taus = task_vectors(theta0, thetas)
    trimmed = [_trim_topk(t, keep_frac) for t in taus]
    return [{k: theta0[k] + trimmed[i][k] for k in theta0} for i in range(len(thetas))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt-subdir", default="checkpoints")
    ap.add_argument("--out", default=str(ROOT / "results/trim_other_methods.json"))
    ap.add_argument("--keep-list", nargs="+", type=float, default=[0.3, 0.5])
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f"Loading artifacts from {args.ckpt_subdir} ...", flush=True)
    theta0, thetas, arts = load_artifacts(TASKS, device, ckpt_subdir=args.ckpt_subdir)

    print("Loading CLIP and eval loaders ...", flush=True)
    arch = "ViT-B-16" if "b16" in args.ckpt_subdir else "ViT-B-32"
    import os
    os.environ["CLIP_ARCH"] = arch
    clip_model, _, _ = load_clip(device=device)
    eval_loaders = build_eval_loaders(TASKS, batch_size=128, num_workers=4, max_test=4000)

    results = {"keep_list": args.keep_list, "methods": {}}

    method_configs = [
        ("task_arithmetic", lambda t0, ts: task_arithmetic(t0, ts, alpha=0.3)),
        ("iso_c",          lambda t0, ts: iso_c(t0, ts, alpha=1.0)),
        ("tsv",            lambda t0, ts: tsv(t0, ts, alpha=0.3, rank_keep=0.5)),
        ("actmat",         lambda t0, ts: actmat(t0, ts, reg_eps=1e-8)),
    ]

    for name, fn in method_configs:
        print(f"\n=== {name} ===", flush=True)
        results["methods"][name] = {}

        # Baseline (no trim)
        merged = fn(theta0, thetas)
        accs = eval_state_on_all(merged, arts, eval_loaders, clip_model, device)
        mean = sum(accs.values()) / len(accs)
        print(f"  baseline (no trim): mean = {mean*100:.2f}%", flush=True)
        results["methods"][name]["baseline"] = {"mean": mean, "per_task": accs}

        for keep in args.keep_list:
            thetas_trimmed = apply_trim_to_thetas(theta0, thetas, keep)
            merged = fn(theta0, thetas_trimmed)
            accs = eval_state_on_all(merged, arts, eval_loaders, clip_model, device)
            mean = sum(accs.values()) / len(accs)
            print(f"  trim k={keep}: mean = {mean*100:.2f}%", flush=True)
            results["methods"][name][f"trim_{keep}"] = {"mean": mean, "per_task": accs}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
