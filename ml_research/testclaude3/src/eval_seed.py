"""Evaluate merging methods on a different-seed set of fine-tuned checkpoints.

Loads from `checkpoints_seed{seed}/` and runs ACTMat, TACT, and TACT-cov.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets_setup import build_task
from src.evaluate_merge import build_eval_loaders, eval_state_on_all
from src.merging import actmat, simple_average, tact, tact_cov_only, task_arithmetic, ties_merge
from src.model import load_clip


TASKS = ["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"]


def load_seeded(tasks, device, seed_dir):
    seed_dir = Path(seed_dir)
    pre = torch.load(seed_dir / "_pretrained_visual.pt", map_location=device, weights_only=True)
    theta0 = {k: v.to(device) for k, v in pre["visual_state_dict"].items()}
    thetas, arts = [], {}
    for t in tasks:
        a = torch.load(seed_dir / f"{t}.pt", map_location=device, weights_only=True)
        thetas.append({k: v.to(device) for k, v in a["visual_state_dict"].items()})
        arts[t] = {
            "text_features": a["text_features"].to(device),
            "classnames": a["classnames"],
            "prompt_template": a["prompt_template"],
            "zero_shot_acc": float(a["zero_shot_acc"]),
            "finetuned_acc": float(a["finetuned_acc"]),
        }
    return theta0, thetas, arts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seed_dir = ROOT / f"checkpoints_seed{args.seed}"
    out = args.out or str(ROOT / f"results/seed{args.seed}.json")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f"Loading from {seed_dir} on {device} ...")
    theta0, thetas, arts = load_seeded(TASKS, device, seed_dir)
    clip_model, _, _ = load_clip(device=device)
    loaders = build_eval_loaders(TASKS, batch_size=256, num_workers=2)

    results = {
        "seed": args.seed,
        "individual": {t: {"zero_shot": arts[t]["zero_shot_acc"],
                           "finetuned": arts[t]["finetuned_acc"]} for t in TASKS},
        "merged": {},
    }

    methods = {
        "average": (simple_average, {}),
        "task_arithmetic": (task_arithmetic, {"alpha": 0.3}),
        "ties": (ties_merge, {"keep_frac": 0.2, "alpha": 1.0}),
        "actmat": (actmat, {}),
        "tact_cov_k0.5": (tact_cov_only, {"keep_frac": 0.5, "use_sign": False}),
        "tact_full_k0.5": (tact, {"keep_frac": 0.5, "use_sign": False, "alpha": 1.0}),
    }
    for name, (fn, kw) in methods.items():
        t0 = time.time()
        merged = fn(theta0, thetas, **kw)
        accs = eval_state_on_all(merged, arts, loaders, clip_model, device)
        mean_acc = sum(accs.values()) / len(accs)
        results["merged"][name] = {"per_task": accs, "mean": mean_acc,
                                    "time_s": time.time() - t0}
        print(f"  {name:20s}  mean={mean_acc*100:.2f}%")

    Path(out).parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
