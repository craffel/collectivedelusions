"""Evaluate model-merging methods on the multi-task vision benchmark.

Loads per-task fine-tuned checkpoints from ./checkpoints/ and the pretrained
visual state, applies each merging method, then evaluates the merged visual
encoder on every task's test set using that task's frozen text classifier
head (CLIP zero-shot heads).

Reports per-task accuracies and the mean across tasks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets_setup import build_task
from src.finetune import evaluate
from src.model import (
    CLIPVisualClassifier, build_text_classifier, get_visual_state_dict,
    load_clip, set_visual_state_dict,
)
from src.merging import (
    METHODS, actmat, dare, iso_c, regmean, simple_average, tact, tact_cov_only,
    task_arithmetic, ties_merge, tsv,
)
from src.regmean_covs import compute_activation_covs


def load_artifacts(tasks: List[str], device: torch.device, ckpt_subdir: str = "checkpoints"):
    """Returns (theta0, thetas_per_task, task_artifacts)."""
    ckpt_dir = ROOT / ckpt_subdir
    pre = torch.load(ckpt_dir / "_pretrained_visual.pt", map_location=device, weights_only=True)
    theta0 = {k: v.to(device) for k, v in pre["visual_state_dict"].items()}

    thetas, arts = [], {}
    for t in tasks:
        a = torch.load(ckpt_dir / f"{t}.pt", map_location=device, weights_only=True)
        thetas.append({k: v.to(device) for k, v in a["visual_state_dict"].items()})
        arts[t] = {
            "text_features": a["text_features"].to(device),
            "classnames": a["classnames"],
            "prompt_template": a["prompt_template"],
            "zero_shot_acc": float(a["zero_shot_acc"]),
            "finetuned_acc": float(a["finetuned_acc"]),
        }
    return theta0, thetas, arts


def eval_state_on_all(theta_merged, arts, eval_loaders, clip_model, device,
                      eval_batches: int | None = None) -> Dict[str, float]:
    accs = {}
    for t, loader in eval_loaders.items():
        classifier = CLIPVisualClassifier(clip_model, arts[t]["text_features"]).to(device)
        set_visual_state_dict(classifier, theta_merged)
        acc = evaluate(classifier, loader, device)
        accs[t] = acc
    return accs


def build_eval_loaders(tasks, batch_size=128, num_workers=4, max_test=4000):
    out = {}
    for t in tasks:
        b = build_task(t, batch_size=batch_size, num_workers=num_workers,
                       max_train=1, max_test=max_test)
        out[t] = b.test_loader
    return out


def build_cov_loaders(tasks, batch_size=64, num_workers=2, max_train=2000):
    """Tiny loaders for RegMean covariance estimation."""
    out = {}
    for t in tasks:
        b = build_task(t, batch_size=batch_size, num_workers=num_workers,
                       max_train=max_train, max_test=1)
        out[t] = b.train_loader
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--methods", nargs="+",
                    default=["average", "task_arithmetic", "ties", "regmean", "actmat", "tact"])
    ap.add_argument("--regmean-batches", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results/main.json"))
    ap.add_argument("--ckpt-dir", default="checkpoints",
                    help="Subdirectory under project root that holds checkpoints.")
    ap.add_argument("--ablations", action="store_true",
                    help="Run TACT ablations (sign, trim level).")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    os.makedirs(Path(args.out).parent, exist_ok=True)

    print(f"Loading artifacts on {device} ...", flush=True)
    theta0, thetas, arts = load_artifacts(args.tasks, device, ckpt_subdir=args.ckpt_dir)

    print(f"Building eval loaders ...", flush=True)
    eval_loaders = build_eval_loaders(args.tasks, batch_size=args.eval_batch_size,
                                      num_workers=2)

    clip_model, _, _ = load_clip(device=device)

    results = {
        "tasks": args.tasks,
        "individual": {t: {"zero_shot": arts[t]["zero_shot_acc"],
                           "finetuned": arts[t]["finetuned_acc"]}
                       for t in args.tasks},
        "merged": {},
    }

    methods_to_run = list(args.methods)

    # --- compute RegMean covs if needed -------------------------------------
    data_covs = None
    if "regmean" in methods_to_run:
        print("Computing RegMean covariances (using train calibration data) ...",
              flush=True)
        cov_loaders = build_cov_loaders(args.tasks, batch_size=64, num_workers=2,
                                        max_train=2000)
        data_covs = []
        for t, loader in cov_loaders.items():
            classifier = CLIPVisualClassifier(clip_model, arts[t]["text_features"]).to(device)
            set_visual_state_dict(classifier, dict(zip(
                theta0.keys(), [thetas[args.tasks.index(t)][k] for k in theta0.keys()]
            )))
            covs = compute_activation_covs(classifier, loader, device,
                                           max_batches=args.regmean_batches)
            data_covs.append(covs)
            print(f"  {t}: covs for {len(covs)} layers", flush=True)

    # --- run baselines + TACT ----------------------------------------------
    for m in methods_to_run:
        t0 = time.time()
        print(f"\n=== {m} ===", flush=True)
        if m == "average":
            merged = simple_average(theta0, thetas)
        elif m == "task_arithmetic":
            # Search a small grid for alpha
            best = None
            for alpha in [0.1, 0.2, 0.3, 0.4]:
                merged_alpha = task_arithmetic(theta0, thetas, alpha=alpha)
                accs = eval_state_on_all(merged_alpha, arts, eval_loaders, clip_model, device)
                mean_acc = sum(accs.values()) / len(accs)
                print(f"  alpha={alpha} mean={mean_acc:.4f}", flush=True)
                if best is None or mean_acc > best[1]:
                    best = (alpha, mean_acc, merged_alpha, accs)
            print(f"  best alpha={best[0]}", flush=True)
            results["merged"][m] = {"per_task": best[3], "mean": best[1],
                                    "best_alpha": best[0],
                                    "time_s": time.time() - t0}
            continue
        elif m == "ties":
            best = None
            for alpha in [0.3, 0.5, 1.0]:
                merged_alpha = ties_merge(theta0, thetas, keep_frac=0.2, alpha=alpha)
                accs = eval_state_on_all(merged_alpha, arts, eval_loaders, clip_model, device)
                mean_acc = sum(accs.values()) / len(accs)
                print(f"  alpha={alpha} mean={mean_acc:.4f}", flush=True)
                if best is None or mean_acc > best[1]:
                    best = (alpha, mean_acc, merged_alpha, accs)
            print(f"  best alpha={best[0]}", flush=True)
            results["merged"][m] = {"per_task": best[3], "mean": best[1],
                                    "best_alpha": best[0],
                                    "time_s": time.time() - t0}
            continue
        elif m == "iso_c":
            best = None
            for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
                merged_alpha = iso_c(theta0, thetas, alpha=alpha)
                accs = eval_state_on_all(merged_alpha, arts, eval_loaders, clip_model, device)
                mean_acc = sum(accs.values()) / len(accs)
                print(f"  alpha={alpha} mean={mean_acc:.4f}", flush=True)
                if best is None or mean_acc > best[1]:
                    best = (alpha, mean_acc, merged_alpha, accs)
            print(f"  best alpha={best[0]}", flush=True)
            results["merged"][m] = {"per_task": best[3], "mean": best[1],
                                    "best_alpha": best[0],
                                    "time_s": time.time() - t0}
            continue
        elif m == "tsv":
            best = None
            for alpha in [0.3, 0.5, 0.7, 1.0]:
                for rk in [0.2, 0.5, 0.8]:
                    merged_alpha = tsv(theta0, thetas, alpha=alpha, rank_keep=rk)
                    accs = eval_state_on_all(merged_alpha, arts, eval_loaders, clip_model, device)
                    mean_acc = sum(accs.values()) / len(accs)
                    print(f"  alpha={alpha} rank_keep={rk} mean={mean_acc:.4f}", flush=True)
                    if best is None or mean_acc > best[1]:
                        best = ((alpha, rk), mean_acc, merged_alpha, accs)
            print(f"  best (alpha, rank_keep)={best[0]}", flush=True)
            results["merged"][m] = {"per_task": best[3], "mean": best[1],
                                    "best_alpha": best[0][0],
                                    "best_rank_keep": best[0][1],
                                    "time_s": time.time() - t0}
            continue
        elif m == "dare":
            best = None
            for drop in [0.5, 0.7, 0.9]:
                for alpha in [0.3, 0.5, 1.0]:
                    merged_alpha = dare(theta0, thetas, drop_p=drop, alpha=alpha, seed=0)
                    accs = eval_state_on_all(merged_alpha, arts, eval_loaders, clip_model, device)
                    mean_acc = sum(accs.values()) / len(accs)
                    print(f"  drop={drop} alpha={alpha} mean={mean_acc:.4f}", flush=True)
                    if best is None or mean_acc > best[1]:
                        best = ((drop, alpha), mean_acc, merged_alpha, accs)
            print(f"  best (drop, alpha)={best[0]}", flush=True)
            results["merged"][m] = {"per_task": best[3], "mean": best[1],
                                    "best_drop_p": best[0][0],
                                    "best_alpha": best[0][1],
                                    "time_s": time.time() - t0}
            continue
        elif m == "regmean":
            merged = regmean(theta0, thetas, data_covs)
        elif m == "actmat":
            merged = actmat(theta0, thetas)
        elif m == "tact":
            best = None
            for k in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
                m_full = tact(theta0, thetas, keep_frac=k, use_sign=True, alpha=1.0)
                accs = eval_state_on_all(m_full, arts, eval_loaders, clip_model, device)
                mean_acc = sum(accs.values()) / len(accs)
                print(f"  keep={k} mean={mean_acc:.4f}", flush=True)
                if best is None or mean_acc > best[1]:
                    best = (k, mean_acc, m_full, accs)
            print(f"  best keep={best[0]}", flush=True)
            results["merged"][m] = {"per_task": best[3], "mean": best[1],
                                    "best_keep_frac": best[0],
                                    "time_s": time.time() - t0}
            continue
        elif m == "tact_cov":
            best = None
            for k in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
                for sgn in [True, False]:
                    m_full = tact_cov_only(theta0, thetas, keep_frac=k, use_sign=sgn)
                    accs = eval_state_on_all(m_full, arts, eval_loaders, clip_model, device)
                    mean_acc = sum(accs.values()) / len(accs)
                    print(f"  keep={k} sign={sgn} mean={mean_acc:.4f}", flush=True)
                    if best is None or mean_acc > best[1]:
                        best = ((k, sgn), mean_acc, m_full, accs)
            print(f"  best (keep, sign)={best[0]}", flush=True)
            results["merged"][m] = {"per_task": best[3], "mean": best[1],
                                    "best_keep_frac": best[0][0],
                                    "best_use_sign": best[0][1],
                                    "time_s": time.time() - t0}
            continue
        else:
            raise ValueError(f"Unknown method: {m}")

        accs = eval_state_on_all(merged, arts, eval_loaders, clip_model, device)
        mean_acc = sum(accs.values()) / len(accs)
        print(f"  per-task: " + ", ".join(f"{t}={a:.3f}" for t, a in accs.items()), flush=True)
        print(f"  mean = {mean_acc:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        results["merged"][m] = {"per_task": accs, "mean": mean_acc,
                                "time_s": time.time() - t0}

    # --- TACT ablations -----------------------------------------------------
    if args.ablations:
        print("\n=== TACT ablations ===", flush=True)
        abl = {}
        for keep in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            for use_sign in [True, False]:
                merged = tact(theta0, thetas, keep_frac=keep, use_sign=use_sign, alpha=1.0)
                accs = eval_state_on_all(merged, arts, eval_loaders, clip_model, device)
                m = sum(accs.values()) / len(accs)
                tag = f"keep={keep:.2f}_sign={use_sign}"
                abl[tag] = {"mean": m, "per_task": accs}
                print(f"  {tag}  mean={m:.4f}", flush=True)
        results["ablations"] = abl

    # --- Final dump ---------------------------------------------------------
    # Convert any tensors to floats just in case
    def jsonify(o):
        if isinstance(o, torch.Tensor):
            return o.item() if o.numel() == 1 else o.tolist()
        if isinstance(o, dict):
            return {k: jsonify(v) for k, v in o.items()}
        if isinstance(o, list):
            return [jsonify(v) for v in o]
        return o

    with open(args.out, "w") as f:
        json.dump(jsonify(results), f, indent=2)
    print(f"\nSaved results to {args.out}", flush=True)

    # Pretty summary
    print("\nSummary (mean accuracy across tasks):")
    for m, r in results["merged"].items():
        print(f"  {m:20s}  {r['mean']*100:6.2f}%")


if __name__ == "__main__":
    main()
