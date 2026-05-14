"""Evaluate Fisher merging on the 7-task CLIP merging benchmark.

For each fine-tuned task encoder, computes diagonal Fisher F_t on a few
calibration batches of that task's training data, then runs Fisher merging
across all 7 tasks and evaluates per-task test accuracy.

Usage:
    python -m src.eval_fisher --gpu 0 --ckpt-dir checkpoints --out results/fisher_seed0.json
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
from src.fisher import compute_diagonal_fisher
from src.merging import fisher_merge
from src.model import (
    CLIPVisualClassifier,
    build_text_classifier,
    load_clip,
    set_visual_state_dict,
)


TASKS = ["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out", default=str(ROOT / "results/fisher.json"))
    ap.add_argument("--fisher-batches", type=int, default=4)
    ap.add_argument("--fisher-batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f"Loading on {device} from {args.ckpt_dir} ...", flush=True)
    theta0, thetas, arts = load_artifacts(TASKS, device, ckpt_subdir=args.ckpt_dir)
    clip_model, _, tokenizer = load_clip(device=device)

    # Build a CLIPVisualClassifier per task with the task's fine-tuned visual weights
    # and the task's text classifier. Compute Fisher on the task's training data.
    fishers = []
    for i, t in enumerate(TASKS):
        print(f"  [{i+1}/7] Fisher for {t} ...", flush=True)
        # Need a task's training loader.
        bundle = build_task(
            t, batch_size=args.fisher_batch_size, num_workers=2,
            max_train=args.fisher_batches * args.fisher_batch_size,
            max_test=1,
        )
        train_loader = bundle.train_loader
        text_feats = build_text_classifier(bundle.classnames, bundle.prompt_template,
                                            tokenizer, clip_model, device)
        # Build classifier on the fly using the task's fine-tuned visual weights
        m = CLIPVisualClassifier(clip_model, text_feats).to(device)
        set_visual_state_dict(m, thetas[i])
        # Freeze nothing; we want gradients w.r.t. all visual params
        for p in m.parameters():
            p.requires_grad_(True)
        fisher_t = compute_diagonal_fisher(m, train_loader, device,
                                           max_batches=args.fisher_batches)
        # Keep only visual.* keys (which is what merging operates on)
        fisher_t = {k: v for k, v in fisher_t.items() if k.startswith("visual.")}
        fishers.append(fisher_t)
        del m
        torch.cuda.empty_cache()

    print("\nRunning Fisher merge ...", flush=True)
    merged = fisher_merge(theta0, thetas, fishers)
    eval_loaders = build_eval_loaders(TASKS, batch_size=256, num_workers=2)
    accs = eval_state_on_all(merged, arts, eval_loaders, clip_model, device)
    mean = sum(accs.values()) / len(accs)
    print(f"\nFisher merge mean accuracy: {mean*100:.2f}%")
    for t, a in accs.items():
        print(f"  {t:10s} {a*100:.2f}%")

    with open(args.out, "w") as f:
        json.dump({"fisher": {"per_task": accs, "mean": mean}}, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
