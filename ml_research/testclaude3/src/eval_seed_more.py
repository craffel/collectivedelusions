"""Evaluate Iso-C, TSV, and RegMean on seed=1 and seed=2 checkpoints."""
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
from src.eval_seed import load_seeded
from src.evaluate_merge import build_eval_loaders, eval_state_on_all
from src.merging import iso_c, regmean, tsv
from src.model import CLIPVisualClassifier, load_clip, set_visual_state_dict
from src.regmean_covs import compute_activation_covs


TASKS = ["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    seed_dir = ROOT / f"checkpoints_seed{args.seed}"
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print(f"Loading from {seed_dir} on {device} ...", flush=True)
    theta0, thetas, arts = load_seeded(TASKS, device, seed_dir)
    clip_model, _, _ = load_clip(device=device)
    loaders = build_eval_loaders(TASKS, batch_size=256, num_workers=2)

    out_path = ROOT / f"results/seed{args.seed}_extra.json"

    # Iso-C (best alpha = 1.0)
    print("Iso-C ...", flush=True)
    m = iso_c(theta0, thetas, alpha=1.0)
    accs_iso = eval_state_on_all(m, arts, loaders, clip_model, device)
    iso_mean = sum(accs_iso.values()) / len(accs_iso)
    print(f"  iso_c mean = {iso_mean*100:.2f}", flush=True)

    # TSV (best alpha=0.3, rank=0.8)
    print("TSV ...", flush=True)
    m = tsv(theta0, thetas, alpha=0.3, rank_keep=0.8)
    accs_tsv = eval_state_on_all(m, arts, loaders, clip_model, device)
    tsv_mean = sum(accs_tsv.values()) / len(accs_tsv)
    print(f"  tsv mean = {tsv_mean*100:.2f}", flush=True)

    # RegMean (requires per-task calibration data)
    print("RegMean (calibration) ...", flush=True)
    data_covs = []
    for i, t in enumerate(TASKS):
        b = build_task(t, batch_size=64, num_workers=2, max_train=2000, max_test=1)
        classifier = CLIPVisualClassifier(clip_model, arts[t]["text_features"]).to(device)
        set_visual_state_dict(classifier, thetas[i])
        cov = compute_activation_covs(classifier, b.train_loader, device, max_batches=8)
        data_covs.append(cov)
    m = regmean(theta0, thetas, data_covs)
    accs_rm = eval_state_on_all(m, arts, loaders, clip_model, device)
    rm_mean = sum(accs_rm.values()) / len(accs_rm)
    print(f"  regmean mean = {rm_mean*100:.2f}", flush=True)

    out = {
        "seed": args.seed,
        "iso_c": {"per_task": accs_iso, "mean": iso_mean},
        "tsv": {"per_task": accs_tsv, "mean": tsv_mean},
        "regmean": {"per_task": accs_rm, "mean": rm_mean},
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
