"""Compute cosine alignment between the data-free covariance estimator C_hat
and the data-derived activation covariance C, for raw Delta vs trimmed Delta.

Saves a JSON summary and a matplotlib bar plot in figures/cov_alignment.pdf.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets_setup import build_task
from src.evaluate_merge import load_artifacts
from src.merging import _trim_topk, is_2d_weight, task_vectors
from src.model import (
    CLIPVisualClassifier, build_text_classifier, load_clip, set_visual_state_dict,
)
from src.regmean_covs import compute_activation_covs


TASKS = ["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"]


def cosine_align(A, B):
    """Frobenius-cosine of two matrices (treated as flat vectors)."""
    a = A.float().flatten()
    b = B.float().flatten()
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--max-batches", type=int, default=8)
    ap.add_argument("--keep-list", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--out", default=str(ROOT / "results/cov_alignment.json"))
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f"Loading on {device} ...", flush=True)
    theta0, thetas, arts = load_artifacts(TASKS, device)
    clip_model, _, _ = load_clip(device=device)

    # Compute data-derived C_t per task (small calibration set)
    print("Computing data-derived activation covariances ...")
    data_covs = []
    for i, t in enumerate(TASKS):
        bundle = build_task(t, batch_size=64, num_workers=2, max_train=1024, max_test=1)
        classifier = CLIPVisualClassifier(clip_model, arts[t]["text_features"]).to(device)
        set_visual_state_dict(classifier, thetas[i])
        cov = compute_activation_covs(classifier, bundle.train_loader, device,
                                       max_batches=args.max_batches)
        data_covs.append(cov)
        print(f"  {t}: covs for {len(cov)} layers")

    # Compute data-free C_hat under various trim levels
    taus = task_vectors(theta0, thetas)
    layer_keys = [k for k, v in theta0.items() if v.ndim == 2 and is_2d_weight(k, v)]
    print(f"Computing data-free covariances over {len(layer_keys)} linear layers ...")

    summary = {"per_layer": {}, "mean_per_keep": {}}
    for keep in args.keep_list:
        per_layer = []
        if keep >= 1.0:
            trimmed = taus
        else:
            trimmed = [_trim_topk(tau, keep) for tau in taus]
        for k in layer_keys:
            cosines = []
            # Translate the model name "visual.transformer.resblocks.0.attn.out_proj.weight" -> Linear param name
            cov_name = k  # In our model, the parameter name is the same
            if cov_name not in data_covs[0]:
                # Try alternate forms; if missing, skip
                continue
            for t in range(len(TASKS)):
                d = trimmed[t][k].to(torch.float32)
                C_hat = d.T @ d  # [D, D] using Linear convention (in_dim = shape[1])
                C = data_covs[t][cov_name].to(C_hat.device)
                cosines.append(cosine_align(C_hat, C))
            mean_c = float(np.mean(cosines))
            per_layer.append((k, mean_c))
        means = [c for _, c in per_layer]
        summary["per_layer"][f"keep_{keep}"] = per_layer
        summary["mean_per_keep"][f"keep_{keep}"] = float(np.mean(means))
        print(f"keep={keep:.2f}  mean cosine alignment = {summary['mean_per_keep'][f'keep_{keep}']:.4f}")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {args.out}")

    # Plot
    keeps = sorted([float(k.split("_")[1]) for k in summary["mean_per_keep"]])
    means = [summary["mean_per_keep"][f"keep_{k}"] for k in keeps]
    plt.figure(figsize=(5.0, 3.4))
    plt.plot(keeps, means, "o-", linewidth=2, markersize=8, color="C0")
    plt.xlabel("Trim keep fraction $k$")
    plt.ylabel(r"$\cos(\hat C_t, C_t)$")
    plt.title("Data-free vs.~data-derived covariance alignment")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = ROOT / "figures/cov_alignment.pdf"
    plt.savefig(fig_path, format="pdf")
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
