"""Compute effective rank of the merged covariance sum at various trim levels.

The effective rank of a matrix is exp(H(singular-value distribution)) where H is
the Shannon entropy of the normalized singular-value distribution. This captures
how many directions the matrix meaningfully spans, regardless of the absolute
scale of singular values.

We compute the effective rank of (Σ_t Δ̃_t^T Δ̃_t) for each layer at each trim
level, average across layers, and produce a plot.
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

from src.evaluate_merge import load_artifacts
from src.merging import _trim_topk, is_2d_weight, task_vectors


TASKS = ["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"]


def effective_rank(M: torch.Tensor, eps: float = 1e-12) -> float:
    """exp(Shannon entropy of normalized singular values)."""
    s = torch.linalg.svdvals(M.float())
    p = s / (s.sum() + eps)
    H = -(p * (p + eps).log()).sum().item()
    return float(np.exp(H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--keep-list", nargs="+", type=float,
                    default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--out", default=str(ROOT / "results/effective_rank.json"))
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f"Loading on {device} ...", flush=True)
    theta0, thetas, _ = load_artifacts(TASKS, device)
    taus = task_vectors(theta0, thetas)
    layer_keys = [k for k, v in theta0.items() if v.ndim == 2 and is_2d_weight(k, v)]
    print(f"Found {len(layer_keys)} linear layers.")

    out = {}
    for keep in args.keep_list:
        if keep >= 1.0:
            trimmed = taus
        else:
            trimmed = [_trim_topk(tau, keep) for tau in taus]
        eranks = []
        for k in layer_keys:
            S = torch.zeros((trimmed[0][k].shape[1], trimmed[0][k].shape[1]),
                            device=device, dtype=torch.float32)
            for t in range(len(TASKS)):
                d = trimmed[t][k].to(torch.float32)
                S = S + d.T @ d
            eranks.append(effective_rank(S))
        out[f"keep_{keep}"] = {
            "mean_effective_rank": float(np.mean(eranks)),
            "median_effective_rank": float(np.median(eranks)),
            "per_layer": [(k, r) for k, r in zip(layer_keys, eranks)],
        }
        print(f"keep={keep:.2f}  mean eff. rank = {out[f'keep_{keep}']['mean_effective_rank']:.1f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    keeps = sorted([float(k.split("_")[1]) for k in out])
    means = [out[f"keep_{k}"]["mean_effective_rank"] for k in keeps]
    plt.figure(figsize=(5.2, 3.4))
    plt.plot(keeps, means, "o-", linewidth=2, markersize=7, color="C2")
    plt.xlabel("Trim keep fraction $k$")
    plt.ylabel(r"Mean effective rank of $\sum_t \tilde\Delta_t^\top \tilde\Delta_t$")
    plt.title("Effective rank vs.\\ trim level")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = ROOT / "figures/effective_rank.pdf"
    plt.savefig(fig_path, format="pdf")
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
