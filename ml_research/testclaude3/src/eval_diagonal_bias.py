"""Empirical validation of Proposition 3.1.

Proposition 3.1 predicts that the data-free covariance estimator
    Ĉ_t = Δ_t^T Δ_t  with  Δ_t = Δ_t^* + N_t
has the form
    E[Ĉ_t] = Δ_t^{*T} Δ_t^* + σ_t^2 · diag(d_o - n_{t,j}),
i.e. the bias is exactly diagonal. This script measures the diagonal-vs-off-diagonal
mass ratio of what magnitude-trim removes from Ĉ_t, and compares to what SVD
truncation removes. If Proposition 3.1 is correct, the trim residual is
predominantly diagonal while the SVD residual is not.

For each task and each Linear-style layer:
  D_trim(k) = Ĉ_t - Ĉ_t^{trim(k)}
  D_svd(r)  = Ĉ_t - Ĉ_t^{svd(r)}
Compute the *diagonal mass ratio* rho(D) = ||diag(D)||_2 / ||D||_F.
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
from src.merging import _trim_topk, _svd_truncate, is_2d_weight, task_vectors


TASKS = ["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"]


def diagonal_mass(M: torch.Tensor) -> float:
    """Ratio ||diag(M)||_2 / ||M||_F. In [0,1]; 1 means M is diagonal."""
    M = M.to(torch.float32)
    d_norm = torch.linalg.norm(torch.diagonal(M)).item()
    f_norm = torch.linalg.norm(M).item()
    if f_norm < 1e-12:
        return float("nan")
    return d_norm / f_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--keep-list", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3, 0.5, 0.7])
    ap.add_argument("--ckpt-subdir", default="checkpoints")
    ap.add_argument("--out", default=str(ROOT / "results/diagonal_bias.json"))
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f"Loading on {device} ...", flush=True)
    theta0, thetas, _arts = load_artifacts(TASKS, device, ckpt_subdir=args.ckpt_subdir)

    # Trim each task vector globally to top-k%.
    taus = task_vectors(theta0, thetas)
    layer_keys = [k for k, v in theta0.items() if v.ndim == 2 and is_2d_weight(k, v)]
    print(f"Analyzing {len(layer_keys)} linear layers across {len(TASKS)} tasks", flush=True)

    summary = {"trim": {}, "svd": {}, "trim_per_layer": {}, "svd_per_layer": {}}

    # Cache full-Δ covariance per layer per task
    Chat_full = {}
    for t in range(len(TASKS)):
        Chat_full[t] = {}
        for k in layer_keys:
            d = taus[t][k].to(torch.float32)
            Chat_full[t][k] = d.T @ d

    # ---- Magnitude trim sweep ----
    for keep in args.keep_list:
        trimmed = [_trim_topk(tau, keep) for tau in taus]
        per_layer_ratios = []
        for k in layer_keys:
            per_task = []
            for t in range(len(TASKS)):
                d_trim = trimmed[t][k].to(torch.float32)
                C_trim = d_trim.T @ d_trim
                D = Chat_full[t][k] - C_trim
                per_task.append(diagonal_mass(D))
            per_layer_ratios.append(float(np.mean(per_task)))
        mean_rho = float(np.mean(per_layer_ratios))
        summary["trim"][f"keep_{keep}"] = mean_rho
        summary["trim_per_layer"][f"keep_{keep}"] = per_layer_ratios
        print(f"trim k={keep:.2f}: mean diag-mass(Ĉ - Ĉ^trim) = {mean_rho:.4f}")

    # ---- SVD truncation sweep (matched residual budgets) ----
    for rk in args.keep_list:
        per_layer_ratios = []
        for k in layer_keys:
            per_task = []
            for t in range(len(TASKS)):
                d_svd = _svd_truncate(taus[t][k], rk).to(torch.float32)
                C_svd = d_svd.T @ d_svd
                D = Chat_full[t][k] - C_svd
                per_task.append(diagonal_mass(D))
            per_layer_ratios.append(float(np.mean(per_task)))
        mean_rho = float(np.mean(per_layer_ratios))
        summary["svd"][f"rank_{rk}"] = mean_rho
        summary["svd_per_layer"][f"rank_{rk}"] = per_layer_ratios
        print(f"svd  r={rk:.2f}: mean diag-mass(Ĉ - Ĉ^svd ) = {mean_rho:.4f}")

    # Random reference: a Wishart-ish PSD matrix's diag mass
    d_in = Chat_full[0][layer_keys[0]].shape[0]
    # For comparison: a random 2x2 with iid Gaussian entries has expected diag/Frobenius ~ sqrt(2/(d^2)) -> ~0 for large d
    summary["random_baseline"] = float(np.sqrt(1.0 / d_in))

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {args.out}")

    # ---- Plot ----
    keeps = sorted([float(k.split("_")[1]) for k in summary["trim"]])
    trim_means = [summary["trim"][f"keep_{k}"] for k in keeps]
    svd_means = [summary["svd"][f"rank_{k}"] for k in keeps]
    rand_ref = summary["random_baseline"]

    plt.figure(figsize=(5.2, 3.6))
    plt.plot(keeps, trim_means, "o-", color="C0", linewidth=2, markersize=8,
             label=r"Magnitude trim:  $\hat C_t-\hat C_t^{\mathrm{trim}(k)}$")
    plt.plot(keeps, svd_means, "s--", color="C3", linewidth=2, markersize=8,
             label=r"SVD truncation: $\hat C_t-\hat C_t^{\mathrm{svd}(k)}$")
    plt.axhline(rand_ref, color="gray", linestyle=":",
                label=f"Random reference ($1/\\sqrt{{d_i}} \\approx {rand_ref:.3f}$)")
    plt.xlabel(r"Fraction retained $k$")
    plt.ylabel(r"Diagonal mass ratio $\|\mathrm{diag}(D)\|_2 / \|D\|_F$")
    plt.title("Proposition 3.1: trim removes a diagonal bias; SVD truncation does not")
    plt.legend(fontsize=8, loc="center right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = ROOT / "figures/diagonal_bias.pdf"
    plt.savefig(fig_path, format="pdf")
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
