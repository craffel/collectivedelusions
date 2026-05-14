"""Compute summary statistics over the expert task vectors for the paper.

Produces a JSON file with:
  - per-layer fraction of entries where the sign of (Wt - W0) agrees with the
    cross-task elected sign (i.e., does NOT get masked by TIES sign-elect)
  - per-layer fraction of entries that survive both TIES trim and sign-election
    at density=0.2
  - histogram of |Δ_t| values

The numbers are used to motivate TRIM in the paper.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.merge import ties_mask
from src.models import build_vision_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="./checkpoints")
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--density", type=float, default=0.2)
    ap.add_argument("--out_json", default="./results/sign_analysis.json")
    ap.add_argument("--out_fig", default="./paper/figs/sign_agreement.pdf")
    args = ap.parse_args()

    if not args.tasks:
        args.tasks = sorted([f[:-3] for f in os.listdir(args.ckpt_dir) if f.endswith(".pt")])
    print("tasks:", args.tasks)

    enc = build_vision_encoder()
    base = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
    experts = []
    for t in args.tasks:
        e = torch.load(os.path.join(args.ckpt_dir, f"{t}.pt"),
                       map_location="cpu", weights_only=False)
        experts.append(e["vision_state_dict"])

    layer_stats = []
    sign_agree_per_layer = []
    survive_per_layer = []
    layer_names = []
    layer_sizes = []
    for k, v0 in base.items():
        if v0.ndim != 2:
            continue
        deltas = torch.stack([(e[k] - v0).to(torch.float32) for e in experts])  # (T, ...)
        T = deltas.shape[0]
        N = deltas.numel() // T
        elected = torch.sign(deltas.sum(0))
        elected[elected == 0] = 1.0
        match = (torch.sign(deltas) == elected.unsqueeze(0))
        # fraction of entries (per-task averaged) whose sign matches elected
        sign_agree = match.float().mean().item()
        # fraction of entries that survive both trim and sign-elect at density
        d_hat = ties_mask(deltas, args.density)
        survive = (d_hat != 0).float().mean().item()
        layer_names.append(k)
        layer_sizes.append(v0.numel())
        sign_agree_per_layer.append(sign_agree)
        survive_per_layer.append(survive)
        layer_stats.append({
            "layer": k,
            "shape": list(v0.shape),
            "params": v0.numel(),
            "sign_agree_frac": sign_agree,
            "survive_frac_at_density": survive,
        })

    overall_sign_agree = float(np.average(sign_agree_per_layer,
                                          weights=layer_sizes))
    overall_survive = float(np.average(survive_per_layer, weights=layer_sizes))

    payload = {
        "tasks": args.tasks,
        "density": args.density,
        "overall_sign_agree_frac": overall_sign_agree,
        "overall_survive_frac": overall_survive,
        "per_layer": layer_stats,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print("overall_sign_agree_frac =", overall_sign_agree)
    print("overall_survive_frac =", overall_survive)
    print("wrote", args.out_json)

    # plot
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    layer_idx = np.arange(len(layer_names))
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.plot(layer_idx, sign_agree_per_layer, label="sign-agree fraction", color="#4c72b0")
    ax.plot(layer_idx, survive_per_layer, label=f"survive (density={args.density})",
            color="#d62728")
    ax.set_xlabel("Linear layer index (in CLIP-ViT-B/32)")
    ax.set_ylabel("Fraction of entries")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, ls=":", color="grey", lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(args.out_fig)
    plt.close(fig)
    print("wrote", args.out_fig)


if __name__ == "__main__":
    main()
