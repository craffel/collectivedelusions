"""Bar chart of per-group accuracy gain from layer-type attribution.

Two side-by-side panels: ViT-B/32 (left) and ViT-B/16 (right). Each panel shows
the per-group accuracy gain (delta from ACTMat baseline) for the 7 selective
TACT variants, with 3-seed error bars.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_deltas(seed_files):
    data = [json.load(open(p)) for p in seed_files]
    order = ["attn_in", "attn_out", "all_attn", "mlp_fc", "mlp_proj", "all_mlp", "all"]
    deltas_per_seed = []
    for d in data:
        none_acc = d["groups"]["none"]["mean_acc"] * 100
        deltas = [d["groups"][g]["mean_acc"] * 100 - none_acc for g in order]
        deltas_per_seed.append(deltas)
    deltas_per_seed = np.array(deltas_per_seed)
    return order, deltas_per_seed.mean(axis=0), deltas_per_seed.std(axis=0, ddof=1)


def _bar_panel(ax, means, stds, title):
    labels = ["attn\nin-proj", "attn\nout-proj", "all\nattn",
              "MLP\nup", "MLP\ndown", "all\nMLP", "all\n(TACT)"]
    colors = ["#9ecae1", "#9ecae1", "#3182bd",
              "#fdae6b", "#fdae6b", "#e6550d",
              "#31a354"]
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=stds, color=colors, capsize=4,
           edgecolor="black", linewidth=0.6)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.03, f"+{m:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(top=max(means + stds) * 1.22)


def main():
    b32_files = [ROOT / "results/layer_type.json",
                 ROOT / "results/layer_type_seed1.json",
                 ROOT / "results/layer_type_seed2.json"]
    b16_files = [ROOT / "results/layer_type_b16.json",
                 ROOT / "results/layer_type_b16_seed1.json",
                 ROOT / "results/layer_type_b16_seed2.json"]

    _, b32_mean, b32_std = _load_deltas(b32_files)
    _, b16_mean, b16_std = _load_deltas(b16_files)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=False)
    _bar_panel(ax_l, b32_mean, b32_std, "ViT-B/32 ($k=0.5$, 3 seeds)")
    _bar_panel(ax_r, b16_mean, b16_std, "ViT-B/16 ($k=0.5$, 3 seeds)")
    ax_l.set_ylabel("Mean acc.\\ gain over ACTMat (pp)")

    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color="#3182bd", label="attention"),
        mpatches.Patch(color="#e6550d", label="MLP"),
        mpatches.Patch(color="#31a354", label="all (TACT)"),
    ]
    ax_l.legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=True)

    plt.tight_layout()
    out_path = ROOT / "figures/layer_type_attribution.pdf"
    plt.savefig(out_path, format="pdf")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
