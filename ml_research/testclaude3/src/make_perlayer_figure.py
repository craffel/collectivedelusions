"""Make a 2-panel figure showing global vs per-layer trim sweep across 3 seeds,
on both B/32 and B/16. Highlights the catastrophic-collapse failure mode of
global trim at very small k.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KEEPS = [0.1, 0.2, 0.3, 0.5, 0.7]
ARCHES = [
    ("B/32", [
        ROOT / "results/per_layer_trim.json",
        ROOT / "results/per_layer_trim_seed1.json",
        ROOT / "results/per_layer_trim_seed2.json",
    ]),
    ("B/16", [
        ROOT / "results/per_layer_trim_b16.json",
        ROOT / "results/per_layer_trim_b16_seed1.json",
        ROOT / "results/per_layer_trim_b16_seed2.json",
    ]),
]


def load_arch(files):
    dicts = [json.load(open(p)) for p in files]
    out = {}
    for k in KEEPS:
        gkey = f"global_k{k:.1f}_signoff"
        pkey = f"perlayer_k{k:.1f}"
        gvals = [d[gkey]["mean"] * 100 for d in dicts]
        pvals = [d[pkey]["mean"] * 100 for d in dicts]
        out[k] = {
            "global_mean": statistics.mean(gvals), "global_std": statistics.stdev(gvals),
            "perlayer_mean": statistics.mean(pvals), "perlayer_std": statistics.stdev(pvals),
            "global_seeds": gvals, "perlayer_seeds": pvals,
        }
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=False)
    for ax, (arch_name, files) in zip(axes, ARCHES):
        data = load_arch(files)
        ks = list(data.keys())
        gmean = np.array([data[k]["global_mean"] for k in ks])
        gstd = np.array([data[k]["global_std"] for k in ks])
        pmean = np.array([data[k]["perlayer_mean"] for k in ks])
        pstd = np.array([data[k]["perlayer_std"] for k in ks])

        ax.errorbar(ks, gmean, yerr=gstd, marker="o", color="C1", capsize=3,
                    label="Global (TIES-style)", linewidth=1.5)
        ax.errorbar(ks, pmean, yerr=pstd, marker="s", color="C0", capsize=3,
                    label="Per-layer (proposed)", linewidth=1.5)
        # individual seed dots for global
        for k in ks:
            for v in data[k]["global_seeds"]:
                ax.scatter([k], [v], color="C1", alpha=0.35, s=14)
            for v in data[k]["perlayer_seeds"]:
                ax.scatter([k], [v], color="C0", alpha=0.35, s=14)

        ax.set_title(f"ViT-{arch_name}")
        ax.set_xlabel("Trim keep fraction $k$")
        ax.set_ylabel("Mean accuracy (%)")
        ax.set_xticks(ks)
        ax.set_ylim(-2, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_path = ROOT / "figures/perlayer_trim_sweep.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
