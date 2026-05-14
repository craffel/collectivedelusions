"""Generate per-task improvement plots for the paper."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"


def _load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def main():
    main = _load("main.json")
    fix = _load("main_actmat_fix.json")
    tact_cov = _load("tact_cov.json")
    tasks = main["tasks"]

    methods_to_plot = {
        "Simple Avg":      ("average", main["merged"]),
        "TIES":            ("ties", main["merged"]),
        "ACTMat":          ("actmat", fix["merged"]),
        "TACT (ours)":     ("tact_cov", tact_cov["merged"]),
    }

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = np.arange(len(tasks))
    width = 0.20
    colors = ["#bdbdbd", "#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (label, (key, src)) in enumerate(methods_to_plot.items()):
        vals = [src[key]["per_task"][t] * 100 for t in tasks]
        ax.bar(x + i * width - 1.5 * width, vals, width, label=label,
               color=colors[i], edgecolor="black", linewidth=0.6)

    # Individual fine-tuned upper bound
    ax.plot(x, [main["individual"][t]["finetuned"] * 100 for t in tasks],
            "k--", linewidth=1, label="Individual FT (upper bound)")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Test accuracy (\\%)")
    ax.set_title("Per-task accuracy across data-free merging methods")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.set_ylim(30, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = FIGS / "per_task_bars.pdf"
    plt.savefig(out, format="pdf")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
