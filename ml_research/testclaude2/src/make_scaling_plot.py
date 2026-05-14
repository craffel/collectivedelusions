"""Plot mean accuracy of ACTMat vs Trim-Mat as a function of number of tasks merged."""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    with open("results/scaling.json") as f:
        d = json.load(f)
    ns = sorted([int(k) for k in d.keys()])
    am = [d[str(n)]["actmat_mean"] * 100 for n in ns]
    tm = [d[str(n)]["trim_mat_mean"] * 100 for n in ns]

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.plot(ns, am, "o-", label="ACTMat",   color="#4c72b0")
    ax.plot(ns, tm, "s-", label="Trim-Mat (ours)", color="#d62728")
    ax.set_xlabel("\\# tasks merged" if False else "# tasks merged")
    ax.set_ylabel("Mean accuracy (\\%)" if False else "Mean accuracy (%)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for n, a, t in zip(ns, am, tm):
        ax.annotate(f"+{t-a:.1f}", (n, t), textcoords="offset points",
                    xytext=(0, 5), fontsize=7, ha="center", color="#d62728")
    plt.tight_layout()
    out = "paper/figs/scaling_tasks.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
