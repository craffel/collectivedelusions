"""Generate figures and tables from results.json for the paper."""
from __future__ import annotations
import argparse
import json
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PRETTY = {
    "pretrained": "Pre-trained (no merge)",
    "weight_average": "Weight averaging",
    "task_arithmetic": "Task arithmetic",
    "ties": "TIES-Merging",
    "iso_c": "Iso-C",
    "actmat": "ACTMat",
    "trim_actmat": "Trim-Mat (ours)",
    "trim_actmat__trim_only": "Trim-Mat (ours)",
    "trim_actmat__sign_only": "ACTMat + sign-elect",
    "trim_actmat__full_d0.5": "ACTMat + trim + sign",
    "random_trim_d0.5_s0": "ACTMat + random-trim",
    "regmean": "RegMean (data oracle)",
}

ORDER = [
    "pretrained", "weight_average", "task_arithmetic", "ties",
    "actmat", "random_trim_d0.5_s0", "trim_actmat__sign_only",
    "trim_actmat__full_d0.5", "trim_actmat",
    "regmean",
]

TASK_PRETTY = {
    "cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "mnist": "MNIST",
    "svhn": "SVHN", "fashionmnist": "F-MNIST", "eurosat": "EuroSAT",
    "gtsrb": "GTSRB", "dtd": "DTD",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results/results.json")
    ap.add_argument("--out", default="./paper/figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.results) as f:
        d = json.load(f)
    tasks = d["tasks"]
    method_results: Dict[str, Dict[str, float]] = d["method_results"]
    method_means = d["method_means"]
    expert_accs = d["expert_accs"]

    # ----------------- main mean-accuracy bar chart -----------------------
    ms = [m for m in ORDER if m in method_results]
    means = [method_means[m] for m in ms]
    pretty = [PRETTY.get(m, m) for m in ms]
    # color: ours in red, oracle in green, others in blue/grey
    colors = []
    for m in ms:
        if m == "trim_actmat":
            colors.append("#d62728")
        elif m == "regmean":
            colors.append("#2ca02c")
        elif m == "pretrained":
            colors.append("#bbbbbb")
        else:
            colors.append("#4c72b0")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bars = ax.bar(pretty, [m * 100 for m in means], color=colors)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{v*100:.1f}",
                ha="center", va="bottom", fontsize=8)
    # expert mean acc as dashed line
    exp_mean = np.mean(list(expert_accs.values())) * 100
    ax.axhline(exp_mean, ls="--", color="black", lw=0.8,
               label=f"Fine-tuned experts ({exp_mean:.1f}%)")
    ax.set_ylabel("Average accuracy (\\%)" if False else "Average accuracy (%)")
    ax.set_ylim(0, max(exp_mean, max(means) * 100) + 5)
    ax.tick_params(axis="x", rotation=25)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "main_bar.pdf"))
    plt.close(fig)

    # density sweep figure
    sweep = d.get("trim_full_sweep_mean_by_density", None)
    # We also have per-task results at densities 0.2/0.3/0.5/0.7/0.9 keyed under method_results
    tdens, tmeans = [], []
    for k in sorted(method_results.keys()):
        if k.startswith("trim_actmat__trim_only_d"):
            try:
                tdens.append(float(k.split("d")[-1]))
                tmeans.append(method_means[k] * 100)
            except ValueError:
                pass
    # The headline "trim_actmat__trim_only" entry came from eval at density=0.5.
    if "trim_actmat__trim_only" in method_means and 0.5 not in tdens:
        tdens.append(0.5)
        tmeans.append(method_means["trim_actmat__trim_only"] * 100)
    if tdens:
        fig, ax = plt.subplots(figsize=(4.0, 2.8))
        order = sorted(range(len(tdens)), key=lambda i: tdens[i])
        x = [tdens[i] for i in order]
        y = [tmeans[i] for i in order]
        ax.plot(x, y, "o-", color="#d62728", label="Trim-Mat")
        ax.axhline(method_means["actmat"] * 100, ls="--", color="#4c72b0", label="ACTMat")
        if "regmean" in method_means:
            ax.axhline(method_means["regmean"] * 100, ls=":", color="#2ca02c", label="RegMean (oracle)")
        ax.set_xlabel("Density $\\rho$")
        ax.set_ylabel("Mean accuracy (%)")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        plt.tight_layout()
        fig.savefig(os.path.join(args.out, "density_sweep.pdf"))
        plt.close(fig)

    # ----------------- per-task bar chart (selected methods) -------------
    sel = [m for m in ["task_arithmetic", "ties",
                       "actmat", "trim_actmat", "regmean"] if m in method_results]
    x = np.arange(len(tasks))
    w = 0.8 / max(1, len(sel))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    method_colors = {
        "task_arithmetic": "#8c8c8c",
        "ties":            "#ff9f1c",
        "actmat":          "#4c72b0",
        "trim_actmat":     "#d62728",
        "regmean":         "#2ca02c",
    }
    for i, m in enumerate(sel):
        vals = [method_results[m][t] * 100 for t in tasks]
        ax.bar(x + (i - len(sel) / 2 + 0.5) * w, vals, width=w,
               label=PRETTY[m], color=method_colors.get(m, "#999"))
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_PRETTY.get(t, t) for t in tasks])
    ax.set_ylabel("Accuracy (%)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.15),
              frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "per_task_bar.pdf"))
    plt.close(fig)

    # ----------------- LaTeX results table --------------------------------
    out_tex = os.path.join(args.out, "results_table.tex")
    with open(out_tex, "w") as f:
        f.write("% auto-generated by make_plots.py\n")
        cols = "l" + "c" * (len(tasks) + 1)
        f.write(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n")
        header = "Method & " + " & ".join(TASK_PRETTY.get(t, t) for t in tasks) + " & Avg \\\\\n"
        f.write(header)
        f.write("\\midrule\n")
        # find best mean among "fair" (data-free, non-pretrained) methods to bold
        fair = [m for m in ms if m not in ("pretrained", "regmean")]
        best_mean = max(method_means[m] for m in fair)
        for m in ms:
            mean = method_means[m]
            cells = []
            for t in tasks:
                v = method_results[m][t]
                cells.append(f"{v*100:.1f}")
            row = (PRETTY[m] + " & " + " & ".join(cells)
                   + " & " + (f"\\textbf{{{mean*100:.1f}}}"
                              if (m in fair and mean == best_mean)
                              else f"{mean*100:.1f}"))
            f.write(row + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print("wrote figures to", args.out)


if __name__ == "__main__":
    main()
