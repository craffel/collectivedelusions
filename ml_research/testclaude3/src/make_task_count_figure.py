"""Plot the task-count-scaling figure.

Reads:
  results/task_count_scaling_seed{0,1,2}.json        (B/32, 3 seeds)
  results/task_count_scaling_b16_seed{0,1,2}.json    (B/16, 3 seeds)

Writes:
  figures/task_count_scaling.pdf
  template/table_task_count.tex
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
TPL = ROOT / "template"


METHOD_LABEL = {
    "average":         ("Simple Average", "#888888", "o"),
    "task_arithmetic": ("Task Arithmetic", "#9467bd", "v"),
    "ties":            ("TIES",            "#2ca02c", "s"),
    "actmat":          ("ACTMat",          "#1f77b4", "^"),
    "tact_cov_k0.5":   ("TACT (ours)",     "#d62728", "D"),
}


def load_arch(prefix: str) -> dict:
    """Return {T: {method: [seed0, seed1, seed2]}}."""
    runs = []
    for s in [0, 1, 2]:
        p = RES / f"{prefix}_seed{s}.json"
        if p.exists():
            with open(p) as f:
                runs.append(json.load(f))
    if not runs:
        return {}
    out = {}
    Ts = sorted(runs[0].keys(), key=int)
    for T in Ts:
        out[int(T)] = {}
        methods = list(runs[0][T].keys())
        for m in methods:
            vals = [100.0 * r[T][m] for r in runs if T in r and m in r[T]]
            out[int(T)][m] = vals
    return out


def write_latex_table(b32: dict, b16: dict, path: Path):
    """One row per method, columns for each T, separate panels for B/32 and B/16."""
    Ts = sorted(b32.keys())
    methods = ["average", "task_arithmetic", "ties", "actmat", "tact_cov_k0.5"]
    lines = []
    lines.append(r"\begin{tabular}{l" + "c" * len(Ts) + "}")
    lines.append(r"\toprule")
    head = " & " + " & ".join([f"$T={T}$" for T in Ts]) + r" \\"
    lines.append(head)
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(Ts) + 1) + r"}{l}{\emph{ViT-B/32 (mean of 3 seeds)}} \\")
    for m in methods:
        label = METHOD_LABEL[m][0]
        if m == "tact_cov_k0.5":
            label = r"\textbf{TACT (ours)}"
        row = label
        for T in Ts:
            vals = b32[T].get(m, [])
            if vals:
                mean = np.mean(vals)
                row += f" & {mean:.2f}"
            else:
                row += " & --"
        row += r" \\"
        lines.append(row)
    if b16:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(len(Ts) + 1) + r"}{l}{\emph{ViT-B/16 (mean of 3 seeds)}} \\")
        for m in methods:
            label = METHOD_LABEL[m][0]
            if m == "tact_cov_k0.5":
                label = r"\textbf{TACT (ours)}"
            row = label
            for T in Ts:
                vals = b16[T].get(m, [])
                if vals:
                    mean = np.mean(vals)
                    row += f" & {mean:.2f}"
                else:
                    row += " & --"
            row += r" \\"
            lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def plot_panel(ax, data: dict, title: str, show_ylabel: bool, show_legend: bool):
    Ts = sorted(data.keys())
    methods = ["average", "task_arithmetic", "ties", "actmat", "tact_cov_k0.5"]
    for m in methods:
        label, color, marker = METHOD_LABEL[m]
        if m == "tact_cov_k0.5":
            label = "TACT (ours)"
        means = []
        stds = []
        for T in Ts:
            vals = data[T].get(m, [])
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0.0)
        means = np.array(means)
        stds = np.array(stds)
        lw = 2.0 if m == "tact_cov_k0.5" else 1.3
        ms = 7 if m == "tact_cov_k0.5" else 5
        zorder = 5 if m == "tact_cov_k0.5" else 3
        ax.errorbar(Ts, means, yerr=stds, label=label, color=color, marker=marker,
                    markersize=ms, linewidth=lw, capsize=2.5, zorder=zorder)
    ax.set_xlabel("Number of merged tasks $T$")
    if show_ylabel:
        ax.set_ylabel(r"Mean test accuracy (\%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.set_xticks(Ts)


def main():
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.size"] = 9

    b32 = load_arch("task_count_scaling")
    b16 = load_arch("task_count_scaling_b16")

    if not b32:
        print("Missing B/32 results; aborting.")
        return

    have_b16 = bool(b16)
    if have_b16:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)
        plot_panel(axes[0], b32, "ViT-B/32", show_ylabel=True,  show_legend=True)
        plot_panel(axes[1], b16, "ViT-B/16", show_ylabel=False, show_legend=False)
    else:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        plot_panel(ax, b32, "ViT-B/32", show_ylabel=True, show_legend=True)

    plt.tight_layout()
    out = FIG / "task_count_scaling.pdf"
    plt.savefig(out, format="pdf")
    plt.close()
    print(f"Wrote {out}")

    # LaTeX table
    write_latex_table(b32, b16, TPL / "table_task_count.tex")

    # Print final-T summary
    Ts = sorted(b32.keys())
    print("\n=== TACT vs ACTMat gap, by T (B/32, mean of 3 seeds): ===")
    for T in Ts:
        a = np.mean(b32[T].get("actmat", []) or [0.0])
        t = np.mean(b32[T].get("tact_cov_k0.5", []) or [0.0])
        print(f"  T={T}  ACTMat={a:.2f}  TACT={t:.2f}  gap={t-a:+.2f}")
    if have_b16:
        print("\n=== TACT vs ACTMat gap, by T (B/16, mean of 3 seeds): ===")
        for T in Ts:
            a = np.mean(b16[T].get("actmat", []) or [0.0])
            t = np.mean(b16[T].get("tact_cov_k0.5", []) or [0.0])
            print(f"  T={T}  ACTMat={a:.2f}  TACT={t:.2f}  gap={t-a:+.2f}")


if __name__ == "__main__":
    main()
