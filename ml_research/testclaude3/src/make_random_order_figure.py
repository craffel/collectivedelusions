"""Random-ordering figure & summary table.

Reads results/task_count_random_seed{0,1,2}.json (each contains ~5 orderings).
For each T, collects all (seed × ordering) TACT/ACTMat numbers and reports:
- mean ± std across seeds × orderings
- TACT-ACTMat gap distribution

Writes:
- figures/task_count_random_gap.pdf  (a bar/violin showing the gap across orderings, per T)
- template/table_task_count_random.tex
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
TPL = ROOT / "template"


def load_all():
    runs = []
    for s in [0, 1, 2]:
        p = RES / f"task_count_random_seed{s}.json"
        if not p.exists():
            continue
        with open(p) as f:
            runs.append((s, json.load(f)))
    return runs


def main():
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.size"] = 9
    runs = load_all()
    if not runs:
        print("No data.")
        return

    # Collect: (T, seed, ordering) -> {method: acc}
    rows = defaultdict(list)  # T -> list of (seed, ord_name, method_accs)
    for seed, run in runs:
        for ord_name, by_T in run.items():
            for T_str, blob in by_T.items():
                T = int(T_str)
                rows[T].append({
                    "seed": seed, "ord": ord_name,
                    "tasks": blob["tasks"],
                    "accs": {m: 100*v for m, v in blob["metrics"].items()},
                })

    Ts = sorted(rows.keys())

    # Gap statistics
    print("\n=== TACT - ACTMat gap, per T (over 3 seeds × all orderings): ===")
    gap_by_T = {T: [] for T in Ts}
    for T in Ts:
        for r in rows[T]:
            gap = r["accs"]["tact_cov_k0.5"] - r["accs"]["actmat"]
            gap_by_T[T].append(gap)
    for T in Ts:
        g = np.array(gap_by_T[T])
        n_wins = (g > 0).sum()
        print(f"T={T}: gap mean={g.mean():.2f}  std={g.std():.2f}  min={g.min():.2f}  max={g.max():.2f}  TACT wins {n_wins}/{len(g)}")

    # Plot: gap distribution per T (boxplot)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    data = [gap_by_T[T] for T in Ts]
    bp = ax.boxplot(data, positions=Ts, widths=0.4, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#d62728")
        patch.set_alpha(0.4)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)
    # Scatter individual (seed, ord) points
    for T in Ts:
        ax.scatter([T]*len(gap_by_T[T]), gap_by_T[T],
                   c="#d62728", s=12, zorder=4, alpha=0.7)
    ax.axhline(0, color="gray", lw=0.7, linestyle="--", zorder=1)
    ax.set_xlabel("Number of merged tasks $T$")
    ax.set_ylabel("TACT $-$ ACTMat (\\%)")
    ax.set_title("TACT gain over ACTMat\n(3 seeds × 4 task orderings)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(Ts)
    plt.tight_layout()
    out = FIG / "task_count_random_gap.pdf"
    plt.savefig(out, format="pdf")
    plt.close()
    print(f"\nWrote {out}")

    # LaTeX table: mean ± std per T per method
    methods = ["average", "task_arithmetic", "ties", "actmat", "tact_cov_k0.5"]
    method_label = {
        "average": "Simple Avg.",
        "task_arithmetic": "Task Arith.",
        "ties": "TIES",
        "actmat": "ACTMat",
        "tact_cov_k0.5": r"\textbf{TACT (ours)}",
    }
    lines = []
    lines.append(r"\begin{tabular}{l" + "c"*len(Ts) + "}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join([f"$T={T}$" for T in Ts]) + r" \\")
    lines.append(r"\midrule")
    for m in methods:
        cells = []
        for T in Ts:
            vals = [r["accs"][m] for r in rows[T]]
            cells.append(f"{np.mean(vals):.2f}")
        lines.append(method_label[m] + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    cells = []
    for T in Ts:
        g = np.array(gap_by_T[T])
        cells.append(f"{g.mean():+.2f}")
    lines.append(r"\emph{TACT $-$ ACTMat} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    tab = TPL / "table_task_count_random.tex"
    tab.write_text("\n".join(lines) + "\n")
    print(f"Wrote {tab}")


if __name__ == "__main__":
    main()
