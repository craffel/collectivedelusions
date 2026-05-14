"""Aggregate trim_other_methods_seed{0,1,2}.json into a LaTeX table.

For each method, reports:
  baseline  vs  trim k=0.3  vs  trim k=0.5
mean and std across 3 seeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def mean_std(values):
    a = np.array(values, dtype=float)
    return float(a.mean()), float(a.std())


def load_runs(arch_suffix=""):
    seeds = [0, 1, 2]
    runs = []
    for s in seeds:
        p = ROOT / f"results/trim_other_methods{arch_suffix}_seed{s}.json"
        if not p.exists():
            print(f"Missing {p}", file=sys.stderr)
            sys.exit(1)
        runs.append(json.loads(p.read_text()))
    return runs


def main():
    runs = load_runs("")

    methods = list(runs[0]["methods"].keys())
    settings = ["baseline", "trim_0.3", "trim_0.5"]

    # Compute mean ± std across seeds for every (method, setting)
    table = {}
    for m in methods:
        table[m] = {}
        for s in settings:
            if s not in runs[0]["methods"][m]:
                continue
            vals = [100.0 * r["methods"][m][s]["mean"] for r in runs]
            mu, std = mean_std(vals)
            table[m][s] = (mu, std)

    # Print summary
    print("\n=== Trim-other-methods summary (3-seed mean ± std, %) ===")
    print(f"{'Method':<18} {'baseline':>14} {'trim k=0.3':>14} {'trim k=0.5':>14}   delta_best")
    delta_summary = {}
    for m in methods:
        b = table[m]["baseline"][0]
        t3 = table[m]["trim_0.3"][0]
        t5 = table[m]["trim_0.5"][0]
        best = max(t3, t5)
        delta = best - b
        delta_summary[m] = delta
        print(f"{m:<18} "
              f"{table[m]['baseline'][0]:>6.2f} ± {table[m]['baseline'][1]:.2f}  "
              f"{table[m]['trim_0.3'][0]:>6.2f} ± {table[m]['trim_0.3'][1]:.2f}  "
              f"{table[m]['trim_0.5'][0]:>6.2f} ± {table[m]['trim_0.5'][1]:.2f}   "
              f"{delta:+.2f}")

    # B/16 table
    print("\n=== B/16 ===")
    runs_b16 = load_runs("_b16")
    table_b16 = {}
    for m in methods:
        table_b16[m] = {}
        for s in settings:
            if s not in runs_b16[0]["methods"][m]:
                continue
            vals = [100.0 * r["methods"][m][s]["mean"] for r in runs_b16]
            mu, std = mean_std(vals)
            table_b16[m][s] = (mu, std)
    for m in methods:
        b = table_b16[m]["baseline"][0]
        t3 = table_b16[m]["trim_0.3"][0]
        t5 = table_b16[m]["trim_0.5"][0]
        best = max(t3, t5)
        delta = best - b
        print(f"  {m:<18} baseline={b:6.2f} trim0.3={t3:6.2f} trim0.5={t5:6.2f} delta={delta:+.2f}")

    # Emit LaTeX table (combined B/32 + B/16)
    method_labels = {
        "task_arithmetic": "Task Arithmetic",
        "iso_c": "Iso-C",
        "tsv": "TSV",
        "actmat": "ACTMat",
    }
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\caption{Generality of magnitude trim across data-free merging methods. "
                 "For each method, we apply identical TIES-style magnitude trim with keep fraction $k$ to each $\\Delta_t$ "
                 "before running the method (3-seed mean $\\pm$ std, all 7 tasks). "
                 "Reported is the best of $k\\in\\{0.3, 0.5\\}$. Only ACTMat's covariance solve benefits substantially on both architectures; "
                 "for the spectral method Iso-C the same trim \\emph{hurts}. The TACT improvement is therefore "
                 "specific to interference-minimizing covariance solves, not a generic side-effect of trimming.}")
    lines.append("\\label{tab:trim_other}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{lcc|cc}")
    lines.append("\\toprule")
    lines.append(" & \\multicolumn{2}{c|}{\\textbf{ViT-B/32}} & \\multicolumn{2}{c}{\\textbf{ViT-B/16}} \\\\")
    lines.append("Method & Baseline & Best ($+\\Delta$) & Baseline & Best ($+\\Delta$) \\\\")
    lines.append("\\midrule")
    for m in ["task_arithmetic", "iso_c", "tsv", "actmat"]:
        if m not in table:
            continue
        # B/32
        b_mu, b_sd = table[m]["baseline"]
        t3_mu, _ = table[m]["trim_0.3"]
        t5_mu, t5_sd = table[m]["trim_0.5"]
        best32 = max(t3_mu, t5_mu)
        delta32 = best32 - b_mu
        delta32_str = f"+{delta32:.2f}" if delta32 > 0 else f"{delta32:.2f}"
        # B/16
        b16_mu, b16_sd = table_b16[m]["baseline"]
        t3b_mu, _ = table_b16[m]["trim_0.3"]
        t5b_mu, t5b_sd = table_b16[m]["trim_0.5"]
        best16 = max(t3b_mu, t5b_mu)
        delta16 = best16 - b16_mu
        delta16_str = f"+{delta16:.2f}" if delta16 > 0 else f"{delta16:.2f}"
        if m == "actmat":
            lines.append(
                f"\\textbf{{{method_labels[m]}}} & \\textbf{{{b_mu:.2f}}} & \\textbf{{{best32:.2f} ({delta32_str})}} "
                f"& \\textbf{{{b16_mu:.2f}}} & \\textbf{{{best16:.2f} ({delta16_str})}} \\\\"
            )
        else:
            lines.append(
                f"{method_labels[m]} & {b_mu:.2f} & {best32:.2f} (${delta32_str}$) "
                f"& {b16_mu:.2f} & {best16:.2f} (${delta16_str}$) \\\\"
            )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    tex = "\n".join(lines) + "\n"
    out = ROOT / "template/table_trim_other.tex"
    out.write_text(tex)
    print(f"\nSaved {out}")
    print(f"\nDeltas summary: {delta_summary}")


if __name__ == "__main__":
    main()
