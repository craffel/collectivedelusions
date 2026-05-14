"""Aggregate per-layer trim ablation across 3 seeds; emit new table_perlayer.tex."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
SEED_FILES = [
    ROOT / "results/per_layer_trim.json",
    ROOT / "results/per_layer_trim_seed1.json",
    ROOT / "results/per_layer_trim_seed2.json",
]
KEEPS = [0.1, 0.2, 0.3, 0.5, 0.7]


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    dicts = [load(p) for p in SEED_FILES]
    rows = []
    for k in KEEPS:
        gkey = f"global_k{k:.1f}_signoff"
        pkey = f"perlayer_k{k:.1f}"
        gvals = [d[gkey]["mean"] * 100 for d in dicts]
        pvals = [d[pkey]["mean"] * 100 for d in dicts]
        gm, gs = statistics.mean(gvals), statistics.stdev(gvals)
        pm, ps = statistics.mean(pvals), statistics.stdev(pvals)
        rows.append((k, gm, gs, pm, ps, pm - gm))
        print(f"k={k}  global={gm:.2f}±{gs:.2f}  per-layer={pm:.2f}±{ps:.2f}  delta={pm-gm:+.2f}")

    out = []
    out.append("\\begin{table}[h]")
    out.append("\\caption{Multi-seed (mean$\\pm$std over 3 fine-tuning seeds, B/32) mean test accuracy (\\%) of TACT (cov-only, no sign election) on the 7-task benchmark, comparing global vs per-layer trim. Per-layer trim is strictly more robust at aggressive trim levels.}")
    out.append("\\label{tab:perlayer}")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\begin{tabular}{lccc}")
    out.append("\\toprule")
    out.append("$k$ & Global & Per-layer & $\\Delta$ (per-global) \\\\")
    out.append("\\midrule")
    for k, gm, gs, pm, ps, dlt in rows:
        bold_open, bold_close = ("", "")
        if k == 0.5:
            bold_open, bold_close = ("\\textbf{", "}")
        out.append(
            f"{bold_open}{k}{bold_close} & {bold_open}{gm:.2f} {{\\scriptsize $\\pm$ {gs:.2f}}}{bold_close} & {bold_open}{pm:.2f} {{\\scriptsize $\\pm$ {ps:.2f}}}{bold_close} & ${dlt:+.2f}$ \\\\"
        )
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")

    out_path = ROOT / "template/table_perlayer.tex"
    out_path.write_text("\n".join(out) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
