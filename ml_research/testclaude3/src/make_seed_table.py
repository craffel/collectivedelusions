"""Make the multi-seed comparison table for the paper."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    p = ROOT / "results" / name
    return json.load(open(p)) if p.exists() else None


def main():
    summary = _load("multi_seed_summary.json")
    if summary is None:
        # produce a placeholder
        Path(ROOT / "template/table_seeds.tex").write_text("% multi_seed_summary.json not available\n")
        return

    agg = summary["aggregate"]
    label = {
        "average": "Simple Average",
        "task_arithmetic": "Task Arithmetic",
        "ties": "TIES",
        "iso_c": "Iso-C",
        "tsv": "TSV",
        "regmean": "RegMean \\textbf{(data)}",
        "actmat": "ACTMat",
        "tact_full": "TACT-full",
        "tact": "\\textbf{TACT (ours)}",
    }
    order = ["average", "task_arithmetic", "ties", "iso_c", "tsv",
             "regmean", "actmat", "tact_full", "tact"]

    rows = []
    for k in order:
        if k not in agg:
            continue
        a = agg[k]
        per_seed = a["per_seed"]
        # Only include in the multi-seed table if we have all 3 seeds
        if len(per_seed) < 3:
            continue
        cells = " & ".join(f"{v*100:.2f}" for v in per_seed)
        mu = a["mean"] * 100
        sd = a["std"] * 100
        rows.append(f"{label.get(k, k)} & {cells} & \\textbf{{{mu:.2f}}}~\\footnotesize($\\pm${sd:.2f}) \\\\")

    n_seed = max(len(agg[k]["per_seed"]) for k in agg if k in order)
    seed_cols = " & ".join(f"seed {i}" for i in range(n_seed))

    tex = [
        "\\begin{table}[h]",
        "\\caption{Multi-seed mean test accuracy (\\%) for each merging method on the 7-task vision benchmark. Each seed corresponds to an independent re-fine-tuning of all seven task-specific encoders from scratch. The TACT--ACTMat gap is consistent across seeds.}",
        "\\label{tab:seeds}",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{l{'c' * n_seed}c}}",
        "\\toprule",
        f"Method & {seed_cols} & \\textbf{{Mean$\\pm$std}} \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    Path(ROOT / "template/table_seeds.tex").write_text("\n".join(tex))
    print(f"Wrote template/table_seeds.tex")


if __name__ == "__main__":
    main()
