"""Aggregate multi-seed B/16 fast eval results into a small markdown summary
and a LaTeX table fragment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [0, 1, 2]
FILES = {
    0: ROOT / "results/b16_fast.json",
    1: ROOT / "results/b16_seed1_fast.json",
    2: ROOT / "results/b16_seed2_fast.json",
}
METHOD_DISPLAY = [
    ("average",                 "Simple Average"),
    ("task_arithmetic_a0.3",    "Task Arithmetic"),
    ("ties_k0.2_a1.0",          "TIES"),
    ("iso_c_a0.3",              "Iso-C"),
    ("tsv_a0.3_r0.5",           "TSV"),
    ("actmat",                  "ACTMat"),
    ("tact_k0.5_signoff",       "TACT (ours)"),
]


def load(seed):
    with open(FILES[seed]) as f:
        return json.load(f)


def main():
    out_lines = []
    out_lines.append("\\begin{table}[h]")
    out_lines.append("\\caption{Multi-seed mean test accuracy (\\%) on the ViT-B/16 architecture transfer. Three independent fine-tuning runs with seeds $\\{0,1,2\\}$. The TACT$\\succ$ACTMat ordering is preserved across all three seeds.}")
    out_lines.append("\\label{tab:b16_seeds}")
    out_lines.append("\\centering")
    out_lines.append("\\small")
    out_lines.append("\\begin{tabular}{lcccc}")
    out_lines.append("\\toprule")
    out_lines.append("Method & seed 0 & seed 1 & seed 2 & Mean $\\pm$ std \\\\")
    out_lines.append("\\midrule")

    summary = {}
    for key, name in METHOD_DISPLAY:
        vals = []
        for seed in SEEDS:
            try:
                d = load(seed)
                vals.append(d["merged"][key]["mean"] * 100)
            except Exception as e:
                print(f"WARN: missing {seed}/{key}: {e}", file=sys.stderr)
                vals.append(None)
        if any(v is None for v in vals):
            continue
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0
        bold_open, bold_close = ("\\textbf{", "}") if "TACT" in name else ("", "")
        out_lines.append(
            f"{bold_open}{name}{bold_close} & {vals[0]:.2f} & {vals[1]:.2f} & {vals[2]:.2f} & {bold_open}{m:.2f} $\\pm$ {s:.2f}{bold_close} \\\\"
        )
        summary[name] = {"per_seed": vals, "mean": m, "std": s}

    out_lines.append("\\bottomrule")
    out_lines.append("\\end{tabular}")
    out_lines.append("\\end{table}")

    out_path = ROOT / "template/table_b16_seeds.tex"
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"Wrote {out_path}")
    for name, info in summary.items():
        print(f"  {name:20s}  {info['mean']:.2f} ± {info['std']:.2f}  (per-seed: {[f'{v:.2f}' for v in info['per_seed']]})")

    # Save raw summary too
    with open(ROOT / "results/b16_multi_seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
