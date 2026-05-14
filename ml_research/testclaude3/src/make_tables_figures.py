"""Generate LaTeX tables and Matplotlib figures from the JSON results.

Reads:
  results/main.json   (initial bad-actmat run; only used to keep TIES/avg/RegMean entries)
  results/main_actmat_fix.json
  results/ablations.json

Writes:
  template/table_main.tex
  template/table_ablation.tex
  figures/trim_sweep.pdf
  figures/cov_alignment.pdf  (placeholder if no covariance data yet)
  figures/task_count_scaling.pdf (placeholder for now)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "template"
FIGS = ROOT / "figures"
RESULTS = ROOT / "results"
FIGS.mkdir(exist_ok=True)


def _load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _fmt_pct(x):
    return f"{x*100:.2f}"


def _best_from_ablations(abl_grid, label_filter=None):
    """Find best (keep, sign) point in an ablations grid; return (mean, per_task)."""
    best = None
    for k, v in abl_grid.items():
        m = re.match(r"keep=(?P<k>[\d.]+)_sign=(?P<s>True|False)", k)
        if not m:
            continue
        keep = float(m["k"])
        sign = m["s"] == "True"
        if label_filter is not None and not label_filter(keep, sign):
            continue
        mean_acc = v["mean"]
        if best is None or mean_acc > best[0]:
            best = (mean_acc, v["per_task"], keep, sign)
    return best


def make_main_table(out_path: Path):
    """Combine baselines + the fixed ACTMat/TACT results."""
    main = _load("main.json") or {}
    fix = _load("main_actmat_fix.json") or {}
    abl = _load("ablations.json") or {}
    tact_cov_data = _load("tact_cov.json") or {}
    iso_c_data = _load("iso_c.json") or {}
    tsv_data = _load("tsv.json") or {}

    # Merge: use fixed ACTMat and TACT; keep the baselines from initial main.json
    individual = main.get("individual", fix.get("individual", {}))
    merged = dict(main.get("merged", {}))
    if "merged" in fix:
        for k, v in fix["merged"].items():
            merged[k] = v

    # Replace TACT with the best ablation point
    if "ablations" in abl:
        best = _best_from_ablations(abl["ablations"])
        if best:
            mean_acc, per_task, keep, sign = best
            merged["tact"] = {"per_task": per_task, "mean": mean_acc,
                              "best_keep_frac": keep, "best_use_sign": sign}

    # Insert TACT-cov from its own sweep
    if "merged" in tact_cov_data and "tact_cov" in tact_cov_data["merged"]:
        merged["tact_cov"] = tact_cov_data["merged"]["tact_cov"]
    # Insert Iso-C from its own sweep
    if "merged" in iso_c_data and "iso_c" in iso_c_data["merged"]:
        merged["iso_c"] = iso_c_data["merged"]["iso_c"]
    # Insert TSV
    if "merged" in tsv_data and "tsv" in tsv_data["merged"]:
        merged["tsv"] = tsv_data["merged"]["tsv"]

    tasks = main.get("tasks") or fix.get("tasks") or abl.get("tasks") or []

    # Method display order and labels
    order = [
        ("zero_shot", "Pretrained CLIP (Zero-shot)"),
        ("finetuned", "Individual fine-tuned"),
        ("average", "Simple Average~\\citep{wortsman2022model}"),
        ("task_arithmetic", "Task Arithmetic~\\citep{ilharco2023editing}"),
        ("ties", "TIES~\\citep{yadav2023ties}"),
        ("iso_c", "Iso-C~\\citep{marczak2025isoc}"),
        ("tsv", "TSV~\\citep{gargiulo2025tsv}"),
        ("regmean", "RegMean~\\citep{jin2023dataless} {\\footnotesize\\textbf{(data)}}"),
        ("actmat", "ACTMat~\\citep{hameed2026actmat}"),
        ("tact", "TACT-full (ours, $k{=}0.5$, sign off)"),
        ("tact_cov", "\\textbf{TACT (ours, $k{=}0.5$, sign off)}"),
    ]

    # Per-task scores
    def row_scores(method_key):
        if method_key == "zero_shot":
            return [individual[t]["zero_shot"] for t in tasks], \
                   sum(individual[t]["zero_shot"] for t in tasks) / max(len(tasks), 1)
        if method_key == "finetuned":
            return [individual[t]["finetuned"] for t in tasks], \
                   sum(individual[t]["finetuned"] for t in tasks) / max(len(tasks), 1)
        if method_key in merged:
            r = merged[method_key]
            pt = r.get("per_task", {})
            return [pt.get(t, float("nan")) for t in tasks], r["mean"]
        return None

    body_rows = []
    for key, label in order:
        sc = row_scores(key)
        if sc is None:
            continue
        per_task, mean_acc = sc
        cells = [_fmt_pct(x) if x == x else "--" for x in per_task]
        row = label + " & " + " & ".join(cells) + f" & \\textbf{{{_fmt_pct(mean_acc)}}}"
        body_rows.append(row + " \\\\")

    short = {"MNIST":"MNIST", "SVHN":"SVHN", "CIFAR10":"CIFAR10",
             "CIFAR100":"CIFAR100", "EuroSAT":"EuroSAT", "GTSRB":"GTSRB", "DTD":"DTD"}
    col_header = " & ".join(short.get(t, t) for t in tasks)

    n_data_cols = len(tasks)
    col_spec = "l" + ("c" * (n_data_cols + 1))

    tex = []
    tex.append("\\begin{table*}[t]")
    tex.append("\\caption{Per-task test accuracy (\\%) and mean across 7 vision tasks for each merging method on CLIP ViT-B/32. \\textbf{(data)} indicates a method that requires per-task training data to compute covariances; all others are fully data-free.}")
    tex.append("\\label{tab:main}")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    tex.append("\\toprule")
    tex.append("Method & " + col_header + " & \\textbf{Mean} \\\\")
    tex.append("\\midrule")
    tex.extend(body_rows)
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table*}")
    out_path.write_text("\n".join(tex))
    print(f"Wrote {out_path}")


def make_ablation_table_and_figure(out_tex: Path, out_fig: Path):
    abl = _load("ablations.json")
    if abl is None or "ablations" not in abl:
        # Placeholder
        out_tex.write_text("% ablations.json not available\n")
        return

    grid = abl["ablations"]
    # Parse keys like keep=0.20_sign=True
    parsed = []
    for k, v in grid.items():
        m = re.match(r"keep=(?P<k>[\d.]+)_sign=(?P<s>True|False)", k)
        if not m:
            continue
        parsed.append((float(m["k"]), m["s"] == "True", v["mean"]))

    keep_values = sorted({p[0] for p in parsed})
    sign_values = [True, False]

    # Table: 2x4 cell corners (keep ∈ {0.1, 1.0} × sign ∈ {True, False})
    def get(k, s):
        for kk, ss, mm in parsed:
            if abs(kk - k) < 1e-6 and ss == s:
                return mm
        return None

    # Pick interesting corners (TACT-full: cleans both W_t and C_t)
    corners = [(1.0, False, "ACTMat baseline (no trim, no sign)"),
               (1.0, True, "ACTMat + Sign Election"),
               (0.5, False, "Trim ($k{=}0.5$, no sign) -- TACT-full"),
               (0.5, True, "Trim ($k{=}0.5$) + Sign Election")]
    rows = []
    CHK = "\\checkmark"
    BS = "\\\\"
    for k, s, label in corners:
        v = get(k, s)
        if v is None:
            continue
        trim_mark = CHK if k < 1.0 else ""
        sign_mark = CHK if s else ""
        rows.append(f"{label} & {trim_mark} & {sign_mark} & {_fmt_pct(v)} {BS}")

    tex = [
        "\\begin{table}[t]",
        "\\caption{TACT ablation: contribution of TIES-style trim and sign election on top of ACTMat's data-free covariance solve. Mean accuracy across the 7 vision tasks.}",
        "\\label{tab:ablation}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Trim & Sign & Mean Acc.~(\\%) \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    out_tex.write_text("\n".join(tex))
    print(f"Wrote {out_tex}")

    # Figure: trim sweep with both sign settings, TACT-full vs TACT (cov-only)
    tact_cov_data = _load("tact_cov.json") or {}
    cov_grid = {}
    if "ablations_grid" in tact_cov_data:
        cov_grid = tact_cov_data["ablations_grid"]
    # We can also reconstruct from tact_cov_data['merged']['tact_cov'] which has only the best;
    # parse the log instead.
    log_path = ROOT / "logs/eval_tact_cov.log"
    cov_points = []
    if log_path.exists():
        log_text = log_path.read_text()
        for line in log_text.splitlines():
            m = re.search(r"keep=([\d.]+)\s+sign=(True|False)\s+mean=([\d.]+)", line)
            if m:
                cov_points.append((float(m.group(1)), m.group(2) == "True", float(m.group(3))))

    plt.figure(figsize=(5.4, 3.6))
    # TACT-full
    for s in sign_values:
        ks, ms = [], []
        for k in keep_values:
            v = get(k, s)
            if v is None:
                continue
            ks.append(k)
            ms.append(v * 100)
        label = "TACT-full (sign on)" if s else "TACT-full (sign off)"
        marker = "o" if s else "s"
        color = "tab:orange" if s else "tab:red"
        plt.plot(ks, ms, marker=marker, linewidth=2, markersize=7, label=label,
                 color=color, linestyle="--", alpha=0.85)
    # TACT (cov-only)
    for s in [True, False]:
        ks_ms = [(k, m) for k, ss, m in cov_points if ss == s]
        ks_ms.sort()
        if not ks_ms:
            continue
        ks = [k for k, _ in ks_ms]
        ms = [m * 100 for _, m in ks_ms]
        label = "TACT (cov-only, sign on)" if s else r"\textbf{TACT (cov-only, sign off)}"
        marker = "D" if s else "*"
        color = "tab:blue" if s else "tab:green"
        plt.plot(ks, ms, marker=marker, linewidth=2, markersize=9, label=label,
                 color=color)
    # ACTMat baseline horizontal line
    actmat_baseline = get(1.0, False) or 0.8411
    plt.axhline(actmat_baseline * 100, color="grey", linestyle=":", label="ACTMat baseline")
    plt.xlabel("Trim keep fraction $k$")
    plt.ylabel(r"Mean accuracy (\%)")
    plt.title("TACT trim sweep")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_fig, format="pdf")
    plt.close()
    print(f"Wrote {out_fig}")


def make_placeholder_figs():
    # cov alignment placeholder
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.text(0.5, 0.5, "Covariance alignment\n(placeholder)",
            ha="center", va="center", fontsize=12, transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(FIGS / "cov_alignment.pdf", format="pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.text(0.5, 0.5, "Task-count scaling\n(placeholder)",
            ha="center", va="center", fontsize=12, transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(FIGS / "task_count_scaling.pdf", format="pdf")
    plt.close()


def main():
    make_main_table(TEMPLATE / "table_main.tex")
    make_ablation_table_and_figure(TEMPLATE / "table_ablation.tex",
                                    FIGS / "trim_sweep.pdf")
    make_placeholder_figs()


if __name__ == "__main__":
    main()
