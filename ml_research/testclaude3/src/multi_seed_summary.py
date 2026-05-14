"""Aggregate multi-seed eval results and report mean ± std per method.

Reads:
  results/main.json (seed=0 baselines)
  results/main_actmat_fix.json (seed=0 ACTMat, TACT)
  results/ablations.json
  results/tact_cov.json
  results/iso_c.json
  results/tsv.json
  results/seed1.json
  results/seed2.json (if available)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.load(open(p))


def collect_seed0():
    """Pull each method's best (seed=0) result from across files."""
    main = _load("main.json") or {}
    fix = _load("main_actmat_fix.json") or {}
    abl = _load("ablations.json") or {}
    tactcov = _load("tact_cov.json") or {}
    isoc = _load("iso_c.json") or {}
    tsv = _load("tsv.json") or {}
    res = {}
    # baselines from main
    for m in ("average", "task_arithmetic", "ties", "regmean"):
        if m in main.get("merged", {}):
            res[m] = main["merged"][m]["mean"]
    # ACTMat from fix
    if "actmat" in fix.get("merged", {}):
        res["actmat"] = fix["merged"]["actmat"]["mean"]
    # tact (best from ablations)
    if "ablations" in abl:
        best = None
        for k, v in abl["ablations"].items():
            mean_acc = v["mean"]
            if best is None or mean_acc > best:
                best = mean_acc
        res["tact_full"] = best
    # tact_cov (best from tact_cov.json's merged)
    if "tact_cov" in tactcov.get("merged", {}):
        res["tact"] = tactcov["merged"]["tact_cov"]["mean"]
    # iso_c, tsv
    if "iso_c" in isoc.get("merged", {}):
        res["iso_c"] = isoc["merged"]["iso_c"]["mean"]
    if "tsv" in tsv.get("merged", {}):
        res["tsv"] = tsv["merged"]["tsv"]["mean"]
    return res


def collect_seed(seed: int):
    """Read results/seed{seed}.json + seed{seed}_extra.json into a dict of mean accs."""
    d = _load(f"seed{seed}.json")
    if d is None:
        return None
    out = {name: r["mean"] for name, r in d["merged"].items()}
    extra = _load(f"seed{seed}_extra.json")
    if extra is not None:
        for name in ("iso_c", "tsv", "regmean"):
            if name in extra:
                out[name] = extra[name]["mean"]
    return out


def main():
    seed0 = collect_seed0()
    seed1 = collect_seed(1)
    seed2 = collect_seed(2)

    print("Per-seed results (mean accuracy):")
    methods = ["average", "task_arithmetic", "ties", "iso_c", "tsv", "regmean",
               "actmat", "tact_full", "tact"]
    # seed1 uses different names
    seed1_aliases = {"tact_cov_k0.5": "tact", "tact_full_k0.5": "tact_full"}
    if seed1:
        seed1_normalized = {seed1_aliases.get(k, k): v for k, v in seed1.items()}
    else:
        seed1_normalized = None
    seed2_normalized = {seed1_aliases.get(k, k): v for k, v in seed2.items()} if seed2 else None

    out = {"seed0": seed0, "seed1": seed1_normalized, "seed2": seed2_normalized,
           "aggregate": {}}

    print(f"\n{'method':25s}  seed0    seed1    seed2    mean±std")
    print("-" * 70)
    for m in methods:
        vals = []
        for s in (seed0, seed1_normalized, seed2_normalized):
            if s and m in s:
                vals.append(s[m])
        if not vals:
            continue
        v0 = seed0.get(m, float("nan"))
        v1 = seed1_normalized.get(m, float("nan")) if seed1_normalized else float("nan")
        v2 = seed2_normalized.get(m, float("nan")) if seed2_normalized else float("nan")
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=0))
        out["aggregate"][m] = {"mean": mu, "std": sd, "n_seeds": len(vals),
                               "per_seed": [float(x) for x in vals]}
        def fmt(v):
            return f"{v*100:6.2f}" if v == v else "  -   "
        print(f"{m:25s}  {fmt(v0)}  {fmt(v1)}  {fmt(v2)}  {mu*100:6.2f} ± {sd*100:.2f}")

    with open(RESULTS / "multi_seed_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {RESULTS}/multi_seed_summary.json")


if __name__ == "__main__":
    main()
