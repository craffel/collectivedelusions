"""Aggregate per-layer trim ablation across 3 seeds on B/16; print summary."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
SEED_FILES = [
    ROOT / "results/per_layer_trim_b16.json",
    ROOT / "results/per_layer_trim_b16_seed1.json",
    ROOT / "results/per_layer_trim_b16_seed2.json",
]
KEEPS = [0.1, 0.2, 0.3, 0.5, 0.7]


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    dicts = [load(p) for p in SEED_FILES]
    print("\n  k  | Global (B/16, 3-seed)              | Per-layer (B/16, 3-seed)            | Δ")
    print("-" * 100)
    for k in KEEPS:
        gkey = f"global_k{k:.1f}_signoff"
        pkey = f"perlayer_k{k:.1f}"
        gvals = [d[gkey]["mean"] * 100 for d in dicts]
        pvals = [d[pkey]["mean"] * 100 for d in dicts]
        gm, gs = statistics.mean(gvals), statistics.stdev(gvals)
        pm, ps = statistics.mean(pvals), statistics.stdev(pvals)
        seeds_g = " ".join(f"{v:6.2f}" for v in gvals)
        seeds_p = " ".join(f"{v:6.2f}" for v in pvals)
        print(f"  {k:.1f}| {seeds_g}  ({gm:6.2f} ± {gs:5.2f}) | {seeds_p}  ({pm:6.2f} ± {ps:5.2f}) | {pm-gm:+6.2f}")


if __name__ == "__main__":
    main()
