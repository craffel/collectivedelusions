"""Quick eval: TACT-global-trim vs TACT-per-layer-trim, on B/32 seed 0."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate_merge import build_eval_loaders, eval_state_on_all, load_artifacts
from src.model import load_clip
from src.merging import tact_cov_only, tact_per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out", default="results/per_layer_trim.json")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arch", default=None,
                    help="If given, sets CLIP_ARCH (e.g., ViT-B-16-quickgelu).")
    ap.add_argument("--tasks", nargs="+",
                    default=["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"])
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    os.makedirs(Path(args.out).parent, exist_ok=True)

    print("Loading artifacts...", flush=True)
    theta0, thetas, arts = load_artifacts(args.tasks, device, ckpt_subdir=args.ckpt_dir)
    print("Loading CLIP...", flush=True)
    clip_model, _, _ = load_clip(device=device)
    print("Building eval loaders...", flush=True)
    eval_loaders = build_eval_loaders(args.tasks, batch_size=128, num_workers=2)

    results = {}

    def run(name, theta_merged):
        t0 = time.time()
        accs = eval_state_on_all(theta_merged, arts, eval_loaders, clip_model, device)
        m = sum(accs.values()) / len(accs)
        results[name] = {"per_task": accs, "mean": m, "time_s": time.time() - t0}
        print(f"  {name:30s}  mean={m*100:6.2f}%  ({time.time()-t0:.1f}s)", flush=True)

    print("\n=== Global vs per-layer trim, k sweep ===\n")
    for k in [0.1, 0.2, 0.3, 0.5, 0.7]:
        run(f"global_k{k:.1f}_signoff", tact_cov_only(theta0, thetas, keep_frac=k, use_sign=False))
        run(f"perlayer_k{k:.1f}",     tact_per_layer(theta0, thetas, keep_frac=k))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
