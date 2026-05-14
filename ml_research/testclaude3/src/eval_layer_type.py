"""Layer-type attribution of TACT's gain.

Starting from the ACTMat baseline (no cleaning), apply TACT-style magnitude trim
only to a *subset* of layer types and measure the accuracy. This isolates which
layer types contribute most to the TACT vs ACTMat gap.

Layer types in CLIP ViT visual encoder:
  - attn_in:   ``attn.in_proj_weight`` (q/k/v concatenation, [3*D, D])
  - attn_out:  ``attn.out_proj.weight`` ([D, D])
  - mlp_fc:    ``mlp.c_fc.weight`` (up-projection, [4D, D])
  - mlp_proj:  ``mlp.c_proj.weight`` (down-projection, [D, 4D])
  - conv1:     ``visual.conv1.weight`` (patch-embedding, conv2d treated separately)

We re-run a customised TACT solve where, for each Linear weight, we use the
*cleaned* covariance \tilde C_t if the layer is in the selected group and the
*raw* covariance otherwise. The merge target is always the raw fine-tuned
matrix (matches the cov-only TACT variant, which is what wins in §5.3).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate_merge import (
    build_eval_loaders, eval_state_on_all, load_artifacts,
)
from src.merging import (
    _solve_regmean, _trim_topk, is_2d_weight, task_vectors,
)
from src.model import load_clip


TASKS = ["MNIST", "SVHN", "CIFAR10", "CIFAR100", "EuroSAT", "GTSRB", "DTD"]


LAYER_GROUPS = {
    "all": lambda name: True,
    "attn_in": lambda name: "attn.in_proj_weight" in name,
    "attn_out": lambda name: "attn.out_proj.weight" in name,
    "mlp_fc": lambda name: "mlp.c_fc.weight" in name,
    "mlp_proj": lambda name: "mlp.c_proj.weight" in name,
    "all_attn": lambda name: "attn.in_proj_weight" in name or "attn.out_proj.weight" in name,
    "all_mlp": lambda name: "mlp.c_fc.weight" in name or "mlp.c_proj.weight" in name,
    "none": lambda name: False,
}


def tact_selective(theta0, thetas, keep_frac=0.5, selector=None, reg_eps=1e-8):
    """Run a TACT-style solve where only layers whose name matches ``selector``
    use the trimmed task vectors for the data-free covariance; other layers use
    the raw task vectors (= ACTMat).

    The merge target is always the full fine-tuned matrix W_t.
    """
    taus = task_vectors(theta0, thetas)
    trimmed = [_trim_topk(t, keep_frac) for t in taus]

    def lookup(name, t):
        base = theta0[name]
        if base.ndim == 2 and is_2d_weight(name, base):
            if selector is not None and selector(name):
                d = trimmed[t][name].to(torch.float32)
            else:
                d = taus[t][name].to(torch.float32)
            return d.T @ d
        return None

    return _solve_regmean(theta0, thetas, lookup, reg_eps=reg_eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--keep-frac", type=float, default=0.5)
    ap.add_argument("--ckpt-subdir", default="checkpoints")
    ap.add_argument("--max-test", type=int, default=4000)
    ap.add_argument("--out", default=str(ROOT / "results/layer_type.json"))
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f"Loading on {device} ...", flush=True)
    theta0, thetas, arts = load_artifacts(TASKS, device, ckpt_subdir=args.ckpt_subdir)
    eval_loaders = build_eval_loaders(TASKS, batch_size=256, num_workers=4,
                                       max_test=args.max_test)
    clip_model, _, _ = load_clip(device=device)

    # First count how many layers each group covers
    n_layers = {g: 0 for g in LAYER_GROUPS}
    for k in theta0:
        if theta0[k].ndim == 2 and is_2d_weight(k, theta0[k]):
            for g, sel in LAYER_GROUPS.items():
                if sel(k):
                    n_layers[g] += 1
    print("Layer counts per group:", n_layers, flush=True)

    summary = {"keep_frac": args.keep_frac, "n_layers_per_group": n_layers,
               "groups": {}}

    for group, selector in LAYER_GROUPS.items():
        t0 = time.time()
        merged = tact_selective(theta0, thetas, keep_frac=args.keep_frac,
                                 selector=selector)
        accs = eval_state_on_all(merged, arts, eval_loaders, clip_model, device)
        mean_acc = sum(accs.values()) / len(accs)
        dt = time.time() - t0
        summary["groups"][group] = {
            "mean_acc": mean_acc, "per_task": accs, "wall_s": dt,
            "n_layers": n_layers[group],
        }
        print(f"group={group:10s} ({n_layers[group]:2d} layers): mean acc = {mean_acc*100:.2f}%  ({dt:.1f}s)",
              flush=True)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
