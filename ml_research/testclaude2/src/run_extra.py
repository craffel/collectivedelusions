"""Extra experiments for the refinement phase, GPU-accelerated.

Runs additional configurations to characterise TRIM:
  - Sign-only TRIM with no trim, at multiple "density" placeholders
    (this is what really tests our hypothesis)
  - TRIM with density 0.7, 0.9 (closer to ACTMat)
  - A version of ACTMat with a sign-elected ∆ used ONLY for the
    covariance estimate, keeping the full W_t in the merge formula
    (cov_clean_only)
  - Counterfactual: a version where we use the absolute value of
    sign-conflicted entries in the covariance (replace negative
    contributions with their absolute value).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_task
from src.models import ClipClassifier, build_processor, build_vision_encoder
from src.merge import (
    actmat, trim_actmat, ties_mask, _pinv_chol,
)
from src.train_expert import make_collate, evaluate


def vision_pretrained_state_dict():
    enc = build_vision_encoder()
    return {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}


def evaluate_merged(merged_vision_sd, experts, processor, device, bs=256,
                   num_workers=2, max_eval_samples=4000, data_root="./data"):
    out = {}
    for task, ckpt in experts.items():
        num_classes = ckpt["num_classes"]
        model = ClipClassifier(num_classes).to(device)
        model.vision.load_state_dict({k: v.to(device) for k, v in merged_vision_sd.items()})
        model.head.load_state_dict({k: v.to(device) for k, v in ckpt["head_state_dict"].items()})
        _, val_ds, _, _ = load_task(task, data_root)
        if len(val_ds) > max_eval_samples:
            g = torch.Generator().manual_seed(1)
            idx = torch.randperm(len(val_ds), generator=g)[:max_eval_samples].tolist()
            val_ds = Subset(val_ds, idx)
        loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=num_workers, collate_fn=make_collate(processor),
                            pin_memory=True)
        out[task] = evaluate(model, loader, device)
        del model
        torch.cuda.empty_cache()
    return out


def cov_sign_clean_actmat(base, experts, sign_clean: bool = True,
                          device=None, eps: float = 1e-5,
                          fallback_alpha: float = 0.3):
    """ACTMat where the ∆ used in W_t merging is full, but the covariance is
    computed from sign-cleaned ∆̂_t.  This isolates the effect on Ĉ_t."""
    dev = device if device is not None else torch.device("cpu")
    out = {}
    for k, v0 in base.items():
        if v0.ndim != 2:
            # fallback
            deltas = torch.stack([e[k].to(torch.float32) - v0.to(torch.float32) for e in experts]).mean(0)
            out[k] = (v0.to(torch.float32) + fallback_alpha * deltas).to(v0.dtype)
            continue
        deltas = torch.stack([(e[k] - v0).to(torch.float64).to(dev) for e in experts])
        if sign_clean:
            elected = torch.sign(deltas.sum(0))
            elected[elected == 0] = 1.0
            match = (torch.sign(deltas) == elected.unsqueeze(0)).to(deltas.dtype)
            d_hat = deltas * match
        else:
            d_hat = deltas
        Cts = [d_hat[t].transpose(-2, -1) @ d_hat[t] for t in range(d_hat.shape[0])]
        Csum = sum(Cts)
        Csum_inv = _pinv_chol(Csum, eps=eps)
        W0 = v0.to(torch.float64).to(dev)
        acc = torch.zeros_like(W0)
        for t in range(len(experts)):
            Wt = experts[t][k].to(torch.float64).to(dev)
            acc = acc + Wt @ (Cts[t] @ Csum_inv)
        out[k] = acc.to(v0.dtype).cpu()
        del W0, Cts, Csum, Csum_inv, acc
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="./checkpoints")
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="./results/extra.json")
    ap.add_argument("--max_eval_samples", type=int, default=4000)
    ap.add_argument("--data_root", default="./data")
    args = ap.parse_args()

    if not args.tasks:
        args.tasks = sorted([f[:-3] for f in os.listdir(args.ckpt_dir) if f.endswith(".pt")])
    print("tasks:", args.tasks)

    experts = {}
    for t in args.tasks:
        experts[t] = torch.load(os.path.join(args.ckpt_dir, f"{t}.pt"),
                                map_location="cpu", weights_only=False)
    base = vision_pretrained_state_dict()
    expert_sds = [experts[t]["vision_state_dict"] for t in args.tasks]
    processor = build_processor()
    dev = torch.device(args.device)

    results = {}

    # 1. cov_sign_cleaned ACTMat: keep full W_t, clean only Ĉ_t.
    print("=== cov_sign_clean (W_t intact, Ĉ sign-cleaned) ===")
    merged = cov_sign_clean_actmat(base, expert_sds, sign_clean=True, device=dev)
    res = evaluate_merged(merged, experts, processor, dev,
                          max_eval_samples=args.max_eval_samples,
                          data_root=args.data_root)
    results["actmat_covclean"] = res
    print("actmat_covclean:", res, " mean=", sum(res.values())/len(res))

    # 2. Sign-only TRIM with no trim (density=1.0)
    print("=== trim_actmat sign_only density=1.0 ===")
    merged = trim_actmat(base, expert_sds, density=1.0, sign_resolve=True, trim=False, device=dev)
    res = evaluate_merged(merged, experts, processor, dev,
                          max_eval_samples=args.max_eval_samples,
                          data_root=args.data_root)
    results["trim_sign_only_full"] = res
    print("trim_sign_only_full:", res, " mean=", sum(res.values())/len(res))

    # 3. TRIM density 0.7
    for d in [0.7, 0.9]:
        print(f"=== trim_actmat full d={d} ===")
        merged = trim_actmat(base, expert_sds, density=d, sign_resolve=True, trim=True, device=dev)
        res = evaluate_merged(merged, experts, processor, dev,
                              max_eval_samples=args.max_eval_samples,
                              data_root=args.data_root)
        results[f"trim_d{d}"] = res
        print(f"trim_d{d}:", res, " mean=", sum(res.values())/len(res))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
