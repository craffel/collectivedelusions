"""Evaluate Trim-Mat vs ACTMat as a function of number of tasks merged.

For each n in {2, 3, 4, 5, 6, 7}, pick a deterministic subset of n tasks,
merge with ACTMat and Trim-Mat (rho=0.5), and report mean accuracy on the
merged tasks (only the n merged tasks are evaluated).

This shows whether Trim-Mat's advantage grows or shrinks with task count.
"""
from __future__ import annotations
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_task
from src.models import ClipClassifier, build_processor, build_vision_encoder
from src.merge import actmat, trim_actmat
from src.train_expert import make_collate, evaluate


def vision_pretrained():
    enc = build_vision_encoder()
    return {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}


def evaluate_merged(merged, experts, processor, device, bs=256, max_eval=4000):
    out = {}
    for task, ckpt in experts.items():
        m = ClipClassifier(ckpt["num_classes"]).to(device)
        m.vision.load_state_dict({k: v.to(device) for k, v in merged.items()})
        m.head.load_state_dict({k: v.to(device) for k, v in ckpt["head_state_dict"].items()})
        _, val_ds, _, _ = load_task(task, "./data")
        if len(val_ds) > max_eval:
            g = torch.Generator().manual_seed(1)
            idx = torch.randperm(len(val_ds), generator=g)[:max_eval].tolist()
            val_ds = Subset(val_ds, idx)
        loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2,
                            collate_fn=make_collate(processor), pin_memory=True)
        out[task] = evaluate(m, loader, device)
        del m
        torch.cuda.empty_cache()
    return out


def main():
    # deterministic order; we keep an "easy" task and a "hard" task in every subset
    full = ["cifar10", "cifar100", "dtd", "eurosat", "fashionmnist", "gtsrb", "mnist", "svhn"]
    subsets_by_n = {
        2: ["cifar10", "mnist"],
        3: ["cifar10", "mnist", "dtd"],
        4: ["cifar10", "mnist", "dtd", "gtsrb"],
        5: ["cifar10", "mnist", "dtd", "gtsrb", "cifar100"],
        6: ["cifar10", "mnist", "dtd", "gtsrb", "cifar100", "fashionmnist"],
        7: ["cifar10", "mnist", "dtd", "gtsrb", "cifar100", "fashionmnist", "svhn"],
        8: full,
    }

    all_experts = {t: torch.load(f"./checkpoints/{t}.pt", map_location="cpu",
                                  weights_only=False) for t in full}
    base = vision_pretrained()
    processor = build_processor()
    dev = torch.device("cuda:0")

    results = {}
    for n, subset in subsets_by_n.items():
        experts = {t: all_experts[t] for t in subset}
        expert_sds = [experts[t]["vision_state_dict"] for t in subset]
        print(f"=== n={n} subset={subset} ===", flush=True)
        # ACTMat
        merged = actmat(base, expert_sds, device=dev)
        res_a = evaluate_merged(merged, experts, processor, dev)
        mean_a = sum(res_a.values()) / len(res_a)
        # Trim-Mat
        merged = trim_actmat(base, expert_sds, density=0.5,
                             sign_resolve=False, trim=True, device=dev)
        res_t = evaluate_merged(merged, experts, processor, dev)
        mean_t = sum(res_t.values()) / len(res_t)
        results[n] = {
            "subset": subset,
            "actmat": res_a, "actmat_mean": mean_a,
            "trim_mat": res_t, "trim_mat_mean": mean_t,
        }
        print(f"  ACTMat mean={mean_a:.4f}  Trim-Mat mean={mean_t:.4f}  diff={mean_t-mean_a:+.4f}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/scaling.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results/scaling.json", flush=True)


if __name__ == "__main__":
    main()
