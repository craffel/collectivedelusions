"""Compute per-task numbers for TIES (best config) and Iso-C, then merge
into results/results.json."""
from __future__ import annotations
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_task
from src.models import ClipClassifier, build_processor, build_vision_encoder
from src.merge import ties, iso_c
from src.train_expert import make_collate, evaluate


def vision_pretrained():
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


def main():
    ckpt_dir = "./checkpoints"
    tasks = sorted([f[:-3] for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    print("tasks:", tasks)
    experts = {t: torch.load(os.path.join(ckpt_dir, f"{t}.pt"),
                              map_location="cpu", weights_only=False) for t in tasks}
    expert_sds = [experts[t]["vision_state_dict"] for t in tasks]
    base = vision_pretrained()
    processor = build_processor()
    dev = torch.device("cuda:0")

    # Load existing results
    with open("results/results.json") as f:
        payload = json.load(f)
    method_results = payload["method_results"]
    method_meta = payload["method_meta"]
    method_means = payload["method_means"]

    # TIES at best from sweep: alpha=0.8, density=0.2.
    print("=== TIES alpha=0.8 density=0.2 ===")
    merged = ties(base, expert_sds, alpha=0.8, density=0.2)
    res = evaluate_merged(merged, experts, processor, dev)
    method_results["ties"] = res
    method_meta["ties"] = {"alpha": 0.8, "density": 0.2}
    method_means["ties"] = sum(res.values()) / len(res)
    print("ties:", res, " mean=", method_means["ties"])

    # Iso-C: try a much wider alpha range. Original alpha 0.5 gave 0.10 which is broken;
    # the issue may be the scale. Original Iso-C scaling is *T* (sum of tasks).
    print("=== Iso-C broader alpha sweep ===")
    best = (-1, None, None)
    for a in [1.0, 2.0, 4.0, 7.0, 10.0]:
        merged = iso_c(base, expert_sds, alpha=a)
        res = evaluate_merged(merged, experts, processor, dev)
        mean = sum(res.values()) / len(res)
        print(f"  alpha={a} mean={mean:.4f}")
        if mean > best[0]:
            best = (mean, a, res)
    method_results["iso_c"] = best[2]
    method_meta["iso_c"] = {"alpha": best[1]}
    method_means["iso_c"] = best[0]
    print("iso_c best:", method_meta["iso_c"], "mean=", best[0])

    payload["method_results"] = method_results
    payload["method_meta"] = method_meta
    payload["method_means"] = method_means
    with open("results/results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("updated results.json")


if __name__ == "__main__":
    main()
