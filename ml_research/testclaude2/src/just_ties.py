"""Compute per-task TIES (best config) and write into results/results.json.

The original eval ran TIES at α∈{0.5,0.8,1.0,1.2} with density=0.2; the best
α was 0.8 (mean=0.5569). We only logged the per-task numbers at α=0.5, so
re-run at α=0.8.
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
from src.merge import ties
from src.train_expert import make_collate, evaluate


def vision_pretrained():
    enc = build_vision_encoder()
    return {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}


def evaluate_merged(merged, experts, processor, device, bs=256):
    out = {}
    for task, ckpt in experts.items():
        m = ClipClassifier(ckpt["num_classes"]).to(device)
        m.vision.load_state_dict({k: v.to(device) for k, v in merged.items()})
        m.head.load_state_dict({k: v.to(device) for k, v in ckpt["head_state_dict"].items()})
        _, val_ds, _, _ = load_task(task, "./data")
        if len(val_ds) > 4000:
            g = torch.Generator().manual_seed(1)
            idx = torch.randperm(len(val_ds), generator=g)[:4000].tolist()
            val_ds = Subset(val_ds, idx)
        loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2,
                            collate_fn=make_collate(processor), pin_memory=True)
        out[task] = evaluate(m, loader, device)
        del m
        torch.cuda.empty_cache()
    return out


def main():
    tasks = sorted([f[:-3] for f in os.listdir("./checkpoints") if f.endswith(".pt")])
    experts = {t: torch.load(f"./checkpoints/{t}.pt", map_location="cpu", weights_only=False)
               for t in tasks}
    expert_sds = [experts[t]["vision_state_dict"] for t in tasks]
    base = vision_pretrained()
    processor = build_processor()
    dev = torch.device("cuda:0")

    with open("results/results.json") as f:
        payload = json.load(f)

    print("=== TIES alpha=0.8 density=0.2 ===", flush=True)
    merged = ties(base, expert_sds, alpha=0.8, density=0.2)
    res = evaluate_merged(merged, experts, processor, dev)
    payload["method_results"]["ties"] = res
    payload["method_meta"]["ties"] = {"alpha": 0.8, "density": 0.2}
    payload["method_means"]["ties"] = sum(res.values()) / len(res)
    print("ties:", res, "mean=", payload["method_means"]["ties"], flush=True)

    with open("results/results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
