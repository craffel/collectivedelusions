"""Seed robustness of SCALE with sign-election (k=0.3, sign=YES).

Runs on each of the three seed checkpoint sets and appends a
`scale_sign` entry to results/seed_robustness.json under each seed,
plus an aggregate summary entry.

Usage:
    python src/run_seed_sign.py --seeds 42,123,456 --device cuda:0
"""
import argparse
import json
import os
import sys
import torch

torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(__file__))
from clip_utils import load_clip, load_visual_state_dict
from merging import scale_merge
from run_experiments import evaluate_on_task


ROOT = "/fsx/craffel/collectivedelusions/ml_research/testclaude"
TASKS = ["MNIST", "CIFAR10", "CIFAR100", "SVHN", "FashionMNIST", "EuroSAT", "GTSRB", "DTD"]


def to_device_sd(sd, device):
    return {k: v.to(device) for k, v in sd.items()}


def load_ckpts(ckpt_dir):
    ckpts = {}
    for t in TASKS:
        path = os.path.join(ckpt_dir, f"{t}.pt")
        if not os.path.exists(path):
            return None
        c = torch.load(path, map_location="cpu", weights_only=False)
        ckpts[t] = c
    return ckpts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--k", type=float, default=0.3)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--path", default=os.path.join(ROOT, "results/seed_robustness.json"))
    args = ap.parse_args()

    device = args.device
    seeds = [int(s) for s in args.seeds.split(",")]

    with open(args.path) as f:
        out = json.load(f)

    pre = torch.load(os.path.join(ROOT, "checkpoints/pretrained.pt"),
                     map_location="cpu", weights_only=False)
    pre_cpu = {k: v.float() for k, v in pre.items()}

    print("Loading CLIP backbone...")
    clip_model, _, _ = load_clip(device=device)

    for seed in seeds:
        ckpt_dir = os.path.join(ROOT, f"checkpoints_seed{seed}")
        ckpts = load_ckpts(ckpt_dir)
        if ckpts is None:
            print(f"Skipping seed {seed}: checkpoints not ready.")
            continue
        text_classifiers = {t: ckpts[t]["text_classifier"].to(device) for t in TASKS}
        task_sds_cpu = [{k: v.float() for k, v in ckpts[t]["state_dict"].items()} for t in TASKS]
        pre_gpu = to_device_sd(pre_cpu, device)
        tsds_gpu = [to_device_sd(sd, device) for sd in task_sds_cpu]

        print(f"[seed {seed}] SCALE (k={args.k}, sign=YES)...")
        merged = scale_merge(pre_gpu, tsds_gpu, keep_frac=args.k,
                             ridge=args.ridge, use_sign_election=True)
        merged_cpu = {k: v.cpu() for k, v in merged.items()}
        load_visual_state_dict(clip_model, merged_cpu, strict=False)
        clip_model.to(device)
        per_task = {}
        for t in TASKS:
            acc = evaluate_on_task(clip_model, t, text_classifiers[t], device)
            per_task[t] = acc
            print(f"  SCALE+sign [{t}]: {acc:.4f}")
        avg = sum(per_task.values()) / len(per_task)
        out["per_seed"][str(seed)]["methods"]["scale_sign"] = {
            "per_task": per_task, "avg": avg
        }
        del tsds_gpu, pre_gpu
        with open(args.path, "w") as f:
            json.dump(out, f, indent=2)

    # Aggregate for scale_sign.
    per_seed = out["per_seed"]
    avgs = [per_seed[str(s)]["methods"]["scale_sign"]["avg"] for s in seeds
            if "scale_sign" in per_seed[str(s)]["methods"]]
    if avgs:
        mean_avg = sum(avgs) / len(avgs)
        std_avg = (sum((a - mean_avg) ** 2 for a in avgs) / len(avgs)) ** 0.5
        per_task_mean = {}
        per_task_std = {}
        for t in TASKS:
            vals = [per_seed[str(s)]["methods"]["scale_sign"]["per_task"][t]
                    for s in seeds if "scale_sign" in per_seed[str(s)]["methods"]]
            m = sum(vals) / len(vals)
            per_task_mean[t] = m
            per_task_std[t] = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        out.setdefault("aggregate", {})["scale_sign"] = {
            "avg_mean": mean_avg, "avg_std": std_avg,
            "per_task_mean": per_task_mean, "per_task_std": per_task_std,
        }
        # Gap scale_sign - actmat
        gaps = [per_seed[str(s)]["methods"]["scale_sign"]["avg"]
                - per_seed[str(s)]["methods"]["actmat"]["avg"]
                for s in seeds if "scale_sign" in per_seed[str(s)]["methods"]]
        mg = sum(gaps) / len(gaps)
        out["aggregate"]["scale_sign_minus_actmat"] = {
            "mean": mg, "std": (sum((g - mg) ** 2 for g in gaps) / len(gaps)) ** 0.5,
            "per_seed": gaps,
        }

    with open(args.path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Updated {args.path}")
    if "scale_sign" in out.get("aggregate", {}):
        v = out["aggregate"]["scale_sign"]
        print(f"SCALE+sign avg: {v['avg_mean']*100:.2f}±{v['avg_std']*100:.2f}")


if __name__ == "__main__":
    main()
