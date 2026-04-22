"""Run SCALE-Merge and ACTMat on each seed's checkpoints and aggregate.

Usage:
    python src/run_seed_robustness.py --seeds 42,123,456 --device cuda:0

Writes results/seed_robustness.json with per-seed and per-task accuracies, plus
mean and std across seeds for each method.
"""
import argparse
import json
import os
import sys
import time
import torch

torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(__file__))
from clip_utils import load_clip, load_visual_state_dict
from merging import actmat, scale_merge
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
    ap.add_argument("--k", type=float, default=0.3, help="SCALE keep fraction")
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/seed_robustness.json"))
    args = ap.parse_args()

    device = args.device
    seeds = [int(s) for s in args.seeds.split(",")]

    # Load pretrained (shared across seeds; pretrained weights are identical for all seeds).
    pre = torch.load(os.path.join(ROOT, "checkpoints/pretrained.pt"), map_location="cpu", weights_only=False)
    pre_cpu = {k: v.float() for k, v in pre.items()}

    print("Loading CLIP backbone...")
    clip_model, _, _ = load_clip(device=device)

    out = {"seeds": seeds, "tasks": TASKS, "k": args.k, "ridge": args.ridge,
           "per_seed": {}}

    for seed in seeds:
        ckpt_dir = os.path.join(ROOT, f"checkpoints_seed{seed}")
        ckpts = load_ckpts(ckpt_dir)
        if ckpts is None:
            print(f"Skipping seed {seed}: checkpoints not ready.")
            continue
        task_sds_full = [ckpts[t]["state_dict"] for t in TASKS]
        text_classifiers = {t: ckpts[t]["text_classifier"].to(device) for t in TASKS}
        task_sds_cpu = [{k: v.float() for k, v in sd.items()} for sd in task_sds_full]
        pre_gpu = to_device_sd(pre_cpu, device)
        tsds_gpu = [to_device_sd(sd, device) for sd in task_sds_cpu]

        seed_out = {"individual_acc": {}, "methods": {}}

        # Individual fine-tuned accuracies (check fidelity of this seed's training)
        for i, t in enumerate(TASKS):
            seed_out["individual_acc"][t] = float(ckpts[t]["acc"])

        # ACTMat
        print(f"[seed {seed}] ACTMat...")
        merged = actmat(pre_gpu, tsds_gpu, ridge=args.ridge)
        merged_cpu = {k: v.cpu() for k, v in merged.items()}
        load_visual_state_dict(clip_model, merged_cpu, strict=False)
        clip_model.to(device)
        per_task = {}
        for t in TASKS:
            acc = evaluate_on_task(clip_model, t, text_classifiers[t], device)
            per_task[t] = acc
            print(f"  ACTMat [{t}]: {acc:.4f}")
        seed_out["methods"]["actmat"] = {"per_task": per_task, "avg": sum(per_task.values())/len(per_task)}

        # SCALE (trim-only, matching paper default)
        print(f"[seed {seed}] SCALE (k={args.k}, no-sign)...")
        merged = scale_merge(pre_gpu, tsds_gpu, keep_frac=args.k, ridge=args.ridge, use_sign_election=False)
        merged_cpu = {k: v.cpu() for k, v in merged.items()}
        load_visual_state_dict(clip_model, merged_cpu, strict=False)
        clip_model.to(device)
        per_task = {}
        for t in TASKS:
            acc = evaluate_on_task(clip_model, t, text_classifiers[t], device)
            per_task[t] = acc
            print(f"  SCALE [{t}]: {acc:.4f}")
        seed_out["methods"]["scale"] = {"per_task": per_task, "avg": sum(per_task.values())/len(per_task)}

        out["per_seed"][seed] = seed_out
        del tsds_gpu, pre_gpu

        # Save incrementally
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    # Aggregate
    if out["per_seed"]:
        agg = {}
        for method in ["actmat", "scale"]:
            avgs = [out["per_seed"][s]["methods"][method]["avg"] for s in out["per_seed"]]
            per_task_means = {}
            per_task_stds = {}
            for t in TASKS:
                vals = [out["per_seed"][s]["methods"][method]["per_task"][t] for s in out["per_seed"]]
                per_task_means[t] = sum(vals) / len(vals)
                m = per_task_means[t]
                per_task_stds[t] = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            mean_avg = sum(avgs) / len(avgs)
            std_avg = (sum((a - mean_avg) ** 2 for a in avgs) / len(avgs)) ** 0.5
            agg[method] = {
                "avg_mean": mean_avg,
                "avg_std": std_avg,
                "per_task_mean": per_task_means,
                "per_task_std": per_task_stds,
            }
        # Per-seed ACTMat/SCALE gaps
        gaps = [out["per_seed"][s]["methods"]["scale"]["avg"] - out["per_seed"][s]["methods"]["actmat"]["avg"]
                for s in out["per_seed"]]
        agg["scale_minus_actmat"] = {"mean": sum(gaps)/len(gaps),
                                      "std": (sum((g - sum(gaps)/len(gaps)) ** 2 for g in gaps) / len(gaps)) ** 0.5,
                                      "per_seed": gaps}
        out["aggregate"] = agg

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")
    if "aggregate" in out:
        for m, v in out["aggregate"].items():
            print(m, v if m == "scale_minus_actmat" else {"avg": f"{v['avg_mean']*100:.2f}±{v['avg_std']*100:.2f}"})


if __name__ == "__main__":
    main()
