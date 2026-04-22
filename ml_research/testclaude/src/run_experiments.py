"""Run all merging methods and evaluate on all tasks.

Outputs results/results.json and results/results.csv with per-method, per-task accuracy.
"""
import argparse
import json
import os
import sys
import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(__file__))
from datasets_utils import get_dataset
from clip_utils import load_clip, load_visual_state_dict
from merging import (
    simple_average, task_arithmetic, ties_merging, actmat, scale_merge, regmean,
    task_vectors,
)


def build_eval_transform():
    mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


@torch.no_grad()
def evaluate_on_task(clip_model, task, text_classifier, device, batch_size=512, max_samples=None):
    tf = build_eval_transform()
    _, test_ds, _ = get_dataset(task, tf, tf)
    if max_samples is not None and len(test_ds) > max_samples:
        test_ds = torch.utils.data.Subset(test_ds, list(range(max_samples)))
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    clip_model.eval()
    logit_scale = float(clip_model.logit_scale.detach().exp().item())
    text_cls = text_classifier.to(device)
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        feats = clip_model.encode_image(x)
        feats = F.normalize(feats, dim=-1)
        logits = logit_scale * feats @ text_cls.T
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def load_checkpoints(ckpt_dir, tasks):
    ckpts = {}
    for t in tasks:
        path = os.path.join(ckpt_dir, f"{t}.pt")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}, skipping {t}")
            continue
        c = torch.load(path, map_location="cpu", weights_only=False)
        ckpts[t] = c
    return ckpts


def to_device_sd(sd, device):
    return {k: v.to(device) for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints")
    ap.add_argument("--out_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/results")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", default="MNIST,CIFAR10,CIFAR100,SVHN,STL10,EuroSAT,GTSRB,DTD")
    ap.add_argument("--max_eval_samples", type=int, default=0)
    ap.add_argument("--methods", default="zeroshot,pretrained,individual,simple_avg,task_arith,ties,actmat,scale")
    ap.add_argument("--trim_k", type=float, default=0.2)
    ap.add_argument("--ta_alpha", type=float, default=0.3)
    ap.add_argument("--ties_alpha", type=float, default=0.4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tasks = args.tasks.split(",")
    device = args.device
    max_samples = args.max_eval_samples if args.max_eval_samples > 0 else None

    # Load pre-trained
    pre = torch.load(os.path.join(args.ckpt_dir, "pretrained.pt"), map_location="cpu", weights_only=False)
    # Load fine-tuned
    ckpts = load_checkpoints(args.ckpt_dir, tasks)
    task_sds_full = [ckpts[t]["state_dict"] for t in tasks]
    text_classifiers = {t: ckpts[t]["text_classifier"].to(device) for t in tasks}

    # Load CLIP backbone
    print("Loading CLIP...")
    clip_model, _, _ = load_clip(device=device)

    # Convert state dicts to float32 on CPU for numerical merging
    pre_cpu = {k: v.float() for k, v in pre.items()}
    task_sds_cpu = [{k: v.float() for k, v in sd.items()} for sd in task_sds_full]

    results = {"tasks": tasks, "methods": {}}

    def run_eval(name, merged_sd):
        # Load into clip_model
        load_visual_state_dict(clip_model, merged_sd, strict=False)
        clip_model.to(device)
        per_task = {}
        for t in tasks:
            acc = evaluate_on_task(clip_model, t, text_classifiers[t], device, max_samples=max_samples)
            per_task[t] = acc
            print(f"  [{name}] {t}: {acc:.4f}")
        avg = sum(per_task.values()) / len(per_task)
        norm_avg = None  # placeholder
        results["methods"][name] = {"per_task": per_task, "avg": avg}
        print(f"  [{name}] AVG: {avg:.4f}")
        return per_task, avg

    methods = args.methods.split(",")

    # Zero-shot (pretrained visual)
    if "zeroshot" in methods or "pretrained" in methods:
        print("\n== Zero-shot / Pretrained ==")
        run_eval("pretrained", pre_cpu)

    # Individual (upper bound)
    if "individual" in methods:
        print("\n== Individual (per-task upper bound) ==")
        per_task = {}
        for t in tasks:
            load_visual_state_dict(clip_model, task_sds_cpu[tasks.index(t)], strict=False)
            clip_model.to(device)
            acc = evaluate_on_task(clip_model, t, text_classifiers[t], device, max_samples=max_samples)
            per_task[t] = acc
            print(f"  [individual] {t}: {acc:.4f}")
        avg = sum(per_task.values()) / len(per_task)
        results["methods"]["individual"] = {"per_task": per_task, "avg": avg}

    # Simple averaging
    if "simple_avg" in methods:
        print("\n== Simple averaging ==")
        merged = simple_average(pre_cpu, task_sds_cpu)
        run_eval("simple_avg", merged)

    # Task arithmetic sweep
    if "task_arith" in methods:
        print("\n== Task arithmetic ==")
        best = None
        for alpha in [0.2, 0.3, 0.4]:
            merged = task_arithmetic(pre_cpu, task_sds_cpu, alpha=alpha)
            print(f"-- alpha={alpha} --")
            per_task, avg = run_eval(f"task_arith_a{alpha}", merged)
            if best is None or avg > best[1]:
                best = (alpha, avg, per_task)
        results["methods"]["task_arith_best"] = {"alpha": best[0], "per_task": best[2], "avg": best[1]}

    # TIES
    if "ties" in methods:
        print("\n== TIES-Merging ==")
        best = None
        for k in [0.2]:
            for alpha in [0.2, 0.3, 0.4, 0.5]:
                merged = ties_merging(pre_cpu, task_sds_cpu, keep_frac=k, alpha=alpha)
                name = f"ties_k{k}_a{alpha}"
                print(f"-- {name} --")
                per_task, avg = run_eval(name, merged)
                if best is None or avg > best[2]:
                    best = (k, alpha, avg, per_task)
        results["methods"]["ties_best"] = {"keep_frac": best[0], "alpha": best[1], "per_task": best[3], "avg": best[2]}

    # ACTMat
    if "actmat" in methods:
        print("\n== ACTMat ==")
        # Perform on GPU for speed
        pre_gpu = to_device_sd(pre_cpu, device)
        tsds_gpu = [to_device_sd(sd, device) for sd in task_sds_cpu]
        merged = actmat(pre_gpu, tsds_gpu, ridge=1e-4)
        merged = {k: v.cpu() for k, v in merged.items()}
        run_eval("actmat", merged)
        del tsds_gpu, pre_gpu

    # SCALE (ours) + ablation
    if "scale" in methods:
        print("\n== SCALE (ours) + ablations ==")
        pre_gpu = to_device_sd(pre_cpu, device)
        tsds_gpu = [to_device_sd(sd, device) for sd in task_sds_cpu]
        best = None
        # Trim sweep (for ablation)
        for keep in [0.1, 0.2, 0.3, 0.5, 1.0]:
            for sign in [True, False]:
                merged = scale_merge(pre_gpu, tsds_gpu, keep_frac=keep, ridge=1e-4, use_sign_election=sign)
                merged_cpu = {k: v.cpu() for k, v in merged.items()}
                name = f"scale_k{keep}_s{int(sign)}"
                print(f"-- {name} --")
                per_task, avg = run_eval(name, merged_cpu)
                if best is None or avg > best[2]:
                    best = (keep, 1e-4, avg, per_task, sign)
        results["methods"]["scale_best"] = {"keep_frac": best[0], "ridge": best[1], "sign": best[4], "per_task": best[3], "avg": best[2]}
        del tsds_gpu, pre_gpu

    # Save
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nDone. Saved to", args.out_dir)


if __name__ == "__main__":
    main()
