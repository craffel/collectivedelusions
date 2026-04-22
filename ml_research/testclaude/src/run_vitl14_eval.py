"""Run merging baselines + SCALE on ViT-L/14 checkpoints.

Designed as a cycle-5 cross-backbone validation. Produces:
  results/vitl14_results.json
with per-method, per-task accuracy and the average.
"""
import argparse
import json
import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(__file__))
from datasets_utils import get_dataset
from clip_utils import load_clip, load_visual_state_dict
from merging import (
    simple_average, task_arithmetic, ties_merging, actmat, scale_merge,
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
def evaluate_on_task(clip_model, task, text_classifier, device, batch_size=256):
    tf = build_eval_transform()
    _, test_ds, _ = get_dataset(task, tf, tf)
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


def to_device_sd(sd, device):
    return {k: v.to(device) for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints_vitl14")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--out", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/results/vitl14_results.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", default="MNIST,CIFAR10,CIFAR100,SVHN,FashionMNIST,EuroSAT,GTSRB,DTD")
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    device = args.device

    pre = torch.load(os.path.join(args.ckpt_dir, "pretrained.pt"), map_location="cpu", weights_only=False)
    ckpts = {}
    for t in tasks:
        c = torch.load(os.path.join(args.ckpt_dir, f"{t}.pt"), map_location="cpu", weights_only=False)
        ckpts[t] = c
    task_sds = [ckpts[t]["state_dict"] for t in tasks]
    text_classifiers = {t: ckpts[t]["text_classifier"].to(device) for t in tasks}
    pre_cpu = {k: v.float() for k, v in pre.items()}
    task_sds_cpu = [{k: v.float() for k, v in sd.items()} for sd in task_sds]

    print(f"Loading CLIP {args.model}...")
    clip_model, _, _ = load_clip(model_name=args.model, pretrained="openai", device=device)

    def run_eval(name, merged_sd):
        load_visual_state_dict(clip_model, merged_sd, strict=False)
        clip_model.to(device)
        per_task = {}
        for t in tasks:
            acc = evaluate_on_task(clip_model, t, text_classifiers[t], device)
            per_task[t] = acc
            print(f"  [{name}] {t}: {acc:.4f}")
        avg = sum(per_task.values()) / len(per_task)
        print(f"  [{name}] AVG: {avg:.4f}")
        return per_task, avg

    results = {"tasks": tasks, "methods": {}}

    # Zero-shot (pretrained)
    print("\n== Pretrained (zero-shot) ==")
    pt, avg = run_eval("pretrained", pre_cpu)
    results["methods"]["pretrained"] = {"per_task": pt, "avg": avg}

    # Individual upper bound
    print("\n== Individual ==")
    per_task = {}
    for t in tasks:
        load_visual_state_dict(clip_model, task_sds_cpu[tasks.index(t)], strict=False)
        clip_model.to(device)
        acc = evaluate_on_task(clip_model, t, text_classifiers[t], device)
        per_task[t] = acc
        print(f"  [individual] {t}: {acc:.4f}")
    avg = sum(per_task.values()) / len(per_task)
    print(f"  [individual] AVG: {avg:.4f}")
    results["methods"]["individual"] = {"per_task": per_task, "avg": avg}

    # Simple averaging
    print("\n== Simple Averaging ==")
    merged = simple_average(pre_cpu, task_sds_cpu)
    pt, avg = run_eval("simple_avg", merged)
    results["methods"]["simple_avg"] = {"per_task": pt, "avg": avg}

    # Task Arithmetic sweep
    print("\n== Task Arithmetic ==")
    best = None
    for alpha in [0.2, 0.3]:
        merged = task_arithmetic(pre_cpu, task_sds_cpu, alpha=alpha)
        pt, avg = run_eval(f"task_arith_a{alpha}", merged)
        if best is None or avg > best[1]:
            best = (alpha, avg, pt)
    results["methods"]["task_arith_best"] = {"alpha": best[0], "avg": best[1], "per_task": best[2]}

    # TIES sweep
    print("\n== TIES ==")
    best = None
    for alpha in [0.3, 0.5]:
        merged = ties_merging(pre_cpu, task_sds_cpu, keep_frac=0.2, alpha=alpha)
        pt, avg = run_eval(f"ties_k0.2_a{alpha}", merged)
        if best is None or avg > best[1]:
            best = (alpha, avg, pt)
    results["methods"]["ties_best"] = {"keep_frac": 0.2, "alpha": best[0], "avg": best[1], "per_task": best[2]}

    # ACTMat
    print("\n== ACTMat ==")
    pre_gpu = to_device_sd(pre_cpu, device)
    tsds_gpu = [to_device_sd(sd, device) for sd in task_sds_cpu]
    merged = actmat(pre_gpu, tsds_gpu, ridge=1e-4)
    merged = {k: v.cpu() for k, v in merged.items()}
    pt, avg = run_eval("actmat", merged)
    results["methods"]["actmat"] = {"per_task": pt, "avg": avg}

    # SCALE sweep
    print("\n== SCALE ==")
    best = None
    for k in [0.2, 0.3, 0.5]:
        merged = scale_merge(pre_gpu, tsds_gpu, keep_frac=k, ridge=1e-4, use_sign_election=False)
        merged_cpu = {kk: v.cpu() for kk, v in merged.items()}
        pt, avg = run_eval(f"scale_k{k}_s0", merged_cpu)
        if best is None or avg > best[1]:
            best = (k, avg, pt)
    results["methods"]["scale_best"] = {"keep_frac": best[0], "avg": best[1], "per_task": best[2]}

    del tsds_gpu, pre_gpu
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")

    # Summary
    print("\n=== Summary ===")
    for name, d in results["methods"].items():
        if "avg" in d:
            extras = ""
            if "keep_frac" in d:
                extras = f" (k={d['keep_frac']})"
            if "alpha" in d:
                extras += f" (alpha={d['alpha']})"
            print(f"  {name:25s} avg={d['avg']*100:.2f}%{extras}")


if __name__ == "__main__":
    main()
