"""Benchmark SCALE-Merge: direct LU solve vs MaTS-style matrix-free conjugate gradient.

For each backbone (ViT-B/32, ViT-L/14) we run both merging variants on the 8-task
checkpoints, reporting:
  - merge wall-clock time
  - peak GPU memory during merge
  - resulting avg accuracy across tasks
  - for CG: mean/max CG iterations per layer

Output: results/cg_timing.json
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
from merging import scale_merge, scale_merge_cg


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


def run_backbone(model_name, ckpt_dir, keep_frac, tasks, device):
    print(f"\n======= Backbone: {model_name} =======")
    pre = torch.load(os.path.join(ckpt_dir, "pretrained.pt"), map_location="cpu", weights_only=False)
    ckpts = {}
    for t in tasks:
        path = os.path.join(ckpt_dir, f"{t}.pt")
        ckpts[t] = torch.load(path, map_location="cpu", weights_only=False)

    pre_cpu = {k: v.float() for k, v in pre.items()}
    task_sds_cpu = [{k: v.float() for k, v in ckpts[t]["state_dict"].items()} for t in tasks]
    text_classifiers = {t: ckpts[t]["text_classifier"].to(device) for t in tasks}

    pre_gpu = to_device_sd(pre_cpu, device)
    tsds_gpu = [to_device_sd(sd, device) for sd in task_sds_cpu]

    clip_model, _, _ = load_clip(model_name=model_name, pretrained="openai", device=device)

    results = {"backbone": model_name, "keep_frac": keep_frac, "tasks": tasks, "variants": {}}

    for variant_name, fn in [("direct", "direct"), ("cg", "cg")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.time()
        if fn == "direct":
            merged = scale_merge(pre_gpu, tsds_gpu, keep_frac=keep_frac, ridge=1e-4,
                                 use_sign_election=False)
            cg_stats = None
        else:
            merged, cg_stats = scale_merge_cg(pre_gpu, tsds_gpu, keep_frac=keep_frac, ridge=1e-4,
                                              use_sign_election=False, cg_iters=50, cg_tol=1e-6,
                                              return_stats=True)
        torch.cuda.synchronize(device)
        t1 = time.time()
        peak_bytes = torch.cuda.max_memory_allocated(device)
        wall = t1 - t0

        merged_cpu = {k: v.cpu() for k, v in merged.items()}
        load_visual_state_dict(clip_model, merged_cpu, strict=False)
        clip_model.to(device)
        per_task = {}
        for t in tasks:
            acc = evaluate_on_task(clip_model, t, text_classifiers[t], device)
            per_task[t] = acc
            print(f"  [{variant_name}] {t}: {acc:.4f}")
        avg = sum(per_task.values()) / len(per_task)
        print(f"  [{variant_name}] AVG: {avg:.4f}  time={wall:.3f}s  peak_gpu={peak_bytes/2**30:.3f}GiB")
        entry = {"per_task": per_task, "avg": avg, "merge_seconds": wall,
                 "peak_gpu_bytes": int(peak_bytes), "peak_gpu_gib": peak_bytes / 2**30}
        if cg_stats is not None:
            n = max(cg_stats["layers"], 1)
            entry["cg_mean_iters"] = cg_stats["iters_total"] / n
            entry["cg_max_iters"] = cg_stats["iters_max"]
            entry["cg_layers"] = cg_stats["layers"]
        results["variants"][variant_name] = entry

        # free merged dict promptly
        del merged
        torch.cuda.empty_cache()

    del tsds_gpu, pre_gpu
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/results/cg_timing.json")
    ap.add_argument("--tasks", default="MNIST,CIFAR10,CIFAR100,SVHN,FashionMNIST,EuroSAT,GTSRB,DTD")
    ap.add_argument("--ckpt_dir_b32", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints")
    ap.add_argument("--ckpt_dir_l14", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints_vitl14")
    ap.add_argument("--skip_l14", action="store_true")
    args = ap.parse_args()
    tasks = args.tasks.split(",")

    out = {"backbones": {}}
    b32 = run_backbone("ViT-B-32", args.ckpt_dir_b32, keep_frac=0.3, tasks=tasks, device=args.device)
    out["backbones"]["ViT-B/32"] = b32
    if not args.skip_l14:
        l14 = run_backbone("ViT-L-14", args.ckpt_dir_l14, keep_frac=0.5, tasks=tasks, device=args.device)
        out["backbones"]["ViT-L/14"] = l14

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}")

    # Summary
    print("\n=== Summary ===")
    for bname, bentry in out["backbones"].items():
        for vname, v in bentry["variants"].items():
            extras = ""
            if "cg_mean_iters" in v:
                extras = f"  cg_iters=mean {v['cg_mean_iters']:.1f}, max {v['cg_max_iters']}"
            print(f"  {bname:<10}  {vname:<7}  avg={v['avg']*100:5.2f}%  "
                  f"merge={v['merge_seconds']:6.2f}s  peak={v['peak_gpu_gib']:6.3f}GiB{extras}")


if __name__ == "__main__":
    main()
