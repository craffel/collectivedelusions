"""Run ablations separate from the main pipeline:
 - Keep-fraction sweep for SCALE (k in {0.05, 0.1, 0.2, 0.3, 0.5, 1.0})
 - Trim-only vs sign-only vs both
 - Output: results/ablations.json
"""
import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(__file__))
from datasets_utils import get_dataset
from clip_utils import load_clip, load_visual_state_dict
from merging import scale_merge, actmat, ties_merging


def build_tf():
    mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


@torch.no_grad()
def eval_task(clip_model, task, text_cls, device, bs=512, max_samples=None):
    tf = build_tf()
    _, test_ds, _ = get_dataset(task, tf, tf)
    if max_samples and len(test_ds) > max_samples:
        test_ds = torch.utils.data.Subset(test_ds, list(range(max_samples)))
    loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    clip_model.eval()
    ls = float(clip_model.logit_scale.detach().exp().item())
    text_cls = text_cls.to(device)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        f = clip_model.encode_image(x); f = F.normalize(f, dim=-1)
        logits = ls * f @ text_cls.T
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints")
    ap.add_argument("--out", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/results/ablations.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", default="MNIST,CIFAR10,CIFAR100,SVHN,FashionMNIST,EuroSAT,GTSRB,DTD")
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    device = args.device
    max_samples = args.max_samples if args.max_samples > 0 else None

    pre = torch.load(os.path.join(args.ckpt_dir, "pretrained.pt"), map_location="cpu", weights_only=False)
    ckpts = {t: torch.load(os.path.join(args.ckpt_dir, f"{t}.pt"), map_location="cpu", weights_only=False) for t in tasks}
    task_sds = [ckpts[t]["state_dict"] for t in tasks]
    text_cls = {t: ckpts[t]["text_classifier"].to(device) for t in tasks}

    clip_model, _, _ = load_clip(device=device)

    pre_cpu = {k: v.float() for k, v in pre.items()}
    tsds_cpu = [{k: v.float() for k, v in sd.items()} for sd in task_sds]
    pre_gpu = {k: v.to(device) for k, v in pre_cpu.items()}
    tsds_gpu = [{k: v.to(device) for k, v in sd.items()} for sd in tsds_cpu]

    def evaluate_sd(sd):
        load_visual_state_dict(clip_model, sd, strict=False)
        clip_model.to(device)
        per = {}
        for t in tasks:
            per[t] = eval_task(clip_model, t, text_cls[t], device, max_samples=max_samples)
        avg = sum(per.values()) / len(per)
        return per, avg

    results = {"tasks": tasks, "methods": {}}

    # Keep-fraction sweep (with sign election)
    print("\n== Keep-fraction sweep (with sign election) ==")
    for k in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        merged = scale_merge(pre_gpu, tsds_gpu, keep_frac=k, ridge=1e-4, use_sign_election=True)
        merged_cpu = {kk: v.cpu() for kk, v in merged.items()}
        per, avg = evaluate_sd(merged_cpu)
        name = f"scale_sign_k{k}"
        results["methods"][name] = {"per_task": per, "avg": avg, "k": k, "sign": True}
        print(f"  {name}: avg={avg:.4f}")

    # Keep-fraction sweep (no sign election)
    print("\n== Keep-fraction sweep (no sign election) ==")
    for k in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        merged = scale_merge(pre_gpu, tsds_gpu, keep_frac=k, ridge=1e-4, use_sign_election=False)
        merged_cpu = {kk: v.cpu() for kk, v in merged.items()}
        per, avg = evaluate_sd(merged_cpu)
        name = f"scale_nosign_k{k}"
        results["methods"][name] = {"per_task": per, "avg": avg, "k": k, "sign": False}
        print(f"  {name}: avg={avg:.4f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to", args.out)


if __name__ == "__main__":
    main()
