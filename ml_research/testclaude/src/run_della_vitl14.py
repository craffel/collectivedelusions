"""Focused DELLA sweep on ViT-L/14 — 4 configs near ViT-B/32 optimum."""
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
from merging import della


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
def eval_loader(clip_model, loader, text_cls, device):
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
    ap.add_argument("--ckpt_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints_vitl14")
    ap.add_argument("--out", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/results/della_vitl14.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", default="MNIST,CIFAR10,CIFAR100,SVHN,FashionMNIST,EuroSAT,GTSRB,DTD")
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    device = args.device
    print(f"[init] loading {len(tasks)} tasks (ViT-L/14)", flush=True)

    pre = torch.load(os.path.join(args.ckpt_dir, "pretrained.pt"), map_location="cpu", weights_only=False)
    ckpts = {t: torch.load(os.path.join(args.ckpt_dir, f"{t}.pt"), map_location="cpu", weights_only=False) for t in tasks}
    task_sds = [ckpts[t]["state_dict"] for t in tasks]
    text_cls = {t: ckpts[t]["text_classifier"].to(device) for t in tasks}

    clip_model, _, _ = load_clip(model_name="ViT-L-14", pretrained="openai", device=device)
    pre_cpu = {k: v.float() for k, v in pre.items()}
    tsds_cpu = [{k: v.float() for k, v in sd.items()} for sd in task_sds]
    pre_gpu = {k: v.to(device) for k, v in pre_cpu.items()}
    tsds_gpu = [{k: v.to(device) for k, v in sd.items()} for sd in tsds_cpu]

    tf = build_tf()
    loaders = {}
    for t in tasks:
        _, test_ds, _ = get_dataset(t, tf, tf)
        loaders[t] = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    print("[init] loaders ready", flush=True)

    def evaluate_sd(sd):
        load_visual_state_dict(clip_model, sd, strict=False)
        clip_model.to(device)
        per = {}
        for t in tasks:
            per[t] = eval_loader(clip_model, loaders[t], text_cls[t], device)
        return per, sum(per.values()) / len(per)

    # Focused sweep around ViT-B/32 optimum (p_low=0.05, p_high=0.8, alpha=0.5).
    # ViT-L/14 has larger parameter count; probe nearby points.
    sweep = [
        (0.05, 0.8, 0.3),
        (0.05, 0.8, 0.5),
        (0.05, 0.9, 0.5),
        (0.1,  0.8, 0.5),
    ]

    out = {"tasks": tasks, "configs": {}}
    best = None
    for i, (p_low, p_high, alpha) in enumerate(sweep):
        t0 = time.time()
        merged = della(pre_gpu, tsds_gpu, p_low=p_low, p_high=p_high, alpha=alpha, seed=0)
        merged = {k: v.cpu() for k, v in merged.items()}
        per, avg = evaluate_sd(merged)
        name = f"della_pl{p_low}_ph{p_high}_a{alpha}"
        out["configs"][name] = {"p_low": p_low, "p_high": p_high, "alpha": alpha,
                                "per_task": per, "avg": avg}
        print(f"[{i+1}/{len(sweep)}] {name}: {avg*100:.2f}  ({time.time()-t0:.1f}s)", flush=True)
        if best is None or avg > best[1]:
            best = (name, avg, per, p_low, p_high, alpha)

    out["best"] = {"name": best[0], "avg": best[1], "per_task": best[2],
                   "p_low": best[3], "p_high": best[4], "alpha": best[5]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nBest: {best[0]} avg={best[1]*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
