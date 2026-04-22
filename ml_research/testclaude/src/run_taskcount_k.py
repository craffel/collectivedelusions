"""Task-count ablation: sweep keep_frac for SCALE at each T.

For each T in {2,4,6,8} and k in {0.1, 0.2, 0.3, 0.5, 1.0}, report SCALE accuracy.
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

torch.backends.cudnn.enabled = False
sys.path.insert(0, os.path.dirname(__file__))
from datasets_utils import get_dataset
from clip_utils import load_clip, load_visual_state_dict
from merging import scale_merge, actmat


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
def eval_task(clip_model, task, text_cls, device, bs=512):
    tf = build_tf()
    _, test_ds, _ = get_dataset(task, tf, tf)
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
    ap.add_argument("--out", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/results/taskcount_k.json")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--tasks", default="MNIST,CIFAR10,CIFAR100,SVHN,FashionMNIST,EuroSAT,GTSRB,DTD")
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    device = args.device

    pre = torch.load(os.path.join(args.ckpt_dir, "pretrained.pt"), map_location="cpu", weights_only=False)
    ckpts = {t: torch.load(os.path.join(args.ckpt_dir, f"{t}.pt"), map_location="cpu", weights_only=False) for t in tasks}
    task_sds = [ckpts[t]["state_dict"] for t in tasks]
    text_cls = {t: ckpts[t]["text_classifier"].to(device) for t in tasks}

    clip_model, _, _ = load_clip(device=device)

    pre_cpu = {k: v.float() for k, v in pre.items()}
    tsds_cpu = [{k: v.float() for k, v in sd.items()} for sd in task_sds]
    pre_gpu = {k: v.to(device) for k, v in pre_cpu.items()}
    tsds_gpu = [{k: v.to(device) for k, v in sd.items()} for sd in tsds_cpu]

    def evaluate_sd(sd, eval_tasks):
        load_visual_state_dict(clip_model, sd, strict=False)
        clip_model.to(device)
        per = {}
        for t in eval_tasks:
            per[t] = eval_task(clip_model, t, text_cls[t], device)
        return per, sum(per.values()) / len(per)

    results = {}
    for T in [2, 4, 6, 8]:
        sub_tasks = tasks[:T]
        sub_tsds_gpu = [tsds_gpu[tasks.index(t)] for t in sub_tasks]
        best = (None, -1, None)
        for k in [0.1, 0.2, 0.3, 0.5, 1.0]:
            merged = scale_merge(pre_gpu, sub_tsds_gpu, keep_frac=k, ridge=1e-4, use_sign_election=False)
            merged_cpu = {kk: v.cpu() for kk, v in merged.items()}
            per, avg = evaluate_sd(merged_cpu, sub_tasks)
            name = f"T{T}_k{k}"
            results[name] = {"T": T, "k": k, "per_task": per, "avg": avg}
            print(f"  {name}: {avg:.4f}", flush=True)
            if avg > best[1]:
                best = (k, avg, per)
        results[f"T{T}_best"] = {"T": T, "k": best[0], "avg": best[1], "per_task": best[2]}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to", args.out)


if __name__ == "__main__":
    main()
