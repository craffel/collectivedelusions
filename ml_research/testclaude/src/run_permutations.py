"""Robustness sweep: SCALE / ACTMat / TIES / Task-Arith at T in {4, 6} across random task permutations.

Confirms that the task-count scaling trend is not an artifact of alphabetical ordering.
"""
import argparse, json, os, random, sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

torch.backends.cudnn.enabled = False
sys.path.insert(0, os.path.dirname(__file__))
from datasets_utils import get_dataset
from clip_utils import load_clip, load_visual_state_dict
from merging import scale_merge, actmat, ties_merging, task_arithmetic


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
    ap.add_argument("--out", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/results/permutations.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_perms", type=int, default=3)
    ap.add_argument("--Ts", default="4,6")
    ap.add_argument("--tasks", default="MNIST,CIFAR10,CIFAR100,SVHN,FashionMNIST,EuroSAT,GTSRB,DTD")
    args = ap.parse_args()

    all_tasks = args.tasks.split(",")
    device = args.device
    Ts = [int(x) for x in args.Ts.split(",")]

    pre = torch.load(os.path.join(args.ckpt_dir, "pretrained.pt"), map_location="cpu", weights_only=False)
    ckpts = {t: torch.load(os.path.join(args.ckpt_dir, f"{t}.pt"), map_location="cpu", weights_only=False) for t in all_tasks}
    task_sds = {t: ckpts[t]["state_dict"] for t in all_tasks}
    text_cls = {t: ckpts[t]["text_classifier"].to(device) for t in all_tasks}

    clip_model, _, _ = load_clip(device=device)

    pre_gpu = {k: v.to(device).float() for k, v in pre.items()}
    tsd_gpu = {t: {k: v.to(device).float() for k, v in sd.items()} for t, sd in task_sds.items()}

    def evaluate_sd(sd, eval_tasks):
        load_visual_state_dict(clip_model, sd, strict=False)
        clip_model.to(device)
        per = {}
        for t in eval_tasks:
            per[t] = eval_task(clip_model, t, text_cls[t], device)
        return per, sum(per.values()) / len(per)

    results = []
    rng = random.Random(0)
    perms = []
    perms.append(list(all_tasks))
    for s in range(args.n_perms - 1):
        p = list(all_tasks)
        rng.shuffle(p)
        perms.append(p)

    # SCALE per-T best-k (from taskcount_k.json)
    scale_k = {2: 1.0, 4: 0.2, 6: 0.3, 8: 0.3}

    for pi, perm in enumerate(perms):
        print(f"\n=== Permutation {pi}: {perm} ===", flush=True)
        for T in Ts:
            sub = perm[:T]
            sub_tsds = [tsd_gpu[t] for t in sub]
            methods = {}
            k = scale_k.get(T, 0.3)
            m = scale_merge(pre_gpu, sub_tsds, keep_frac=k, ridge=1e-4, use_sign_election=False)
            per, avg = evaluate_sd({kk: v.cpu() for kk, v in m.items()}, sub)
            methods[f"scale_k{k}"] = {"per_task": per, "avg": avg}
            print(f"  T={T} scale(k={k}): {avg:.4f}", flush=True)

            m = actmat(pre_gpu, sub_tsds, ridge=1e-4)
            per, avg = evaluate_sd({kk: v.cpu() for kk, v in m.items()}, sub)
            methods["actmat"] = {"per_task": per, "avg": avg}
            print(f"  T={T} actmat:       {avg:.4f}", flush=True)

            m = ties_merging(pre_gpu, sub_tsds, alpha=0.5, keep_frac=0.2)
            per, avg = evaluate_sd({kk: v.cpu() for kk, v in m.items()}, sub)
            methods["ties"] = {"per_task": per, "avg": avg}
            print(f"  T={T} ties:         {avg:.4f}", flush=True)

            m = task_arithmetic(pre_gpu, sub_tsds, alpha=0.2)
            per, avg = evaluate_sd({kk: v.cpu() for kk, v in m.items()}, sub)
            methods["task_arith"] = {"per_task": per, "avg": avg}
            print(f"  T={T} task_arith:   {avg:.4f}", flush=True)

            results.append({"perm_idx": pi, "T": T, "tasks": sub, "methods": methods})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to", args.out)


if __name__ == "__main__":
    main()
