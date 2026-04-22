"""Phase-5 refinement experiments.

Runs:
  1) New baselines:   DARE, DARE+TIES, Fisher-approx.
  2) Ridge sensitivity sweep for SCALE (rho in {1e-6..1e-1}).
  3) Task-count ablation: merge T in {2,4,6,8} tasks (deterministic subsets).
  4) Sign-agreement diagnostic: fraction of majority-sign agreement in raw vs trimmed
     task vectors (averaged across all merged linear layers).

Outputs: results/extras.json
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
    dare, dare_ties, fisher_approx_merge, task_vectors, topk_mask, _is_matrix_param,
    _reshape_to_matrix,
)


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


def sign_agreement_diagnostic(pre_sd, task_sds, keep_fracs=(1.0, 0.3, 0.2, 0.1)):
    """For each keep-fraction, compute mean majority-sign-agreement across all 2D+ params."""
    tvs = task_vectors(pre_sd, task_sds)
    results = {}
    for k in keep_fracs:
        agree_weighted = 0.0
        total_weight = 0.0
        for name in pre_sd:
            theta0 = pre_sd[name]
            if not _is_matrix_param(name, theta0):
                continue
            D_list = []
            for tv in tvs:
                if name in tv:
                    D, _ = _reshape_to_matrix(tv[name])
                    D_list.append(D.float())
            if len(D_list) < 2:
                continue
            # trim per-task
            if k < 1.0:
                D_list = [D * topk_mask(D, k).to(D.dtype) for D in D_list]
            stack = torch.stack(D_list, dim=0)  # [T, out, in]
            signs = torch.sign(stack)
            # majority sign per entry (ignoring zeros)
            pos = (signs > 0).sum(dim=0).float()
            neg = (signs < 0).sum(dim=0).float()
            nz = (signs != 0).sum(dim=0).float()  # nonzero count per entry
            # majority-sign agreement: max(pos, neg) / nz, where nz > 0
            maj = torch.maximum(pos, neg)
            # For entries with any nonzero, compute maj/nz
            valid = nz > 0
            if valid.sum() == 0:
                continue
            r = (maj[valid] / nz[valid]).mean().item()
            w = valid.sum().item()
            agree_weighted += r * w
            total_weight += w
        results[f"k{k}"] = agree_weighted / max(total_weight, 1.0)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints")
    ap.add_argument("--out", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/results/extras.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", default="MNIST,CIFAR10,CIFAR100,SVHN,FashionMNIST,EuroSAT,GTSRB,DTD")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--skip", default="")
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    skip = set(args.skip.split(",")) if args.skip else set()
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

    def evaluate_sd(sd, eval_tasks=None):
        load_visual_state_dict(clip_model, sd, strict=False)
        clip_model.to(device)
        use_tasks = eval_tasks if eval_tasks is not None else tasks
        per = {}
        for t in use_tasks:
            per[t] = eval_task(clip_model, t, text_cls[t], device, max_samples=max_samples)
        avg = sum(per.values()) / len(per)
        return per, avg

    out = {"tasks": tasks, "methods": {}, "ridge_sweep": {}, "task_count": {}, "sign_agreement": {}}

    # ---- (1) New baselines ----
    if "baselines" not in skip:
        print("\n== New baselines ==")
        # DARE (sweep alpha)
        best_dare = None
        for alpha in [0.2, 0.3, 0.5]:
            merged = dare(pre_cpu, tsds_cpu, drop_p=0.9, alpha=alpha, seed=0)
            per, avg = evaluate_sd(merged)
            name = f"dare_p0.9_a{alpha}"
            out["methods"][name] = {"per_task": per, "avg": avg}
            print(f"  {name}: {avg:.4f}")
            if best_dare is None or avg > best_dare[1]:
                best_dare = (name, avg, per, alpha)
        out["methods"]["dare_best"] = {"per_task": best_dare[2], "avg": best_dare[1], "alpha": best_dare[3]}

        # DARE + TIES
        best_dt = None
        for alpha in [0.3, 0.5, 0.7]:
            merged = dare_ties(pre_cpu, tsds_cpu, drop_p=0.9, keep_frac=0.2, alpha=alpha, seed=0)
            per, avg = evaluate_sd(merged)
            name = f"dare_ties_a{alpha}"
            out["methods"][name] = {"per_task": per, "avg": avg}
            print(f"  {name}: {avg:.4f}")
            if best_dt is None or avg > best_dt[1]:
                best_dt = (name, avg, per, alpha)
        out["methods"]["dare_ties_best"] = {"per_task": best_dt[2], "avg": best_dt[1], "alpha": best_dt[3]}

        # Fisher-approx
        merged = fisher_approx_merge(pre_cpu, tsds_cpu, ridge=1e-6)
        per, avg = evaluate_sd(merged)
        out["methods"]["fisher_approx"] = {"per_task": per, "avg": avg}
        print(f"  fisher_approx: {avg:.4f}")

    # ---- (2) Ridge sensitivity for SCALE (no-sign, k=0.3) ----
    if "ridge" not in skip:
        print("\n== Ridge sensitivity (SCALE nosign k=0.3) ==")
        for rho in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            merged = scale_merge(pre_gpu, tsds_gpu, keep_frac=0.3, ridge=rho, use_sign_election=False)
            merged_cpu = {k: v.cpu() for k, v in merged.items()}
            per, avg = evaluate_sd(merged_cpu)
            out["ridge_sweep"][f"rho{rho}"] = {"rho": rho, "per_task": per, "avg": avg}
            print(f"  rho={rho}: {avg:.4f}")

    # ---- (3) Task-count ablation: merge T=2,4,6,8 tasks ----
    if "taskcount" not in skip:
        print("\n== Task-count ablation ==")
        # Use deterministic subsets (first T tasks by name order)
        for T in [2, 4, 6, 8]:
            sub_tasks = tasks[:T]
            sub_tsds_gpu = [tsds_gpu[tasks.index(t)] for t in sub_tasks]
            sub_tsds_cpu = [tsds_cpu[tasks.index(t)] for t in sub_tasks]
            # SCALE
            merged = scale_merge(pre_gpu, sub_tsds_gpu, keep_frac=0.3, ridge=1e-4, use_sign_election=False)
            merged_cpu = {k: v.cpu() for k, v in merged.items()}
            per, avg = evaluate_sd(merged_cpu, eval_tasks=sub_tasks)
            out["task_count"][f"scale_T{T}"] = {"T": T, "tasks": sub_tasks, "per_task": per, "avg": avg}
            print(f"  scale T={T}: {avg:.4f}")
            # ACTMat
            merged = actmat(pre_gpu, sub_tsds_gpu, ridge=1e-4)
            merged_cpu = {k: v.cpu() for k, v in merged.items()}
            per, avg = evaluate_sd(merged_cpu, eval_tasks=sub_tasks)
            out["task_count"][f"actmat_T{T}"] = {"T": T, "tasks": sub_tasks, "per_task": per, "avg": avg}
            print(f"  actmat T={T}: {avg:.4f}")
            # TIES
            merged = ties_merging(pre_cpu, sub_tsds_cpu, keep_frac=0.2, alpha=0.5)
            per, avg = evaluate_sd(merged, eval_tasks=sub_tasks)
            out["task_count"][f"ties_T{T}"] = {"T": T, "tasks": sub_tasks, "per_task": per, "avg": avg}
            print(f"  ties T={T}: {avg:.4f}")
            # Task Arithmetic
            merged = task_arithmetic(pre_cpu, sub_tsds_cpu, alpha=0.2)
            per, avg = evaluate_sd(merged, eval_tasks=sub_tasks)
            out["task_count"][f"task_arith_T{T}"] = {"T": T, "tasks": sub_tasks, "per_task": per, "avg": avg}
            print(f"  task_arith T={T}: {avg:.4f}")

    # ---- (4) Sign-agreement diagnostic ----
    if "signagree" not in skip:
        print("\n== Sign-agreement diagnostic ==")
        sa = sign_agreement_diagnostic(pre_cpu, tsds_cpu, keep_fracs=(1.0, 0.5, 0.3, 0.2, 0.1, 0.05))
        out["sign_agreement"] = sa
        for k, v in sa.items():
            print(f"  {k}: agreement={v:.4f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to", args.out)


if __name__ == "__main__":
    main()
