"""Cycle-4 refinement experiments.

(A) Consensus Merging baseline (Wang et al. 2024 / TALL-masks): sweep (keep_frac, min_agree, alpha).
(B) Per-layer routing diagnostic: compare ACTMat vs SCALE's "task-routing matrix"
    P_t = C_t (Σ_s C_s)^{-1} and measure how much of P_t's Frobenius mass lies on
    task-t-active columns (top-10% of per-column Δ_t norm).

Outputs: results/consensus.json, results/routing_diagnostic.json
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
from merging import (
    consensus_merging, task_vectors, topk_mask, _is_matrix_param, _reshape_to_matrix,
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


def routing_diagnostic(pre_sd_gpu, task_sds_gpu, keep_fracs=(1.0, 0.5, 0.3, 0.2, 0.1),
                        use_sign_election_for_scale=False, active_col_frac=0.1, ridge=1e-4):
    """For each keep_frac k and each matrix layer, compute:
      C_t(k) = cleaned Δ_t^T Δ_t
      P_t(k) = C_t(k) · (Σ_s C_s(k) + ρ̄ I)^{-1}   (task-routing matrix)
      active cols of task t  S_t = top-active_col_frac columns of ||Δ_t||_col (on raw Δ_t)
      mass ratio r_t(k) = ||P_t(k)|_{S_t}||_F^2 / ||P_t(k)||_F^2
      cross-coupling x_t(k) = ||P_t(k)|_{complement of S_t}||_F^2 / ||P_t(k)||_F^2  (= 1 - r_t)

    Returns mean mass ratio averaged over tasks and layers.
    """
    tvs = task_vectors(pre_sd_gpu, task_sds_gpu)
    T = len(task_sds_gpu)
    per_k = {}
    layer_count = 0
    for k in keep_fracs:
        total_r = 0.0    # mean on-active mass ratio across (task, layer)
        total_offdiag_t = 0.0  # mean off-target mass ratio across (t != t')
        n_on = 0
        n_off = 0
        layer_count_this = 0
        for name in pre_sd_gpu:
            theta0 = pre_sd_gpu[name]
            if not _is_matrix_param(name, theta0):
                continue
            D_list = []
            raw_D_list = []
            for tv in tvs:
                if name in tv:
                    D, _ = _reshape_to_matrix(tv[name])
                    D_list.append(D.float())
                    raw_D_list.append(D.float().clone())
            if len(D_list) < 2:
                continue
            # Trim
            if k < 1.0:
                D_list = [D * topk_mask(D, k).to(D.dtype) for D in D_list]
            # Optional sign election (SCALE's choice)
            if use_sign_election_for_scale and k < 1.0:
                stack = torch.stack(D_list, dim=0)
                elected = torch.sign(stack.sum(dim=0))
                agree = (torch.sign(stack) == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)
                D_list = [torch.where(agree[i], stack[i], torch.zeros_like(stack[i])) for i in range(stack.shape[0])]

            din = D_list[0].shape[1]
            C_list = [D.T @ D for D in D_list]
            C_sum = sum(C_list)
            # ridge
            rho = ridge * (C_sum.diagonal().mean().item() + 1e-12)
            C_sum_r = C_sum + rho * torch.eye(din, device=C_sum.device)
            C_sum_inv = torch.linalg.inv(C_sum_r)

            # active columns per task from raw D (top active_col_frac by col-norm)
            active_S = []
            for rawD in raw_D_list:
                col_norm = rawD.pow(2).sum(dim=0).sqrt()  # [in]
                keep_cols = max(1, int(active_col_frac * col_norm.numel()))
                _, idx = torch.topk(col_norm, keep_cols)
                m = torch.zeros_like(col_norm, dtype=torch.bool)
                m[idx] = True
                active_S.append(m)

            # For each task-pair (t_weight, t_active): compute mass of P_{t_weight}
            # restricted to active_S[t_active] columns (columns of P correspond to "output" dim).
            for tw in range(T):
                P = C_list[tw] @ C_sum_inv  # [in, in]; rows "in", cols "in"
                P_total = P.pow(2).sum().item() + 1e-20
                # diagonal: on task's own active cols (columns dim of P = in dim)
                on_mask = active_S[tw]
                # ||P[:, on_mask]||_F^2
                on_mass = P[:, on_mask].pow(2).sum().item()
                total_r += on_mass / P_total
                n_on += 1
                # off-diagonal: avg over other tasks' active cols
                for to in range(T):
                    if to == tw:
                        continue
                    off_mass = P[:, active_S[to]].pow(2).sum().item()
                    # Compare "on-target" vs "off-target": how does mass on t's own active cols
                    # compare to mass on another task's active cols.
                    # report the ratio: mass on own active cols / mean mass on other tasks' active cols.
                    total_offdiag_t += off_mass / P_total
                    n_off += 1
            layer_count_this += 1
        per_k[f"k{k}"] = {
            "mean_on_mass_frac": total_r / max(n_on, 1),
            "mean_off_mass_frac": total_offdiag_t / max(n_off, 1),
            "on_over_off": (total_r / max(n_on, 1)) / max(total_offdiag_t / max(n_off, 1), 1e-12),
            "active_col_frac_ref": active_col_frac,  # null baseline ≈ active_col_frac if uniform
            "layers_aggregated": layer_count_this,
        }
        layer_count = layer_count_this
    return {"per_k": per_k, "layers_aggregated": layer_count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints")
    ap.add_argument("--out_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/results")
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

    # ---- (A) Consensus Merging baseline ----
    if "consensus" not in skip:
        print("\n== Consensus Merging baseline (Wang et al. 2024) ==")
        results = {}
        best = None
        # Sweep over (keep_frac, min_agree, alpha)
        for keep in [0.1, 0.2, 0.3]:
            for min_agree in [2, 3, 4]:
                for alpha in [0.2, 0.3, 0.5]:
                    merged = consensus_merging(pre_cpu, tsds_cpu, keep_frac=keep, min_agree=min_agree, alpha=alpha)
                    per, avg = evaluate_sd(merged)
                    name = f"consensus_k{keep}_m{min_agree}_a{alpha}"
                    results[name] = {"keep": keep, "min_agree": min_agree, "alpha": alpha, "per_task": per, "avg": avg}
                    print(f"  {name}: {avg:.4f}")
                    if best is None or avg > best["avg"]:
                        best = {**results[name], "name": name}
        results["best"] = best
        with open(os.path.join(args.out_dir, "consensus.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Best consensus: {best['name']} avg={best['avg']:.4f}")

    # ---- (B) Per-layer routing diagnostic ----
    if "routing" not in skip:
        print("\n== Routing diagnostic ==")
        keep_fracs = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05)
        out_no_sign = routing_diagnostic(pre_gpu, tsds_gpu, keep_fracs=keep_fracs,
                                          use_sign_election_for_scale=False,
                                          active_col_frac=0.1, ridge=1e-4)
        for k, v in out_no_sign["per_k"].items():
            print(f"  no-sign {k}: on={v['mean_on_mass_frac']:.4f} off={v['mean_off_mass_frac']:.4f} "
                  f"on/off={v['on_over_off']:.2f}")
        out_sign = routing_diagnostic(pre_gpu, tsds_gpu, keep_fracs=keep_fracs,
                                       use_sign_election_for_scale=True,
                                       active_col_frac=0.1, ridge=1e-4)
        for k, v in out_sign["per_k"].items():
            print(f"  w/sign  {k}: on={v['mean_on_mass_frac']:.4f} off={v['mean_off_mass_frac']:.4f} "
                  f"on/off={v['on_over_off']:.2f}")
        with open(os.path.join(args.out_dir, "routing_diagnostic.json"), "w") as f:
            json.dump({"no_sign": out_no_sign, "sign": out_sign,
                       "keep_fracs": list(keep_fracs),
                       "active_col_frac": 0.1,
                       "null_baseline_expected": 0.1}, f, indent=2)
        print("\n  Saved routing_diagnostic.json")


if __name__ == "__main__":
    main()
