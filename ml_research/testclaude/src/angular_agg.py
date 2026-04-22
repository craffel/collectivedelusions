"""Extended angular-distance validation.

Beyond per-task  angle(Δ^T Δ, C_t), measure:
  (A) Summed  angle(Σ_t Δ̃_t^T Δ̃_t, Σ_t C_t)   — the quantity that actually
      enters the linear system Σ_t C_t W_t = (Σ_t C_t) W^*.
  (B) Top-r principal-subspace alignment between Δ̃_t^T Δ̃_t and C_t:
      1/r Σ_{i=1..r}  ||P_A u^B_i||_2^2  where P_A is the projector onto
      the top-r eigenspace of A and u^B_i are the top-r eigenvectors of B.
      r=32 by default (much smaller than d_in=768/3072).
  (C) Angular distance on the top-r eigenspace projection only.

Loads the per-task true covariances saved by `angular_validation.py --save_covariances`.
"""
import argparse
import os
import sys
import json
import time
import torch

torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(__file__))
from merging import topk_mask

ALL_TASKS = ["MNIST", "CIFAR10", "CIFAR100", "SVHN", "FashionMNIST", "EuroSAT", "GTSRB", "DTD"]
TARGET_LAYERS = [
    "visual.transformer.resblocks.0.attn.in_proj_weight",
    "visual.transformer.resblocks.0.mlp.c_fc.weight",
    "visual.transformer.resblocks.3.attn.in_proj_weight",
    "visual.transformer.resblocks.3.mlp.c_fc.weight",
    "visual.transformer.resblocks.6.attn.in_proj_weight",
    "visual.transformer.resblocks.6.mlp.c_fc.weight",
    "visual.transformer.resblocks.9.attn.in_proj_weight",
    "visual.transformer.resblocks.9.mlp.c_fc.weight",
    "visual.transformer.resblocks.11.attn.in_proj_weight",
    "visual.transformer.resblocks.11.mlp.c_fc.weight",
    "visual.transformer.resblocks.11.mlp.c_proj.weight",
]


def angular(A, B):
    a = A.reshape(-1).double(); b = B.reshape(-1).double()
    denom = a.norm() * b.norm() + 1e-20
    return torch.acos(((a @ b) / denom).clamp(-1.0, 1.0)).item()


def topr_subspace_overlap(A, B, r=32):
    """Return (1/r) * sum of squared projections of top-r eigvecs of B onto
    top-r eigenspace of A (in [0,1]; 1 = perfect alignment)."""
    A = A.double()
    B = B.double()
    # Symmetric PSD expected
    eA, UA = torch.linalg.eigh((A + A.T) / 2)
    eB, UB = torch.linalg.eigh((B + B.T) / 2)
    # eigh returns ascending; take last r
    UA_r = UA[:, -r:]
    UB_r = UB[:, -r:]
    # Projector onto col(UA_r): P = UA_r UA_r^T. Squared projection of UB_r columns:
    PuB = UA_r.T @ UB_r  # [r, r]
    # Squared norm of each projection column: sum over rows of PuB^2
    sq = (PuB * PuB).sum(dim=0)  # [r]
    return sq.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints")
    ap.add_argument("--covariance_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/covariances")
    ap.add_argument("--out_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/results")
    ap.add_argument("--tasks", default=",".join(ALL_TASKS))
    ap.add_argument("--r", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tasks = args.tasks.split(",")

    # Load pretrained & task sds
    pre = torch.load(os.path.join(args.ckpt_dir, "pretrained.pt"), map_location="cpu", weights_only=False)
    pre_f = {k: v.float() for k, v in pre.items()}
    task_sds = {}
    for t in tasks:
        d = torch.load(os.path.join(args.ckpt_dir, f"{t}.pt"), map_location="cpu", weights_only=False)
        task_sds[t] = {k: v.float() for k, v in d["state_dict"].items()}

    # Load covariances
    true_covs = {}
    for t in tasks:
        true_covs[t] = torch.load(os.path.join(args.covariance_dir, f"{t}.pt"), map_location="cpu", weights_only=True)

    k_values = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
    sign_options = (False, True)

    results = {"tasks": tasks, "layers": TARGET_LAYERS, "k_values": k_values, "r": args.r,
               "summed_angle_deg": {}, "subspace_alignment": {}}

    for ln in TARGET_LAYERS:
        print(f"[agg] {ln}", flush=True)
        # Build Δ_t 2D per task
        deltas_2d = []
        for t in tasks:
            D = task_sds[t][ln] - pre_f[ln]
            if D.ndim != 2:
                D = D.reshape(D.shape[0], -1)
            deltas_2d.append(D.float())

        # True C (sum across tasks)
        Csum = sum([true_covs[t][ln] for t in tasks]).double()

        # For each (k, sign) build trimmed gram for every task and sum.
        for k in k_values:
            trimmed = [d * topk_mask(d, k) for d in deltas_2d]
            stack = torch.stack(trimmed, dim=0)  # [T, out, in]
            for use_sign in sign_options:
                if use_sign:
                    elected = torch.sign(stack.sum(dim=0))
                    agree = (torch.sign(stack) == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)
                    cleaned = torch.where(agree, stack, torch.zeros_like(stack))
                    tv_list = cleaned
                else:
                    tv_list = stack
                # Sum Δ^T Δ
                Gsum = torch.zeros(deltas_2d[0].shape[1], deltas_2d[0].shape[1])
                for ti in range(len(tasks)):
                    D_t = tv_list[ti]
                    Gsum = Gsum + D_t.T @ D_t
                # Angle
                ang = angular(Gsum, Csum) * 180 / 3.141592653589793
                # Subspace overlap
                r = min(args.r, Gsum.shape[0])
                overlap = topr_subspace_overlap(Gsum, Csum, r=r)

                key = f"k={k},sign={int(use_sign)}"
                results["summed_angle_deg"].setdefault(key, {})[ln] = ang
                results["subspace_alignment"].setdefault(key, {})[ln] = overlap

    # Averages across layers
    results["summed_angle_mean"] = {}
    results["subspace_alignment_mean"] = {}
    for key in results["summed_angle_deg"]:
        vals = list(results["summed_angle_deg"][key].values())
        results["summed_angle_mean"][key] = sum(vals) / len(vals)
        vals2 = list(results["subspace_alignment"][key].values())
        results["subspace_alignment_mean"][key] = sum(vals2) / len(vals2)

    # Also compute the "ACTMat" reference: no trim, no sign, per task angular — just for completeness.
    with open(os.path.join(args.out_dir, "angular_agg.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summed angular distance (Σ Δ̃^T Δ̃ vs Σ C_t), degrees ===")
    for k in k_values:
        for s in (0, 1):
            key = f"k={k},sign={s}"
            print(f"  {key}: angle={results['summed_angle_mean'][key]:.2f}°   top{args.r}-subspace-overlap={results['subspace_alignment_mean'][key]:.3f}")

    print("Saved", os.path.join(args.out_dir, "angular_agg.json"))


if __name__ == "__main__":
    main()
