"""Empirically validate the ACTMat angular-distance claim and the effect of
trimming and sign-election on it.

For a selection of linear layers in the visual encoder:
  (1) Compute the TRUE activation covariance C_t = E[z z^T] from task-t
      training data, passed through the fine-tuned model W_t (up to that layer).
  (2) Compute the data-free covariance estimates:
        - ACTMat:      Δ_t^T Δ_t
        - SCALE-trim:  (Δ_t ⊙ m_t)^T (Δ_t ⊙ m_t)  with m_t = top-k% magnitude mask
        - SCALE-full:  use trim + sign-election (cross-task sign agreement)
  (3) Measure the angular distance
        θ(A, B) = arccos( <A,B>_F / (||A||_F ||B||_F) )
      between each estimate and the true C_t.

Averages across layers and tasks, sweep k, save JSON + plot-ready data.
"""
import argparse
import os
import sys
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(__file__))
from datasets_utils import get_dataset
from clip_utils import load_clip, load_visual_state_dict
from merging import topk_mask


ALL_TASKS = ["MNIST", "CIFAR10", "CIFAR100", "SVHN", "FashionMNIST", "EuroSAT", "GTSRB", "DTD"]

# Representative layers (across depth of encoder). Keep in_dim = 768 for cheap math.
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


def build_tf():
    mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def resolve_module(model, dotted):
    """Resolve a dotted name ending in '.weight' or 'in_proj_weight' -> (module, is_in_proj)."""
    is_in_proj = dotted.endswith("in_proj_weight")
    if is_in_proj:
        # visual.transformer.resblocks.N.attn.in_proj_weight -> module is the MultiheadAttention
        mod_path = dotted[: -len(".in_proj_weight")]
    else:
        mod_path = dotted[: -len(".weight")]
    mod = model
    for part in mod_path.split("."):
        mod = getattr(mod, part)
    return mod, is_in_proj


@torch.no_grad()
def compute_true_covariance_for_task(clip_model, task, device, target_layers, max_samples=256, batch_size=64):
    """Pass task-specific training data through clip_model and collect input
    activations at each target layer. Return dict: layer_name -> C_t [in, in]."""
    tf = build_tf()
    train_ds, _, _ = get_dataset(task, tf, tf)
    if len(train_ds) > max_samples:
        train_ds = Subset(train_ds, list(range(max_samples)))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    clip_model.eval()
    # Accumulators
    covs = {}
    counts = {}

    hooks = []
    def make_hook(name):
        def hook(mod, inputs, output):
            x = inputs[0]
            # Flatten leading dims except the feature dim.
            x = x.reshape(-1, x.shape[-1]).float()
            c = covs.get(name)
            if c is None:
                c = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                covs[name] = c
                counts[name] = 0
            covs[name] = c + x.T @ x
            counts[name] += x.shape[0]
        return hook

    for name in target_layers:
        mod, is_in_proj = resolve_module(clip_model, name)
        # For MultiheadAttention with in_proj_weight, the input is the first positional arg.
        hooks.append(mod.register_forward_hook(make_hook(name)))

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = clip_model.encode_image(x)

    for h in hooks:
        h.remove()

    # Normalize
    out = {}
    for name, c in covs.items():
        out[name] = (c / max(counts[name], 1)).detach().cpu()
    return out


def angular_distance(A, B):
    """Angle (radians) between two matrices via Frobenius inner product."""
    a = A.reshape(-1).double()
    b = B.reshape(-1).double()
    denom = a.norm() * b.norm() + 1e-20
    cos_t = (a @ b) / denom
    cos_t = cos_t.clamp(-1.0, 1.0)
    return torch.acos(cos_t).item()


def build_gram_estimates(delta_list, task_idx, k_values, use_sign_options=(False, True)):
    """For task task_idx, return dict: (k, use_sign) -> Gram estimate
    Δ̃_t^T Δ̃_t (at this layer), where trimming is per-task and sign-election is
    cross-task (using trimmed sum across tasks).
    """
    T = len(delta_list)
    # Reshape each Δ_t to 2D [out, in] (already 2D here).
    D2 = [d.float() for d in delta_list]
    out = {}
    for k in k_values:
        # Compute trimmed per task
        trimmed = [d * topk_mask(d, k) for d in D2]
        stack = torch.stack(trimmed, dim=0)  # [T, out, in]
        for use_sign in use_sign_options:
            if use_sign:
                elected = torch.sign(stack.sum(dim=0))
                agree = (torch.sign(stack) == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)
                cleaned = torch.where(agree, stack, torch.zeros_like(stack))
                D_t = cleaned[task_idx]
            else:
                D_t = stack[task_idx]
            # Gram: [in, in]
            out[(k, use_sign)] = (D_t.T @ D_t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints")
    ap.add_argument("--out_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/results")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", default=",".join(ALL_TASKS))
    ap.add_argument("--max_samples", type=int, default=256)
    ap.add_argument("--save_covariances", action="store_true",
                    help="Also write true covariances to disk (for RegMean).")
    ap.add_argument("--covariance_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/covariances")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tasks = args.tasks.split(",")
    device = args.device

    # Load pre-trained
    pre = torch.load(os.path.join(args.ckpt_dir, "pretrained.pt"), map_location="cpu", weights_only=False)
    pre_f = {k: v.float() for k, v in pre.items()}

    # Load all task state dicts (need them to (a) compute C_t via fine-tuned model,
    # (b) compute Δ_t = W_t - W_0)
    task_sds = {}
    for t in tasks:
        d = torch.load(os.path.join(args.ckpt_dir, f"{t}.pt"), map_location="cpu", weights_only=False)
        task_sds[t] = {k: v.float() for k, v in d["state_dict"].items()}

    # Load a clip_model once; we'll swap in each task's weights.
    clip_model, _, _ = load_clip(device=device)

    # Step 1: compute true covariances per task (via forward hooks)
    t0 = time.time()
    true_covs = {}  # task -> {layer_name: C_t [in, in]}
    for t in tasks:
        print(f"[true-cov] task={t}", flush=True)
        load_visual_state_dict(clip_model, task_sds[t], strict=False)
        clip_model.to(device)
        true_covs[t] = compute_true_covariance_for_task(
            clip_model, t, device, TARGET_LAYERS, max_samples=args.max_samples)
        print(f"  done in {time.time()-t0:.1f}s")

    # Optionally persist covariances so RegMean can reuse.
    if args.save_covariances:
        os.makedirs(args.covariance_dir, exist_ok=True)
        for t in tasks:
            torch.save(true_covs[t], os.path.join(args.covariance_dir, f"{t}.pt"))

    # Step 2: for each task and each layer, compute angular distance of:
    #   - Δ^T Δ    vs C_t
    #   - trimmed Δ^T Δ (k in sweep)   vs C_t
    #   - trimmed+sign Δ̃^T Δ̃            vs C_t
    k_values = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
    per_layer_results = {}
    for ln in TARGET_LAYERS:
        print(f"[angle] layer={ln}", flush=True)
        # Build list of Δ_t for this layer across tasks (2D [out, in])
        deltas = []
        for t in tasks:
            D = task_sds[t][ln] - pre_f[ln]
            if D.ndim != 2:
                D = D.reshape(D.shape[0], -1)
            deltas.append(D)

        # Precompute grams: per-task dict of (k, use_sign) -> in×in gram
        grams_per_task = []
        for ti in range(len(tasks)):
            grams_per_task.append(build_gram_estimates(deltas, ti, k_values))

        # Now measure per task per k per sign
        layer_entry = {"k_values": k_values, "per_task": {}}
        for ti, t in enumerate(tasks):
            Ct = true_covs[t][ln]
            rec = {}
            for k in k_values:
                for use_sign in (False, True):
                    G = grams_per_task[ti][(k, use_sign)]
                    rec[f"k={k},sign={int(use_sign)}"] = angular_distance(G, Ct)
            layer_entry["per_task"][t] = rec
        per_layer_results[ln] = layer_entry

    # Summary: average across tasks and layers for each (k, sign)
    summary = {"k_values": k_values, "averages": {}}
    for k in k_values:
        for use_sign in (False, True):
            key = f"k={k},sign={int(use_sign)}"
            vals = []
            for ln in TARGET_LAYERS:
                for t in tasks:
                    vals.append(per_layer_results[ln]["per_task"][t][key])
            summary["averages"][key] = {"mean_deg": sum(vals)/len(vals) * 180.0 / 3.141592653589793,
                                          "n": len(vals)}

    out = {"tasks": tasks, "layers": TARGET_LAYERS, "per_layer": per_layer_results, "summary": summary}
    with open(os.path.join(args.out_dir, "angular_validation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== Summary (average angular distance in degrees) ===")
    for k in k_values:
        for use_sign in (False, True):
            key = f"k={k},sign={int(use_sign)}"
            print(f"  {key}: {summary['averages'][key]['mean_deg']:.2f}")
    print("Saved to", os.path.join(args.out_dir, "angular_validation.json"))


if __name__ == "__main__":
    main()
