"""Properly-randomized T=6 subset experiment: 5 random 6-task subsets from 8, all 4 methods."""
import argparse, json, os, random, sys, itertools, time
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
    ap.add_argument("--ckpt_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints")
    ap.add_argument("--out", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/results/t6_subsets.json")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--n_subsets", type=int, default=5)
    ap.add_argument("--T", type=int, default=6)
    ap.add_argument("--tasks", default="MNIST,CIFAR10,CIFAR100,SVHN,FashionMNIST,EuroSAT,GTSRB,DTD")
    ap.add_argument("--bs", type=int, default=512)
    args = ap.parse_args()

    all_tasks = args.tasks.split(",")
    device = args.device
    T = args.T

    # Sample n_subsets distinct random T-subsets from all_tasks.
    rng = random.Random(42)
    all_combos = list(itertools.combinations(all_tasks, T))
    rng.shuffle(all_combos)
    subsets = [list(s) for s in all_combos[: args.n_subsets]]
    print(f"[init] selected {len(subsets)} random {T}-subsets from C({len(all_tasks)},{T})={len(all_combos)}")
    for i, s in enumerate(subsets):
        print(f"  subset {i}: {s}")

    pre = torch.load(os.path.join(args.ckpt_dir, "pretrained.pt"), map_location="cpu", weights_only=False)
    ckpts = {t: torch.load(os.path.join(args.ckpt_dir, f"{t}.pt"), map_location="cpu", weights_only=False) for t in all_tasks}
    task_sds = {t: ckpts[t]["state_dict"] for t in all_tasks}
    text_cls = {t: ckpts[t]["text_classifier"].to(device) for t in all_tasks}

    clip_model, _, _ = load_clip(device=device)
    pre_gpu = {k: v.to(device).float() for k, v in pre.items()}
    tsd_gpu = {t: {k: v.to(device).float() for k, v in sd.items()} for t, sd in task_sds.items()}

    tf = build_tf()
    loaders = {}
    for t in all_tasks:
        _, test_ds, _ = get_dataset(t, tf, tf)
        loaders[t] = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    print("[init] loaders ready", flush=True)

    def evaluate_sd(sd, eval_tasks):
        load_visual_state_dict(clip_model, sd, strict=False)
        clip_model.to(device)
        per = {}
        for t in eval_tasks:
            per[t] = eval_loader(clip_model, loaders[t], text_cls[t], device)
        return per, sum(per.values()) / len(per)

    # Use SCALE k=0.3 (best for T=6 from the main results).
    scale_k = 0.3

    results = []
    for si, sub in enumerate(subsets):
        t0 = time.time()
        print(f"\n=== Subset {si}: {sub} ===", flush=True)
        sub_tsds = [tsd_gpu[t] for t in sub]
        methods = {}

        m = scale_merge(pre_gpu, sub_tsds, keep_frac=scale_k, ridge=1e-4, use_sign_election=False)
        per, avg = evaluate_sd({kk: v.cpu() for kk, v in m.items()}, sub)
        methods[f"scale_k{scale_k}"] = {"per_task": per, "avg": avg}
        print(f"  scale(k={scale_k}): {avg:.4f}", flush=True)

        m = actmat(pre_gpu, sub_tsds, ridge=1e-4)
        per, avg = evaluate_sd({kk: v.cpu() for kk, v in m.items()}, sub)
        methods["actmat"] = {"per_task": per, "avg": avg}
        print(f"  actmat:       {avg:.4f}", flush=True)

        m = ties_merging(pre_gpu, sub_tsds, alpha=0.5, keep_frac=0.2)
        per, avg = evaluate_sd({kk: v.cpu() for kk, v in m.items()}, sub)
        methods["ties"] = {"per_task": per, "avg": avg}
        print(f"  ties:         {avg:.4f}", flush=True)

        m = task_arithmetic(pre_gpu, sub_tsds, alpha=0.2)
        per, avg = evaluate_sd({kk: v.cpu() for kk, v in m.items()}, sub)
        methods["task_arith"] = {"per_task": per, "avg": avg}
        print(f"  task_arith:   {avg:.4f}", flush=True)

        results.append({"subset_idx": si, "T": T, "tasks": sub, "methods": methods})
        print(f"[subset {si}] elapsed {time.time()-t0:.1f}s", flush=True)

    # Aggregate
    import numpy as np
    agg = {}
    for method in ["scale_k0.3", "actmat", "ties", "task_arith"]:
        avgs = [r["methods"][method]["avg"] for r in results]
        agg[method] = {"mean": float(np.mean(avgs)), "std": float(np.std(avgs, ddof=1) if len(avgs) > 1 else 0.0),
                        "per_subset": avgs}
    out = {"T": T, "subsets": subsets, "results": results, "aggregate": agg}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== Aggregate over", len(subsets), "subsets ===")
    for m, v in agg.items():
        print(f"  {m}: {v['mean']*100:.2f} ± {v['std']*100:.2f}")
    print("Saved to", args.out)


if __name__ == "__main__":
    main()
