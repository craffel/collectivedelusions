"""Fine-tune CLIP ViT-B/32 vision encoder on a single task."""
import argparse
import os
import sys
import time
import torch
torch.backends.cudnn.enabled = False  # driver/cudnn mismatch workaround
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from datasets_utils import get_dataset
from clip_utils import load_clip, compute_text_classifier, ClipClassifier, get_visual_state_dict


def build_transforms(preprocess):
    """Create training and val transforms matching CLIP preprocess."""
    # preprocess = Compose([Resize, CenterCrop, _convert_image_to_rgb, ToTensor, Normalize(mean,std)])
    # Extract mean/std from the last Normalize
    mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    train_tf = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = train_tf  # keep identical; tasks vary enough
    return train_tf, val_tf


def run_finetune(args):
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    if args.seed is not None:
        import random, numpy as np
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        print(f"[{args.task}] Seed set to {args.seed}")
    print(f"[{args.task}] Loading CLIP on {device}...")
    model, preprocess, tokenizer = load_clip(args.model, args.pretrained, device)
    train_tf, val_tf = build_transforms(preprocess)

    print(f"[{args.task}] Loading dataset...")
    train_ds, test_ds, classnames = get_dataset(args.task, train_tf, val_tf)

    # Save pre-trained visual state dict once per machine (only if not exists)
    if args.save_base:
        base_path = os.path.join(args.out_dir, "pretrained.pt")
        if not os.path.exists(base_path):
            torch.save(get_visual_state_dict(model), base_path)
            print(f"[{args.task}] Saved pretrained visual state to {base_path}")

    text_classifier = compute_text_classifier(model, tokenizer, classnames, device=device)
    print(f"[{args.task}] Num classes: {len(classnames)}, text classifier shape: {tuple(text_classifier.shape)}")

    clf = ClipClassifier(model, text_classifier).to(device)

    # Freeze text encoder + token embedding (we only train visual)
    for name, p in clf.named_parameters():
        if name.startswith("clip_model.visual"):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    # Quick eval baseline (zero-shot)
    def evaluate(loader):
        clf.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device); y = y.to(device)
                logits = clf(x)
                correct += (logits.argmax(-1) == y).sum().item()
                total += y.numel()
        return correct / max(total, 1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    zs_acc = evaluate(test_loader)
    print(f"[{args.task}] Zero-shot acc: {zs_acc:.4f}")

    opt = torch.optim.AdamW([p for p in clf.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.1)

    total_steps = args.epochs * len(train_loader)
    warmup = min(500, total_steps // 10)
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        import math
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    clf.train()
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        clf.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits = clf(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in clf.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[{args.task}] ep {ep} step {step}/{total_steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} elapsed {time.time()-t0:.0f}s")
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    acc = evaluate(test_loader)
    elapsed = time.time() - t0
    print(f"[{args.task}] FINAL acc: {acc:.4f} (zs: {zs_acc:.4f}) time: {elapsed:.0f}s")

    # Save fine-tuned visual state dict
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.task}.pt")
    torch.save({
        "state_dict": get_visual_state_dict(clf.clip_model),
        "text_classifier": text_classifier.detach().cpu(),
        "classnames": classnames,
        "acc": acc,
        "zs_acc": zs_acc,
        "task": args.task,
    }, ckpt_path)
    print(f"[{args.task}] Saved to {ckpt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--save_base", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_dir", default="/fsx/craffel/collectivedelusions/ml_research/testclaude/checkpoints")
    args = ap.parse_args()
    run_finetune(args)


if __name__ == "__main__":
    main()
