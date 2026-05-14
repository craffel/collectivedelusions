"""Fine-tune CLIP ViT-B/32 visual encoder on a single task.

Usage:
    python -m src.finetune --task EuroSAT --gpu 0 --epochs 3 --lr 1e-5

Saves checkpoints to checkpoints/{task}.pt containing the *visual* state dict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets_setup import build_task
from src.model import (
    CLIPVisualClassifier, build_text_classifier, get_visual_state_dict,
    load_clip,
)


def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(dtype=torch.float16):
                logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=20000,
                    help="Cap number of training samples per task to keep cost bounded.")
    ap.add_argument("--max-test", type=int, default=4000)
    ap.add_argument("--out-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--log-dir", default=str(ROOT / "logs"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    import random as _random
    import numpy as _np
    torch.manual_seed(args.seed)
    _np.random.seed(args.seed)
    _random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    log_path = Path(args.log_dir) / f"ft_{args.task}.log"
    log_f = open(log_path, "w")
    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    log(f"[{args.task}] device={device} epochs={args.epochs} lr={args.lr}")

    # Build data
    bundle = build_task(args.task, batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        max_train=args.max_train, max_test=args.max_test)
    log(f"[{args.task}] classes={len(bundle.classnames)} train_batches={len(bundle.train_loader)}")

    # Build model
    clip_model, _, tokenizer = load_clip(device=device)
    text_feats = build_text_classifier(bundle.classnames, bundle.prompt_template,
                                       tokenizer, clip_model, device)
    classifier = CLIPVisualClassifier(clip_model, text_feats).to(device)

    # Zero-shot baseline
    zs_acc = evaluate(classifier, bundle.test_loader, device)
    log(f"[{args.task}] zero-shot acc = {zs_acc:.4f}")

    # Save pretrained (zero-shot) visual state dict for *only* the first task we encounter.
    pretrained_path = Path(args.out_dir) / "_pretrained_visual.pt"
    if not pretrained_path.exists():
        torch.save({"visual_state_dict": get_visual_state_dict(classifier)},
                   pretrained_path)
        log(f"Saved pretrained visual state dict to {pretrained_path}")

    # Fine-tune
    params = [p for p in classifier.visual.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * len(bundle.train_loader)
    )
    scaler = GradScaler()
    classifier.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for x, y in bundle.train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                logits = classifier(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            step += 1
            if step % 50 == 0:
                log(f"[{args.task}] step {step} loss {loss.item():.4f}")
        # Per-epoch eval (cheap because test is small)
        acc = evaluate(classifier, bundle.test_loader, device)
        log(f"[{args.task}] epoch {epoch+1} val_acc {acc:.4f}")
        classifier.train()

    final_acc = evaluate(classifier, bundle.test_loader, device)
    elapsed = time.time() - t0
    log(f"[{args.task}] done in {elapsed:.1f}s final_acc {final_acc:.4f}")

    out = {
        "task": args.task,
        "visual_state_dict": get_visual_state_dict(classifier),
        "text_features": text_feats.detach().cpu(),
        "classnames": bundle.classnames,
        "prompt_template": bundle.prompt_template,
        "zero_shot_acc": zs_acc,
        "finetuned_acc": final_acc,
    }
    out_path = Path(args.out_dir) / f"{args.task}.pt"
    torch.save(out, out_path)
    log(f"[{args.task}] saved to {out_path}")
    log_f.close()


if __name__ == "__main__":
    main()
