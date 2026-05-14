"""Fine-tune a CLIP-ViT-B/32 classifier on a single task.

Usage:
  python -m src.train_expert --task cifar10 --device cuda:0 --epochs 2 --bs 256 --lr 3e-5
Saves checkpoint to checkpoints/<task>.pt with keys:
  vision_state_dict, head_state_dict, class_names, val_acc, train_loss
"""
from __future__ import annotations
import argparse
import json
import os
import time
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_task
from src.models import ClipClassifier, build_processor, vision_state_dict, head_state_dict


def make_collate(processor):
    def _collate(batch):
        imgs = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = processor(images=imgs, return_tensors="pt")
        return enc["pixel_values"], labels
    return _collate


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for pv, y in loader:
            pv = pv.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pv)
            pred = logits.argmax(-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--max_train_samples", type=int, default=20000,
                    help="Cap on training examples per epoch for speed.")
    ap.add_argument("--max_eval_samples", type=int, default=4000)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out", default="./checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device(args.device)
    print(f"[{args.task}] device={device}")

    processor = build_processor()
    train_ds, val_ds, num_classes, class_names = load_task(args.task, args.data_root)
    print(f"[{args.task}] train={len(train_ds)} val={len(val_ds)} classes={num_classes}")

    # Cap train/val sizes for speed.
    if len(train_ds) > args.max_train_samples:
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(train_ds), generator=g)[: args.max_train_samples].tolist()
        train_ds = torch.utils.data.Subset(train_ds, idx)
    if len(val_ds) > args.max_eval_samples:
        g = torch.Generator().manual_seed(args.seed + 1)
        idx = torch.randperm(len(val_ds), generator=g)[: args.max_eval_samples].tolist()
        val_ds = torch.utils.data.Subset(val_ds, idx)

    collate = make_collate(processor)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=True, persistent_workers=args.num_workers > 0)

    model = ClipClassifier(num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * len(train_loader)))
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    best_acc = 0.0
    for ep in range(args.epochs):
        model.train()
        losses = []
        for i, (pv, y) in enumerate(train_loader):
            pv = pv.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pv)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
        acc = evaluate(model, val_loader, device)
        print(f"[{args.task}] ep{ep} loss={sum(losses)/len(losses):.4f} val_acc={acc:.4f} elapsed={time.time()-t0:.1f}s")
        best_acc = max(best_acc, acc)

    # Save vision state dict (we don't merge heads -- those are task-specific)
    save_path = os.path.join(args.out, f"{args.task}.pt")
    torch.save({
        "task": args.task,
        "num_classes": num_classes,
        "class_names": class_names,
        "vision_state_dict": vision_state_dict(model),
        "head_state_dict": head_state_dict(model),
        "val_acc": best_acc,
        "args": vars(args),
    }, save_path)
    print(f"[{args.task}] saved -> {save_path} (best_acc={best_acc:.4f}) total_time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
