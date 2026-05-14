"""Diagonal Fisher information for Fisher merging (Matena & Raffel, 2022).

For each named parameter p, accumulates F_p = (1/N) Σ_x (∂ L(x;θ)/∂p)^2,
where L is the per-example cross-entropy loss against the model's own
predicted label (empirical Fisher = true-label cross-entropy gradient squared).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


def compute_diagonal_fisher(model, loader, device,
                             max_batches: int = 8) -> Dict[str, torch.Tensor]:
    """Return {param_name: tensor} -- the diagonal Fisher for each parameter.

    Uses the predicted label rather than the true label (this is the "empirical
    Fisher" variant of Matena & Raffel, 2022, which works well as a parameter-
    importance estimate even though it is not the exact Fisher).
    """
    fisher: Dict[str, torch.Tensor] = {
        n: torch.zeros_like(p, dtype=torch.float32) for n, p in model.named_parameters()
        if p.requires_grad
    }
    n_seen = 0
    n_batches = 0
    # Cache reference param objects for fast .grad access
    name_to_param = {n: p for n, p in model.named_parameters() if p.requires_grad}

    for batch in loader:
        x, _y = batch
        x = x.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        # We need gradients, so don't wrap in no_grad. Use float32 to avoid
        # half-precision gradient instabilities.
        logits = model(x)  # [B, C]
        # Empirical Fisher: cross-entropy against the model's argmax label
        with torch.no_grad():
            target = logits.argmax(dim=-1)
        loss = F.cross_entropy(logits, target, reduction="sum")
        loss.backward()
        for name, p in name_to_param.items():
            if p.grad is None:
                continue
            fisher[name] += p.grad.detach().to(torch.float32) ** 2
        n_seen += x.shape[0]
        n_batches += 1
        if n_batches >= max_batches:
            break

    # Normalize: (1/N) Σ grad^2
    fisher = {n: (f / max(n_seen, 1)).cpu() for n, f in fisher.items()}
    return fisher
