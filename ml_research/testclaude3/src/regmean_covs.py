"""Estimate per-layer activation covariance matrices for RegMean.

We hook every Linear-like 2-D weight in the visual encoder and accumulate
E[z zᵀ] over a small calibration set (the task's train loader). The result is
a dict mapping each parameter's *weight name* (e.g. "visual.transformer...weight")
to a [D_in, D_in] covariance.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast


def _module_to_param_name(model: nn.Module) -> Dict[nn.Linear, str]:
    """Map every Linear module to its parameter name "module_path.weight"."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            out[mod] = f"{name}.weight"
    return out


def compute_activation_covs(model: nn.Module, loader, device,
                            max_batches: int = 16,
                            dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    """Return {param_name: D x D activation second-moment matrix}.

    Only Linear modules are covered. We accumulate sum(x xᵀ) / N where x is
    the *input* to the linear layer flattened to [batch*seq, D_in].

    The model is assumed to have a `visual.*` submodule; only those covs are
    used by the merger. We hook *all* Linear modules to be safe.
    """
    mod2name = _module_to_param_name(model)
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    handles = []

    def make_hook(name):
        def hook(_mod, inputs, _output):
            x = inputs[0]
            # Flatten leading dims
            x = x.reshape(-1, x.shape[-1]).to(dtype)
            if name not in sums:
                D = x.shape[-1]
                sums[name] = torch.zeros((D, D), device=x.device, dtype=dtype)
                counts[name] = 0
            sums[name].addmm_(x.t(), x)
            counts[name] += x.shape[0]
        return hook

    for mod, name in mod2name.items():
        handles.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    n = 0
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device, non_blocking=True)
            with autocast(dtype=torch.float16):
                _ = model(x)
            n += 1
            if n >= max_batches:
                break

    for h in handles:
        h.remove()

    covs = {k: (s / max(counts[k], 1)).cpu() for k, s in sums.items()}
    return covs
