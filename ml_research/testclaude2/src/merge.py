"""Model merging methods operating on dicts of parameter tensors.

All methods take:
  base   : state_dict of the pretrained vision encoder (W_0)
  experts: list of state_dicts of the fine-tuned vision encoders (W_t)
and return a merged state_dict with the same keys.

For methods that operate per linear layer (RegMean, ACTMat, TRIM) we apply the
layer-wise rule only to keys in ``linear_weight_keys`` and use simple averaging
of the task vectors (with a scaling coefficient) for the remaining parameters.

Notation per linear key W:
  ∆_t = W_t - W_0          (task vector)
  τ̄    = (1/T) Σ ∆_t       (mean task vector)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Iterable

import math
import torch


# ---------------- baselines -----------------------------------------------

def weight_average(base, experts):
    out = {}
    for k, v0 in base.items():
        stack = torch.stack([e[k].to(v0.dtype) for e in experts])
        out[k] = stack.mean(0)
    return out


def task_arithmetic(base, experts, alpha: float = 0.3):
    out = {}
    for k, v0 in base.items():
        delta = torch.stack([e[k] - v0 for e in experts]).sum(0)
        out[k] = v0 + alpha * delta
    return out


def ties(base, experts, alpha: float = 1.0, density: float = 0.2):
    """TIES-Merging: trim small ∆, elect majority sign, disjoint merge.

    density: fraction of ∆ entries kept per task (top-k by magnitude).
    alpha:   scaling on the merged ∆.
    """
    out = {}
    for k, v0 in base.items():
        deltas = [e[k].to(torch.float32) - v0.to(torch.float32) for e in experts]
        deltas = torch.stack(deltas)  # (T, ...)
        T = deltas.shape[0]
        # 1. Trim per task, keep top-density by |.|
        flat = deltas.reshape(T, -1)
        if 0 < density < 1.0 and flat.shape[1] > 1:
            k_keep = max(1, int(round(density * flat.shape[1])))
            mag = flat.abs()
            # threshold per row
            thr = torch.topk(mag, k_keep, dim=1).values.min(dim=1).values  # (T,)
            keep = mag >= thr.unsqueeze(1)
            flat = flat * keep
        trimmed = flat.reshape_as(deltas)
        # 2. Elect sign: per-element sign of sum of trimmed
        signed_sum = trimmed.sum(0)
        elected = torch.sign(signed_sum)
        elected[elected == 0] = 1.0
        # 3. Disjoint merge: average only entries matching elected sign
        match = (torch.sign(trimmed) == elected.unsqueeze(0)).to(trimmed.dtype)
        kept = trimmed * match
        cnt = match.sum(0).clamp_min(1.0)
        merged_delta = kept.sum(0) / cnt
        out[k] = (v0.to(torch.float32) + alpha * merged_delta).to(v0.dtype)
    return out


def ties_mask(deltas: torch.Tensor, density: float) -> torch.Tensor:
    """Return the TIES-cleaned ∆̂_t for a stack of task vectors.

    deltas: (T, ...)
    Returns a tensor of the same shape with sign-inconsistent or trimmed
    entries zeroed out per-task. The merged TIES vector would be the
    mean (over T, with denom = count of non-zero per element) of the
    return value.
    """
    T = deltas.shape[0]
    flat = deltas.reshape(T, -1)
    if 0 < density < 1.0 and flat.shape[1] > 1:
        k_keep = max(1, int(round(density * flat.shape[1])))
        mag = flat.abs()
        thr = torch.topk(mag, k_keep, dim=1).values.min(dim=1).values
        keep = mag >= thr.unsqueeze(1)
        flat = flat * keep
    trimmed = flat.reshape_as(deltas)
    elected = torch.sign(trimmed.sum(0))
    elected[elected == 0] = 1.0
    match = (torch.sign(trimmed) == elected.unsqueeze(0)).to(trimmed.dtype)
    return trimmed * match


def iso_c(base, experts, alpha: float = 0.3, restrict_to_linear: bool = True):
    """Iso-C: flatten the singular spectrum of the *summed* task matrix.

    Operates on the SUM of task vectors (Σ τ_t), then SVDs and replaces the
    singular values with their mean before reconstructing and adding to W_0.

    By default only applies the spectral flattening to 2D attention/MLP linear
    weight matrices (skips embeddings and 1D parameters where SVD-flattening is
    not well-defined). The rest of the parameters use a uniform task-vector
    average with the same scaling α.
    """
    out = {}
    for k, v0 in base.items():
        deltas = torch.stack([e[k].to(torch.float32) - v0.to(torch.float32) for e in experts])
        summed = deltas.sum(0)
        # Apply spectral flattening only to true 2D linear weights of size > 1
        is_linear = (summed.ndim == 2 and min(summed.shape) > 1 and
                     "embed" not in k.lower() and "position" not in k.lower())
        if (not restrict_to_linear or is_linear) and summed.ndim == 2 and min(summed.shape) > 1:
            U, S, Vh = torch.linalg.svd(summed, full_matrices=False)
            S_iso = torch.full_like(S, S.mean())
            iso_mat = U @ torch.diag(S_iso) @ Vh
            out[k] = (v0.to(torch.float32) + alpha * iso_mat).to(v0.dtype)
        else:
            # uniform average of task vectors (Σ/T)
            mean = deltas.mean(0)
            out[k] = (v0.to(torch.float32) + alpha * mean).to(v0.dtype)
    return out


# ---------------- layer-wise covariance-style methods ---------------------

def _pinv_chol(A: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Cholesky-stabilised pseudo-inverse of a symmetric PSD matrix."""
    n = A.shape[-1]
    Ar = A + eps * torch.eye(n, dtype=A.dtype, device=A.device) * (A.diagonal().abs().mean() + 1.0)
    try:
        L = torch.linalg.cholesky(Ar)
        I = torch.eye(n, dtype=A.dtype, device=A.device)
        return torch.cholesky_solve(I, L)
    except Exception:
        return torch.linalg.pinv(Ar)


def _layerwise_covariance_merge(
    base,
    experts,
    cov_per_task: Dict[str, List[torch.Tensor]],
    use_delta_for_W: bool = False,
    alpha: float = 1.0,
    eps: float = 1e-5,
    fallback_alpha: float = 0.3,
    device: Optional[torch.device] = None,
):
    """Closed-form RegMean / ACTMat-style merge given pre-computed C_t per layer.

    For each linear weight W in ``base`` with provided covariance list
    ``cov_per_task[k] = [C_1, ..., C_T]`` (each Di×Di), compute
        W* = Σ_t W_t · C_t · (Σ_t C_t)^†
    For all other parameters fall back to ``task_arithmetic`` style averaging.

    If ``use_delta_for_W`` is True, replace W_t by ∆_t (TIES-cleaned or raw) when
    we want to interpret the result as W_0 + Σ_t ∆_t C_t (Σ C)^†.
    """
    dev = device if device is not None else torch.device("cpu")
    out = {}
    for k, v0 in base.items():
        if k not in cov_per_task:
            deltas = torch.stack([e[k].to(torch.float32) - v0.to(torch.float32) for e in experts]).mean(0)
            out[k] = (v0.to(torch.float32) + fallback_alpha * deltas).to(v0.dtype)
            continue
        W0 = v0.to(torch.float64).to(dev)
        Wts = [e[k].to(torch.float64).to(dev) for e in experts]
        Cts = [c.to(torch.float64).to(dev) for c in cov_per_task[k]]
        Csum = torch.zeros_like(Cts[0])
        for C in Cts:
            Csum = Csum + C
        Csum_inv = _pinv_chol(Csum, eps=eps)
        acc = torch.zeros_like(W0)
        for Wt, Ct in zip(Wts, Cts):
            mat = Wt if not use_delta_for_W else (Wt - W0)
            acc = acc + mat @ (Ct @ Csum_inv)
        if use_delta_for_W:
            merged = W0 + alpha * acc
        else:
            merged = acc
        out[k] = merged.to(v0.dtype).cpu()
    return out


def actmat(base, experts, eps: float = 1e-5, fallback_alpha: float = 0.3,
           device: Optional[torch.device] = None):
    """ACTMat: data-free covariance estimate C_t = ∆_t^T ∆_t."""
    dev = device if device is not None else torch.device("cpu")
    cov = {}
    for k, v0 in base.items():
        if v0.ndim != 2:
            continue
        cov[k] = []
        for e in experts:
            d = (e[k] - v0).to(torch.float64).to(dev)
            cov[k].append((d.transpose(-2, -1) @ d).cpu())
    return _layerwise_covariance_merge(base, experts, cov, use_delta_for_W=False,
                                       eps=eps, fallback_alpha=fallback_alpha,
                                       device=dev)


def trim_actmat(base, experts, density: float = 0.2, eps: float = 1e-5,
                fallback_alpha: float = 0.3, sign_resolve: bool = True,
                trim: bool = True, device: Optional[torch.device] = None,
                random_trim: bool = False, seed: int = 0):
    """TRIM (ours): apply TIES-style trim + sign-elect to ∆_t, then ACTMat."""
    dev = device if device is not None else torch.device("cpu")
    cov = {}
    cleaned_deltas = {}
    for k, v0 in base.items():
        if v0.ndim != 2:
            continue
        deltas = torch.stack([(e[k] - v0).to(torch.float64).to(dev) for e in experts])
        if trim and sign_resolve:
            d_hat = ties_mask(deltas, density)
        elif trim and not sign_resolve:
            T = deltas.shape[0]
            flat = deltas.reshape(T, -1)
            if 0 < density < 1.0:
                if random_trim:
                    # control: keep a *random* fraction `density` per task
                    g = torch.Generator(device=flat.device).manual_seed(seed)
                    rand = torch.rand(flat.shape, generator=g, device=flat.device)
                    keep = rand < density
                    flat = flat * keep
                else:
                    k_keep = max(1, int(round(density * flat.shape[1])))
                    mag = flat.abs()
                    thr = torch.topk(mag, k_keep, dim=1).values.min(dim=1).values
                    keep = mag >= thr.unsqueeze(1)
                    flat = flat * keep
            d_hat = flat.reshape_as(deltas)
        elif sign_resolve and not trim:
            elected = torch.sign(deltas.sum(0))
            elected[elected == 0] = 1.0
            match = (torch.sign(deltas) == elected.unsqueeze(0)).to(deltas.dtype)
            d_hat = deltas * match
        else:
            d_hat = deltas
        cleaned_deltas[k] = d_hat
        cov[k] = [d_hat[t].transpose(-2, -1) @ d_hat[t] for t in range(d_hat.shape[0])]

    out = {}
    for k, v0 in base.items():
        if k not in cov:
            deltas = torch.stack([e[k].to(torch.float32) - v0.to(torch.float32) for e in experts]).mean(0)
            out[k] = (v0.to(torch.float32) + fallback_alpha * deltas).to(v0.dtype)
            continue
        W0 = v0.to(torch.float64).to(dev)
        Csum = sum(cov[k])
        Csum_inv = _pinv_chol(Csum, eps=eps)
        acc = torch.zeros_like(W0)
        for t in range(len(experts)):
            acc = acc + cleaned_deltas[k][t] @ (cov[k][t] @ Csum_inv)
        out[k] = (W0 + acc).to(v0.dtype).cpu()
        # free GPU memory between layers
        del W0, Csum, Csum_inv, acc
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def regmean(base, experts, cov_per_task: Dict[str, List[torch.Tensor]], eps: float = 1e-5,
            fallback_alpha: float = 0.3, device: Optional[torch.device] = None):
    """RegMean (data oracle) — uses provided activation covariances C_t = E[zz^T].

    cov_per_task[layer_key] must be a list of T tensors of shape (Di, Di).
    """
    return _layerwise_covariance_merge(base, experts, cov_per_task,
                                       use_delta_for_W=False, eps=eps,
                                       fallback_alpha=fallback_alpha,
                                       device=device)
