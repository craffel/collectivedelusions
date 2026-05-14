"""Model-merging methods used in the paper.

All methods operate on *task vectors* tau_t = theta_t - theta_0, defined per
parameter tensor. They take:
  - theta0: dict[str, Tensor]  -- pretrained (shared) state
  - thetas: list[dict[str, Tensor]] -- task-specific states
and return:
  - dict[str, Tensor]   -- merged state

A few methods (RegMean) optionally consume per-layer activation covariance
matrices `covs: list[dict[str, Tensor]]` (one Tensor per linear layer per task);
data-free methods (ACTMat, TACT) derive covariances from the task vectors.

We treat any 2-D weight tensor as a "linear layer" for the purposes of
matrix-shaped merging methods (RegMean / ACTMat / TACT). All other tensors are
merged via the same scalar rule as the method's underlying base (e.g., simple
average for ACTMat/TACT and weighted task arithmetic for non-2D tensors).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch


StateDict = Dict[str, torch.Tensor]

# -------- helpers --------------------------------------------------------------

def task_vectors(theta0: StateDict, thetas: List[StateDict]) -> List[StateDict]:
    return [{k: thetas[i][k] - theta0[k] for k in theta0} for i in range(len(thetas))]


def add_to_pretrained(theta0: StateDict, tau: StateDict, scale: float = 1.0) -> StateDict:
    return {k: theta0[k] + scale * tau[k] for k in theta0}


def merged_from_thetas(thetas: List[StateDict], weights: Optional[List[float]] = None) -> StateDict:
    if weights is None:
        weights = [1.0 / len(thetas)] * len(thetas)
    out: StateDict = {}
    for k in thetas[0]:
        acc = torch.zeros_like(thetas[0][k])
        for w, t in zip(weights, thetas):
            acc = acc + w * t[k]
        out[k] = acc
    return out


_EMBEDDING_KEYWORDS = (
    "positional_embedding", "pos_embed", "class_embedding",
    "token_embedding", "embed.weight", "embeddings",
)
# Tensors that are 2-D but use the non-standard [in, out] layout (i.e. y = x @ W)
# rather than the nn.Linear [out, in] layout (y = F.linear(x, W) = x @ Wᵀ).
# We exclude them from matrix-form merging for simplicity (averaged instead).
_NON_STD_LAYOUT_KEYWORDS = ("visual.proj",)


def is_2d_weight(name: str, tensor: torch.Tensor) -> bool:
    """Whether a 2-D tensor represents a true Linear-style weight matrix.

    Returns True for nn.Linear-style weights ([out_dim, in_dim]) and False for
    embedding tables, additive positional encodings, and non-standard
    projection tensors.
    """
    if tensor.ndim != 2:
        return False
    lowered = name.lower()
    for kw in _EMBEDDING_KEYWORDS:
        if kw in lowered:
            return False
    for kw in _NON_STD_LAYOUT_KEYWORDS:
        if kw in name:
            return False
    return True


# -------- baselines ------------------------------------------------------------

def simple_average(theta0: StateDict, thetas: List[StateDict]) -> StateDict:
    return merged_from_thetas(thetas)


def task_arithmetic(theta0: StateDict, thetas: List[StateDict], alpha: float = 0.3) -> StateDict:
    taus = task_vectors(theta0, thetas)
    summed = {k: torch.zeros_like(theta0[k]) for k in theta0}
    for t in taus:
        for k in summed:
            summed[k] += t[k]
    return {k: theta0[k] + alpha * summed[k] for k in theta0}


def tsv(theta0: StateDict, thetas: List[StateDict], alpha: float = 0.3,
        rank_keep: float = 0.5) -> StateDict:
    """TSV (Gargiulo et al., 2025): Task Singular Vectors.

    Decompose each task vector via SVD, retain only the top ``rank_keep``
    fraction of singular values per task vector, and merge the reconstructed
    low-rank task vectors. This reduces ``singular task interference'' by
    keeping only the dominant directions of each per-task update.

    Simple version (no decorrelation across tasks): per-task low-rank truncation
    followed by summation.
    """
    taus = task_vectors(theta0, thetas)
    out: StateDict = {}
    for k, base in theta0.items():
        if base.ndim == 2 and is_2d_weight(k, base):
            delta_sum = torch.zeros_like(base, dtype=torch.float32)
            for t in range(len(thetas)):
                d = taus[t][k].to(torch.float32)
                try:
                    U, S, Vh = torch.linalg.svd(d, full_matrices=False)
                except Exception:
                    delta_sum = delta_sum + d
                    continue
                r = max(1, int(rank_keep * S.shape[0]))
                d_lowrank = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
                delta_sum = delta_sum + d_lowrank
            out[k] = base + alpha * delta_sum.to(base.dtype)
        else:
            out[k] = torch.stack([thetas[t][k] for t in range(len(thetas))], 0).mean(0)
    return out


def dare(theta0: StateDict, thetas: List[StateDict],
         drop_p: float = 0.7, alpha: float = 0.3, seed: int = 0) -> StateDict:
    """DARE (Yu et al., 2024): randomly drop ``drop_p`` fraction of each
    task vector entry, rescale survivors by ``1/(1-drop_p)``, then merge
    by weighted task arithmetic with scaling ``alpha``.

    DARE provably preserves the expectation of the task vector while reducing
    parameter-level interference between tasks.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    taus = task_vectors(theta0, thetas)
    summed = {k: torch.zeros_like(theta0[k]) for k in theta0}
    survive = 1.0 - drop_p
    for tau in taus:
        for k in summed:
            mask = (torch.rand(tau[k].shape, generator=g, device='cpu') < survive)
            mask = mask.to(tau[k].device, dtype=tau[k].dtype)
            summed[k] = summed[k] + (tau[k] * mask) / max(survive, 1e-12)
    return {k: theta0[k] + alpha * summed[k] for k in theta0}


def iso_c(theta0: StateDict, thetas: List[StateDict], alpha: float = 0.3,
          eps_rel: float = 1e-6) -> StateDict:
    """Iso-C (Marczak et al., 2025): spectrum-flattening data-free merge.

    For each 2-D Linear weight: form the summed task vector Δ_sum = Σ Δ_t,
    take its SVD U Σ V^T, and reconstruct using the mean singular value:
        Δ_iso = U (mean(σ) · I_r) V^T = mean(σ) · U V^T,
    where r is the effective rank (singular values above a relative threshold).
    Add the result back to θ_0 scaled by α. Non-2-D tensors are averaged.

    This implements the simple, fully data-free formulation: flatten the
    summed-task-vector spectrum to balance dominant and underrepresented
    weight-space directions.
    """
    taus = task_vectors(theta0, thetas)
    out: StateDict = {}
    for k, base in theta0.items():
        if base.ndim == 2 and is_2d_weight(k, base):
            delta_sum = torch.zeros_like(base, dtype=torch.float32)
            for t in range(len(thetas)):
                delta_sum = delta_sum + taus[t][k].to(torch.float32)
            try:
                U, S, Vh = torch.linalg.svd(delta_sum, full_matrices=False)
            except Exception:
                # extremely rare; fall back to weighted task arithmetic
                out[k] = base + alpha * delta_sum.to(base.dtype)
                continue
            # Effective rank: retain singular values above eps_rel * max
            thr = eps_rel * S[0].clamp(min=1e-12)
            mask = S >= thr
            mean_sigma = S[mask].mean()
            S_flat = torch.where(mask, mean_sigma, torch.zeros_like(S))
            delta_iso = (U * S_flat.unsqueeze(0)) @ Vh
            out[k] = base + alpha * delta_iso.to(base.dtype)
        else:
            out[k] = torch.stack([thetas[t][k] for t in range(len(thetas))], 0).mean(0)
    return out


# -------- TIES-merging (Yadav et al. 2023) -------------------------------------

def _trim_topk(tau: StateDict, keep_frac: float) -> StateDict:
    """Per-tensor: zero out all but the top-(keep_frac) by magnitude.

    Note: the original TIES paper uses a *global* threshold across the entire
    flattened parameter vector. We follow that more faithful definition.
    """
    # Concatenate magnitudes
    parts = []
    keys = list(tau.keys())
    for k in keys:
        parts.append(tau[k].abs().flatten())
    cat = torch.cat(parts)
    k_keep = max(1, int(keep_frac * cat.numel()))
    thresh = torch.kthvalue(cat, cat.numel() - k_keep + 1).values
    out: StateDict = {}
    for k in keys:
        m = (tau[k].abs() >= thresh).to(tau[k].dtype)
        out[k] = tau[k] * m
    return out


def _trim_topk_per_layer(tau: StateDict, keep_frac: float) -> StateDict:
    """Per-layer: each tensor independently keeps top-(keep_frac) by magnitude.

    Unlike the global threshold of `_trim_topk`, this preserves the same
    fraction of entries in every layer. Useful as a per-layer adaptive trim
    baseline (each layer self-allocates its budget).
    """
    out: StateDict = {}
    for k, v in tau.items():
        flat = v.abs().flatten()
        k_keep = max(1, int(keep_frac * flat.numel()))
        if k_keep >= flat.numel():
            out[k] = v.clone()
            continue
        thresh = torch.kthvalue(flat, flat.numel() - k_keep + 1).values
        m = (v.abs() >= thresh).to(v.dtype)
        out[k] = v * m
    return out


def tact_per_layer(theta0: StateDict, thetas: List[StateDict],
                   keep_frac: float = 0.5,
                   reg_eps: float = 1e-8) -> StateDict:
    """TACT with per-layer trim threshold (each layer independently keeps the top
    `keep_frac` fraction of |Delta_t| entries). Cov-only variant: merge target is the
    full W_t.
    """
    taus = task_vectors(theta0, thetas)
    trimmed = [_trim_topk_per_layer(t, keep_frac) for t in taus]

    def lookup(name, t):
        base = theta0[name]
        if base.ndim == 2 and is_2d_weight(name, base):
            d = trimmed[t][name].to(torch.float32)
            return d.T @ d
        return None

    return _solve_regmean(theta0, thetas, lookup, reg_eps=reg_eps)


def _elect_sign(trimmed: List[StateDict]) -> StateDict:
    """For each parameter, the sign with the largest *total magnitude*."""
    sign: StateDict = {}
    for k in trimmed[0]:
        agg = torch.zeros_like(trimmed[0][k])
        for t in trimmed:
            agg = agg + t[k]
        sign[k] = torch.sign(agg)
    return sign


def _disjoint_merge(trimmed: List[StateDict], sign: StateDict) -> StateDict:
    """Average only the entries whose sign matches the elected sign."""
    out: StateDict = {}
    for k in trimmed[0]:
        s = sign[k]
        # mask out conflicting entries; average non-zero
        mat = torch.stack([t[k] for t in trimmed], dim=0)  # [T, ...]
        masked = mat * (torch.sign(mat) == s.unsqueeze(0)).to(mat.dtype)
        denom = (masked != 0).sum(dim=0).clamp(min=1).to(masked.dtype)
        out[k] = masked.sum(dim=0) / denom
    return out


def ties_merge(theta0: StateDict, thetas: List[StateDict],
               keep_frac: float = 0.2, alpha: float = 0.3) -> StateDict:
    taus = task_vectors(theta0, thetas)
    trimmed = [_trim_topk(t, keep_frac) for t in taus]
    sign = _elect_sign(trimmed)
    tau_merge = _disjoint_merge(trimmed, sign)
    return {k: theta0[k] + alpha * tau_merge[k] for k in theta0}


# -------- RegMean / ACTMat / TACT -----------------------------------------------

def _stable_solve(num: torch.Tensor, S: torch.Tensor,
                  reg_eps: float = 1e-4) -> torch.Tensor:
    """Solve num @ S^+ robustly via SVD-truncated pseudo-inverse.

    Uses ``torch.linalg.pinv`` with a relative tolerance ``rcond=reg_eps``,
    which thresholds singular values below ``rcond * sigma_max(S)``. This is
    well-behaved even when ``S`` is rank-deficient (e.g., when the number of
    merged tasks T is small relative to the layer's input dimension).
    """
    # pinv handles symmetric PSD matrices well via SVD.
    Sinv = torch.linalg.pinv(S, rcond=reg_eps)
    return num @ Sinv


def _solve_regmean(theta0: StateDict, thetas: List[StateDict],
                   covs_for_layer: Callable[[str, int], Optional[torch.Tensor]],
                   reg_eps: float = 1e-4,
                   non2d_weighted_avg: bool = True) -> StateDict:
    """Closed-form regularized RegMean:
        W* = (Σ W_t C_t + λ \bar W) (Σ C_t + λ I)^{-1},
    where λ = reg_eps * mean(diag(Σ C_t)) and \bar W is the unweighted average
    of the task-specific matrices.  The Tikhonov prior toward \bar W fills in
    the null space of (Σ C_t) (which is large when T is small) and yields a
    well-defined, numerically stable solve regardless of T.

    `covs_for_layer(name, t)` returns the C_t tensor for the t-th task's
    parameter named `name`, or None if no covariance is available (in which
    case that layer is averaged).
    """
    out: StateDict = {}
    T = len(thetas)
    for k, base in theta0.items():
        if base.ndim == 2 and is_2d_weight(k, base):
            covs = [covs_for_layer(k, t) for t in range(T)]
            if any(c is None for c in covs):
                out[k] = torch.stack([thetas[t][k] for t in range(T)], 0).mean(0)
                continue
            d = base.shape[1]
            device = base.device
            S = torch.zeros((d, d), device=device, dtype=torch.float32)
            num = torch.zeros_like(base, dtype=torch.float32)
            for t in range(T):
                Ct = covs[t].to(device=device, dtype=torch.float32)
                S = S + Ct
                num = num + thetas[t][k].to(torch.float32) @ Ct
            # Tikhonov prior toward unweighted average of the task matrices.
            W_avg = torch.stack([thetas[t][k] for t in range(T)], 0).mean(0).to(torch.float32)
            scale = S.diag().mean().clamp(min=1e-6)
            lam = reg_eps * scale
            I = torch.eye(d, device=device, dtype=torch.float32)
            S_reg = S + lam * I
            num_reg = num + lam * W_avg
            try:
                W = torch.linalg.solve(S_reg.transpose(-2, -1), num_reg.transpose(-2, -1)).transpose(-2, -1)
            except Exception:
                W = num_reg @ torch.linalg.pinv(S_reg)
            out[k] = W.to(base.dtype)
        else:
            if non2d_weighted_avg:
                out[k] = torch.stack([thetas[t][k] for t in range(T)], 0).mean(0)
            else:
                out[k] = base.clone()
    return out


def regmean(theta0: StateDict, thetas: List[StateDict],
            data_covs: List[Dict[str, torch.Tensor]],
            reg_eps: float = 1e-8) -> StateDict:
    """RegMean with data-derived covariances `data_covs[t][layer_name] = C_t`."""
    def lookup(name, t):
        return data_covs[t].get(name, None)
    return _solve_regmean(theta0, thetas, lookup, reg_eps=reg_eps)


def fisher_merge(theta0: StateDict, thetas: List[StateDict],
                 fishers: List[Dict[str, torch.Tensor]],
                 eps: float = 1e-8) -> StateDict:
    """Fisher merging (Matena & Raffel, 2022).

    Per-parameter weighted average using the diagonal Fisher information F_t
    as importance:
        theta^*_i = (Σ_t F_{t,i} θ_{t,i}) / (Σ_t F_{t,i}).
    Where Σ_t F_t is zero, falls back to simple averaging.
    """
    T = len(thetas)
    out: StateDict = {}
    for k, base in theta0.items():
        num = torch.zeros_like(base, dtype=torch.float32)
        den = torch.zeros_like(base, dtype=torch.float32)
        for t in range(T):
            F_t = fishers[t].get(k)
            if F_t is None:
                continue
            F_t = F_t.to(device=base.device, dtype=torch.float32)
            num = num + F_t * thetas[t][k].to(torch.float32)
            den = den + F_t
        avg = torch.stack([thetas[t][k] for t in range(T)], 0).mean(0).to(torch.float32)
        merged = torch.where(den > eps, num / (den + eps), avg)
        out[k] = merged.to(base.dtype)
    return out


def actmat(theta0: StateDict, thetas: List[StateDict], reg_eps: float = 1e-8) -> StateDict:
    """ACTMat (Hameed et al. 2026): data-free C_t ≈ Δ_tᵀ Δ_t."""
    taus = task_vectors(theta0, thetas)
    def lookup(name, t):
        base = theta0[name]
        if base.ndim == 2 and is_2d_weight(name, base):
            d = taus[t][name].to(torch.float32)
            return d.T @ d  # [in, in]
        return None
    return _solve_regmean(theta0, thetas, lookup, reg_eps=reg_eps)


# -------- proposed: TACT --------------------------------------------------------

def _svd_truncate(d: torch.Tensor, rank_keep: float) -> torch.Tensor:
    """Truncate a 2-D tensor to its top-r singular values where r = rank_keep * min(d_o, d_i)."""
    df = d.to(torch.float32)
    try:
        U, S, Vh = torch.linalg.svd(df, full_matrices=False)
    except Exception:
        return d
    r = max(1, int(rank_keep * S.shape[0]))
    return (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]


def tact_svd_cov_only(theta0: StateDict, thetas: List[StateDict],
                      rank_keep: float = 0.5,
                      reg_eps: float = 1e-4) -> StateDict:
    """SVD-TACT: clean each task vector by SVD-truncating to its top-r components,
    then form the data-free covariance from the cleaned task vectors and run the
    ACTMat solve with the full fine-tuned matrices as the merge target.

    This complements TACT's magnitude trim with rank-truncation cleaning.
    """
    taus = task_vectors(theta0, thetas)
    cleaned_taus: List[StateDict] = []
    for t in range(len(thetas)):
        cleaned = {}
        for k, v in taus[t].items():
            if v.ndim == 2 and is_2d_weight(k, v):
                cleaned[k] = _svd_truncate(v, rank_keep)
            else:
                cleaned[k] = v
        cleaned_taus.append(cleaned)

    def lookup(name, t):
        base = theta0[name]
        if base.ndim == 2 and is_2d_weight(name, base):
            d = cleaned_taus[t][name].to(torch.float32)
            return d.T @ d
        return None

    return _solve_regmean(theta0, thetas, lookup, reg_eps=reg_eps)


def tact_cov_only(theta0: StateDict, thetas: List[StateDict],
                  keep_frac: float = 0.2,
                  use_sign: bool = True,
                  reg_eps: float = 1e-8) -> StateDict:
    """TACT-cov: only the *covariance* uses the cleaned (trimmed + sign-elected)
    task vectors; the merge targets remain the full fine-tuned matrices W_t.

    Solves W* = (Σ W_t C̃_t) (Σ C̃_t)^+, with C̃_t = Δ̃_t^T Δ̃_t.
    This isolates the effect of cleaning the *estimator* of activation covariance
    from the effect of cleaning the merge targets themselves.
    """
    taus = task_vectors(theta0, thetas)
    trimmed = [_trim_topk(t, keep_frac) for t in taus]
    if use_sign:
        sign = _elect_sign(trimmed)
        for t in range(len(trimmed)):
            for k in trimmed[t]:
                s = sign[k]
                m = (torch.sign(trimmed[t][k]) == s).to(trimmed[t][k].dtype)
                trimmed[t][k] = trimmed[t][k] * m

    def lookup(name, t):
        base = theta0[name]
        if base.ndim == 2 and is_2d_weight(name, base):
            d = trimmed[t][name].to(torch.float32)
            return d.T @ d
        return None

    return _solve_regmean(theta0, thetas, lookup, reg_eps=reg_eps)


def tact(theta0: StateDict, thetas: List[StateDict],
         keep_frac: float = 0.2,
         use_sign: bool = True,
         alpha: float = 1.0,
         reg_eps: float = 1e-8) -> StateDict:
    """Trim-And-Covariance-Transform (proposed).

    1. Trim each task vector globally to top-`keep_frac` by magnitude.
    2. (Optional) Sign-elect using TIES rule on the trimmed vectors; for
       each task, zero entries whose sign disagrees with the elected sign.
    3. Compute data-free covariance C_t = Δ̃_tᵀ Δ̃_t on the cleaned task
       vectors (for 2-D weights).
    4. Solve W* = (Σ W̃_t C_t)(Σ C_t)^+, with W̃_t = θ_0 + α · Δ̃_t.
    """
    taus = task_vectors(theta0, thetas)
    trimmed = [_trim_topk(t, keep_frac) for t in taus]
    if use_sign:
        sign = _elect_sign(trimmed)
        for t in range(len(trimmed)):
            for k in trimmed[t]:
                s = sign[k]
                m = (torch.sign(trimmed[t][k]) == s).to(trimmed[t][k].dtype)
                trimmed[t][k] = trimmed[t][k] * m

    # Construct task-specific weight matrices using the cleaned task vectors
    cleaned_thetas: List[StateDict] = []
    for t in range(len(trimmed)):
        cleaned_thetas.append({k: theta0[k] + alpha * trimmed[t][k] for k in theta0})

    def lookup(name, t):
        base = theta0[name]
        if base.ndim == 2 and is_2d_weight(name, base):
            d = trimmed[t][name].to(torch.float32)
            return d.T @ d
        return None

    return _solve_regmean(theta0, cleaned_thetas, lookup, reg_eps=reg_eps)


# -------- Driver registry ------------------------------------------------------

METHODS: Dict[str, Callable] = {
    "average": simple_average,
    "task_arithmetic": task_arithmetic,
    "ties": ties_merge,
    "iso_c": iso_c,
    "tsv": tsv,
    "dare": dare,
    "actmat": actmat,
    "tact": tact,
    "tact_cov": tact_cov_only,
}
