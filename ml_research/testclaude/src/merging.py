"""Merging methods for task-specific fine-tuned checkpoints.

Each merging method takes:
    pretrained_sd: dict[str -> Tensor]   (base model weights)
    task_sds:      list[dict[str -> Tensor]]   (one fine-tuned state dict per task)
and returns a merged state dict.

All merging is performed parameter-wise on the intersection of parameter names.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


# ----------------------------- helpers -----------------------------

def task_vectors(pretrained_sd: Dict[str, torch.Tensor], task_sds: List[Dict[str, torch.Tensor]]):
    """Compute list of Δ_t = θ_t - θ_0 per parameter."""
    tvs = []
    for sd in task_sds:
        tv = {k: (sd[k] - pretrained_sd[k]).detach() for k in pretrained_sd if k in sd}
        tvs.append(tv)
    return tvs


def apply_task_vector(pretrained_sd: Dict[str, torch.Tensor], tv: Dict[str, torch.Tensor], alpha: float = 1.0):
    return {k: (pretrained_sd[k] + alpha * tv[k]) if k in tv else pretrained_sd[k].clone() for k in pretrained_sd}


def topk_mask(tensor: torch.Tensor, k: float) -> torch.Tensor:
    """Return boolean mask keeping top-k fraction of |tensor| entries."""
    if k >= 1.0:
        return torch.ones_like(tensor, dtype=torch.bool)
    flat = tensor.abs().flatten()
    n = flat.numel()
    keep = max(1, int(n * k))
    if keep >= n:
        return torch.ones_like(tensor, dtype=torch.bool)
    thresh = torch.topk(flat, keep, sorted=False).values.min()
    return tensor.abs() >= thresh


# ----------------------------- methods -----------------------------

def simple_average(pretrained_sd, task_sds):
    """Averaging of fine-tuned weights (no pretrained anchor)."""
    out = {}
    for k in pretrained_sd:
        acc = None
        count = 0
        for sd in task_sds:
            if k in sd:
                acc = sd[k].detach().clone() if acc is None else acc + sd[k]
                count += 1
        out[k] = (acc / count) if count > 0 else pretrained_sd[k].clone()
    return out


def task_arithmetic(pretrained_sd, task_sds, alpha: float = 0.3):
    """θ* = θ_0 + α Σ Δ_t."""
    tvs = task_vectors(pretrained_sd, task_sds)
    out = {}
    for k in pretrained_sd:
        acc = torch.zeros_like(pretrained_sd[k], dtype=torch.float32)
        for tv in tvs:
            if k in tv:
                acc = acc + tv[k].float()
        out[k] = pretrained_sd[k] + alpha * acc.to(pretrained_sd[k].dtype)
    return out


def ties_merging(pretrained_sd, task_sds, keep_frac: float = 0.2, alpha: float = 0.4):
    """TIES-Merging (Yadav et al. 2023): trim, elect sign, disjoint-merge, add.

    Algorithm:
      1. For each task, keep top-keep_frac% of |Δ|, zero the rest.
      2. For each parameter, sign = sign(Σ_t Δ̂_t). (Global elected sign.)
      3. For each parameter, average Δ̂_t only over tasks whose sign matches the elected sign.
      4. θ* = θ_0 + α · mean_merged_Δ.
    """
    tvs = task_vectors(pretrained_sd, task_sds)
    out = {}
    for k in pretrained_sd:
        trimmed = []
        for tv in tvs:
            if k in tv:
                t = tv[k].float()
                mask = topk_mask(t, keep_frac)
                trimmed.append(t * mask)
        if not trimmed:
            out[k] = pretrained_sd[k].clone()
            continue
        stack = torch.stack(trimmed, dim=0)  # [T, *]
        elected = torch.sign(stack.sum(dim=0))  # [*]
        # mask entries where sign != elected (elected==0 means keep zero)
        agree = (torch.sign(stack) == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)
        agree_sum = torch.where(agree, stack, torch.zeros_like(stack)).sum(dim=0)
        agree_count = agree.sum(dim=0).clamp(min=1)
        merged = agree_sum / agree_count
        out[k] = pretrained_sd[k] + alpha * merged.to(pretrained_sd[k].dtype)
    return out


# --- RegMean / ACTMat / SCALE: layer-wise covariance merging ---
#
# These methods operate on linear / conv / attention weight matrices W_t and merge as
#   W* = (Σ_t W_t C_t) (Σ_t C_t)^† .
# For the bias (and non-matrix params like LayerNorm scale), we fall back to simple averaging
# (or task-arithmetic for params that are already handled).
#
# For each matrix-parameter, we flatten it to a 2D matrix [out, in]. Covariance is an
# [in, in] matrix:
#   * RegMean: covariance C_t is computed from auxiliary activation data Z^T Z.
#   * ACTMat : covariance C_t ≈ Δ_t^T Δ_t (data-free).
#   * SCALE  : same as ACTMat but on TIES-preprocessed Δ̃_t.

MATRIX_KEY_PATTERNS = ("proj", "fc", "conv", "linear", "mlp", "attn", "in_proj", "out_proj", "c_proj", "c_fc", ".weight")


def _is_matrix_param(k, t):
    if t.ndim < 2:
        return False
    # Exclude LN / BN scale etc.
    return True


def _reshape_to_matrix(t):
    """Reshape an N-D weight tensor to 2D [out, in]."""
    if t.ndim == 2:
        return t, t.shape
    # Conv weight: [out, in, kH, kW] -> [out, in*kH*kW]
    return t.reshape(t.shape[0], -1), t.shape


def _unreshape(m, orig_shape):
    return m.reshape(orig_shape)


def actmat(pretrained_sd, task_sds, ridge: float = 1e-4):
    """ACTMat (Hameed et al. 2025): covariance-free merging where C_t ≈ Δ_t^T Δ_t.

    θ* computed per matrix as:
      W*  = (Σ_t  W_t Δ_t^T Δ_t ) · (Σ_t Δ_t^T Δ_t + ridge·I)^{-1}
    For non-matrix parameters, fall back to simple averaging.
    """
    tvs = task_vectors(pretrained_sd, task_sds)
    out = {}
    for k in pretrained_sd:
        theta0 = pretrained_sd[k]
        W_t_list = []
        delta_t_list = []
        for i, sd in enumerate(task_sds):
            if k in sd:
                W_t_list.append(sd[k])
                delta_t_list.append(tvs[i][k])
        if not W_t_list:
            out[k] = theta0.clone()
            continue
        if not _is_matrix_param(k, theta0):
            # Average fine-tuned weights
            acc = torch.stack([w.float() for w in W_t_list], dim=0).mean(dim=0)
            out[k] = acc.to(theta0.dtype)
            continue
        # Reshape to 2D [out, in]
        W2_list = []
        shape = theta0.shape
        for W in W_t_list:
            W2, _ = _reshape_to_matrix(W)
            W2_list.append(W2.float())
        D2_list = []
        for D in delta_t_list:
            D2, _ = _reshape_to_matrix(D)
            D2_list.append(D2.float())

        din = W2_list[0].shape[1]
        # Covariance matrices C_t = Δ_t^T Δ_t  of shape [in, in]
        C_sum = torch.zeros(din, din, device=W2_list[0].device)
        WC_sum = torch.zeros_like(W2_list[0])
        for W2, D2 in zip(W2_list, D2_list):
            C_t = D2.T @ D2
            C_sum = C_sum + C_t
            WC_sum = WC_sum + W2 @ C_t
        # Ridge
        C_sum = C_sum + ridge * torch.eye(din, device=C_sum.device) * (C_sum.diagonal().mean().item() + 1e-12)
        W_star = torch.linalg.solve(C_sum.T, WC_sum.T).T  # solve X^T C^T = WC^T
        out[k] = _unreshape(W_star, shape).to(theta0.dtype)
    return out


def scale_merge(pretrained_sd, task_sds, keep_frac: float = 0.2, ridge: float = 1e-4,
                use_sign_election: bool = True, sign_source: str = "trimmed"):
    """SCALE-Merge (our method): TIES-style trim & sign-election, then covariance merge.

    Steps (per matrix parameter, reshaped to [out, in]):
      1. Compute Δ_t = W_t - W_0.
      2. Trim Δ_t: keep top-keep_frac fraction of entries by |Δ_t|, zero the rest.
      3. Elect sign γ = sign(Σ_t Δ̂_t). Zero entries of Δ̂_t that disagree with γ.
      4. Let Δ̃_t = Δ̂_t after trimming and sign alignment.
      5. Covariance C_t = Δ̃_t^T Δ̃_t.
      6. Merged W* = (Σ_t W_t C_t) (Σ_t C_t + ridge I)^{-1}.
    """
    tvs = task_vectors(pretrained_sd, task_sds)
    out = {}
    for k in pretrained_sd:
        theta0 = pretrained_sd[k]
        W_t_list = []
        D_t_list = []
        for i, sd in enumerate(task_sds):
            if k in sd:
                W_t_list.append(sd[k])
                D_t_list.append(tvs[i][k])
        if not W_t_list:
            out[k] = theta0.clone()
            continue
        if not _is_matrix_param(k, theta0):
            acc = torch.stack([w.float() for w in W_t_list], dim=0).mean(dim=0)
            out[k] = acc.to(theta0.dtype)
            continue

        shape = theta0.shape
        W2_list = []
        D2_list = []
        for W, D in zip(W_t_list, D_t_list):
            W2, _ = _reshape_to_matrix(W)
            D2, _ = _reshape_to_matrix(D)
            W2_list.append(W2.float())
            D2_list.append(D2.float())

        # Step 1-2: Trim top-k entries per task
        trimmed = []
        for D2 in D2_list:
            mask = topk_mask(D2, keep_frac)
            trimmed.append(D2 * mask)

        # Step 3: Elect sign and zero disagreeing entries
        if use_sign_election:
            stack = torch.stack(trimmed, dim=0)  # [T, out, in]
            if sign_source == "trimmed":
                elected = torch.sign(stack.sum(dim=0))  # [out, in]
            else:
                elected = torch.sign(torch.stack(D2_list, dim=0).sum(dim=0))
            agree = (torch.sign(stack) == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)
            cleaned_stack = torch.where(agree, stack, torch.zeros_like(stack))
            trimmed = [cleaned_stack[t] for t in range(cleaned_stack.shape[0])]

        # Steps 4-5: covariance from cleaned Δ̃
        din = W2_list[0].shape[1]
        C_sum = torch.zeros(din, din, device=W2_list[0].device)
        WC_sum = torch.zeros_like(W2_list[0])
        for W2, D2_clean in zip(W2_list, trimmed):
            C_t = D2_clean.T @ D2_clean
            C_sum = C_sum + C_t
            WC_sum = WC_sum + W2 @ C_t
        C_sum = C_sum + ridge * torch.eye(din, device=C_sum.device) * (C_sum.diagonal().mean().item() + 1e-12)
        W_star = torch.linalg.solve(C_sum.T, WC_sum.T).T
        out[k] = _unreshape(W_star, shape).to(theta0.dtype)
    return out


def scale_merge_cg(pretrained_sd, task_sds, keep_frac: float = 0.3, ridge: float = 1e-4,
                   use_sign_election: bool = False, sign_source: str = "trimmed",
                   cg_iters: int = 50, cg_tol: float = 1e-6,
                   return_stats: bool = False):
    """SCALE-Merge with MaTS-style matrix-free conjugate gradient solve.

    Identical output (up to CG residual) to ``scale_merge`` but never materialises
    the d_in x d_in gram matrix Σ_t Δ̃_t^T Δ̃_t. Instead, each CG matvec applies
    v -> Σ_t Δ̃_t^T (Δ̃_t v) + ρλ̄ v, which costs O(T * d_out * d_in) per matvec
    and uses only O(T * d_out * d_in) memory for the trimmed task vectors.
    Columns of the right-hand side are solved in parallel with column-wise α/β.
    """
    tvs = task_vectors(pretrained_sd, task_sds)
    out = {}
    stats = {"layers": 0, "iters_total": 0, "iters_max": 0}
    for k in pretrained_sd:
        theta0 = pretrained_sd[k]
        W_t_list, D_t_list = [], []
        for i, sd in enumerate(task_sds):
            if k in sd:
                W_t_list.append(sd[k])
                D_t_list.append(tvs[i][k])
        if not W_t_list:
            out[k] = theta0.clone()
            continue
        if not _is_matrix_param(k, theta0):
            acc = torch.stack([w.float() for w in W_t_list], dim=0).mean(dim=0)
            out[k] = acc.to(theta0.dtype)
            continue

        shape = theta0.shape
        W2_list, D2_list = [], []
        for W, D in zip(W_t_list, D_t_list):
            W2, _ = _reshape_to_matrix(W); D2, _ = _reshape_to_matrix(D)
            W2_list.append(W2.float()); D2_list.append(D2.float())

        trimmed = []
        for D2 in D2_list:
            mask = topk_mask(D2, keep_frac)
            trimmed.append(D2 * mask)

        if use_sign_election:
            stack = torch.stack(trimmed, dim=0)
            base = stack if sign_source == "trimmed" else torch.stack(D2_list, dim=0)
            elected = torch.sign(base.sum(dim=0))
            agree = (torch.sign(stack) == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)
            cleaned = torch.where(agree, stack, torch.zeros_like(stack))
            trimmed = [cleaned[t] for t in range(cleaned.shape[0])]

        din = W2_list[0].shape[1]
        trace = 0.0
        for d in trimmed:
            trace = trace + (d * d).sum().item()
        lam_bar = trace / max(din, 1)
        rho_eff = ridge * (lam_bar + 1e-12)

        def matvec(V):
            acc = torch.zeros_like(V)
            for d in trimmed:
                acc = acc + d.T @ (d @ V)
            return acc + rho_eff * V

        B = torch.zeros(din, W2_list[0].shape[0], device=W2_list[0].device, dtype=W2_list[0].dtype)
        for W2, d in zip(W2_list, trimmed):
            B = B + d.T @ (d @ W2.T)

        X = torch.zeros_like(B)
        R = B - matvec(X)
        P = R.clone()
        rs_old = (R * R).sum(dim=0)
        it_used = 0
        for it in range(cg_iters):
            AP = matvec(P)
            denom = (P * AP).sum(dim=0).clamp(min=1e-30)
            alpha = rs_old / denom
            X = X + alpha * P
            R = R - alpha * AP
            rs_new = (R * R).sum(dim=0)
            it_used = it + 1
            if rs_new.max().sqrt().item() < cg_tol:
                break
            beta = rs_new / rs_old.clamp(min=1e-30)
            P = R + beta * P
            rs_old = rs_new

        W_star = X.T
        out[k] = _unreshape(W_star, shape).to(theta0.dtype)
        stats["layers"] += 1
        stats["iters_total"] += it_used
        stats["iters_max"] = max(stats["iters_max"], it_used)

    if return_stats:
        return out, stats
    return out


def dare(pretrained_sd, task_sds, drop_p: float = 0.9, alpha: float = 0.3, seed: int = 0):
    """DARE (Yu et al. 2024): Drop And REscale task vectors, then apply task arithmetic.

    Each Δ_t is sparsified by dropping a random fraction drop_p of entries and rescaling
    the survivors by 1/(1-drop_p) so the expectation is preserved. The surviving task
    vectors are then summed with task-arithmetic coefficient alpha.
    """
    tvs = task_vectors(pretrained_sd, task_sds)
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k in pretrained_sd:
        acc = torch.zeros_like(pretrained_sd[k], dtype=torch.float32)
        for tv in tvs:
            if k in tv:
                t = tv[k].float()
                mask = (torch.rand(t.shape, generator=g) > drop_p).to(t.device).to(t.dtype)
                t_drop = t * mask / max(1e-6, (1.0 - drop_p))
                acc = acc + t_drop
        out[k] = pretrained_sd[k] + alpha * acc.to(pretrained_sd[k].dtype)
    return out


def breadcrumbs(pretrained_sd, task_sds, drop_small: float = 0.9, drop_large: float = 0.01,
                alpha: float = 0.3):
    """Model Breadcrumbs (Davari & Belilovsky 2024).

    Sparsify each task vector by zeroing the bottom `drop_small` fraction of entries by |Δ|
    AND the top `drop_large` fraction. Keep the middle-magnitude band. Sum across tasks
    with task-arithmetic coefficient alpha.
    """
    tvs = task_vectors(pretrained_sd, task_sds)
    out = {}
    for k in pretrained_sd:
        acc = torch.zeros_like(pretrained_sd[k], dtype=torch.float32)
        for tv in tvs:
            if k not in tv:
                continue
            t = tv[k].float()
            a = t.abs().flatten()
            n = a.numel()
            if n == 0:
                continue
            lo = max(1, min(n, int(drop_small * n)))
            hi = max(lo + 1, min(n, int((1.0 - drop_large) * n)))
            # O(n) threshold extraction via kthvalue (no full sort).
            lo_thr = torch.kthvalue(a, lo).values
            hi_thr = torch.kthvalue(a, hi).values
            mask = (t.abs() >= lo_thr) & (t.abs() <= hi_thr)
            acc = acc + (t * mask.to(t.dtype))
        out[k] = pretrained_sd[k] + alpha * acc.to(pretrained_sd[k].dtype)
    return out


def dare_ties(pretrained_sd, task_sds, drop_p: float = 0.9, keep_frac: float = 0.2,
              alpha: float = 0.3, seed: int = 0):
    """DARE + TIES: apply DARE drop-and-rescale, then TIES elect-sign merge."""
    tvs = task_vectors(pretrained_sd, task_sds)
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k in pretrained_sd:
        trimmed = []
        for tv in tvs:
            if k in tv:
                t = tv[k].float()
                m_drop = (torch.rand(t.shape, generator=g) > drop_p).to(t.device).to(t.dtype)
                t2 = t * m_drop / max(1e-6, (1.0 - drop_p))
                m_trim = topk_mask(t2, keep_frac)
                trimmed.append(t2 * m_trim)
        if not trimmed:
            out[k] = pretrained_sd[k].clone(); continue
        stack = torch.stack(trimmed, dim=0)
        elected = torch.sign(stack.sum(dim=0))
        agree = (torch.sign(stack) == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)
        agree_sum = torch.where(agree, stack, torch.zeros_like(stack)).sum(dim=0)
        agree_count = agree.sum(dim=0).clamp(min=1)
        merged = agree_sum / agree_count
        out[k] = pretrained_sd[k] + alpha * merged.to(pretrained_sd[k].dtype)
    return out


def della(pretrained_sd, task_sds, p_low: float = 0.1, p_high: float = 0.9,
          alpha: float = 0.3, seed: int = 0):
    """DELLA (Deep et al. 2024): magnitude-ranked Bernoulli sampling, rescale, then TIES elect-sign.

    Per task vector Δ_t (flattened):
      1. Sort entries by |Δ| ascending; assign keep-probability p(i) linear in rank:
         p(i) = p_low + (p_high - p_low) * rank_i / (n-1)
         so low-magnitude entries get p_low, high-magnitude get p_high.
      2. Bernoulli-sample a mask with these probabilities. Rescale survivors by 1/p(i).
      3. Stack across tasks, elect the global sign, average agreeing entries (TIES step).
      4. θ* = θ_0 + α · merged.
    """
    tvs = task_vectors(pretrained_sd, task_sds)
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k in pretrained_sd:
        sampled = []
        for tv in tvs:
            if k not in tv:
                continue
            t = tv[k].float()
            flat = t.flatten()
            n = flat.numel()
            if n == 0:
                continue
            # Rank of each entry by |Δ| (ascending: 0 = smallest, n-1 = largest).
            abs_flat = flat.abs()
            order = torch.argsort(abs_flat)
            ranks = torch.empty_like(order)
            ranks[order] = torch.arange(n, device=flat.device)
            rank_frac = ranks.float() / max(1, n - 1)
            p = p_low + (p_high - p_low) * rank_frac  # per-entry keep probability
            u = torch.rand(n, generator=g).to(flat.device)
            mask = (u < p).float()
            rescaled = (flat * mask / p.clamp(min=1e-6)).view_as(t)
            sampled.append(rescaled)
        if not sampled:
            out[k] = pretrained_sd[k].clone(); continue
        stack = torch.stack(sampled, dim=0)  # [T, *]
        elected = torch.sign(stack.sum(dim=0))
        agree = (torch.sign(stack) == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)
        agree_sum = torch.where(agree, stack, torch.zeros_like(stack)).sum(dim=0)
        agree_count = agree.sum(dim=0).clamp(min=1)
        merged = agree_sum / agree_count
        out[k] = pretrained_sd[k] + alpha * merged.to(pretrained_sd[k].dtype)
    return out


def fisher_approx_merge(pretrained_sd, task_sds, ridge: float = 1e-4):
    """Data-free Fisher-weighted averaging: weight each W_t parameter-wise by |Δ_t|^2.

    Approximates Fisher information (Matena & Raffel 2022) in the data-free regime by
    noting that, under a few assumptions, the diagonal Fisher is proportional to
    E[(∂L/∂W)^2] ≈ (W_t − W_0)^2 / η^2 after K SGD steps (Pascanu & Bengio, 2014).
    θ* = Σ_t F_t W_t / Σ_t F_t with F_t = Δ_t^2 (per parameter, elementwise).
    """
    tvs = task_vectors(pretrained_sd, task_sds)
    out = {}
    for k in pretrained_sd:
        num = torch.zeros_like(pretrained_sd[k], dtype=torch.float32)
        den = torch.zeros_like(pretrained_sd[k], dtype=torch.float32)
        for i, sd in enumerate(task_sds):
            if k in sd:
                F = tvs[i][k].float().pow(2) + ridge
                num = num + F * sd[k].float()
                den = den + F
        den = den.clamp(min=ridge)
        out[k] = (num / den).to(pretrained_sd[k].dtype)
    return out


def regmean(pretrained_sd, task_sds, task_covariances: List[Dict[str, torch.Tensor]], ridge: float = 1e-4):
    """RegMean: covariance C_t computed from auxiliary activation data (per-layer Z_t^T Z_t)."""
    out = {}
    for k in pretrained_sd:
        theta0 = pretrained_sd[k]
        W_t_list = [sd[k] for sd in task_sds if k in sd]
        if not W_t_list:
            out[k] = theta0.clone(); continue
        if not _is_matrix_param(k, theta0) or not any(k in tc for tc in task_covariances):
            acc = torch.stack([w.float() for w in W_t_list], dim=0).mean(dim=0)
            out[k] = acc.to(theta0.dtype); continue

        shape = theta0.shape
        W2_list = []
        for W in W_t_list:
            W2, _ = _reshape_to_matrix(W)
            W2_list.append(W2.float())
        din = W2_list[0].shape[1]
        C_sum = torch.zeros(din, din, device=W2_list[0].device)
        WC_sum = torch.zeros_like(W2_list[0])
        for W2, tc in zip(W2_list, task_covariances):
            if k in tc:
                C_t = tc[k].to(W2.device).float()
                C_sum = C_sum + C_t
                WC_sum = WC_sum + W2 @ C_t
        C_sum = C_sum + ridge * torch.eye(din, device=C_sum.device) * (C_sum.diagonal().mean().item() + 1e-12)
        W_star = torch.linalg.solve(C_sum.T, WC_sum.T).T
        out[k] = _unreshape(W_star, shape).to(theta0.dtype)
    return out


def consensus_merging(pretrained_sd, task_sds, keep_frac: float = 0.2, min_agree: int = 2,
                      alpha: float = 0.3):
    """Consensus Merging / TALL-masks (Wang et al. 2024).

    Per parameter:
      1. Build per-task binary mask m_t = top-keep_frac entries of |Δ_t|.
      2. Consensus mask M = (Σ_t m_t >= min_agree).
      3. θ* = θ_0 + α · M · Σ_t Δ_t.
    Keeps only parameter entries that at least ``min_agree`` out of T tasks all
    consider important, and discards the rest.
    """
    tvs = task_vectors(pretrained_sd, task_sds)
    out = {}
    for k in pretrained_sd:
        acc = torch.zeros_like(pretrained_sd[k], dtype=torch.float32)
        mask_count = torch.zeros_like(pretrained_sd[k], dtype=torch.float32)
        any_found = False
        for tv in tvs:
            if k in tv:
                any_found = True
                t = tv[k].float()
                m = topk_mask(t, keep_frac).to(t.dtype)
                mask_count = mask_count + m
                acc = acc + t
        if not any_found:
            out[k] = pretrained_sd[k].clone(); continue
        consensus = (mask_count >= min_agree).to(acc.dtype)
        out[k] = pretrained_sd[k] + alpha * (consensus * acc).to(pretrained_sd[k].dtype)
    return out
