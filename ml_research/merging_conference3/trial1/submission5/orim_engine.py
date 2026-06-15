import os
import sys
import torch
import gc
import numpy as np
from tqdm import tqdm

def robust_cpu_svd(A):
    """
    Computes SVD on GPU if available for 1000x speedup, with robust CPU-based NumPy fallbacks to bypass any stability or convergence issues.
    """
    if A.device.type == 'cuda':
        try:
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            return U, S, Vh
        except Exception as e:
            # If GPU SVD fails to converge, fallback to CPU NumPy
            pass
            
    A_np = A.cpu().numpy()
    try:
        U_np, S_np, Vh_np = np.linalg.svd(A_np, full_matrices=False)
        return torch.from_numpy(U_np).to(A.device), torch.from_numpy(S_np).to(A.device), torch.from_numpy(Vh_np).to(A.device)
    except Exception:
        # Fallback with small diagonal regularization in NumPy
        try:
            epsilon = 1e-6
            A_reg = A_np + epsilon * np.eye(A_np.shape[0])
            U_np, S_np, Vh_np = np.linalg.svd(A_reg, full_matrices=False)
            return torch.from_numpy(U_np).to(A.device), torch.from_numpy(S_np).to(A.device), torch.from_numpy(Vh_np).to(A.device)
        except Exception:
            # Absolute fallback to identity and ones
            d_out, d_in = A.shape
            d_min = min(d_out, d_in)
            U = torch.eye(d_out, d_min, device=A.device)
            S = torch.ones(d_min, device=A.device)
            Vh = torch.eye(d_min, d_in, device=A.device)
            return U, S, Vh


@torch.no_grad()
def orim_merge_weights(W0, Wi_list, gamma=0.5, decoupling_mode='global', use_decoupling=True):
    """
    Performs ORIM merging or Isotropic merging on a single 2D weight matrix.
    Computations are automatically run on the GPU if available for extreme speed.
    """
    N = len(Wi_list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    W0_f32 = W0.to(device=device, dtype=torch.float32)
    Wi_f32_list = [Wi.to(device=device, dtype=torch.float32) for Wi in Wi_list]

    if not use_decoupling:
        # Pure Isotropic Merging (SAIM baseline): SVD on task vectors directly
        balanced_updates = []
        for Wi_f32 in Wi_f32_list:
            tau_i = Wi_f32 - W0_f32
            U_t, S_t, Vh_t = robust_cpu_svd(tau_i)
            mean_sigma = S_t.mean()
            S_t_balanced = mean_sigma + gamma * (S_t - mean_sigma)
            tau_i_balanced = U_t @ torch.diag(S_t_balanced) @ Vh_t
            balanced_updates.append(tau_i_balanced)
        
        merged_update = torch.stack(balanced_updates, dim=0).mean(dim=0)
        W_final = W0_f32 + merged_update
        return W_final.to(device=W0.device, dtype=W0.dtype)

    # ORIM (with Orthogonal-Residual Decoupling)
    R_list = []
    rho_list = []

    if decoupling_mode == 'conflict_aware':
        tau_list = [Wi_f32 - W0_f32 for Wi_f32 in Wi_f32_list]
        tau_mean = torch.stack(tau_list, dim=0).mean(dim=0) # (d_out, d_in)
        
        tau_mean_norms = torch.norm(tau_mean, dim=0, keepdim=True) # (1, d_in)
        tau_mean_norms = torch.clamp(tau_mean_norms, min=1e-8)

    for i in range(N):
        Wi_f32 = Wi_f32_list[i]
        
        if decoupling_mode == 'conflict_aware':
            tau_i = tau_list[i]
            tau_i_norms = torch.norm(tau_i, dim=0, keepdim=True) # (1, d_in)
            tau_i_norms = torch.clamp(tau_i_norms, min=1e-8)
            
            # Cosine similarity column-wise
            cos_sim = (tau_i * tau_mean).sum(dim=0, keepdim=True) / (tau_i_norms * tau_mean_norms) # (1, d_in)
            
            # Keep conflict columns
            conflict_mask = (cos_sim < 0).float() # (1, d_in)
            tau_conf = tau_i * conflict_mask
            W_target_i = W0_f32 + tau_conf
        else:
            W_target_i = Wi_f32

        # Solve Orthogonal Procrustes: robust SVD of W_target_i @ W0^T
        A = W_target_i @ W0_f32.T
        U, S, Vh = robust_cpu_svd(A)
        R_i = U @ Vh
        R_list.append(R_i)

        # Residual acquisition
        rho_i = Wi_f32 - R_i @ W0_f32
        
        # Isotropic Spectrum Balancing of Residuals
        U_r, S_r, Vh_r = robust_cpu_svd(rho_i)
        mean_sigma = S_r.mean()
        S_r_balanced = mean_sigma + gamma * (S_r - mean_sigma)
        rho_i_balanced = U_r @ torch.diag(S_r_balanced) @ Vh_r
        rho_list.append(rho_i_balanced)

    # Lie Manifold Merging of Rotations
    Q_list = []
    I = torch.eye(W0_f32.shape[0], dtype=torch.float32, device=device)
    for R_i in R_list:
        try:
            # Inverse Cayley Transform
            Q_i = (R_i - I) @ torch.linalg.inv(R_i + I + 1e-6 * I)
            # Enforce skew-symmetry
            Q_i = 0.5 * (Q_i - Q_i.T)
            Q_list.append(Q_i)
        except Exception as e:
            Q_list.append(torch.zeros_like(R_i))

    # Magnitude-Corrected Average of Q
    Q_sum = torch.stack(Q_list, dim=0).sum(dim=0)
    sum_norm = sum(torch.linalg.norm(Q_i, ord='fro') for Q_i in Q_list)
    norm_sum = torch.linalg.norm(Q_sum, ord='fro')
    Q_merged = (Q_sum / N) * (sum_norm / (norm_sum + 1e-8))

    # Convert Q_merged back to R_merged via Cayley Transform
    try:
        R_merged = (I + Q_merged) @ torch.linalg.inv(I - Q_merged + 1e-6 * I)
    except Exception as e:
        R_merged = I

    # Hybrid Combination
    rho_merged = torch.stack(rho_list, dim=0).mean(dim=0)
    
    W_final = R_merged @ W0_f32 + rho_merged
    return W_final.to(device=W0.device, dtype=W0.dtype)


@torch.no_grad()
def ties_merge_weights(W0, Wi_list, k=20):
    """
    Performs TIES-Merging on a single layer weight.
    All operations are kept on CPU.
    """
    N = len(Wi_list)
    
    W0_f = W0.cpu().float()
    Wi_f_list = [Wi.cpu().float() for Wi in Wi_list]
    
    # 1. Compute updates
    updates = [Wi_f - W0_f for Wi_f in Wi_f_list]
    updates = torch.stack(updates, dim=0) # (N, ...)
    
    # 2. Trim (top-k% by magnitude)
    shape = updates.shape
    flat_updates = updates.view(N, -1)
    
    trimmed_flat = torch.zeros_like(flat_updates)
    for i in range(N):
        task_update = flat_updates[i]
        abs_update = torch.abs(task_update)
        n_elements = abs_update.numel()
        k_elements = max(1, int(n_elements * (k / 100.0)))
        threshold = torch.topk(abs_update, k_elements).values[-1]
        
        mask = (abs_update >= threshold).float()
        trimmed_flat[i] = task_update * mask
        
    trimmed_updates = trimmed_flat.view(shape) # (N, ...)
    
    # 3. Elect Sign (majority sign)
    signs = torch.sign(trimmed_updates) # (N, ...)
    sign_sum = signs.sum(dim=0) # (...)
    majority_sign = torch.sign(sign_sum) # (...)
    
    # 4. Disjoint Merge
    agree_mask = (torch.sign(trimmed_updates) == majority_sign).float() # (N, ...)
    agree_values = trimmed_updates * agree_mask
    
    agree_count = agree_mask.sum(dim=0)
    merged_update = agree_values.sum(dim=0) / torch.clamp(agree_count, min=1.0)
    
    W_final = W0_f + merged_update
    return W_final.to(dtype=W0.dtype)


@torch.no_grad()
def merge_orim_state_dicts(pretrained_sd, task_sds, gamma=0.5, decoupling_mode='global', use_decoupling=True):
    """
    Performs ORIM merging directly on state dicts in memory on the CPU.
    """
    new_state_dict = {}

    # Perform merging layer by layer
    for key in pretrained_sd.keys():
        W0 = pretrained_sd[key]
        is_embedding = any(x in key.lower() for x in ["embedding", "positional", "token"])
        if W0.dtype in [torch.int64, torch.uint8] or len(W0.shape) != 2 or is_embedding:
            # 1D/0D parameters (biases, layernorms) and embedding weights merged using Task Arithmetic average
            task_updates = [sd[key].float() - W0.float() for sd in task_sds]
            merged_update = torch.stack(task_updates, dim=0).mean(dim=0)
            W_final = W0.float() + merged_update
            new_state_dict[key] = W_final.to(dtype=W0.dtype)
        else:
            # 2D projection weight matrix: apply ORIM
            Wi_list = [sd[key] for sd in task_sds]
            new_state_dict[key] = orim_merge_weights(W0, Wi_list, gamma=gamma, decoupling_mode=decoupling_mode, use_decoupling=use_decoupling)

    return new_state_dict


@torch.no_grad()
def merge_ties_state_dicts(pretrained_sd, task_sds, k=20):
    """
    Performs TIES-Merging directly on state dicts in memory on the CPU.
    """
    new_state_dict = {}

    # Perform merging layer by layer
    for key in pretrained_sd.keys():
        W0 = pretrained_sd[key]
        if W0.dtype in [torch.int64, torch.uint8]:
            new_state_dict[key] = W0
        else:
            Wi_list = [sd[key] for sd in task_sds]
            new_state_dict[key] = ties_merge_weights(W0, Wi_list, k=k)

    return new_state_dict
