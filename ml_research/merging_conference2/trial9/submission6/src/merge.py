import torch
import torch.nn as nn
import numpy as np

def get_task_vectors(expert_states, progenitor_state):
    """
    Computes task vectors: tau_k = W_k - W_init
    """
    task_vectors = []
    for expert_state in expert_states:
        task_vector = {}
        for key in progenitor_state.keys():
            # Only compute task vectors for float/half weight/bias parameters
            if progenitor_state[key].is_floating_point():
                task_vector[key] = expert_state[key] - progenitor_state[key]
            else:
                task_vector[key] = progenitor_state[key].clone()
        task_vectors.append(task_vector)
    return task_vectors

def ties_merging(task_vectors, progenitor_state, reset_thresh=20):
    """
    TIES-Merging:
    1. Trim the bottom reset_thresh% task vector values per parameter tensor.
    2. Elect sign based on dominant sign.
    3. Resolve sign conflicts and average.
    """
    merged_task_vector = {}
    
    # We only merge floating point tensors
    for key in progenitor_state.keys():
        if not progenitor_state[key].is_floating_point():
            merged_task_vector[key] = progenitor_state[key].clone()
            continue
            
        tensors = [tv[key] for tv in task_vectors]
        stacked = torch.stack(tensors, dim=0) # [K, ...]
        
        # 1. Trim bottom reset_thresh% of values
        # Compute threshold per task vector
        trimmed_tensors = []
        for t in tensors:
            flat_t = t.flatten()
            k = int(len(flat_t) * (reset_thresh / 100.0))
            if k > 0:
                threshold = torch.topk(torch.abs(flat_t), k, largest=False).values[-1]
                mask = torch.abs(t) >= threshold
                trimmed_tensors.append(t * mask)
            else:
                trimmed_tensors.append(t.clone())
        
        trimmed_stacked = torch.stack(trimmed_tensors, dim=0) # [K, ...]
        
        # 2. Elect sign
        signs = torch.sign(trimmed_stacked)
        sum_signs = torch.sum(signs, dim=0) # [...]
        elected_sign = torch.sign(sum_signs)
        
        # 3. Discard sign-conflicting values and average
        # Keep values that match the elected sign
        same_sign_mask = (signs == elected_sign.unsqueeze(0)) & (signs != 0)
        # Average only the sign-consistent values
        sum_values = torch.sum(trimmed_stacked * same_sign_mask, dim=0)
        num_values = torch.sum(same_sign_mask.float(), dim=0)
        
        # Avoid division by zero
        merged_val = torch.where(num_values > 0, sum_values / num_values, torch.zeros_like(sum_values))
        merged_task_vector[key] = merged_val
        
    return merged_task_vector

def dare_merging(task_vectors, progenitor_state, drop_rate=0.2):
    """
    DARE-Merging:
    1. Randomly drop drop_rate% of coordinates (set to 0) and rescale by 1/(1 - drop_rate).
    2. Average the modified task vectors.
    """
    merged_task_vector = {}
    
    for key in progenitor_state.keys():
        if not progenitor_state[key].is_floating_point():
            merged_task_vector[key] = progenitor_state[key].clone()
            continue
            
        tensors = [tv[key] for tv in task_vectors]
        
        modified_tensors = []
        for t in tensors:
            # Create a dropout mask
            mask = (torch.rand_like(t) >= drop_rate).float()
            # Rescale remaining values
            rescaled_t = (t * mask) / (1.0 - drop_rate)
            modified_tensors.append(rescaled_t)
            
        stacked = torch.stack(modified_tensors, dim=0)
        merged_task_vector[key] = torch.mean(stacked, dim=0)
        
    return merged_task_vector

def weight_averaging(expert_states, progenitor_state):
    """
    Simple Weight Averaging of state dicts (backbone)
    """
    merged_state = {}
    for key in progenitor_state.keys():
        if progenitor_state[key].is_floating_point():
            tensors = [state[key] for state in expert_states]
            merged_state[key] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            merged_state[key] = progenitor_state[key].clone()
    return merged_state

def get_merged_task_vector(merge_method, task_vectors, progenitor_state, **kwargs):
    if merge_method == "ta":
        # Task Arithmetic: standard average of task vectors (scaling is done in calibration/evaluation)
        merged_task_vector = {}
        for key in progenitor_state.keys():
            if progenitor_state[key].is_floating_point():
                tensors = [tv[key] for tv in task_vectors]
                merged_task_vector[key] = torch.stack(tensors, dim=0).mean(dim=0)
            else:
                merged_task_vector[key] = progenitor_state[key].clone()
        return merged_task_vector
    elif merge_method == "ties":
        reset_thresh = kwargs.get("reset_thresh", 20)
        return ties_merging(task_vectors, progenitor_state, reset_thresh=reset_thresh)
    elif merge_method == "dare":
        drop_rate = kwargs.get("drop_rate", 0.2)
        return dare_merging(task_vectors, progenitor_state, drop_rate=drop_rate)
    else:
        raise ValueError(f"Unknown merge method: {merge_method}")

def calibrate_model(merged_task_vector, task_vectors, progenitor_state, calib_method, **kwargs):
    """
    Calibrates the merged task vector using various methods.
    Returns the calibrated weight state dict: W_cal = W_init + T_cal
    """
    calibrated_state = {}
    K = len(task_vectors)
    
    # We only calibrate floating point weights
    for key in progenitor_state.keys():
        if not progenitor_state[key].is_floating_point():
            calibrated_state[key] = progenitor_state[key].clone()
            continue
            
        t_merged = merged_task_vector[key]
        
        # If the tensor is 1D (like bias) or has fewer than 2 dimensions, fall back to U-IPR
        if t_merged.ndim < 2:
            if calib_method == "none":
                calibrated_state[key] = progenitor_state[key] + t_merged
            else:
                # Isotropic fallback
                norm_experts = torch.stack([torch.norm(tv[key]) for tv in task_vectors]).mean()
                norm_merged = torch.norm(t_merged) + 1e-8
                scale = (norm_experts / norm_merged).clamp(0.1, 10.0)
                calibrated_state[key] = progenitor_state[key] + scale * t_merged
            continue
            
        # 2D or 4D weight tensors
        if calib_method == "none":
            scale = kwargs.get("scale", 1.0)
            calibrated_state[key] = progenitor_state[key] + scale * t_merged
            
        elif calib_method == "u_ipr":
            # Isotropic Parameter Resonance (layer-wise)
            norm_experts = torch.stack([torch.norm(tv[key]) for tv in task_vectors]).mean()
            norm_merged = torch.norm(t_merged) + 1e-8
            scale = norm_experts / norm_merged
            scale = scale.clamp(0.1, 10.0)
            calibrated_state[key] = progenitor_state[key] + scale * t_merged
            
        elif calib_method == "hns":
            # Holographic Norm Scaling (channel-wise)
            C_out = t_merged.shape[0]
            calibrated_t = torch.zeros_like(t_merged)
            for c in range(C_out):
                norm_experts_c = torch.stack([torch.norm(tv[key][c]) for tv in task_vectors]).mean()
                norm_merged_c = torch.norm(t_merged[c]) + 1e-8
                scale_c = norm_experts_c / norm_merged_c
                scale_c = scale_c.clamp(0.1, 10.0)
                calibrated_t[c] = scale_c * t_merged[c]
            calibrated_state[key] = progenitor_state[key] + calibrated_t
            
        elif calib_method == "qr_ipr":
            # Quantization-Robust IPR (using Median and MAD to clamp channel-wise scales)
            C_out = t_merged.shape[0]
            scales = []
            for c in range(C_out):
                norm_experts_c = torch.stack([torch.norm(tv[key][c]) for tv in task_vectors]).mean()
                norm_merged_c = torch.norm(t_merged[c]) + 1e-8
                scales.append((norm_experts_c / norm_merged_c).item())
                
            scales = np.array(scales)
            median = np.median(scales)
            mad = np.median(np.abs(scales - median))
            mad = max(mad, 1e-4)
            gamma = kwargs.get("gamma", 2.0)
            L = max(0.1, median - gamma * mad)
            U = min(4.0, median + gamma * mad)
            
            clamped_scales = np.clip(scales, L, U)
            calibrated_t = torch.zeros_like(t_merged)
            for c in range(C_out):
                calibrated_t[c] = clamped_scales[c] * t_merged[c]
            calibrated_state[key] = progenitor_state[key] + calibrated_t
            
        elif calib_method == "wcpr":
            # Wasserstein-Calibrated Parameter Resonance (channel-by-channel)
            C_out = t_merged.shape[0]
            calibrated_t = torch.zeros_like(t_merged)
            
            for c in range(C_out):
                mc = t_merged[c].flatten()
                Ic = torch.argsort(mc)
                
                # Get sorted expert updates
                expert_sorted = []
                for tv in task_vectors:
                    expert_sorted.append(torch.sort(tv[key][c].flatten()).values)
                
                s_target = torch.stack(expert_sorted, dim=0).mean(dim=0)
                
                cflat = torch.zeros_like(mc)
                cflat[Ic] = s_target
                calibrated_t[c] = cflat.view_as(t_merged[c])
                
            calibrated_state[key] = progenitor_state[key] + calibrated_t
            
        elif calib_method == "qr_sc_wcpr":
            # OUR PROPOSED METHOD: Quantization-Robust Sparsity-Compensated WCPR
            C_out = t_merged.shape[0]
            calibrated_t = torch.zeros_like(t_merged)
            gamma = kwargs.get("gamma", 2.0)
            compensation_type = kwargs.get("compensation", "inverse") # "inverse" or "none"
            
            for c in range(C_out):
                mc = t_merged[c].flatten()
                
                # Zeros remain strictly zero (sparsity preservation)
                active_mask = mc != 0
                num_active = active_mask.sum().item()
                M = len(mc)
                
                if num_active == 0:
                    # All elements pruned, keep as zero
                    calibrated_t[c] = torch.zeros_like(t_merged[c])
                    continue
                    
                p_c = num_active / M
                
                # 1. Get sorted active merged parameters
                active_indices = torch.where(active_mask)[0]
                mc_active = mc[active_indices]
                Ic_active = torch.argsort(mc_active)
                
                # 2. Compute full-size target Wasserstein barycenter
                expert_sorted = []
                for tv in task_vectors:
                    expert_sorted.append(torch.sort(tv[key][c].flatten()).values)
                s_target_full = torch.stack(expert_sorted, dim=0).mean(dim=0)
                
                # 3. Sub-sample/interpolate the target barycenter to match num_active elements
                # Ensure correct quantile mapping
                indices = torch.linspace(0, M - 1, steps=num_active, device=t_merged.device).long()
                s_target_active = s_target_full[indices]
                
                # 4. Sparsity Compensation
                if compensation_type == "inverse":
                    # Scale by 1/sqrt(p_c) to restore expected variance
                    s_target_active = s_target_active / np.sqrt(p_c)
                elif compensation_type == "sqrt":
                    # Scale by sqrt(p_c)
                    s_target_active = s_target_active * np.sqrt(p_c)
                
                # 5. Quantization Robustness via Dynamic Clamping of effective scaling factors
                # Effective scaling factor per active coordinate: g_i = target / merged
                g = s_target_active / (mc_active[Ic_active] + 1e-8)
                g_np = g.cpu().numpy()
                
                median = np.median(g_np)
                mad = np.median(np.abs(g_np - median))
                mad = max(mad, 1e-4)
                
                L = max(0.1, median - gamma * mad)
                U = min(4.0, median + gamma * mad)
                
                g_clamped = g.clamp(L, U)
                
                # Reconstruct active calibrated parameters
                calibrated_active = g_clamped * mc_active[Ic_active]
                
                cflat = torch.zeros_like(mc)
                cflat[active_indices[Ic_active]] = calibrated_active
                calibrated_t[c] = cflat.view_as(t_merged[c])
                
            calibrated_state[key] = progenitor_state[key] + calibrated_t
            
        else:
            raise ValueError(f"Unknown calibration method: {calib_method}")
            
    return calibrated_state

def average_batchnorm_stats(expert_states, merged_state):
    """
    BatchNorm parameters and running statistics are physically averaged across experts.
    """
    for key in merged_state.keys():
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            tensors = [state[key] for state in expert_states]
            if tensors[0].is_floating_point():
                merged_state[key] = torch.stack(tensors, dim=0).mean(dim=0)
            else:
                merged_state[key] = tensors[0].clone()
    return merged_state
