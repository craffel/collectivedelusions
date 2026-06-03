import torch
import torch.nn as nn
import numpy as np
import math

def get_task_vectors(backbones, initial_backbone):
    task_vectors = []
    state_init = initial_backbone.state_dict()
    for bb in backbones:
        state_bb = bb.state_dict()
        tv = {}
        for key in state_init.keys():
            if state_init[key].dtype.is_floating_point:
                tv[key] = state_bb[key] - state_init[key]
            else:
                tv[key] = state_bb[key].clone()
        task_vectors.append(tv)
    return task_vectors

def weight_averaging(backbones):
    merged = {}
    for key in backbones[0].state_dict().keys():
        tensors = [bb.state_dict()[key] for bb in backbones]
        if tensors[0].dtype.is_floating_point:
            merged[key] = torch.mean(torch.stack(tensors), dim=0)
        else:
            merged[key] = tensors[0].clone()
    return merged

def task_arithmetic(backbones, initial_backbone, lam=0.5):
    merged = {}
    state_init = initial_backbone.state_dict()
    task_vectors = get_task_vectors(backbones, initial_backbone)
    
    for key in state_init.keys():
        if state_init[key].dtype.is_floating_point:
            tvs = [tv[key] for tv in task_vectors]
            merged_tv = torch.mean(torch.stack(tvs), dim=0)
            merged[key] = state_init[key] + lam * merged_tv
        else:
            merged[key] = state_init[key].clone()
    return merged

def ties_merging(backbones, initial_backbone, fraction=0.2):
    """
    TIES-Merging: Pruning bottom fraction updates, sign election, disagreement resolution.
    """
    merged = {}
    state_init = initial_backbone.state_dict()
    task_vectors = get_task_vectors(backbones, initial_backbone)
    
    for key in state_init.keys():
        tensor_init = state_init[key]
        if tensor_init.dtype.is_floating_point:
            tvs = [tv[key] for tv in task_vectors]
            
            # Step 1: Pruning (Trimming the bottom fraction by magnitude)
            pruned_tvs = []
            for tv in tvs:
                flat_tv = tv.view(-1)
                if flat_tv.numel() == 0:
                    pruned_tvs.append(tv.clone())
                    continue
                k = int(flat_tv.numel() * fraction)
                if k > 0:
                    threshold = torch.kthvalue(torch.abs(flat_tv), k).values
                    mask = torch.abs(tv) >= threshold
                    pruned_tv = tv * mask
                else:
                    pruned_tv = tv.clone()
                pruned_tvs.append(pruned_tv)
            
            # Step 2: Sign Election
            stacked_tvs = torch.stack(pruned_tvs)
            signs = torch.sign(stacked_tvs)
            sign_sum = torch.sum(signs, dim=0)
            winning_sign = torch.sign(sign_sum)
            
            # Step 3: Disagreement Resolution
            resolved_tvs = []
            for pruned_tv in pruned_tvs:
                # Keep values that align with winning sign, zero out others
                resolved_tv = pruned_tv * (torch.sign(pruned_tv) == winning_sign)
                resolved_tvs.append(resolved_tv)
            
            # Average sign-consistent parameters
            stacked_resolved = torch.stack(resolved_tvs)
            # Count non-zero contributions per coordinate to avoid dividing by K if some are zeroed
            non_zeros = torch.sum((stacked_resolved != 0).float(), dim=0)
            sum_resolved = torch.sum(stacked_resolved, dim=0)
            # Avoid division by zero
            mean_resolved = torch.where(non_zeros > 0, sum_resolved / non_zeros, torch.zeros_like(sum_resolved))
            
            merged[key] = tensor_init + mean_resolved
        else:
            merged[key] = tensor_init.clone()
    return merged

def dare_merging(backbones, initial_backbone, fraction=0.2):
    """
    DARE-Merging: Randomly dropping updates and rescaling.
    """
    merged = {}
    state_init = initial_backbone.state_dict()
    task_vectors = get_task_vectors(backbones, initial_backbone)
    
    for key in state_init.keys():
        tensor_init = state_init[key]
        if tensor_init.dtype.is_floating_point:
            tvs = [tv[key] for tv in task_vectors]
            rescaled_tvs = []
            for tv in tvs:
                mask = (torch.rand_like(tv) >= fraction).float()
                rescaled_tv = (tv * mask) / (1.0 - fraction)
                rescaled_tvs.append(rescaled_tv)
            
            merged_tv = torch.mean(torch.stack(rescaled_tvs), dim=0)
            merged[key] = tensor_init + merged_tv
        else:
            merged[key] = tensor_init.clone()
    return merged

def wcpr_merging(backbones, initial_backbone, mode='unified'):
    """
    Wasserstein-Calibrated Parameter Resonance (WCPR) channel-by-channel.
    """
    merged = {}
    state_init = initial_backbone.state_dict()
    task_vectors = get_task_vectors(backbones, initial_backbone)
    
    for key in state_init.keys():
        tensor_init = state_init[key]
        if not tensor_init.dtype.is_floating_point:
            merged[key] = tensor_init.clone()
            continue
            
        tvs = [tv[key] for tv in task_vectors]
        merged_tv = torch.mean(torch.stack(tvs), dim=0)
        
        # Apply Channel-Wise WCPR for weights with dimension >= 2
        if tensor_init.dim() >= 2:
            C_out = tensor_init.size(0)
            T_WCPR = torch.zeros_like(merged_tv)
            for c in range(C_out):
                m_c = merged_tv[c].flatten()
                if m_c.numel() == 0:
                    continue
                I_c = torch.argsort(m_c)
                
                s_ks = []
                for tv in tvs:
                    s_ks.append(torch.sort(tv[c].flatten())[0])
                
                s_target_c = torch.mean(torch.stack(s_ks), dim=0)
                
                c_flat = torch.zeros_like(m_c)
                c_flat[I_c] = s_target_c
                T_WCPR[c] = c_flat.view_as(merged_tv[c])
            merged[key] = tensor_init + T_WCPR
        else:
            # Fallback to standard U-IPR for biases/1D parameters
            sum_norm = sum(torch.norm(tv, p='fro') for tv in tvs) / len(tvs)
            merged_norm = torch.norm(merged_tv, p='fro')
            S = sum_norm / (merged_norm + 1e-8)
            S_clamped = torch.clamp(S, 0.1, 10.0)
            merged[key] = tensor_init + S_clamped * merged_tv
            
    return merged

def qr_ipr_merging(backbones, initial_backbone, gamma=2.0):
    """
    Quantization-Robust Parameter Resonance (QR-IPR) channel-by-channel.
    """
    merged = {}
    state_init = initial_backbone.state_dict()
    task_vectors = get_task_vectors(backbones, initial_backbone)
    
    for key in state_init.keys():
        tensor_init = state_init[key]
        if not tensor_init.dtype.is_floating_point:
            merged[key] = tensor_init.clone()
            continue
            
        tvs = [tv[key] for tv in task_vectors]
        merged_tv = torch.mean(torch.stack(tvs), dim=0)
        
        if tensor_init.dim() >= 2:
            C_out = tensor_init.size(0)
            scs = []
            for c in range(C_out):
                nm = torch.norm(merged_tv[c], p=2)
                ne = sum(torch.norm(tv[c], p=2) for tv in tvs) / len(tvs)
                sc = ne / (nm + 1e-8)
                scs.append(sc)
            scs = torch.stack(scs)
            
            # Robust Outlier Clipping
            median_val = torch.median(scs)
            mad_val = torch.median(torch.abs(scs - median_val))
            L = torch.max(torch.tensor(0.1, device=tensor_init.device), median_val - gamma * mad_val)
            U = torch.min(torch.tensor(4.0, device=tensor_init.device), median_val + gamma * mad_val)
            scs_robust = torch.clamp(scs, L, U)
            
            T_QR = torch.zeros_like(merged_tv)
            for c in range(C_out):
                T_QR[c] = scs_robust[c] * merged_tv[c]
            merged[key] = tensor_init + T_QR
        else:
            # Layer-wise robust scaling for 1D parameters
            nm = torch.norm(merged_tv, p='fro')
            ne = sum(torch.norm(tv, p='fro') for tv in tvs) / len(tvs)
            sc = ne / (nm + 1e-8)
            sc_robust = torch.clamp(sc, 0.1, 3.0)
            merged[key] = tensor_init + sc_robust * merged_tv
            
    return merged

def qr_sp_wcpr_merging(backbones, initial_backbone, sign_merger='ties', fraction=0.2, gamma=2.0, scale_compensation=True):
    """
    Quantization-Robust & Sparsity-Preserving Wasserstein-Calibrated Parameter Resonance (QR-SP-WCPR).
    """
    merged = {}
    state_init = initial_backbone.state_dict()
    task_vectors = get_task_vectors(backbones, initial_backbone)

    # Step 1: Compute baseline merged task vector (with sign conflict resolution and pruning)
    # This generates the sparsity mask
    if sign_merger == 'ties':
        # Re-use the resolved TIES-Merging update logic to get the sparsified task vector
        base_merged = ties_merging(backbones, initial_backbone, fraction=fraction)
    elif sign_merger == 'dare':
        base_merged = dare_merging(backbones, initial_backbone, fraction=fraction)
    else:
        # Vanilla task arithmetic without sparsification
        base_merged = task_arithmetic(backbones, initial_backbone, lam=1.0)

    for key in state_init.keys():
        tensor_init = state_init[key]
        if not tensor_init.dtype.is_floating_point:
            merged[key] = tensor_init.clone()
            continue

        tvs = [tv[key] for tv in task_vectors]
        # Use unpruned, dense task vector for accurate sorting and mapping (as in WCPR)
        merged_tv = torch.mean(torch.stack(tvs), dim=0)

        if tensor_init.dim() >= 2:
            C_out = tensor_init.size(0)
            T_calibrated = torch.zeros_like(merged_tv)

            for c in range(C_out):
                m_c = merged_tv[c].flatten()
                if m_c.numel() == 0:
                    continue

                # argsort of dense merged task vector
                I_c = torch.argsort(m_c)

                # Get the sorted expert updates for channel c
                s_ks = []
                for tv in tvs:
                    s_ks.append(torch.sort(tv[c].flatten())[0])

                s_target_c = torch.mean(torch.stack(s_ks), dim=0)

                # Outlier-suppressive robust clamping on the dense distribution
                # using robust statistics (Median & Standard Deviation) to prevent quantization noise
                median_c = torch.median(s_target_c)
                std_c = torch.std(s_target_c)
                s_target_c_clamped = torch.clamp(s_target_c, min=median_c - gamma * std_c, max=median_c + gamma * std_c)

                # Map back to coordinates
                c_flat = torch.zeros_like(m_c)
                c_flat[I_c] = s_target_c_clamped

                T_calibrated[c] = c_flat.view_as(merged_tv[c])

            # Now apply the sparsity mask from our baseline TIES/DARE model
            mask = (base_merged[key] - tensor_init != 0)
            calibrated_update = T_calibrated * mask

            # Sparsity Compensation if enabled
            if scale_compensation:
                p_c = mask.float().mean().item()
                calibrated_update = calibrated_update / math.sqrt(max(p_c, 1e-5))

            merged[key] = tensor_init + calibrated_update
        else:
            # For 1D parameters, keep them unscaled but masked
            mask = (base_merged[key] - tensor_init != 0)
            merged[key] = tensor_init + merged_tv * mask

    return merged
def quantize_tensor(tensor, num_bits=8, per_channel=False):
    """
    Symmetric Uniform Quantization to num_bits (e.g., 8-bit).
    """
    if num_bits is None:
        return tensor
    qmax = 2**(num_bits - 1) - 1
    if not per_channel or tensor.dim() < 2:
        max_val = torch.max(torch.abs(tensor))
        if max_val == 0:
            return tensor
        scale = max_val / qmax
        quantized = torch.clamp(torch.round(tensor / scale), -qmax, qmax) * scale
        return quantized
    else:
        quantized = tensor.clone()
        for c in range(tensor.size(0)):
            max_val = torch.max(torch.abs(tensor[c]))
            if max_val == 0:
                continue
            scale = max_val / qmax
            quantized[c] = torch.clamp(torch.round(tensor[c] / scale), -qmax, qmax) * scale
        return quantized

def apply_quantization_to_model(model, num_bits=8, per_channel=False):
    """
    Quantize all floating-point weights of the model.
    """
    state_dict = model.state_dict()
    quantized_state = {}
    for key, val in state_dict.items():
        if val.dtype.is_floating_point:
            quantized_state[key] = quantize_tensor(val, num_bits=num_bits, per_channel=per_channel)
        else:
            quantized_state[key] = val.clone()
    model.load_state_dict(quantized_state)
