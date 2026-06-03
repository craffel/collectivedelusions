import torch
import copy

def merge_weight_averaging(model_A_state, model_B_state, alpha=0.5):
    """
    Standard Weight Averaging (WA): W_merged = (1 - alpha) * W_A + alpha * W_B
    """
    merged_state = {}
    for key in model_A_state.keys():
        if key not in model_B_state:
            merged_state[key] = copy.deepcopy(model_A_state[key])
            continue
        
        # Only merge floating point tensors
        if torch.is_tensor(model_A_state[key]) and torch.is_floating_point(model_A_state[key]):
            merged_state[key] = (1.0 - alpha) * model_A_state[key] + alpha * model_B_state[key]
        else:
            # For non-tensor or integer buffers, copy from model_A
            merged_state[key] = copy.deepcopy(model_A_state[key])
            
    return merged_state

def merge_task_arithmetic(base_model_state, model_A_state, model_B_state, lambda_val=0.5):
    """
    Task Arithmetic (TA): W_merged = W_base + lambda * (tv_A + tv_B)
    where tv_i = W_i - W_base
    """
    merged_state = {}
    for key in base_model_state.keys():
        if key not in model_A_state or key not in model_B_state:
            merged_state[key] = copy.deepcopy(base_model_state[key])
            continue
            
        if torch.is_tensor(base_model_state[key]) and torch.is_floating_point(base_model_state[key]):
            tv_A = model_A_state[key] - base_model_state[key]
            tv_B = model_B_state[key] - base_model_state[key]
            merged_state[key] = base_model_state[key] + lambda_val * (tv_A + tv_B)
        else:
            merged_state[key] = copy.deepcopy(base_model_state[key])
            
    return merged_state

def merge_ties(base_model_state, model_A_state, model_B_state, trim_percent=0.20, lambda_val=0.5):
    """
    TIES-Merging:
    1. Resolve Task Vectors: tv_A = W_A - W_base, tv_B = W_B - W_base
    2. Trim: Keep top (1 - trim_percent) values by absolute magnitude.
    3. Sign Resolution: Choose majority sign and zero out non-matching signs.
    4. Disjoint Merge: Average non-zero values.
    5. Add back: W_merged = W_base + lambda * tv_merged
    """
    merged_state = {}
    for key in base_model_state.keys():
        if key not in model_A_state or key not in model_B_state:
            merged_state[key] = copy.deepcopy(base_model_state[key])
            continue
            
        if torch.is_tensor(base_model_state[key]) and torch.is_floating_point(base_model_state[key]):
            tv_A = model_A_state[key] - base_model_state[key]
            tv_B = model_B_state[key] - base_model_state[key]
            
            # Pack task vectors
            tvs = torch.stack([tv_A, tv_B], dim=0) # [2, ...]
            
            # 1. Trimming
            trimmed_tvs = []
            for i in range(2):
                tv = tvs[i]
                abs_tv = torch.abs(tv)
                threshold = torch.quantile(abs_tv.flatten(), trim_percent)
                mask = abs_tv >= threshold
                trimmed_tvs.append(tv * mask)
            trimmed_tvs = torch.stack(trimmed_tvs, dim=0)
            
            # 2. Sign Resolution
            signs = torch.sign(trimmed_tvs) # [2, ...]
            # Sum signs to find majority sign
            sign_sum = torch.sum(signs, dim=0) # [...]
            majority_sign = torch.sign(sign_sum) # [...]
            # If sign_sum is 0, default to positive majority sign
            majority_sign[majority_sign == 0] = 1.0
            
            # Filter by majority sign: keep only values whose sign matches majority sign
            # and are non-zero.
            filtered_tvs = []
            for i in range(2):
                tv = trimmed_tvs[i]
                mask = (torch.sign(tv) == majority_sign) & (tv != 0)
                filtered_tvs.append(tv * mask)
            filtered_tvs = torch.stack(filtered_tvs, dim=0)
            
            # 3. Disjoint Merge (Average the non-zero values)
            non_zero_counts = torch.sum((filtered_tvs != 0).float(), dim=0)
            tv_sum = torch.sum(filtered_tvs, dim=0)
            
            # Avoid division by zero
            tv_merged = torch.where(non_zero_counts > 0, tv_sum / torch.clamp(non_zero_counts, min=1.0), torch.zeros_like(tv_sum))
            
            merged_state[key] = base_model_state[key] + lambda_val * tv_merged
        else:
            merged_state[key] = copy.deepcopy(base_model_state[key])
            
    return merged_state

def merge_dare(base_model_state, model_A_state, model_B_state, drop_prob=0.5, lambda_val=0.5):
    """
    DARE-Merging:
    1. tv_A = W_A - W_base, tv_B = W_B - W_base
    2. Randomly drop values with probability drop_prob and rescale by 1/(1-drop_prob)
    3. Average and add back: W_merged = W_base + lambda * tv_merged
    """
    merged_state = {}
    for key in base_model_state.keys():
        if key not in model_A_state or key not in model_B_state:
            merged_state[key] = copy.deepcopy(base_model_state[key])
            continue
            
        if torch.is_tensor(base_model_state[key]) and torch.is_floating_point(base_model_state[key]):
            tv_A = model_A_state[key] - base_model_state[key]
            tv_B = model_B_state[key] - base_model_state[key]
            
            # Create drop masks
            mask_A = (torch.rand_like(tv_A) > drop_prob).float()
            mask_B = (torch.rand_like(tv_B) > drop_prob).float()
            
            # Rescale
            scale = 1.0 / (1.0 - drop_prob)
            tv_A_scaled = tv_A * mask_A * scale
            tv_B_scaled = tv_B * mask_B * scale
            
            # Average rescaled task vectors
            tv_merged = 0.5 * (tv_A_scaled + tv_B_scaled)
            
            merged_state[key] = base_model_state[key] + lambda_val * tv_merged
        else:
            merged_state[key] = copy.deepcopy(base_model_state[key])
            
    return merged_state


def merge_multi_weight_averaging(model_states, weights=None):
    """
    General Multi-Task Weight Averaging (WA): W_merged = sum(w_k * W_k)
    """
    if weights is None:
        weights = [1.0 / len(model_states)] * len(model_states)
    
    merged_state = {}
    first_state = model_states[0]
    for key in first_state.keys():
        if not all(key in state for state in model_states):
            merged_state[key] = copy.deepcopy(first_state[key])
            continue
            
        if torch.is_tensor(first_state[key]) and torch.is_floating_point(first_state[key]):
            merged_val = torch.zeros_like(first_state[key])
            for state, w in zip(model_states, weights):
                merged_val += w * state[key]
            merged_state[key] = merged_val
        else:
            merged_state[key] = copy.deepcopy(first_state[key])
            
    return merged_state


def merge_multi_task_arithmetic(base_model_state, model_states, lambda_val=0.5):
    """
    General Multi-Task Task Arithmetic (TA): W_merged = W_base + lambda * sum(tv_k)
    where tv_k = W_k - W_base
    """
    merged_state = {}
    for key in base_model_state.keys():
        if not all(key in state for state in model_states):
            merged_state[key] = copy.deepcopy(base_model_state[key])
            continue
            
        if torch.is_tensor(base_model_state[key]) and torch.is_floating_point(base_model_state[key]):
            tv_sum = torch.zeros_like(base_model_state[key])
            for state in model_states:
                tv_sum += (state[key] - base_model_state[key])
            merged_state[key] = base_model_state[key] + lambda_val * tv_sum
        else:
            merged_state[key] = copy.deepcopy(base_model_state[key])
            
    return merged_state


def merge_multi_ties(base_model_state, model_states, trim_percent=0.20, lambda_val=0.5):
    """
    General Multi-Task TIES-Merging:
    1. Resolve Task Vectors: tv_k = W_k - W_base
    2. Trim: Keep top (1 - trim_percent) values by absolute magnitude.
    3. Sign Resolution: Choose majority sign and zero out non-matching signs.
    4. Disjoint Merge: Average non-zero values.
    5. Add back: W_merged = W_base + lambda * tv_merged
    """
    merged_state = {}
    num_models = len(model_states)
    for key in base_model_state.keys():
        if not all(key in state for state in model_states):
            merged_state[key] = copy.deepcopy(base_model_state[key])
            continue
            
        if torch.is_tensor(base_model_state[key]) and torch.is_floating_point(base_model_state[key]):
            trimmed_tvs = []
            for state in model_states:
                tv = state[key] - base_model_state[key]
                abs_tv = torch.abs(tv)
                if abs_tv.numel() > 0:
                    threshold = torch.quantile(abs_tv.flatten().float(), trim_percent)
                    mask = abs_tv >= threshold
                    trimmed_tvs.append(tv * mask)
                else:
                    trimmed_tvs.append(tv)
                    
            trimmed_tvs = torch.stack(trimmed_tvs, dim=0)
            
            # Sign resolution
            signs = torch.sign(trimmed_tvs)
            sign_sum = torch.sum(signs, dim=0)
            majority_sign = torch.sign(sign_sum)
            majority_sign[majority_sign == 0] = 1.0
            
            # Filter by majority sign
            filtered_tvs = []
            for i in range(num_models):
                tv = trimmed_tvs[i]
                mask = (torch.sign(tv) == majority_sign) & (tv != 0)
                filtered_tvs.append(tv * mask)
            filtered_tvs = torch.stack(filtered_tvs, dim=0)
            
            # Disjoint Merge
            non_zero_counts = torch.sum((filtered_tvs != 0).float(), dim=0)
            tv_sum = torch.sum(filtered_tvs, dim=0)
            
            tv_merged = torch.where(non_zero_counts > 0, tv_sum / torch.clamp(non_zero_counts, min=1.0), torch.zeros_like(tv_sum))
            
            merged_state[key] = base_model_state[key] + lambda_val * tv_merged
        else:
            merged_state[key] = copy.deepcopy(base_model_state[key])
            
    return merged_state


def merge_multi_dare(base_model_state, model_states, drop_prob=0.5, lambda_val=0.5):
    """
    General Multi-Task DARE-Merging:
    1. tv_k = W_k - W_base
    2. Randomly drop values with probability drop_prob and rescale by 1/(1-drop_prob)
    3. Average and add back: W_merged = W_base + lambda * tv_merged
    """
    merged_state = {}
    num_models = len(model_states)
    for key in base_model_state.keys():
        if not all(key in state for state in model_states):
            merged_state[key] = copy.deepcopy(base_model_state[key])
            continue
            
        if torch.is_tensor(base_model_state[key]) and torch.is_floating_point(base_model_state[key]):
            tv_scaled_sum = torch.zeros_like(base_model_state[key])
            scale = 1.0 / (1.0 - drop_prob)
            for state in model_states:
                tv = state[key] - base_model_state[key]
                mask = (torch.rand_like(tv) > drop_prob).float()
                tv_scaled_sum += tv * mask * scale
                
            tv_merged = tv_scaled_sum / num_models
            merged_state[key] = base_model_state[key] + lambda_val * tv_merged
        else:
            merged_state[key] = copy.deepcopy(base_model_state[key])
            
    return merged_state
