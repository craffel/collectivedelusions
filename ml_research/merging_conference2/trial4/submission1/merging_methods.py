import torch
import copy

def merge_weight_averaging(base_weights, expert_weights_list):
    """
    Standard Weight Averaging (WA): Average of all expert state dicts.
    """
    merged_state_dict = copy.deepcopy(base_weights)
    for key in merged_state_dict.keys():
        if merged_state_dict[key].is_floating_point():
            tensors = [expert[key] for expert in expert_weights_list]
            merged_state_dict[key] = torch.stack(tensors, dim=0).mean(dim=0)
    return merged_state_dict

def merge_task_arithmetic(base_weights, expert_weights_list, scale=1.0):
    """
    Task Arithmetic (TA): W_merged = W_0 + scale * sum(W_k - W_0)
    """
    merged_state_dict = copy.deepcopy(base_weights)
    for key in merged_state_dict.keys():
        if merged_state_dict[key].is_floating_point():
            task_vectors = [expert[key] - base_weights[key] for expert in expert_weights_list]
            sum_task_vectors = torch.stack(task_vectors, dim=0).sum(dim=0)
            merged_state_dict[key] = base_weights[key] + scale * sum_task_vectors
    return merged_state_dict

def merge_ties(base_weights, expert_weights_list, keep_rate=0.2, scale=1.0):
    """
    TIES-Merging:
    1. Compute task vectors.
    2. Prune (keep top keep_rate by magnitude).
    3. Sign consensus.
    4. Average and scale.
    """
    merged_state_dict = copy.deepcopy(base_weights)
    for key in merged_state_dict.keys():
        if merged_state_dict[key].is_floating_point():
            # 1. Compute task vectors
            task_vectors = [expert[key] - base_weights[key] for expert in expert_weights_list]
            
            # 2. Prune top keep_rate by magnitude
            pruned_vectors = []
            for tv in task_vectors:
                if tv.numel() == 0:
                    pruned_vectors.append(tv)
                    continue
                flat_tv = tv.view(-1)
                k = int(flat_tv.numel() * keep_rate)
                if k > 0:
                    threshold = torch.topk(flat_tv.abs(), k).values[-1]
                    mask = tv.abs() >= threshold
                    pruned_vectors.append(tv * mask)
                else:
                    pruned_vectors.append(torch.zeros_like(tv))
            
            # 3. Sign consensus
            stacked = torch.stack(pruned_vectors, dim=0) # [K, ...]
            signs = stacked.sign()
            
            # Sum of magnitudes for positive and negative signs
            pos_mask = stacked > 0
            neg_mask = stacked < 0
            
            pos_sum = (stacked * pos_mask).sum(dim=0)
            neg_sum = (stacked * neg_mask).sum(dim=0).abs()
            
            # Consensus sign
            consensus_sign = torch.zeros_like(base_weights[key])
            consensus_sign[pos_sum > neg_sum] = 1.0
            consensus_sign[neg_sum > pos_sum] = -1.0
            
            # Disagreement masking & average of agreeing updates
            agree_mask = signs == consensus_sign.unsqueeze(0)
            agree_count = agree_mask.sum(dim=0).float()
            
            sum_agree = (stacked * agree_mask).sum(dim=0)
            
            # Avoid division by zero
            mean_agree = torch.zeros_like(sum_agree)
            nz_mask = agree_count > 0
            mean_agree[nz_mask] = sum_agree[nz_mask] / agree_count[nz_mask]
            
            # 4. Scale and apply
            current_scale = scale
            if isinstance(scale, dict):
                current_scale = scale.get('default', 1.0)
                for pattern, s in scale.items():
                    if pattern in key:
                        current_scale = s
                        break
            merged_state_dict[key] = base_weights[key] + current_scale * mean_agree
            
    return merged_state_dict

def merge_dare(base_weights, expert_weights_list, drop_rate=0.9, scale=1.0):
    """
    DARE-Task Arithmetic:
    1. Compute task vectors.
    2. Drop parameters randomly with probability drop_rate and scale by 1/(1-drop_rate).
    3. Sum and apply scale.
    """
    merged_state_dict = copy.deepcopy(base_weights)
    keep_prob = 1.0 - drop_rate
    for key in merged_state_dict.keys():
        if merged_state_dict[key].is_floating_point():
            task_vectors = [expert[key] - base_weights[key] for expert in expert_weights_list]
            
            dropped_vectors = []
            for tv in task_vectors:
                # Bernoulli mask
                mask = (torch.rand_like(tv) < keep_prob).float()
                # Scale remaining parameters by 1 / keep_prob
                dropped_tv = tv * mask / keep_prob
                dropped_vectors.append(dropped_tv)
                
            sum_vectors = torch.stack(dropped_vectors, dim=0).sum(dim=0)
            current_scale = scale
            if isinstance(scale, dict):
                current_scale = scale.get('default', 1.0)
                for pattern, s in scale.items():
                    if pattern in key:
                        current_scale = s
                        break
            merged_state_dict[key] = base_weights[key] + current_scale * sum_vectors
            
    return merged_state_dict

def merge_layerwise_scaling(base_weights, expert_weights_list, layer_scales=None):
    """
    Layer-wise Weight Scaling (LWS) for Task Arithmetic:
    Apply different scales for different layers/blocks to prevent activation explosion
    while counteracting representation collapse in deep layers.
    layer_scales is a dict mapping layer names/patterns to scale factors.
    """
    if layer_scales is None:
        layer_scales = {'layer1': 0.3, 'layer2': 0.3, 'layer3': 0.3, 'layer4': 0.3, 'default': 0.3}
        
    merged_state_dict = copy.deepcopy(base_weights)
    for key in merged_state_dict.keys():
        if merged_state_dict[key].is_floating_point():
            task_vectors = [expert[key] - base_weights[key] for expert in expert_weights_list]
            sum_task_vectors = torch.stack(task_vectors, dim=0).sum(dim=0)
            
            # Find the scale for this key
            scale = layer_scales.get('default', 0.3)
            for pattern, s in layer_scales.items():
                if pattern in key:
                    scale = s
                    break
                    
            merged_state_dict[key] = base_weights[key] + scale * sum_task_vectors
            
    return merged_state_dict

