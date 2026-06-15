import torch
import math

def trim_tensor(tensor, keep_ratio=0.5):
    flat = tensor.flatten()
    k = int(keep_ratio * flat.numel())
    if k == 0:
        return torch.zeros_like(tensor)
    threshold = torch.topk(flat.abs(), k).values[-1]
    return tensor * (tensor.abs() >= threshold)

@torch.no_grad()
def merge_models(theta_pre, theta_prev_merged, theta_expert, task_idx, method='isotropic', alpha=0.5, lambda_val=0.0):
    """
    Merges task expert model parameters with previous merged model parameters.
    
    Args:
        theta_pre (dict): State dict of the initial pre-trained model (theta_0).
        theta_prev_merged (dict): State dict of the previous merged model (theta_{t-1}).
        theta_expert (dict): State dict of the current task expert (theta_{T_t}).
        task_idx (int): Current task index t (1-indexed).
        method (str): Merging method ('isotropic', 'spectral_dampening', 'task_arithmetic').
        alpha (float): Scaling factor for the merged update.
        lambda_val (float): Balance coefficient between history and new task (default 0.0).
        
    Returns:
        dict: State dict of the merged model (theta_t).
    """
    merged_state_dict = {}
    
    for key in theta_pre.keys():
        w_pre = theta_pre[key]
        w_prev = theta_prev_merged[key]
        w_exp = theta_expert[key]
        
        # If parameters are integers (like non-trainable step counters), just copy from expert
        if not torch.is_floating_point(w_pre):
            merged_state_dict[key] = w_exp.clone()
            continue
            
        # Compute task vectors
        # delta_cum = theta_{t-1} - theta_0
        delta_cum = w_prev - w_pre
        # delta_T_t = theta_{T_t} - theta_{t-1}
        delta_T_t = w_exp - w_prev
        
        # Combined candidate update:
        # delta_com = (1 + lambda) * delta_cum + (1 - lambda) * delta_T_t
        delta_com = (1.0 + lambda_val) * delta_cum + (1.0 - lambda_val) * delta_T_t
        
        # Check dimensionality of the parameter tensor
        dims = delta_com.dim()
        
        if dims < 2 or method == 'task_arithmetic':
            # 1D or 0D parameters, or standard Task Arithmetic
            delta_merged = delta_com
            
        elif method == 'spectral_dampening':
            # Our proposed simplified baseline: scale by 1 / sqrt(t)
            gamma_t = 1.0 / math.sqrt(task_idx)
            delta_merged = gamma_t * delta_com
            
        elif method == 'norm_matching':
            # Our proposed norm-matching baseline: normalize Frobenius norm of combined update 
            # to match the average Frobenius norm of the input updates.
            # To avoid the t=1 zero-vector confounding artifact, we only average non-zero updates.
            norm_cum = torch.linalg.norm(delta_cum)
            norm_exp = torch.linalg.norm(delta_T_t)
            if norm_cum > 1e-8:
                avg_norm = 0.5 * (norm_cum + norm_exp)
            else:
                # At t=1, delta_cum is 0. The only non-zero input is the task expert update.
                # To prevent artificial update shrinkage, we match the norm of the candidate update delta_com.
                avg_norm = torch.linalg.norm(delta_com)
            
            norm_com = torch.linalg.norm(delta_com)
            if norm_com > 1e-8:
                delta_merged = delta_com * (avg_norm / norm_com)
            else:
                delta_merged = delta_com
            
        elif method == 'scale_calibrated':
            # Scale-Calibrated baseline: scale combined update to match the Frobenius norm of the current task expert directly.
            # This avoids any compounding scale shrinkage.
            norm_exp = torch.linalg.norm(delta_T_t)
            norm_com = torch.linalg.norm(delta_com)
            if norm_com > 1e-8:
                delta_merged = delta_com * (norm_exp / norm_com)
            else:
                delta_merged = delta_com
            
        elif method == 'ties_merging':
            # TIES-Merging sequential adaptation: Trim, Sign Election, Disjoint Merge
            keep_ratio = 0.5
            u1 = (1.0 + lambda_val) * delta_cum
            u2 = (1.0 - lambda_val) * delta_T_t
            
            u1_trimmed = trim_tensor(u1, keep_ratio=keep_ratio)
            u2_trimmed = trim_tensor(u2, keep_ratio=keep_ratio)
            
            total_sign = u1_trimmed.sign() + u2_trimmed.sign()
            elected_sign = total_sign.sign()
            
            u1_cons = u1_trimmed * (u1_trimmed.sign() == elected_sign)
            u2_cons = u2_trimmed * (u2_trimmed.sign() == elected_sign)
            
            num_non_zero = (u1_cons != 0).float() + (u2_cons != 0).float()
            delta_merged = (u1_cons + u2_cons) / torch.clamp(num_non_zero, min=1.0)
            
        elif method == 'dare':
            # DARE baseline: randomly sparsify combined update and scale remaining elements
            drop_rate = 0.5
            mask = (torch.rand_like(delta_com) >= drop_rate).float()
            delta_merged = delta_com * mask / (1.0 - drop_rate)
            
        elif method == 'isotropic':
            # SVD-based adaptive isotropic merging
            # If 4D (conv layers), reshape to 2D
            orig_shape = delta_com.shape
            if dims > 2:
                # Flatten to 2D: [out_dim, rest]
                delta_com_2d = delta_com.view(orig_shape[0], -1)
            else:
                delta_com_2d = delta_com
                
            try:
                # Run SVD
                # delta_com_2d = U * diag(S) * V^T
                U, S, V = torch.linalg.svd(delta_com_2d, full_matrices=False)
                
                # Compute mean of singular values
                mean_s = S.mean()
                
                # Interpolate singular values: S_hat = mean_s + (S - mean_s) * (1 / sqrt(t))
                gamma_t = 1.0 / math.sqrt(task_idx)
                S_hat = mean_s + (S - mean_s) * gamma_t
                
                # Reconstruct
                delta_merged_2d = U @ torch.diag(S_hat) @ V
                
                # Reshape back to original shape
                delta_merged = delta_merged_2d.view(orig_shape)
                
            except RuntimeError:
                # SVD did not converge, fallback to task arithmetic for this layer
                delta_merged = delta_com
        else:
            raise ValueError(f"Unknown merging method: {method}")
            
        # Synthesize final merged weights: theta_t = theta_0 + alpha * delta_merged
        w_merged = w_pre + alpha * delta_merged
        merged_state_dict[key] = w_merged
        
    return merged_state_dict
