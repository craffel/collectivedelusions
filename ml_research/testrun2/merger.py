import torch
import copy
from typing import List, Dict

def ties_purify(task_vectors: List[torch.Tensor], k: float = 0.2):
    """
    Apply TIES-style trimming and sign election to purify task vectors.
    k: fraction of parameters to keep (e.g., 0.2 for top 20%)
    """
    # 1. Trimming
    trimmed_vectors = []
    for tv in task_vectors:
        flat_tv = tv.view(-1)
        k_val = int(len(flat_tv) * k)
        if k_val == 0:
            trimmed_vectors.append(torch.zeros_like(tv))
            continue
        
        values, indices = torch.topk(torch.abs(flat_tv), k_val)
        mask = torch.zeros_like(flat_tv)
        mask[indices] = 1
        trimmed_vectors.append((flat_tv * mask).view(tv.shape))
    
    # 2. Elect Sign
    # Sum across tasks to find the dominant sign per parameter
    total_trimmed = torch.stack(trimmed_vectors).sum(dim=0)
    elected_signs = torch.sign(total_trimmed)
    
    # 3. Disjoint Merge Mask (Filter values that don't match elected sign)
    purified_vectors = []
    for tv in trimmed_vectors:
        # Keep only values that match the elected sign and are non-zero
        mask = (torch.sign(tv) == elected_signs) & (tv != 0)
        purified_vectors.append(tv * mask)
        
    return purified_vectors

def solve_cg(A, B, x0=None, iterations=100, tol=1e-5):
    """
    Solve AX = B for X using Conjugate Gradient.
    """
    device = A.device
    d_out, d_in = B.shape
    
    if x0 is None:
        X = torch.zeros((d_in, d_out), device=device)
    else:
        X = x0.t().contiguous() # (d_in, d_out)
        
    Target = B.t() # (d_in, d_out)
    
    R = Target - torch.matmul(A, X)
    P = R.clone()
    rsold = torch.sum(R * R, dim=0) # (d_out,)
    
    for i in range(iterations):
        AP = torch.matmul(A, P)
        alpha = rsold / (torch.sum(P * AP, dim=0) + 1e-10) # (d_out,)
        X = X + alpha.unsqueeze(0) * P
        R = R - alpha.unsqueeze(0) * AP
        rsnew = torch.sum(R * R, dim=0)
        
        if torch.max(rsnew) < tol:
            break
            
        beta = rsnew / (rsold + 1e-10)
        P = R + beta.unsqueeze(0) * P
        rsold = rsnew
        
    return X.t()

def sr_purify(task_vectors: List[torch.Tensor], k: float = 0.2):
    """
    Apply TIES-style trimming and sign election to purify task vectors.
    """
    # 1. Trimming
    trimmed_vectors = []
    for tv in task_vectors:
        flat_tv = tv.view(-1)
        k_val = int(len(flat_tv) * k)
        if k_val == 0:
            trimmed_vectors.append(torch.zeros_like(tv))
            continue
        
        values, indices = torch.topk(torch.abs(flat_tv), k_val)
        mask = torch.zeros_like(flat_tv)
        mask[indices] = 1
        trimmed_vectors.append((flat_tv * mask).view(tv.shape))
    
    # 2. Elect Sign
    total_trimmed = torch.stack(trimmed_vectors).sum(dim=0)
    elected_signs = torch.sign(total_trimmed)
    
    # 3. Disjoint Merge Mask
    purified_vectors = []
    for tv in trimmed_vectors:
        mask = (torch.sign(tv) == elected_signs) & (tv != 0)
        purified_vectors.append(tv * mask)
        
    return purified_vectors

def sr_actmat_merge(base_model_sd: Dict[str, torch.Tensor], 
                   expert_sds: List[Dict[str, torch.Tensor]], 
                   k: float = 0.2, 
                   use_cg: bool = True):
    """
    Sign-Resolved ACTMat Merging.
    """
    merged_sd = copy.deepcopy(base_model_sd)
    keys = base_model_sd.keys()
    
    for key in keys:
        if not key.endswith('.weight') or base_model_sd[key].dim() != 2:
            experts_params = [sd[key] for sd in expert_sds]
            merged_sd[key] = torch.stack(experts_params).mean(dim=0)
            continue
            
        w0 = base_model_sd[key]
        w_ts = [sd[key] for sd in expert_sds]
        deltas = [w - w0 for w in w_ts]
        
        # 1. Purify task vectors (TIES style)
        purified_deltas = sr_purify(deltas, k=k)
        
        # 2. Compute Proxy Covariances
        cs = [torch.matmul(d.t(), d) for d in purified_deltas]
        
        # 3. Formulate linear system
        sum_c = torch.stack(cs).sum(dim=0)
        sum_c += torch.eye(sum_c.shape[0], device=sum_c.device) * 1e-6
        sum_wc = torch.stack([torch.matmul(w, c) for w, c in zip(w_ts, cs)]).sum(dim=0)
        
        # 4. Solve
        if use_cg:
            ta_delta = torch.stack(purified_deltas).mean(dim=0)
            w_init = w0 + ta_delta
            merged_sd[key] = solve_cg(sum_c, sum_wc, x0=w_init)
        else:
            merged_sd[key] = torch.matmul(sum_wc, torch.pinverse(sum_c))
            
    return merged_sd

def asr_actmat_merge(base_model_sd: Dict[str, torch.Tensor], 
                    expert_sds: List[Dict[str, torch.Tensor]], 
                    k: float = 0.5, 
                    soft_lambda: float = 0.1,
                    use_cg: bool = True):
    """
    Adaptive Sign-Resolved ACTMat Merging.
    """
    merged_sd = copy.deepcopy(base_model_sd)
    keys = base_model_sd.keys()
    
    for key in keys:
        if not key.endswith('.weight') or base_model_sd[key].dim() != 2:
            experts_params = [sd[key] for sd in expert_sds]
            merged_sd[key] = torch.stack(experts_params).mean(dim=0)
            continue
            
        w0 = base_model_sd[key]
        w_ts = [sd[key] for sd in expert_sds]
        deltas = [w - w0 for w in w_ts]
        
        # 1. Purify task vectors (ASR style)
        purified_deltas = asr_purify(deltas, k=k, soft_lambda=soft_lambda)
        
        # 2. Compute Proxy Covariances
        cs = [torch.matmul(d.t(), d) for d in purified_deltas]
        
        # 3. Formulate linear system
        sum_c = torch.stack(cs).sum(dim=0)
        sum_c += torch.eye(sum_c.shape[0], device=sum_c.device) * 1e-6
        sum_wc = torch.stack([torch.matmul(w, c) for w, c in zip(w_ts, cs)]).sum(dim=0)
        
        # 4. Solve
        if use_cg:
            ta_delta = torch.stack(purified_deltas).mean(dim=0)
            w_init = w0 + ta_delta
            merged_sd[key] = solve_cg(sum_c, sum_wc, x0=w_init)
        else:
            merged_sd[key] = torch.matmul(sum_wc, torch.pinverse(sum_c))
            
    return merged_sd

def asr_purify(task_vectors: List[torch.Tensor], k: float = 0.5, soft_lambda: float = 0.1):
    """
    Adaptive Sign-Resolved purification.
    soft_lambda: scaling factor for updates that disagree with the elected sign.
    """
    # 1. Trimming (Hard trim to remove noise)
    trimmed_vectors = []
    for tv in task_vectors:
        flat_tv = tv.view(-1)
        k_val = int(len(flat_tv) * k)
        if k_val == 0:
            trimmed_vectors.append(torch.zeros_like(tv))
            continue
        
        values, indices = torch.topk(torch.abs(flat_tv), k_val)
        mask = torch.zeros_like(flat_tv)
        mask[indices] = 1
        trimmed_vectors.append((flat_tv * mask).view(tv.shape))
    
    # 2. Elect Sign
    total_trimmed = torch.stack(trimmed_vectors).sum(dim=0)
    elected_signs = torch.sign(total_trimmed)
    
    # 3. Soft Sign Resolution
    purified_vectors = []
    for tv in task_vectors: # Use original tv for soft resolution, or trimmed? Let's use trimmed.
        agree_mask = (torch.sign(tv) == elected_signs) | (elected_signs == 0)
        # Apply soft_lambda to disagreeing signs
        mask = torch.where(agree_mask, torch.ones_like(tv), torch.ones_like(tv) * soft_lambda)
        purified_vectors.append(tv * mask)
        
    return purified_vectors

def psr_actmat_merge(base_model_sd: Dict[str, torch.Tensor], 
                    expert_sds: List[Dict[str, torch.Tensor]], 
                    k: float = 0.5, 
                    use_cg: bool = True):
    """
    Partial Sign-Resolved ACTMat Merging.
    Purifies target weights but uses full task vectors for covariance estimation.
    """
    merged_sd = copy.deepcopy(base_model_sd)
    keys = base_model_sd.keys()
    
    for key in keys:
        if not key.endswith('.weight') or base_model_sd[key].dim() != 2:
            experts_params = [sd[key] for sd in expert_sds]
            merged_sd[key] = torch.stack(experts_params).mean(dim=0)
            continue
            
        w0 = base_model_sd[key]
        w_ts = [sd[key] for sd in expert_sds]
        deltas = [w - w0 for w in w_ts]
        
        # 1. Compute Proxy Covariances using FULL deltas
        cs = [torch.matmul(d.t(), d) for d in deltas]
        
        # 2. Purify deltas for the target weights
        purified_deltas = sr_purify(deltas, k=k)
        purified_w_ts = [w0 + d for d in purified_deltas]
        
        # 3. Formulate linear system
        sum_c = torch.stack(cs).sum(dim=0)
        sum_c += torch.eye(sum_c.shape[0], device=sum_c.device) * 1e-6
        sum_wc = torch.stack([torch.matmul(w, c) for w, c in zip(purified_w_ts, cs)]).sum(dim=0)
        
        # 4. Solve
        if use_cg:
            ta_delta = torch.stack(purified_deltas).mean(dim=0)
            w_init = w0 + ta_delta
            merged_sd[key] = solve_cg(sum_c, sum_wc, x0=w_init)
        else:
            merged_sd[key] = torch.matmul(sum_wc, torch.pinverse(sum_c))
            
    return merged_sd

def asr_actmat_merge(base_model_sd: Dict[str, torch.Tensor], 
                    expert_sds: List[Dict[str, torch.Tensor]], 
                    k: float = 0.5, 
                    soft_lambda: float = 0.1,
                    use_cg: bool = True):
    """
    Adaptive Sign-Resolved ACTMat Merging.
    """
    merged_sd = copy.deepcopy(base_model_sd)
    keys = base_model_sd.keys()
    
    for key in keys:
        if not key.endswith('.weight') or base_model_sd[key].dim() != 2:
            experts_params = [sd[key] for sd in expert_sds]
            merged_sd[key] = torch.stack(experts_params).mean(dim=0)
            continue
            
        w0 = base_model_sd[key]
        w_ts = [sd[key] for sd in expert_sds]
        deltas = [w - w0 for w in w_ts]
        
        # 1. Purify task vectors (ASR style)
        purified_deltas = asr_purify(deltas, k=k, soft_lambda=soft_lambda)
        
        # 2. Compute Proxy Covariances
        cs = [torch.matmul(d.t(), d) for d in purified_deltas]
        
        # 3. Formulate linear system
        sum_c = torch.stack(cs).sum(dim=0)
        sum_c += torch.eye(sum_c.shape[0], device=sum_c.device) * 1e-6
        sum_wc = torch.stack([torch.matmul(w, c) for w, c in zip(w_ts, cs)]).sum(dim=0)
        
        # 4. Solve
        if use_cg:
            ta_delta = torch.stack(purified_deltas).mean(dim=0)
            w_init = w0 + ta_delta
            merged_sd[key] = solve_cg(sum_c, sum_wc, x0=w_init)
        else:
            merged_sd[key] = torch.matmul(sum_wc, torch.pinverse(sum_c))
            
    return merged_sd

def ace_merging(base_model_sd: Dict[str, torch.Tensor], 
                expert_sds: List[Dict[str, torch.Tensor]], 
                gamma: float = 0.1,
                use_cg: bool = True):
    """
    ACE-Merging: Adaptive Covariance Estimation.
    gamma: weight for the collective structural prior.
    """
    merged_sd = copy.deepcopy(base_model_sd)
    keys = base_model_sd.keys()
    
    for key in keys:
        if not key.endswith('.weight') or base_model_sd[key].dim() != 2:
            experts_params = [sd[key] for sd in expert_sds]
            merged_sd[key] = torch.stack(experts_params).mean(dim=0)
            continue
            
        w0 = base_model_sd[key]
        w_ts = [sd[key] for sd in expert_sds]
        deltas = [w - w0 for w in w_ts]
        
        # 1. Compute and Normalize Covariances (ACN)
        cs = []
        for d in deltas:
            c = torch.matmul(d.t(), d)
            trace = torch.trace(c)
            cs.append(c / (trace + 1e-10))
            
        # 2. Collective Structural Prior (CSP)
        mean_c = torch.stack(cs).mean(dim=0)
        refined_cs = [(1 - gamma) * c + gamma * mean_c for c in cs]
        
        # 3. Formulate linear system
        sum_c = torch.stack(refined_cs).sum(dim=0)
        sum_c += torch.eye(sum_c.shape[0], device=sum_c.device) * 1e-6
        sum_wc = torch.stack([torch.matmul(w, c) for w, c in zip(w_ts, refined_cs)]).sum(dim=0)
        
        # 4. Solve
        if use_cg:
            # Initialize with Task Arithmetic or mean
            ta_delta = torch.stack(deltas).mean(dim=0)
            w_init = w0 + ta_delta
            merged_sd[key] = solve_cg(sum_c, sum_wc, x0=w_init)
        else:
            merged_sd[key] = torch.matmul(sum_wc, torch.pinverse(sum_c))
            
    return merged_sd

def sr_ace_merging(base_model_sd: Dict[str, torch.Tensor], 
                   expert_sds: List[Dict[str, torch.Tensor]], 
                   k: float = 0.5,
                   gamma: float = 0.1,
                   use_cg: bool = True):
    """
    Sign-Resolved ACE-Merging.
    """
    merged_sd = copy.deepcopy(base_model_sd)
    keys = base_model_sd.keys()
    
    for key in keys:
        if not key.endswith('.weight') or base_model_sd[key].dim() != 2:
            experts_params = [sd[key] for sd in expert_sds]
            merged_sd[key] = torch.stack(experts_params).mean(dim=0)
            continue
            
        w0 = base_model_sd[key]
        w_ts = [sd[key] for sd in expert_sds]
        deltas = [w - w0 for w in w_ts]
        
        # 1. Purify task vectors
        purified_deltas = sr_purify(deltas, k=k)
        
        # 2. Compute and Normalize Covariances (ACN)
        cs = []
        for d in purified_deltas:
            c = torch.matmul(d.t(), d)
            trace = torch.trace(c)
            cs.append(c / (trace + 1e-10))
            
        # 3. Collective Structural Prior (CSP)
        mean_c = torch.stack(cs).mean(dim=0)
        refined_cs = [(1 - gamma) * c + gamma * mean_c for c in cs]
        
        # 4. Formulate linear system
        sum_c = torch.stack(refined_cs).sum(dim=0)
        sum_c += torch.eye(sum_c.shape[0], device=sum_c.device) * 1e-6
        # Target weights also purified
        purified_w_ts = [w0 + d for d in purified_deltas]
        sum_wc = torch.stack([torch.matmul(w, c) for w, c in zip(purified_w_ts, refined_cs)]).sum(dim=0)
        
        # 5. Solve
        if use_cg:
            ta_delta = torch.stack(purified_deltas).mean(dim=0)
            w_init = w0 + ta_delta
            merged_sd[key] = solve_cg(sum_c, sum_wc, x0=w_init)
        else:
            merged_sd[key] = torch.matmul(sum_wc, torch.pinverse(sum_c))
            
    return merged_sd

def asr_ace_merging(base_model_sd: Dict[str, torch.Tensor], 
                    expert_sds: List[Dict[str, torch.Tensor]], 
                    k: float = 0.5,
                    soft_lambda: float = 0.1,
                    gamma: float = 0.1,
                    use_cg: bool = True):
    """
    Adaptive Sign-Resolved ACE-Merging.
    """
    merged_sd = copy.deepcopy(base_model_sd)
    keys = base_model_sd.keys()
    
    for key in keys:
        if not key.endswith('.weight') or base_model_sd[key].dim() != 2:
            experts_params = [sd[key] for sd in expert_sds]
            merged_sd[key] = torch.stack(experts_params).mean(dim=0)
            continue
            
        w0 = base_model_sd[key]
        w_ts = [sd[key] for sd in expert_sds]
        deltas = [w - w0 for w in w_ts]
        
        # 1. Purify task vectors (Soft Resolution)
        purified_deltas = asr_purify(deltas, k=k, soft_lambda=soft_lambda)
        
        # 2. Compute and Normalize Covariances (ACN)
        cs = []
        for d in purified_deltas:
            c = torch.matmul(d.t(), d)
            trace = torch.trace(c)
            cs.append(c / (trace + 1e-10))
            
        # 3. Collective Structural Prior (CSP)
        mean_c = torch.stack(cs).mean(dim=0)
        refined_cs = [(1 - gamma) * c + gamma * mean_c for c in cs]
        
        # 4. Formulate linear system
        sum_c = torch.stack(refined_cs).sum(dim=0)
        sum_c += torch.eye(sum_c.shape[0], device=sum_c.device) * 1e-6
        # Target weights also purified
        purified_w_ts = [w0 + d for d in purified_deltas]
        sum_wc = torch.stack([torch.matmul(w, c) for w, c in zip(purified_w_ts, refined_cs)]).sum(dim=0)
        
        # 5. Solve
        if use_cg:
            ta_delta = torch.stack(purified_deltas).mean(dim=0)
            w_init = w0 + ta_delta
            merged_sd[key] = solve_cg(sum_c, sum_wc, x0=w_init)
        else:
            merged_sd[key] = torch.matmul(sum_wc, torch.pinverse(sum_c))
            
    return merged_sd

def spectral_ace_merging(base_model_sd: Dict[str, torch.Tensor], 
                         expert_sds: List[Dict[str, torch.Tensor]], 
                         p: float = 1.0,
                         gamma: float = 0.1,
                         use_cg: bool = True):
    """
    Spectral ACE-Merging.
    p: spectral power. p > 1 sharpens, p < 1 flattens.
    """
    merged_sd = copy.deepcopy(base_model_sd)
    keys = base_model_sd.keys()
    
    for key in keys:
        if not key.endswith('.weight') or base_model_sd[key].dim() != 2:
            experts_params = [sd[key] for sd in expert_sds]
            merged_sd[key] = torch.stack(experts_params).mean(dim=0)
            continue
            
        w0 = base_model_sd[key]
        w_ts = [sd[key] for sd in expert_sds]
        deltas = [w - w0 for w in w_ts]
        
        # 1. Compute, Normalize and Rescale Eigenvalues
        cs = []
        device = w0.device
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for d in deltas:
            d_comp = d.to(compute_device)
            c = torch.matmul(d_comp.t(), d_comp)
            
            if p != 1.0:
                try:
                    # Use SVD for better stability than eigh on some platforms
                    U, S, Vh = torch.linalg.svd(c)
                    S = torch.clamp(S, min=1e-10)
                    S = S ** p
                    c = torch.matmul(U, torch.matmul(torch.diag(S), Vh))
                except RuntimeError:
                    # Fallback to identity or original if SVD fails
                    print(f"Warning: SVD failed for key {key}, using original covariance.")
            
            trace = torch.trace(c)
            c = c / (trace + 1e-10)
            cs.append(c.to(device))
            
        # 2. Collective Structural Prior (CSP)
        mean_c = torch.stack(cs).mean(dim=0)
        refined_cs = [(1 - gamma) * c + gamma * mean_c for c in cs]
        
        # 3. Formulate linear system
        sum_c = torch.stack(refined_cs).sum(dim=0)
        sum_c += torch.eye(sum_c.shape[0], device=sum_c.device) * 1e-6
        sum_wc = torch.stack([torch.matmul(w, c) for w, c in zip(w_ts, refined_cs)]).sum(dim=0)
        
        # 4. Solve
        if use_cg:
            ta_delta = torch.stack(deltas).mean(dim=0)
            w_init = w0 + ta_delta
            merged_sd[key] = solve_cg(sum_c, sum_wc, x0=w_init)
        else:
            merged_sd[key] = torch.matmul(sum_wc, torch.pinverse(sum_c))
            
    return merged_sd

def base_prior_ace_merging(base_model_sd: Dict[str, torch.Tensor], 
                          expert_sds: List[Dict[str, torch.Tensor]], 
                          alpha: float = 0.1,
                          beta: float = 0.1,
                          use_cg: bool = True):
    """
    ACE-Merging with Base Model Prior.
    alpha: weight for the average expert covariance.
    beta: weight for the base model weight covariance.
    """
    merged_sd = copy.deepcopy(base_model_sd)
    keys = base_model_sd.keys()
    
    for key in keys:
        if not key.endswith('.weight') or base_model_sd[key].dim() != 2:
            experts_params = [sd[key] for sd in expert_sds]
            merged_sd[key] = torch.stack(experts_params).mean(dim=0)
            continue
            
        w0 = base_model_sd[key]
        w_ts = [sd[key] for sd in expert_sds]
        deltas = [w - w0 for w in w_ts]
        
        # 1. Expert Covariances
        cs = []
        for d in deltas:
            c = torch.matmul(d.t(), d)
            trace = torch.trace(c)
            cs.append(c / (trace + 1e-10))
            
        # 2. Average Expert Covariance
        mean_c = torch.stack(cs).mean(dim=0)
        
        # 3. Base Model Prior
        c0 = torch.matmul(w0.t(), w0)
        trace0 = torch.trace(c0)
        c0_norm = c0 / (trace0 + 1e-10)
        
        # 4. Refine
        refined_cs = [(1 - alpha - beta) * c + alpha * mean_c + beta * c0_norm for c in cs]
        
        # 5. Formulate linear system
        sum_c = torch.stack(refined_cs).sum(dim=0)
        sum_c += torch.eye(sum_c.shape[0], device=sum_c.device) * 1e-6
        sum_wc = torch.stack([torch.matmul(w, c) for w, c in zip(w_ts, refined_cs)]).sum(dim=0)
        
        # 6. Solve
        if use_cg:
            ta_delta = torch.stack(deltas).mean(dim=0)
            w_init = w0 + ta_delta
            merged_sd[key] = solve_cg(sum_c, sum_wc, x0=w_init)
        else:
            merged_sd[key] = torch.matmul(sum_wc, torch.pinverse(sum_c))
            
    return merged_sd

def task_arithmetic_merge(base_model_sd, expert_sds, scaling_factor=0.4):
    merged_sd = copy.deepcopy(base_model_sd)
    for key in base_model_sd.keys():
        w0 = base_model_sd[key]
        deltas = [sd[key] - w0 for sd in expert_sds]
        merged_sd[key] = w0 + scaling_factor * torch.stack(deltas).sum(dim=0)
    return merged_sd

def adaptive_ace_merging(base_model_sd: Dict[str, torch.Tensor], 
                         expert_sds: List[Dict[str, torch.Tensor]], 
                         base_gamma: float = 0.05,
                         max_gamma: float = 0.3,
                         use_cg: bool = True):
    """
    Adaptive ACE-Merging.
    Adjusts gamma per layer based on task covariance disagreement.
    """
    merged_sd = copy.deepcopy(base_model_sd)
    keys = base_model_sd.keys()
    
    for key in keys:
        if not key.endswith('.weight') or base_model_sd[key].dim() != 2:
            experts_params = [sd[key] for sd in expert_sds]
            merged_sd[key] = torch.stack(experts_params).mean(dim=0)
            continue
            
        w0 = base_model_sd[key]
        w_ts = [sd[key] for sd in expert_sds]
        deltas = [w - w0 for w in w_ts]
        
        # 1. Expert Covariances
        cs = []
        for d in deltas:
            c = torch.matmul(d.t(), d)
            trace = torch.trace(c)
            cs.append(c / (trace + 1e-10))
            
        # 2. Compute Interference (Cosine Similarity between task covariances)
        num_tasks = len(cs)
        if num_tasks > 1:
            similarities = []
            for i in range(num_tasks):
                for j in range(i + 1, num_tasks):
                    # Frobenius inner product
                    sim = torch.sum(cs[i] * cs[j]) / (torch.norm(cs[i]) * torch.norm(cs[j]) + 1e-10)
                    similarities.append(sim)
            
            avg_sim = torch.stack(similarities).mean()
            interference = 1.0 - torch.clamp(avg_sim, 0, 1)
            # Map interference to [base_gamma, max_gamma]
            gamma = base_gamma + (max_gamma - base_gamma) * interference
        else:
            gamma = base_gamma
            
        # 3. Collective Structural Prior (CSP)
        mean_c = torch.stack(cs).mean(dim=0)
        refined_cs = [(1 - gamma) * c + gamma * mean_c for c in cs]
        
        # 4. Formulate linear system
        sum_c = torch.stack(refined_cs).sum(dim=0)
        sum_c += torch.eye(sum_c.shape[0], device=sum_c.device) * 1e-6
        sum_wc = torch.stack([torch.matmul(w, c) for w, c in zip(w_ts, refined_cs)]).sum(dim=0)
        
        # 5. Solve
        if use_cg:
            ta_delta = torch.stack(deltas).mean(dim=0)
            w_init = w0 + ta_delta
            merged_sd[key] = solve_cg(sum_c, sum_wc, x0=w_init)
        else:
            merged_sd[key] = torch.matmul(sum_wc, torch.pinverse(sum_c))
            
    return merged_sd

def ties_merge(base_model_sd, expert_sds, k=0.2, scaling_factor=1.0):
    merged_sd = copy.deepcopy(base_model_sd)
    for key in base_model_sd.keys():
        if not key.endswith('.weight'):
            experts_params = [sd[key] for sd in expert_sds]
            merged_sd[key] = torch.stack(experts_params).mean(dim=0)
            continue

        w0 = base_model_sd[key]
        deltas = [sd[key] - w0 for sd in expert_sds]
        
        # 1. Trim
        trimmed_deltas = []
        for d in deltas:
            flat_d = d.view(-1)
            k_val = int(len(flat_d) * k)
            if k_val == 0:
                trimmed_deltas.append(torch.zeros_like(d))
                continue
            v, i = torch.topk(torch.abs(flat_d), k_val)
            mask = torch.zeros_like(flat_d)
            mask[i] = 1
            trimmed_deltas.append((flat_d * mask).view(d.shape))
            
        # 2. Elect Sign
        total_trimmed = torch.stack(trimmed_deltas).sum(dim=0)
        elected_signs = torch.sign(total_trimmed)
        
        # 3. Disjoint Merge
        sum_delta = torch.zeros_like(w0)
        counts = torch.zeros_like(w0)
        for td in trimmed_deltas:
            mask = (torch.sign(td) == elected_signs) & (td != 0)
            sum_delta += td * mask
            counts += mask.float()
            
        merged_sd[key] = w0 + scaling_factor * sum_delta / (counts + 1e-10)
    return merged_sd

def actmat_merge(base_model_sd, expert_sds):
    """Standard ACTMat (no purification)"""
    merged_sd = copy.deepcopy(base_model_sd)
    for key in base_model_sd.keys():
        if not key.endswith('.weight') or base_model_sd[key].dim() != 2:
            experts_params = [sd[key] for sd in expert_sds]
            merged_sd[key] = torch.stack(experts_params).mean(dim=0)
            continue
            
        w0 = base_model_sd[key]
        w_ts = [sd[key] for sd in expert_sds]
        deltas = [w - w0 for w in w_ts]
        
        cs = [torch.matmul(d.t(), d) for d in deltas]
        sum_c = torch.stack(cs).sum(dim=0)
        sum_c += torch.eye(sum_c.shape[0], device=sum_c.device) * 1e-6
        sum_wc = torch.stack([torch.matmul(w, c) for w, c in zip(w_ts, cs)]).sum(dim=0)
        
        merged_sd[key] = torch.matmul(sum_wc, torch.pinverse(sum_c))
        
    return merged_sd
