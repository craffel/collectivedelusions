import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from tqdm import tqdm
import os

def get_task_vector(pretrained_model, finetuned_model):
    task_vector = {}
    pretrained_params = {n: p for n, p in pretrained_model.named_parameters()}
    finetuned_params = {n: p for n, p in finetuned_model.named_parameters()}
    
    for name in pretrained_params:
        task_vector[name] = finetuned_params[name] - pretrained_params[name]
    return task_vector

def estimate_kfac_v1_abs(task_vectors):
    kfac_stats = {}
    for i, tv in enumerate(task_vectors):
        kfac_stats[i] = {}
        for name, delta in tv.items():
            if len(delta.shape) == 2:
                d = delta.cpu().float()
                ztz = torch.matmul(d.t(), d)
                optp = torch.diag(torch.mean(torch.abs(d), dim=1))
                kfac_stats[i][name] = (ztz, optp)
    return kfac_stats

def estimate_kfac_v2_sq(task_vectors):
    kfac_stats = {}
    for i, tv in enumerate(task_vectors):
        kfac_stats[i] = {}
        for name, delta in tv.items():
            if len(delta.shape) == 2:
                d = delta.cpu().float()
                ztz = torch.matmul(d.t(), d)
                optp = torch.diag(torch.mean(d**2, dim=1))
                kfac_stats[i][name] = (ztz, optp)
    return kfac_stats

def merge_models_df_bfm_sq(pretrained_model, finetuned_models, iterations=50):
    task_vectors = [get_task_vector(pretrained_model, ft_model) for ft_model in finetuned_models]
    task_vectors = [{k: v.cpu().float() for k, v in tv.items()} for tv in task_vectors]
    print("Estimating K-FAC components (v2 - Sq)...")
    kfac_stats = estimate_kfac_v2_sq(task_vectors)
    pretrained_sd = {k: v.cpu().float() for k, v in pretrained_model.state_dict().items()}
    merged_state_dict = {}
    for name, param in tqdm(pretrained_model.named_parameters(), desc="Merging v2"):
        if len(param.shape) == 2:
            A_list = [kfac_stats[t][name][0] for t in range(len(finetuned_models))]
            B_list = [kfac_stats[t][name][1] for t in range(len(finetuned_models))]
            target_b = torch.zeros(param.shape)
            for t, ft_model in enumerate(finetuned_models):
                W_t = ft_model.state_dict()[name].cpu().float()
                target_b += torch.matmul(B_list[t], torch.matmul(W_t, A_list[t]))
            sum_delta = torch.stack([tv[name] for tv in task_vectors]).sum(dim=0)
            X_init = pretrained_sd[name] + (1.0 / len(finetuned_models)) * sum_delta
            merged_w = conjugate_gradient_kfac(A_list, B_list, target_b, X_init, iterations=iterations)
            merged_state_dict[name] = merged_w
        else:
            others = torch.stack([ft_model.state_dict()[name].cpu().float() for ft_model in finetuned_models])
            merged_state_dict[name] = others.mean(dim=0)
    return merged_state_dict

def conjugate_gradient_kfac(A_list, B_list, target_b, X_init, iterations=50, tol=1e-6):
    # Ensure all on same device (likely CPU)
    device = X_init.device
    def mat_vec_mul(X):
        res = torch.zeros_like(X)
        for A, B in zip(A_list, B_list):
            res += torch.matmul(B, torch.matmul(X, A))
        return res

    X = X_init.clone()
    R = target_b - mat_vec_mul(X)
    P = R.clone()
    rsold = torch.sum(R * R)

    for i in range(iterations):
        Ap = mat_vec_mul(P)
        alpha = rsold / (torch.sum(P * Ap) + 1e-10)
        X = X + alpha * P
        R = R - alpha * Ap
        rsnew = torch.sum(R * R)
        if torch.sqrt(rsnew) < tol:
            break
        P = R + (rsnew / rsold) * P
        rsold = rsnew
    return X

def merge_models_df_bfm(pretrained_model, finetuned_models, iterations=50):
    """
    Perform Data-Free Block-Diagonal Fisher Merging (v1 - Abs).
    Perform on CPU.
    """
    # Move models to CPU temporarily for merging if they are on GPU
    task_vectors = [get_task_vector(pretrained_model, ft_model) for ft_model in finetuned_models]
    # Free task vectors from GPU
    task_vectors = [{k: v.cpu().float() for k, v in tv.items()} for tv in task_vectors]
    
    print("Estimating K-FAC components (v1 - Abs)...")
    kfac_stats = estimate_kfac_v1_abs(task_vectors)
    
    pretrained_sd = {k: v.cpu().float() for k, v in pretrained_model.state_dict().items()}
    merged_state_dict = {}
    
    for name, param in tqdm(pretrained_model.named_parameters(), desc="Merging v1"):
        name_sd = name.replace('module.', '') # Handle potential DP wrapper
        if len(param.shape) == 2:
            A_list = [kfac_stats[t][name][0] for t in range(len(finetuned_models))]
            B_list = [kfac_stats[t][name][1] for t in range(len(finetuned_models))]
            
            target_b = torch.zeros(param.shape, dtype=torch.float32)
            for t, ft_model in enumerate(finetuned_models):
                W_t = ft_model.state_dict()[name].cpu().float()
                target_b += torch.matmul(B_list[t], torch.matmul(W_t, A_list[t]))
            
            sum_delta = torch.stack([tv[name] for tv in task_vectors]).sum(dim=0)
            X_init = pretrained_sd[name] + (1.0 / len(finetuned_models)) * sum_delta
            
            merged_w = conjugate_gradient_kfac(A_list, B_list, target_b, X_init, iterations=iterations)
            merged_state_dict[name] = merged_w
        else:
            others = torch.stack([ft_model.state_dict()[name].cpu().float() for ft_model in finetuned_models])
            merged_state_dict[name] = others.mean(dim=0)
            
    # Include missing buffers/parameters from pretrained if any
    for k, v in pretrained_sd.items():
        if k not in merged_state_dict:
            merged_state_dict[k] = v
            
    return merged_state_dict

def simple_average(models):
    avg_state_dict = {}
    keys = models[0].keys()
    for key in keys:
        try:
            avg_state_dict[key] = torch.stack([m[key].cpu().float() for m in models]).mean(dim=0)
        except KeyError:
            continue
    return avg_state_dict

def task_arithmetic(pretrained_state_dict, task_vectors, scaling=0.4):
    merged_state_dict = {}
    for key in pretrained_state_dict.keys():
        try:
            sum_delta = torch.stack([tv[key].cpu().float() for tv in task_vectors]).sum(dim=0)
            merged_state_dict[key] = pretrained_state_dict[key].cpu().float() + scaling * sum_delta
        except KeyError:
            merged_state_dict[key] = pretrained_state_dict[key].cpu().float()
    return merged_state_dict

def ties_merging(pretrained_state_dict, task_vectors, k=20, scaling=0.4):
    merged_state_dict = {}
    trimmed_task_vectors = []
    for tv in task_vectors:
        trimmed_tv = {}
        for key in pretrained_state_dict.keys():
            if key not in tv: continue
            delta = tv[key].cpu().float()
            if delta.dim() < 1:
                trimmed_tv[key] = delta
                continue
            flat_delta = delta.flatten()
            k_val = max(1, int(len(flat_delta) * (k / 100)))
            if len(flat_delta) > 0:
                threshold = torch.topk(torch.abs(flat_delta), k_val).values[-1]
                mask = (torch.abs(delta) >= threshold)
                trimmed_tv[key] = delta * mask
            else:
                trimmed_tv[key] = delta
        trimmed_task_vectors.append(trimmed_tv)
    
    for key in pretrained_state_dict.keys():
        try:
            deltas = torch.stack([ttv[key] for ttv in trimmed_task_vectors])
            total_mass = deltas.sum(dim=0)
            elected_sign = torch.sign(total_mass)
            mask = (torch.sign(deltas) == elected_sign)
            count = mask.sum(dim=0).float()
            merged_delta = (deltas * mask).sum(dim=0) / torch.clamp(count, min=1.0)
            merged_state_dict[key] = pretrained_state_dict[key].cpu().float() + scaling * merged_delta
        except KeyError:
            merged_state_dict[key] = pretrained_state_dict[key].cpu().float()
            
    return merged_state_dict

def actmat_merging(pretrained_model, finetuned_models):
    """
    Data-free RegMean (ACTMat).
    Equivalent to BFM with O'^T O' = I.
    Perform on CPU.
    """
    task_vectors = [get_task_vector(pretrained_model, ft_model) for ft_model in finetuned_models]
    task_vectors = [{k: v.cpu().float() for k, v in tv.items()} for tv in task_vectors]
    
    pretrained_sd = {k: v.cpu().float() for k, v in pretrained_model.state_dict().items()}
    merged_state_dict = {}
    
    for name, param in tqdm(pretrained_model.named_parameters(), desc="ACTMat Merging (CPU)"):
        if len(param.shape) == 2:
            # Z^T Z approx Delta^T Delta
            A_list = [torch.matmul(tv[name].t(), tv[name]) for tv in task_vectors]
            # O'^T O' = I
            B_list = [torch.eye(param.shape[0]) for _ in range(len(finetuned_models))]
            
            target_b = torch.zeros(param.shape, dtype=torch.float32)
            for t, ft_model in enumerate(finetuned_models):
                W_t = ft_model.state_dict()[name].cpu().float()
                target_b += torch.matmul(W_t, A_list[t])
            
            # Initial guess: simple average
            X_init = torch.stack([ft_model.state_dict()[name].cpu().float() for ft_model in finetuned_models]).mean(dim=0)
            
            merged_w = conjugate_gradient_kfac(A_list, B_list, target_b, X_init, iterations=20)
            merged_state_dict[name] = merged_w
        else:
            others = torch.stack([ft_model.state_dict()[name].cpu().float() for ft_model in finetuned_models])
            merged_state_dict[name] = others.mean(dim=0)
            
    # Include missing buffers/parameters from pretrained if any
    for k, v in pretrained_sd.items():
        if k not in merged_state_dict:
            merged_state_dict[k] = v
            
    return merged_state_dict

if __name__ == "__main__":
    # Test loading
    model_id = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading pretrained model {model_id}...")
    pretrained_model = CLIPModel.from_pretrained(model_id).to(device)
    
    # Task IDs (simplified for demo)
    tasks = ["mnist", "eurosat", "gtsrb"] # Subset for testing
    finetuned_models = []
    
    for task in tasks:
        ft_id = f"tanganke/clip-vit-base-patch32_{task}"
        print(f"Loading finetuned model {ft_id}...")
        try:
            ft_model = CLIPModel.from_pretrained(ft_id).to(device)
            finetuned_models.append(ft_model)
        except Exception as e:
            print(f"Failed to load {ft_id}: {e}")
    
    if len(finetuned_models) > 1:
        merged_sd = merge_models_df_bfm(pretrained_model, finetuned_models)
        torch.save(merged_sd, "merged_df_bfm_vision.pt")
        print("Merged model saved.")
