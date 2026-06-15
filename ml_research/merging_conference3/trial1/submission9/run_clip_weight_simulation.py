import time
import torch
import numpy as np
import open_clip

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading official OpenAI pretrained CLIP ViT-B/32 model...")
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)

# We will collect projection layers from CLIP's visual transformer:
# 1. attn.out_proj.weight (768, 768)
# 2. mlp.c_fc.weight (3072, 768)
# 3. mlp.c_proj.weight (768, 3072)
target_layers = []
for i in range(12):
    block = model.visual.transformer.resblocks[i]
    target_layers.append((f"Block {i} Attn Out Proj", block.attn.out_proj.weight))
    target_layers.append((f"Block {i} MLP c_fc", block.mlp.c_fc.weight))
    target_layers.append((f"Block {i} MLP c_proj", block.mlp.c_proj.weight))

print(f"Successfully extracted {len(target_layers)} real-world high-dimensional layers from CLIP visual encoder.")

K = 3     # 3 task experts
seeds = [101, 102, 103]

print(f"--- Running CLIP ViT-B/32 Real-Layer Weight Merging Simulation ---")
print(f"Evaluating across {len(seeds)} random seeds and {len(target_layers)} real CLIP layers...")

# Helper for Ties-Merging
def ties_merging_matrix(task_vectors, prune_ratio=0.4):
    stacked = torch.stack(task_vectors, dim=0)
    K, d1, d2 = stacked.shape
    
    # Step 1: Trim (keep top p% by magnitude)
    flat = stacked.view(K, -1)
    thresholds = torch.quantile(torch.abs(flat), prune_ratio, dim=1, keepdim=True)
    mask = torch.abs(flat) >= thresholds
    trimmed = flat * mask
    
    # Step 2: Elect Sign
    signs = torch.sign(trimmed)
    cumulative_pos = torch.sum(torch.where(signs > 0, trimmed, 0.0), dim=0)
    cumulative_neg = torch.sum(torch.where(signs < 0, torch.abs(trimmed), 0.0), dim=0)
    elected_sign = torch.where(cumulative_pos >= cumulative_neg, 1.0, -1.0)
    
    # Step 3: Disjoint Merge
    same_sign_mask = (torch.sign(trimmed) == elected_sign) | (trimmed == 0)
    masked_trimmed = trimmed * same_sign_mask
    
    active_count = torch.sum((masked_trimmed != 0).float(), dim=0)
    active_count = torch.clamp(active_count, min=1.0)
    merged_flat = torch.sum(masked_trimmed, dim=0) / active_count
    
    return merged_flat.view(d1, d2)

# Dictionary to store metrics
# We will compute metrics layer-by-layer and average them.
methods = ['ta_default', 'ties_default', 'svd', 'rms', 'pf_rms']
layer_metrics = {m: {'time': [], 'align': [], 'std_align': []} for m in methods}

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    for layer_name, W_pre_orig in target_layers:
        W_pre = W_pre_orig.detach().to(device).float()
        d1, d2 = W_pre.shape  # output_dim, input_dim
        
        # Simulate high-dimensional input features X (batch_size = 100, dimension = d2)
        X = torch.randn(100, d2, device=device)
        
        # Simulate task vectors under extreme scale mismatches on real weights
        # Task 1: Fine-tuned under small LR (scale = 0.1)
        # Task 2: Fine-tuned under medium LR (scale = 0.5)
        # Task 3: Fine-tuned under large LR (scale = 2.0)
        scales = [0.1, 0.5, 2.0]
        task_vectors = []
        
        for k in range(K):
            # Generate random update direction with the same shape as W_pre
            raw_update = torch.randn(d1, d2, device=device) / np.sqrt(d2)
            # Normalize to unit RMS and apply scale
            rms_raw = torch.sqrt((raw_update**2).mean() + 1e-8)
            tau = scales[k] * (raw_update / rms_raw)
            task_vectors.append(tau)
            
        # Helper to calculate alignment (cosine similarity of feature updates in activation space)
        def calculate_alignment(W_merged):
            alignments = []
            for k in range(K):
                # Task-specific expert feature update: Y_k = X * tau_k^T
                Y_target = torch.matmul(X, task_vectors[k].t())
                # Merged model's actual update: Y_merged = X * (W_merged - W_pre)^T
                Y_actual = torch.matmul(X, (W_merged - W_pre).t())
                
                # Compute cosine similarity
                num = torch.sum(Y_target * Y_actual)
                denom = torch.sqrt(torch.sum(Y_target**2) + 1e-8) * torch.sqrt(torch.sum(Y_actual**2) + 1e-8)
                alignments.append((num / denom).item())
            return alignments

        # 1. Task Arithmetic
        t0 = time.perf_counter()
        tau_ta = sum(task_vectors) / K
        W_ta = W_pre + tau_ta
        t_ta = (time.perf_counter() - t0) * 1000
        aligns_ta = calculate_alignment(W_ta)
        layer_metrics['ta_default']['time'].append(t_ta)
        layer_metrics['ta_default']['align'].append(np.mean(aligns_ta))
        layer_metrics['ta_default']['std_align'].append(np.std(aligns_ta))
        
        # 2. Ties-Merging
        t0 = time.perf_counter()
        tau_ties = ties_merging_matrix(task_vectors, prune_ratio=0.4)
        W_ties = W_pre + tau_ties
        t_ties = (time.perf_counter() - t0) * 1000
        aligns_ties = calculate_alignment(W_ties)
        layer_metrics['ties_default']['time'].append(t_ties)
        layer_metrics['ties_default']['align'].append(np.mean(aligns_ties))
        layer_metrics['ties_default']['std_align'].append(np.std(aligns_ties))

        # 3. SVD Isotropic Merging (SAIM-like)
        t0 = time.perf_counter()
        svd_tensors = []
        svd_scales = []
        for tau in task_vectors:
            # SVD is computed on d1 x d2 matrix
            U, S, V = torch.svd(tau)
            mean_s = torch.mean(S)
            if mean_s < 1e-8: mean_s = 1e-8
            svd_scales.append(mean_s.item())
            S_norm = S / mean_s
            recon = torch.matmul(U, torch.matmul(torch.diag(S_norm), V.t()))
            svd_tensors.append(recon)
            
        avg_scale = sum(svd_scales) / len(svd_scales)
        avg_direction = sum(svd_tensors) / len(svd_tensors)
        tau_svd = avg_scale * avg_direction
        W_svd = W_pre + tau_svd
        t_svd = (time.perf_counter() - t0) * 1000
        aligns_svd = calculate_alignment(W_svd)
        layer_metrics['svd']['time'].append(t_svd)
        layer_metrics['svd']['align'].append(np.mean(aligns_svd))
        layer_metrics['svd']['std_align'].append(np.std(aligns_svd))

        # 4. RMS-Scale (Ours, lambda=1.0)
        t0 = time.perf_counter()
        rms_vals = [torch.sqrt((t**2).mean() + 1e-8) for t in task_vectors]
        avg_rms = sum(rms_vals) / len(rms_vals)
        tau_rms = avg_rms * sum(t / r for t, r in zip(task_vectors, rms_vals)) / len(task_vectors)
        W_rms = W_pre + tau_rms
        t_rms = (time.perf_counter() - t0) * 1000
        aligns_rms = calculate_alignment(W_rms)
        layer_metrics['rms']['time'].append(t_rms)
        layer_metrics['rms']['align'].append(np.mean(aligns_rms))
        layer_metrics['rms']['std_align'].append(np.std(aligns_rms))

        # 5. Parameter-Free RMS-Scale (PF-RMS)
        t0 = time.perf_counter()
        rms_vals = [torch.sqrt((t**2).mean() + 1e-8) for t in task_vectors]
        avg_rms = sum(rms_vals) / len(rms_vals)
        norm_vectors = [t / r for t, r in zip(task_vectors, rms_vals)]
        avg_norm = sum(norm_vectors) / len(norm_vectors)
        alpha = torch.sqrt((avg_norm**2).mean() + 1e-8)
        tau_pf_rms = (avg_rms / alpha) * avg_norm
        W_pf_rms = W_pre + tau_pf_rms
        t_pf_rms = (time.perf_counter() - t0) * 1000
        aligns_pf_rms = calculate_alignment(W_pf_rms)
        layer_metrics['pf_rms']['time'].append(t_pf_rms)
        layer_metrics['pf_rms']['align'].append(np.mean(aligns_pf_rms))
        layer_metrics['pf_rms']['std_align'].append(np.std(aligns_pf_rms))

# Print summary report
print("\n" + "="*85)
print(f"{'Method':<25} | {'Mean Time per Layer (ms)':<25} | {'Avg Cosine Align (%)':<22} | {'Alignment Std (%)':<15}")
print("="*85)

summary_results = {}
for name in methods:
    t_mean = np.mean(layer_metrics[name]['time'])
    t_std = np.std(layer_metrics[name]['time'])
    align_mean = np.mean(layer_metrics[name]['align']) * 100
    align_std = np.std(layer_metrics[name]['align']) * 100
    std_align_mean = np.mean(layer_metrics[name]['std_align']) * 100
    
    print(f"{name:<25} | {t_mean:6.2f} ± {t_std:5.2f} ms       | {align_mean:5.2f} ± {align_std:4.2f}%       | {std_align_mean:5.2f}%")
    summary_results[name] = {
        'time_mean': t_mean, 'time_std': t_std,
        'align_mean': align_mean, 'align_std': align_std,
        'std_align_mean': std_align_mean
    }
print("="*85)

# Save result to json
import json
with open('clip_weight_simulation_results.json', 'w') as f:
    json.dump(summary_results, f, indent=2)
print("CLIP weight layer simulation complete. Saved results to clip_weight_simulation_results.json")
