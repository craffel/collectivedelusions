import torch
import torch.nn as nn
import numpy as np
from simulate_sandbox import load_digits, set_seed, compute_accuracy, run_gravimerge

def run_temporal_eval():
    D = 192
    K = 4
    num_seeds = 5
    
    digits = load_digits()
    X = digits.data
    y = digits.target
    task_labels = [0, 1, 2, 3]
    
    print("Running true temporal streaming experiment across 5 seeds...")
    print("Task stream consists of block-wise non-stationary sequence (50 samples per task sequentially).")
    print("-" * 75)
    
    # We will compare EMA, ChemMerge, and GraviMerge with temporal state carryover
    # and measure accuracy and cross-query jitter (jitter between successive queries).
    
    results = {
        'SABLE (Stateless)': {'accuracy': [], 'cross_query_jitter': [], 'layer_jitter': []},
        'EMA (Temporal)': {'accuracy': [], 'cross_query_jitter': [], 'layer_jitter': []},
        'ChemMerge (Temporal)': {'accuracy': [], 'cross_query_jitter': [], 'layer_jitter': []},
        'GraviMerge (Temporal)': {'accuracy': [], 'cross_query_jitter': [], 'layer_jitter': []}
    }
    
    for seed in range(num_seeds):
        curr_seed = 42 + seed
        set_seed(curr_seed)
        
        projection_matrix = torch.randn(64, D)
        projection_matrix = projection_matrix / torch.norm(projection_matrix, dim=0, keepdim=True)
        
        task_samples = []
        for k in range(K):
            indices = np.where(y == task_labels[k])[0]
            indices = indices[:114] # 64 calibration + 50 test
            data_k = torch.tensor(X[indices], dtype=torch.float32)
            projected_k = data_k @ projection_matrix
            task_samples.append(projected_k)
            
        centroids = torch.zeros(K, D)
        for k in range(K):
            mean_h3 = torch.mean(task_samples[k][:64], dim=0)
            centroids[k] = mean_h3 / torch.norm(mean_h3)
            
        # Build non-stationary task stream (50 samples per task, sequentially)
        stream_samples_list = []
        stream_true_tasks_list = []
        for k in range(K):
            stream_samples_list.append(task_samples[k][64:])
            stream_true_tasks_list.append(torch.full((50,), k, dtype=torch.long))
            
        stream_samples = torch.cat(stream_samples_list, dim=0) # shape: (200, D)
        stream_true_tasks = torch.cat(stream_true_tasks_list, dim=0) # shape: (200,)
        
        N = stream_samples.shape[0]
        
        # --- 1. SABLE (Stateless) ---
        h = stream_samples.clone()
        sable_alphas = []
        for t in range(N):
            sample_alphas = []
            h_sample = h[t:t+1]
            for l in range(4, 15):
                h_norm = h_sample / torch.norm(h_sample, dim=1, keepdim=True)
                cos_sim = h_norm @ centroids.t()
                alpha = torch.softmax(cos_sim / 0.05, dim=1)
                sample_alphas.append(alpha.squeeze(0))
                
                blended = alpha @ centroids
                h_sample = h_sample + 0.3 * (blended - h_sample)
                h_sample = h_sample / torch.norm(h_sample, dim=1, keepdim=True)
            h[t] = h_sample
            sable_alphas.append(torch.stack(sample_alphas)) # shape: (11, K)
        sable_alphas = torch.stack(sable_alphas) # shape: (N, 11, K)
        
        # Calculate accuracy and jitters
        acc_sable = compute_accuracy(h, centroids, stream_true_tasks)
        lj_sable = torch.mean(torch.abs(sable_alphas[:, 1:, :] - sable_alphas[:, :-1, :])).item()
        cj_sable = torch.mean(torch.abs(sable_alphas[1:, 0, :] - sable_alphas[:-1, -1, :])).item()
        
        results['SABLE (Stateless)']['accuracy'].append(acc_sable)
        results['SABLE (Stateless)']['layer_jitter'].append(lj_sable)
        results['SABLE (Stateless)']['cross_query_jitter'].append(cj_sable)
        
        # --- 2. EMA (Temporal Persistence) ---
        h = stream_samples.clone()
        ema_alphas = []
        # Carry-over state for EMA
        state_ema = torch.full((1, K), 1.0 / K)
        for t in range(N):
            sample_alphas = []
            h_sample = h[t:t+1]
            for l in range(4, 15):
                h_norm = h_sample / torch.norm(h_sample, dim=1, keepdim=True)
                cos_sim = h_norm @ centroids.t()
                alpha_inst = torch.softmax(cos_sim / 0.05, dim=1)
                
                state_ema = 0.1 * alpha_inst + 0.9 * state_ema
                sample_alphas.append(state_ema.squeeze(0))
                
                blended = state_ema @ centroids
                h_sample = h_sample + 0.3 * (blended - h_sample)
                h_sample = h_sample / torch.norm(h_sample, dim=1, keepdim=True)
            h[t] = h_sample
            ema_alphas.append(torch.stack(sample_alphas))
        ema_alphas = torch.stack(ema_alphas)
        
        acc_ema = compute_accuracy(h, centroids, stream_true_tasks)
        lj_ema = torch.mean(torch.abs(ema_alphas[:, 1:, :] - ema_alphas[:, :-1, :])).item()
        cj_ema = torch.mean(torch.abs(ema_alphas[1:, 0, :] - ema_alphas[:-1, -1, :])).item()
        
        results['EMA (Temporal)']['accuracy'].append(acc_ema)
        results['EMA (Temporal)']['layer_jitter'].append(lj_ema)
        results['EMA (Temporal)']['cross_query_jitter'].append(cj_ema)
        
        # --- 3. ChemMerge (Temporal Persistence) ---
        h = stream_samples.clone()
        chem_alphas = []
        # Carry-over state for ChemMerge concentrations
        state_chem = torch.full((1, K), 1.0 / K)
        for t in range(N):
            sample_alphas = []
            h_sample = h[t:t+1]
            for l in range(4, 15):
                h_norm = h_sample / torch.norm(h_sample, dim=1, keepdim=True)
                cos_sim = h_norm @ centroids.t()
                k_forward = torch.softmax(cos_sim / 0.05, dim=1)
                
                rate_sum = k_forward + 0.3
                decay = torch.exp(-rate_sum * 0.5)
                state_chem = state_chem * decay + (k_forward / rate_sum) * (1.0 - decay)
                alpha = state_chem / torch.sum(state_chem, dim=1, keepdim=True)
                sample_alphas.append(alpha.squeeze(0))
                
                blended = alpha @ centroids
                h_sample = h_sample + 0.3 * (blended - h_sample)
                h_sample = h_sample / torch.norm(h_sample, dim=1, keepdim=True)
            h[t] = h_sample
            chem_alphas.append(torch.stack(sample_alphas))
        chem_alphas = torch.stack(chem_alphas)
        
        acc_chem = compute_accuracy(h, centroids, stream_true_tasks)
        lj_chem = torch.mean(torch.abs(chem_alphas[:, 1:, :] - chem_alphas[:, :-1, :])).item()
        cj_chem = torch.mean(torch.abs(chem_alphas[1:, 0, :] - chem_alphas[:-1, -1, :])).item()
        
        results['ChemMerge (Temporal)']['accuracy'].append(acc_chem)
        results['ChemMerge (Temporal)']['layer_jitter'].append(lj_chem)
        results['ChemMerge (Temporal)']['cross_query_jitter'].append(cj_chem)
        
        # --- 4. GraviMerge (Temporal Persistence) ---
        h = stream_samples.clone()
        grav_alphas = []
        
        # Carry-over states for GraviMerge spacecraft velocity
        # We carry over velocity to ensure physical momentum across successive queries!
        prev_v = torch.zeros((1, D))
        
        for t in range(N):
            h_sample = h[t:t+1]
            
            # Use unified run_gravimerge from simulate_sandbox
            h_sample_new, alphas, last_v = run_gravimerge(
                h_sample, centroids, tau=0.05, G=0.05, epsilon=0.8, drag=0.9, dt=1.0,
                prev_v=prev_v, return_velocity=True
            )
            
            prev_v = last_v.clone() # save velocity for next query
            h[t] = h_sample_new.squeeze(0)
            grav_alphas.append(alphas.squeeze(0))
        grav_alphas = torch.stack(grav_alphas)
        
        acc_grav = compute_accuracy(h, centroids, stream_true_tasks)
        lj_grav = torch.mean(torch.abs(grav_alphas[:, 1:, :] - grav_alphas[:, :-1, :])).item()
        cj_grav = torch.mean(torch.abs(grav_alphas[1:, 0, :] - grav_alphas[:-1, -1, :])).item()
        
        results['GraviMerge (Temporal)']['accuracy'].append(acc_grav)
        results['GraviMerge (Temporal)']['layer_jitter'].append(lj_grav)
        results['GraviMerge (Temporal)']['cross_query_jitter'].append(cj_grav)
        
    print(f"\n{'Method':<25} | {'Accuracy (%)':<15} | {'Layer Jitter':<15} | {'Cross-Query Jitter':<15}")
    print("-" * 80)
    for m in results:
        mean_acc = np.mean(results[m]['accuracy']) * 100
        mean_lj = np.mean(results[m]['layer_jitter'])
        mean_cj = np.mean(results[m]['cross_query_jitter'])
        print(f"{m:<25} | {mean_acc:<15.2f} | {mean_lj:<15.6f} | {mean_cj:<15.6f}")

if __name__ == '__main__':
    run_temporal_eval()
