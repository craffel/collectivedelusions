import os
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from train_and_evaluate import LDSKineticsRouter, compute_similarity, set_seed

# Device
device = torch.device('cpu')

# Base settings
L = 14
gamma_V = 0.05
kappa_scale = 0.0385

def generate_signatures_scaled(K, D):
    orthogonal_indices = [list(range(k * 48, (k + 1) * 48)) for k in range(K)]
    v = torch.zeros(K, D)
    for k in range(K):
        v[k, orthogonal_indices[k]] = 1.0 / math.sqrt(48.0)
    return v, orthogonal_indices

def normalize_vector(z, eps=1e-6):
    return z / (torch.norm(z, dim=-1, keepdim=True) + eps)

def extract_coordinates(z, indices, K, eps=1e-6):
    z_norm = normalize_vector(z, eps)
    e = []
    for k in range(K):
        val = torch.norm(z_norm[..., indices[k]], dim=-1)
        e.append(val)
    return torch.stack(e, dim=-1)

def generate_samples(y, signatures, indices, sigmas):
    D = signatures.shape[1]
    h3 = []
    for label in y:
        noise = torch.randn(D) * sigmas[label.item()]
        h3.append(signatures[label.item()] + noise)
    return torch.stack(h3, dim=0)

def propagate_layers(h3, alpha_seq, signatures, layer_to_block_mapping):
    T = h3.shape[0]
    h = h3.clone()
    for l in range(4, L + 1):
        block_idx = layer_to_block_mapping[l - 4]
        alpha_layer = alpha_seq[block_idx]  # shape (T, K)
        expert_diff = signatures.unsqueeze(0) - h.unsqueeze(1)  # (T, K, D)
        scaled_diff = expert_diff * alpha_layer.unsqueeze(-1)  # (T, K, D)
        update = torch.sum(scaled_diff, dim=1) * gamma_V  # (T, D)
        h = h + update
    return h

def compute_metrics(h_L, y, signatures, biases):
    T = h_L.shape[0]
    K = signatures.shape[0]
    h_L_sq = torch.sum(h_L ** 2, dim=-1, keepdim=True)  # (T, 1)
    sigs_sq = torch.sum(signatures ** 2, dim=-1).unsqueeze(0)  # (1, K)
    dot_prod = torch.matmul(h_L, signatures.t())  # (T, K)
    dists_sq = h_L_sq + sigs_sq - 2.0 * dot_prod  # (T, K)
    
    biases_tensor = torch.tensor(biases, device=device).unsqueeze(0)  # (1, K)
    logits = -dists_sq + biases_tensor  # (T, K)
    
    target_dists_sq = dists_sq[torch.arange(T, device=device), y]  # (T,)
    accs = torch.exp(-kappa_scale * target_dists_sq)  # (T,)
    
    return logits, accs

def train_router_scaled(K, M, layer_mapping, seed, regularized=True, lr=0.005, epochs=100, T_cal=32):
    set_seed(seed)
    D = K * 48
    signatures, indices = generate_signatures_scaled(K, D)
    
    # Setup scaled sigmas and biases
    base_sigmas = [0.05, 0.15, 0.40, 1.20]
    base_biases = [0.0, 0.0, -0.90, -2.30]
    sigmas = (base_sigmas * (K // 4 + 1))[:K]
    biases = (base_biases * (K // 4 + 1))[:K]
    
    samples_per_expert = T_cal // K
    if samples_per_expert == 0:
        samples_per_expert = 1
        T_cal = K
    y_cal = []
    for k in range(K):
        y_cal.extend([k] * samples_per_expert)
    y_cal = torch.tensor(y_cal[:T_cal])
    
    h3_cal = generate_samples(y_cal, signatures, indices, sigmas)
    e_cal = extract_coordinates(h3_cal, indices, K)
    sim_cal = compute_similarity(e_cal)
    
    router = LDSKineticsRouter(M, K)
    optimizer = optim.Adam(router.parameters(), lr=lr)
    
    lambda_param = 0.5
    L_max = 5.0
    a_blocks = float(T_cal) / 4.0
    sigma_0_sq = 5.0
    
    u0 = torch.zeros(M, K)
    W0 = torch.stack([torch.eye(K) for _ in range(M)])
    w0 = torch.ones(M, K) * math.log(0.05)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        alphas = router(e_cal, sim_cal)
        h_L = propagate_layers(h3_cal, alphas, signatures, layer_mapping)
        logits, accs = compute_metrics(h_L, y_cal, signatures, biases)
        
        p = torch.softmax(logits, dim=-1)
        loss_ce = -torch.log(p[torch.arange(logits.shape[0]), y_cal] + 1e-8)
        truncated_loss = torch.clamp(loss_ce, max=L_max)
        R_hat = torch.mean(truncated_loss)
        
        if regularized:
            kl = (torch.sum((router.u - u0) ** 2) + 
                  torch.sum((router.W - W0) ** 2) + 
                  torch.sum((router.w - w0) ** 2)) / (2.0 * sigma_0_sq)
            loss = (lambda_param / L_max) * R_hat + (1.0 / (a_blocks * sigma_0_sq)) * kl
        else:
            loss = R_hat
            
        loss.backward()
        optimizer.step()
        
    return router, signatures, indices, sigmas, biases

def evaluate_model_scaled(K, M, router, signatures, indices, sigmas, biases, test_seed, stream_type='homogeneous'):
    set_seed(test_seed)
    T_test = 200
    
    if stream_type == 'homogeneous':
        y_test = []
        queries_per_expert = math.ceil(T_test / K)
        for k in range(K):
            y_test.extend([k] * queries_per_expert)
        y_test = torch.tensor(y_test[:T_test])
    else:
        random.seed(test_seed)
        y_test = torch.tensor([random.randint(0, K - 1) for _ in range(T_test)])
        
    h3_test = generate_samples(y_test, signatures, indices, sigmas)
    e_test = extract_coordinates(h3_test, indices, K)
    sim_test = compute_similarity(e_test)
    
    with torch.no_grad():
        alphas = router(e_test, sim_test)
        
    if M == 1:
        layer_mapping = [0] * 11
    else:
        layer_mapping = list(range(11))
        
    h_L = propagate_layers(h3_test, alphas, signatures, layer_mapping)
    logits, accs = compute_metrics(h_L, y_test, signatures, biases)
    
    # Calculate soft classification accuracy consistent with Table 1 and 2
    acc = torch.mean(accs).item() * 100.0
    
    # Routing jitter calculation
    jitter = 0.0
    for m in range(M):
        for t in range(1, T_test):
            jitter += torch.sum(torch.abs(alphas[m][t] - alphas[m][t - 1])).item()
    jitter /= (M * (T_test - 1))
    
    return acc, jitter

def run_scaling_sweep():
    print("=== Running K-Expert Scaling Sweep ===")
    K_vals = [4, 8, 12, 16]
    seeds = [101, 102, 103, 104, 105]
    
    results = {
        'K': K_vals,
        'm1_homo_acc': [], 'm1_homo_jit': [], 'm11_homo_acc': [], 'm11_homo_jit': [],
        'm1_hetero_acc': [], 'm1_hetero_jit': [], 'm11_hetero_acc': [], 'm11_hetero_jit': [],
        'm1_lat_us': [], 'm11_lat_us': []
    }
    
    m1_mapping = [0] * 11
    m11_mapping = list(range(11))
    
    # Warmup latency
    dummy_e = torch.randn(200, 16)
    dummy_sim = torch.rand(200)
    dummy_router = LDSKineticsRouter(11, 16)
    for _ in range(50):
        _ = dummy_router(dummy_e, dummy_sim)
        
    for K_val in K_vals:
        print(f"\nEvaluating K = {K_val} task experts...")
        
        # Benchmark latencies
        e_lat = torch.randn(200, K_val)
        sim_lat = torch.rand(200)
        
        router_m1 = LDSKineticsRouter(1, K_val)
        router_m11 = LDSKineticsRouter(11, K_val)
        
        # M=1 Latency
        t0 = time.perf_counter()
        for _ in range(100):
            _ = router_m1(e_lat, sim_lat)
        t1 = time.perf_counter()
        lat_m1_us = ((t1 - t0) / (100 * 200)) * 1e6 # microsecond per step
        
        # M=11 Latency
        t0 = time.perf_counter()
        for _ in range(100):
            _ = router_m11(e_lat, sim_lat)
        t1 = time.perf_counter()
        lat_m11_us = ((t1 - t0) / (100 * 200)) * 1e6 # microsecond per step
        
        results['m1_lat_us'].append(lat_m1_us)
        results['m11_lat_us'].append(lat_m11_us)
        
        # Metrics storage
        m1_hom_acc_list, m1_hom_jit_list = [], []
        m1_het_acc_list, m1_het_jit_list = [], []
        
        m11_hom_acc_list, m11_hom_jit_list = [], []
        m11_het_acc_list, m11_het_jit_list = [], []
        
        for seed in seeds:
            # Global M=1
            r_m1, sigs_m1, ind_m1, sigmas, biases = train_router_scaled(K_val, 1, m1_mapping, seed)
            acc_hom, jit_hom = evaluate_model_scaled(K_val, 1, r_m1, sigs_m1, ind_m1, sigmas, biases, seed, 'homogeneous')
            acc_het, jit_het = evaluate_model_scaled(K_val, 1, r_m1, sigs_m1, ind_m1, sigmas, biases, seed, 'heterogeneous')
            m1_hom_acc_list.append(acc_hom)
            m1_hom_jit_list.append(jit_hom)
            m1_het_acc_list.append(acc_het)
            m1_het_jit_list.append(jit_het)
            
            # Decoupled M=11
            r_m11, sigs_m11, ind_m11, _, _ = train_router_scaled(K_val, 11, m11_mapping, seed)
            acc_hom11, jit_hom11 = evaluate_model_scaled(K_val, 11, r_m11, sigs_m11, ind_m11, sigmas, biases, seed, 'homogeneous')
            acc_het11, jit_het11 = evaluate_model_scaled(K_val, 11, r_m11, sigs_m11, ind_m11, sigmas, biases, seed, 'heterogeneous')
            m11_hom_acc_list.append(acc_hom11)
            m11_hom_jit_list.append(jit_hom11)
            m11_het_acc_list.append(acc_het11)
            m11_het_jit_list.append(jit_het11)
            
        results['m1_homo_acc'].append((np.mean(m1_hom_acc_list), np.std(m1_hom_acc_list)))
        results['m1_homo_jit'].append((np.mean(m1_hom_jit_list), np.std(m1_hom_jit_list)))
        results['m11_homo_acc'].append((np.mean(m11_hom_acc_list), np.std(m11_hom_acc_list)))
        results['m11_homo_jit'].append((np.mean(m11_hom_jit_list), np.std(m11_hom_jit_list)))
        
        results['m1_hetero_acc'].append((np.mean(m1_het_acc_list), np.std(m1_het_acc_list)))
        results['m1_hetero_jit'].append((np.mean(m1_het_jit_list), np.std(m1_het_jit_list)))
        results['m11_hetero_acc'].append((np.mean(m11_het_acc_list), np.std(m11_het_acc_list)))
        results['m11_hetero_jit'].append((np.mean(m11_het_jit_list), np.std(m11_het_jit_list)))
        
        print(f"  Global M=1: Homo Acc={np.mean(m1_hom_acc_list):.2f}% ± {np.std(m1_hom_acc_list):.2f}%, Homo Jitter={np.mean(m1_hom_jit_list):.4f} ± {np.std(m1_hom_jit_list):.4f} | Hetero Acc={np.mean(m1_het_acc_list):.2f}% ± {np.std(m1_het_acc_list):.2f}%, Hetero Jitter={np.mean(m1_het_jit_list):.4f} ± {np.std(m1_het_jit_list):.4f}")
        print(f"  LDS-Kinetics M=11: Homo Acc={np.mean(m11_hom_acc_list):.2f}% ± {np.std(m11_hom_acc_list):.2f}%, Homo Jitter={np.mean(m11_hom_jit_list):.4f} ± {np.std(m11_hom_jit_list):.4f} | Hetero Acc={np.mean(m11_het_acc_list):.2f}% ± {np.std(m11_het_acc_list):.2f}%, Hetero Jitter={np.mean(m11_het_jit_list):.4f} ± {np.std(m11_het_jit_list):.4f}")
        print(f"  Latencies: M=1: {lat_m1_us:.2f} us/step | M=11: {lat_m11_us:.2f} us/step")
        
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Latency vs K
    axes[0].plot(K_vals, results['m1_lat_us'], 'o-', label='Global M=1', color='tab:blue', linewidth=2)
    axes[0].plot(K_vals, results['m11_lat_us'], 's--', label='LDS-Kinetics M=11', color='tab:orange', linewidth=2)
    axes[0].set_xlabel('Number of Experts ($K$)', fontsize=12)
    axes[0].set_ylabel('Inference Latency per Step ($\mu$s)', fontsize=12)
    axes[0].set_title('Inference Latency vs K', fontsize=13, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=10)
    
    # Panel 2: Heterogeneous Accuracy vs K
    m1_acc_mean = [x[0] for x in results['m1_hetero_acc']]
    m1_acc_std = [x[1] for x in results['m1_hetero_acc']]
    m11_acc_mean = [x[0] for x in results['m11_hetero_acc']]
    m11_acc_std = [x[1] for x in results['m11_hetero_acc']]
    
    axes[1].errorbar(K_vals, m1_acc_mean, yerr=m1_acc_std, fmt='o-', label='Global M=1', color='tab:blue', capsize=4, linewidth=2)
    axes[1].errorbar(K_vals, m11_acc_mean, yerr=m11_acc_std, fmt='s--', label='LDS-Kinetics M=11', color='tab:orange', capsize=4, linewidth=2)
    axes[1].set_xlabel('Number of Experts ($K$)', fontsize=12)
    axes[1].set_ylabel('Heterogeneous Accuracy (%)', fontsize=12)
    axes[1].set_title('Heterogeneous Accuracy vs K', fontsize=13, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=10)
    
    # Panel 3: Jitter vs K
    m1_jit_mean = [x[0] for x in results['m1_hetero_jit']]
    m11_jit_mean = [x[0] for x in results['m11_hetero_jit']]
    
    axes[2].plot(K_vals, m1_jit_mean, 'o-', label='Global M=1', color='tab:blue', linewidth=2)
    axes[2].plot(K_vals, m11_jit_mean, 's--', label='LDS-Kinetics M=11', color='tab:orange', linewidth=2)
    axes[2].set_xlabel('Number of Experts ($K$)', fontsize=12)
    axes[2].set_ylabel('Heterogeneous Routing Jitter', fontsize=12)
    axes[2].set_title('Routing Jitter vs K', fontsize=13, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(fontsize=10)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/fig4_scaling_sweep.png', dpi=300)
    plt.close()
    print("\nSaved scaling sweep plot to results/fig4_scaling_sweep.png")
    
    # Write a markdown report
    with open('results/scaling_report.md', 'w') as f:
        f.write("# Scaling Expert Sweeps Analysis (K=4 to K=16)\n\n")
        f.write("This table summarizes the performance of Global PAC-Kinetics (M=1) and Layer-Decoupled Stateful Kinetics (M=11) across various numbers of task experts K.\n\n")
        f.write("| Experts (K) | Model | Homo Acc (%) | Hetero Acc (%) | Hetero Jitter | Step Latency (us) |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for i, K_val in enumerate(K_vals):
            m1_hom_acc, m1_hom_std = results['m1_homo_acc'][i]
            m1_het_acc, m1_het_std = results['m1_hetero_acc'][i]
            m1_jit_mean, _ = results['m1_hetero_jit'][i]
            m1_lat = results['m1_lat_us'][i]
            
            m11_hom_acc, m11_hom_std = results['m11_homo_acc'][i]
            m11_het_acc, m11_het_std = results['m11_hetero_acc'][i]
            m11_jit_mean, _ = results['m11_hetero_jit'][i]
            m11_lat = results['m11_lat_us'][i]
            
            f.write(f"| K={K_val} | Global M=1 | {m1_hom_acc:.2f}% ± {m1_hom_std:.2f}% | {m1_het_acc:.2f}% ± {m1_het_std:.2f}% | {m1_jit_mean:.4f} | {m1_lat:.2f} us |\n")
            f.write(f"| | LDS-Kinetics M=11 | {m11_hom_acc:.2f}% ± {m11_hom_std:.2f}% | {m11_het_acc:.2f}% ± {m11_het_std:.2f}% | {m11_jit_mean:.4f} | {m11_lat:.2f} us |\n")
            
if __name__ == '__main__':
    run_scaling_sweep()
