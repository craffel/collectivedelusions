import time
import numpy as np
import torch
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
from train_and_evaluate import (
    train_router, evaluate_model, generate_signatures, 
    LDSKineticsRouter, set_seed, K, D,
    generate_samples, extract_coordinates, compute_similarity,
    propagate_layers, compute_metrics
)

def run_paired_ttests():
    print("=== Running Paired t-tests (LDS-Kinetics M=11 vs Global PAC-Kinetics M=1) ===")
    configs = ['orthogonal', 'overlapping']
    streams = ['homogeneous', 'heterogeneous']
    seeds = [101, 102, 103, 104, 105]
    
    m1_mapping = [0] * 11
    m11_mapping = list(range(11))
    
    for config_type in configs:
        print(f"\nConfiguration: {config_type.upper()}")
        signatures, indices = generate_signatures(config_type)
        
        for stream_type in streams:
            acc_m1 = []
            acc_m11 = []
            
            for seed in seeds:
                # Train M=1
                router_m1, sigs, ind = train_router(config_type, 1, m1_mapping, seed, regularized=True)
                acc1, _ = evaluate_model(config_type, 'pac_kinetics', router_m1, sigs, ind, seed, stream_type)
                acc_m1.append(acc1)
                
                # Train M=11
                router_m11, sigs, ind = train_router(config_type, 11, m11_mapping, seed, regularized=True)
                acc11, _ = evaluate_model(config_type, 'lds_m11', router_m11, sigs, ind, seed, stream_type)
                acc_m11.append(acc11)
                
            # Perform paired t-test
            t_stat, p_val = stats.ttest_rel(acc_m11, acc_m1)
            print(f"  Stream: {stream_type.capitalize()}")
            print(f"    M=1 Accs:  {np.mean(acc_m1):.4f}% ± {np.std(acc_m1):.4f}% | Raw: {[round(x, 4) for x in acc_m1]}")
            print(f"    M=11 Accs: {np.mean(acc_m11):.4f}% ± {np.std(acc_m11):.4f}% | Raw: {[round(x, 4) for x in acc_m11]}")
            print(f"    t-statistic: {t_stat:.4f} | p-value: {p_val:.6f}")
            if p_val < 0.05:
                print("    => Statistically SIGNIFICANT (p < 0.05)")
            else:
                print("    => NOT statistically significant (p >= 0.05)")

def run_latency_benchmark():
    print("\n=== Running Latency and Computational Overhead Benchmark ===")
    device = torch.device('cpu')
    T_test = 200
    trials = 1000
    
    # Generate dummy input coordinates and similarities
    e = torch.randn(T_test, K, device=device)
    sim_seq = torch.rand(T_test, device=device)
    
    routers = {
        'Global (M=1)': LDSKineticsRouter(1, K).to(device),
        'Tri-Block (M=3)': LDSKineticsRouter(3, K).to(device),
        'Fully Decoupled (M=11)': LDSKineticsRouter(11, K).to(device)
    }
    
    # Warmup
    for name, router in routers.items():
        for _ in range(50):
            _ = router(e, sim_seq)
            
    # Measure
    latencies = {}
    for name, router in routers.items():
        t0 = time.perf_counter()
        for _ in range(trials):
            _ = router(e, sim_seq)
        t1 = time.perf_counter()
        mean_lat_ms = ((t1 - t0) / trials) * 1000.0  # in milliseconds for sequence
        mean_step_us = (mean_lat_ms / T_test) * 1000.0  # in microseconds per step
        latencies[name] = (mean_lat_ms, mean_step_us)
        print(f"  {name}:")
        print(f"    Mean Sequence Latency (T=200): {mean_lat_ms:.4f} ms")
        print(f"    Mean Step Latency:            {mean_step_us:.4f} us")
        
    base_seq_ms, base_step_us = latencies['Global (M=1)']
    for name in ['Tri-Block (M=3)', 'Fully Decoupled (M=11)']:
        seq_ms, step_us = latencies[name]
        overhead_pct = ((seq_ms - base_seq_ms) / base_seq_ms) * 100.0
        print(f"  Overhead of {name} relative to Global (M=1): {overhead_pct:.2f}%")

def run_adaptive_grouping_sweep():
    print("\n=== Running Adaptive Layer Grouping Sweep ===")
    configs = ['orthogonal', 'overlapping']
    streams = ['homogeneous', 'heterogeneous']
    seeds = [101, 102, 103, 104, 105]
    
    # Define alternative layer mappings for M=3
    mappings = {
        'Static Equal (L4-7, L8-11, L12-14)': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
        'Early-Heavy (L4, L5, L6-14)': [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        'Late-Heavy (L4-12, L13, L14)': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
    }
    
    for config_type in configs:
        print(f"\nConfiguration: {config_type.upper()}")
        signatures, indices = generate_signatures(config_type)
        
        for mapping_name, layer_mapping in mappings.items():
            print(f"  Mapping: {mapping_name}")
            for stream_type in streams:
                accs = []
                jits = []
                for seed in seeds:
                    router, sigs, ind = train_router(config_type, 3, layer_mapping, seed, regularized=True)
                    acc, jit = evaluate_model(config_type, 'lds_m3', router, sigs, ind, seed, stream_type, layer_mapping=layer_mapping)
                    accs.append(acc)
                    jits.append(jit)
                print(f"    {stream_type.capitalize()} Acc: {np.mean(accs):.4f}% ± {np.std(accs):.4f}% | Jitter: {np.mean(jits):.4f} ± {np.std(jits):.4f}")

def compute_sequence_risk(router, signatures, indices, seed, T, is_calibration=False):
    set_seed(seed)
    if is_calibration:
        # Construct calibration structured sequence of length T
        y_seq = []
        samples_per_expert = T // K
        for k in range(K):
            y_seq.extend([k] * samples_per_expert)
    else:
        # Construct heterogeneous sequence of length T (using a deterministic random sequence per seed)
        random.seed(seed)
        y_seq = [random.randint(0, K - 1) for _ in range(T)]
        
    y_seq = torch.tensor(y_seq)
    h3 = generate_samples(y_seq, signatures, indices)
    e = extract_coordinates(h3, indices)
    sim = compute_similarity(e)
    
    with torch.no_grad():
        alphas = router(e, sim)
    
    if router.M == 1:
        layer_mapping = [0] * 11
    elif router.M == 3:
        layer_mapping = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    else:
        layer_mapping = list(range(11))
        
    h_L = propagate_layers(h3, alphas, signatures, layer_mapping)
    logits, _ = compute_metrics(h_L, y_seq, signatures)
    
    p = torch.softmax(logits, dim=-1)
    loss_ce = -torch.log(p[torch.arange(logits.shape[0]), y_seq] + 1e-8)
    truncated_loss = torch.clamp(loss_ce, max=5.0)
    return torch.mean(truncated_loss).item()

def run_calibration_size_sweep():
    print("\n=== Running Calibration Sequence Length (T) Sweep ===")
    config_type = 'overlapping'
    T_vals = [32, 64, 128, 256]
    
    # Store results for plotting
    acc_results = {'lds_m11': {t: [] for t in T_vals}, 'erm_m11': {t: [] for t in T_vals}}
    gap_results = {'lds_m11': {t: [] for t in T_vals}, 'erm_m11': {t: [] for t in T_vals}}
    
    signatures, indices = generate_signatures(config_type)
    m11_mapping = list(range(11))
    
    for T in T_vals:
        # Determine number of seeds and epochs dynamically to prevent timeouts
        if T == 32:
            seeds = [101, 102, 103, 104, 105]
            epochs = 40
        elif T == 64:
            seeds = [101, 102, 103, 104, 105]
            epochs = 35
        elif T == 128:
            seeds = [101, 102, 103]
            epochs = 25
        else: # T == 256
            seeds = [101, 102, 103]
            epochs = 20
            
        print(f"  Calibration Size T = {T} (Seeds: {seeds}, Epochs: {epochs})")
        for seed in seeds:
            # 1. LDS-Kinetics (M=11, regularized)
            router_lds, _, _ = train_router(config_type, 11, m11_mapping, seed, regularized=True, T_cal=T, epochs=epochs)
            acc_lds, _ = evaluate_model(config_type, 'lds_m11', router_lds, signatures, indices, seed, 'heterogeneous')
            acc_results['lds_m11'][T].append(acc_lds)
            
            # Compute risks for generalization gap
            train_risk_lds = compute_sequence_risk(router_lds, signatures, indices, seed, T, is_calibration=True)
            test_risk_lds = compute_sequence_risk(router_lds, signatures, indices, seed, 200, is_calibration=False)
            gap_results['lds_m11'][T].append(test_risk_lds - train_risk_lds)
            
            # 2. Decoupled ERM (M=11, unregularized)
            router_erm, _, _ = train_router(config_type, 11, m11_mapping, seed, regularized=False, T_cal=T, epochs=epochs)
            acc_erm, _ = evaluate_model(config_type, 'erm_m11', router_erm, signatures, indices, seed, 'heterogeneous')
            acc_results['erm_m11'][T].append(acc_erm)
            
            # Compute risks for generalization gap
            train_risk_erm = compute_sequence_risk(router_erm, signatures, indices, seed, T, is_calibration=True)
            test_risk_erm = compute_sequence_risk(router_erm, signatures, indices, seed, 200, is_calibration=False)
            gap_results['erm_m11'][T].append(test_risk_erm - train_risk_erm)
            
        mean_lds = np.mean(acc_results['lds_m11'][T])
        std_lds = np.std(acc_results['lds_m11'][T])
        mean_gap_lds = np.mean(gap_results['lds_m11'][T])
        
        mean_erm = np.mean(acc_results['erm_m11'][T])
        std_erm = np.std(acc_results['erm_m11'][T])
        mean_gap_erm = np.mean(gap_results['erm_m11'][T])
        
        print(f"    LDS-Kinetics (M=11):  Hetero Acc = {mean_lds:.4f}% ± {std_lds:.4f}% | Gen Gap = {mean_gap_lds:.4f}")
        print(f"    Decoupled ERM (M=11): Hetero Acc = {mean_erm:.4f}% ± {std_erm:.4f}% | Gen Gap = {mean_gap_erm:.4f}")
        
    # Produce Beautiful Figures
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Serving Accuracy vs Calibration size
    lds_accs = [np.mean(acc_results['lds_m11'][t]) for t in T_vals]
    lds_accs_std = [np.std(acc_results['lds_m11'][t]) if len(acc_results['lds_m11'][t]) > 1 else 0.0 for t in T_vals]
    erm_accs = [np.mean(acc_results['erm_m11'][t]) for t in T_vals]
    erm_accs_std = [np.std(acc_results['erm_m11'][t]) if len(acc_results['erm_m11'][t]) > 1 else 0.0 for t in T_vals]
    
    axes[0].errorbar(T_vals, lds_accs, yerr=lds_accs_std, fmt='-o', color='blue', label='LDS-Kinetics (M=11)', capsize=5, lw=2)
    axes[0].errorbar(T_vals, erm_accs, yerr=erm_accs_std, fmt='-s', color='red', label='Decoupled ERM (M=11)', capsize=5, lw=2)
    axes[0].set_title("Serving Accuracy (%) vs. Calibration Sequence Length (T)")
    axes[0].set_xlabel("Calibration Sequence Length (T)")
    axes[0].set_ylabel("Heterogeneous Serving Accuracy (%)")
    axes[0].set_xticks(T_vals)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    # Right plot: Generalization Gap vs Calibration size
    lds_gaps = [np.mean(gap_results['lds_m11'][t]) for t in T_vals]
    lds_gaps_std = [np.std(gap_results['lds_m11'][t]) if len(gap_results['lds_m11'][t]) > 1 else 0.0 for t in T_vals]
    erm_gaps = [np.mean(gap_results['erm_m11'][t]) for t in T_vals]
    erm_gaps_std = [np.std(gap_results['erm_m11'][t]) if len(gap_results['erm_m11'][t]) > 1 else 0.0 for t in T_vals]
    
    axes[1].errorbar(T_vals, lds_gaps, yerr=lds_gaps_std, fmt='-o', color='blue', label='LDS-Kinetics (M=11)', capsize=5, lw=2)
    axes[1].errorbar(T_vals, erm_gaps, yerr=erm_gaps_std, fmt='-s', color='red', label='Decoupled ERM (M=11)', capsize=5, lw=2)
    axes[1].set_title("Generalization Gap (Test - Train Risk) vs. Calibration Length (T)")
    axes[1].set_xlabel("Calibration Sequence Length (T)")
    axes[1].set_ylabel("Generalization Gap (Truncated CE Risk)")
    axes[1].set_xticks(T_vals)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()
    
    plt.suptitle(f"Generalization and Specialization Sweeps over Calibration Budgets (Overlapping Manifolds)", fontsize=14)
    plt.tight_layout()
    plt.savefig('results/fig3_calibration_sweep.png', dpi=300)
    plt.close()
    print("\nSuccessfully generated results/fig3_calibration_sweep.png!")

if __name__ == "__main__":
    run_paired_ttests()
    run_latency_benchmark()
    run_adaptive_grouping_sweep()
    run_calibration_size_sweep()
