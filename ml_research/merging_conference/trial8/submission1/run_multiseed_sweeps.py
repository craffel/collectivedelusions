import torch
import numpy as np
import os
from run_experiments import load_expert, get_datasets, build_streams, evaluate_method

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 1. Load experts and base model
    base_model = load_expert("base_model.pt")
    expert0 = load_expert("expert_mnist.pt").to(device)
    expert1 = load_expert("expert_kmnist.pt").to(device)
    
    base_state = base_model.state_dict()
    exp0_state = expert0.state_dict()
    exp1_state = expert1.state_dict()
    
    # 2. Extract task vectors
    task_vectors = {}
    for name, param in base_state.items():
        if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
            continue
        v_task0 = exp0_state[name] - param
        v_task1 = exp1_state[name] - param
        task_vectors[name] = torch.stack([v_task0, v_task1]).to(device)
        
    # 3. Extract expert BN buffers for Soft BN Buffer Fusion
    expert_bn_buffers = {}
    for name in base_state:
        if 'running_mean' in name or 'running_var' in name:
            buf0 = exp0_state[name]
            buf1 = exp1_state[name]
            expert_bn_buffers[name] = [buf0.to(device), buf1.to(device)]
            
    # 4. Load datasets
    mnist, kmnist, fashion = get_datasets()
    
    seeds = [42, 43, 44]
    configs = [
        (0.01, 1.0),
        (0.05, 1.0),
        (0.10, 1.0),
        (0.10, 1.5)
    ]
    
    results_baseline = {config: [] for config in configs}
    results_dmtr = {config: [] for config in configs}
    
    for seed in seeds:
        print(f"\nEvaluating Seed: {seed}")
        _, alt_stream, _ = build_streams(mnist, kmnist, fashion, seed=seed)
        
        for lr, beta_base in configs:
            # DF-Bayes-TTMM (Baseline) with static prior beta = beta_base (not scaling with delta_H)
            acc_baseline, _ = evaluate_method(
                "DF-Bayes-TTMM", alt_stream, base_model, expert0, expert1, expert_bn_buffers, task_vectors, device,
                use_dmtr=False, beta_base=beta_base, lr=lr, seed=seed
            )
            results_baseline[(lr, beta_base)].append(acc_baseline)
            
            # DMTR (Ours) with adaptive beta scaling
            acc_dmtr, _ = evaluate_method(
                "DMTR", alt_stream, base_model, expert0, expert1, expert_bn_buffers, task_vectors, device,
                use_dmtr=True, beta_base=beta_base, kappa=5.0, lr=lr, seed=seed
            )
            results_dmtr[(lr, beta_base)].append(acc_dmtr)
            print(f"lr: {lr}, beta: {beta_base} -> Baseline: {acc_baseline:.2f}%, DMTR: {acc_dmtr:.2f}%")
            
    # Print results
    print("\n=== MULTI-SEED LR & BETA_BASE SWEEP (kappa=5.0) ===")
    print("| lr | beta_base | DF-Bayes-TTMM | DMTR (Ours) |")
    print("| :---: | :---: | :---: | :---: |")
    for lr, beta_base in configs:
        baseline_accs = results_baseline[(lr, beta_base)]
        dmtr_accs = results_dmtr[(lr, beta_base)]
        
        mean_base = np.mean(baseline_accs)
        std_base = np.std(baseline_accs)
        mean_dmtr = np.mean(dmtr_accs)
        std_dmtr = np.std(dmtr_accs)
        
        print(f"| {lr:<4} | {beta_base:<9} | {mean_base:.2f}% ± {std_base:.2f}% | {mean_dmtr:.2f}% ± {std_dmtr:.2f}% |")

if __name__ == "__main__":
    main()
