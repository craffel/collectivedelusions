import torch
import numpy as np
import os
from run_experiments import (
    load_expert, get_datasets,
    build_streams as build_clean_streams,
    evaluate_method as evaluate_clean_method
)
from run_noise_experiments import (
    build_streams as build_noise_streams,
    evaluate_method as evaluate_noise_method
)

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
    methods = ["Static", "AdaMerging", "DR-Fisher", "DF-Bayes-TTMM", "DMTR"]
    stream_names = ["Closed Sequential", "Closed Alternating", "Open-World"]
    
    # Structure to hold clean results: {method: {stream: [acc_seed1, acc_seed2, acc_seed3]}}
    clean_results = {m: {s: [] for s in stream_names} for m in methods}
    # Structure to hold noisy results: {noise: {method: {stream: [acc_seed1, acc_seed2, acc_seed3]}}}
    noise_results = {0.3: {m: {s: [] for s in stream_names} for m in methods},
                     0.6: {m: {s: [] for s in stream_names} for m in methods}}
    
    for seed in seeds:
        print(f"\n==================== RUNNING EVALUATION WITH SEED {seed} ====================")
        
        # --- Evaluate Clean ---
        print("\n--- Evaluating Clean Streams ---")
        seq_stream, alt_stream, ow_stream = build_clean_streams(mnist, kmnist, fashion, seed=seed)
        streams = {
            "Closed Sequential": seq_stream,
            "Closed Alternating": alt_stream,
            "Open-World": ow_stream
        }
        for s_name, stream in streams.items():
            for method in methods:
                use_dmtr = (method == "DMTR")
                acc, _ = evaluate_clean_method(
                    method, stream, base_model, expert0, expert1, expert_bn_buffers, task_vectors, device, use_dmtr=use_dmtr, seed=seed
                )
                clean_results[method][s_name].append(acc)
                print(f"[Seed {seed}] Clean - {s_name} - {method:<15}: {acc:.2f}%")
                
        # --- Evaluate Noisy ---
        for noise in [0.3, 0.6]:
            print(f"\n--- Evaluating Noisy Streams (std={noise}) ---")
            seq_stream_n, alt_stream_n, ow_stream_n = build_noise_streams(mnist, kmnist, fashion, noise_std=noise, seed=seed)
            streams_n = {
                "Closed Sequential": seq_stream_n,
                "Closed Alternating": alt_stream_n,
                "Open-World": ow_stream_n
            }
            for s_name, stream in streams_n.items():
                for method in methods:
                    use_dmtr = (method == "DMTR")
                    acc = evaluate_noise_method(
                        method, stream, base_model, expert0, expert1, expert_bn_buffers, task_vectors, device, use_dmtr=use_dmtr, noise_std=noise, seed=seed
                    )
                    noise_results[noise][method][s_name].append(acc)
                    print(f"[Seed {seed}] Noisy({noise}) - {s_name} - {method:<15}: {acc:.2f}%")
                    
    # Compile and print results
    print("\n\n" + "="*30 + " FINAL RESULTS (MEAN ± STD) " + "="*30)
    
    print("\n### 1. Clean Streams Evaluation")
    print("| Method | Closed Sequential | Closed Alternating | Open-World |")
    print("| :--- | :---: | :---: | :---: |")
    for m in methods:
        row = f"| **{m}**"
        for s in stream_names:
            accs = clean_results[m][s]
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            row += f" | {mean_acc:.2f}% ± {std_acc:.2f}%"
        row += " |"
        print(row)
        
    for noise in [0.3, 0.6]:
        print(f"\n### 2. Noisy Streams Evaluation (std={noise})")
        print("| Method | Closed Sequential | Closed Alternating | Open-World |")
        print("| :--- | :---: | :---: | :---: |")
        for m in methods:
            row = f"| **{m}**"
            for s in stream_names:
                accs = noise_results[noise][m][s]
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                row += f" | {mean_acc:.2f}% ± {std_acc:.2f}%"
            row += " |"
            print(row)

if __name__ == "__main__":
    main()
