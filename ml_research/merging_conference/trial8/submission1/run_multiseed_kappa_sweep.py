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
    kappas = [0.0, 1.0, 2.5, 5.0, 7.5, 10.0]
    
    results = {kappa: [] for kappa in kappas}
    
    for seed in seeds:
        _, alt_stream, _ = build_streams(mnist, kmnist, fashion, seed=seed)
        for kappa in kappas:
            acc, _ = evaluate_method(
                "DMTR", alt_stream, base_model, expert0, expert1, expert_bn_buffers, task_vectors, device,
                use_dmtr=True, beta_base=1.0, kappa=kappa, lr=0.05, seed=seed
            )
            results[kappa].append(acc)
            
    # Print results
    print("\n=== MULTI-SEED KAPPA ABLATION STUDY ===")
    print("| kappa | Accuracy |")
    print("| :---: | :---: |")
    for kappa in kappas:
        accs = results[kappa]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"| {kappa:<5} | {mean_acc:.2f}% ± {std_acc:.2f}% |")

if __name__ == "__main__":
    main()
