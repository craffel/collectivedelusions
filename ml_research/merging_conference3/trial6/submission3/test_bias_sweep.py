import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model_routing import (
    generate_synthetic_dataset,
    train_experts,
    PCAPreprojector,
    BWS_Router,
    train_router,
    evaluate_router,
    set_seed
)

SEEDS = [42, 43, 44, 45, 46]
BIAS_VALUES = [-2.0, -1.0, 0.0, 1.0, 2.0]
LRS = [1e-2, 5e-2]
WD = 1e-4 # standard robust weight decay

print("=" * 70)
print("RUNNING BIAS INITIALIZATION EMPIRICAL SWEEP ACROSS 5 SEEDS")
print("=" * 70)

# Cache datasets and experts to make the sweep fast and robust
cached_runs = {}
for seed in SEEDS:
    print(f"Pre-generating and caching experts for Seed {seed}...")
    set_seed(seed)
    train_data, test_data, calib_data = generate_synthetic_dataset(seed)
    experts = train_experts(train_data, test_data)
    
    pca_proj = PCAPreprojector(n_components=4)
    pca_proj.fit(calib_data[0])
    
    cached_runs[seed] = {
        'train': train_data,
        'test': test_data,
        'calib': calib_data,
        'experts': experts,
        'pca': pca_proj
    }

print("\nCaching completed successfully! Starting the sweep...\n")

# Store results for table formatting
results_summary = {}

for lr in LRS:
    results_summary[lr] = {}
    for bias in BIAS_VALUES:
        joint_accs = []
        for seed in SEEDS:
            run = cached_runs[seed]
            calib = run['calib']
            test = run['test']
            experts = run['experts']
            pca_proj = run['pca']
            
            # Re-seed for router init/training consistency
            set_seed(seed)
            
            router = BWS_Router(
                L=12,
                G=4,
                d=4,
                K=4,
                activation='Sigmoid',
                init_bias=bias
            )
            
            # Train router
            train_router(
                router,
                pca_proj,
                experts,
                calib,
                epochs=100,
                lr=lr,
                lambda_wd=WD
            )
            
            # Evaluate router
            _, joint_acc = evaluate_router(
                router,
                pca_proj,
                experts,
                test,
                mode='Homogeneous_B256'
            )
            joint_accs.append(joint_acc)
            
        mean_acc = np.mean(joint_accs)
        std_acc = np.std(joint_accs)
        results_summary[lr][bias] = (mean_acc, std_acc)
        print(f"LR: {lr:.3f} | Bias: {bias:+1.1f} => Joint Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")

print("\n" + "=" * 70)
print("FINAL RESULTS TABLE IN MARKDOWN")
print("=" * 70)
print("| Initial Bias (B) | Joint Acc at lr=1e-2 (%) | Joint Acc at lr=5e-2 (%) |")
print("| :---: | :---: | :---: |")
for bias in BIAS_VALUES:
    mean_1, std_1 = results_summary[1e-2][bias]
    mean_5, std_5 = results_summary[5e-2][bias]
    print(f"| {bias:+1.1f} | {mean_1:.2f} ± {std_1:.2f}% | {mean_5:.2f} ± {std_5:.2f}% |")
