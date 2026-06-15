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
DIMS = [2, 3, 4, 6, 8, 12, 16]
LR = 5e-2 # optimal learning rate for Sigmoid
WD = 1e-4 # optimal weight decay

print("=" * 70)
print("RUNNING PCA SUBSPACE DIMENSION SWEEP ACROSS 5 SEEDS")
print("=" * 70)

# Cache datasets and experts to make the sweep fast
cached_runs = {}
for seed in SEEDS:
    print(f"Pre-generating and caching experts for Seed {seed}...")
    set_seed(seed)
    train_data, test_data, calib_data = generate_synthetic_dataset(seed)
    experts = train_experts(train_data, test_data)
    
    cached_runs[seed] = {
        'train': train_data,
        'test': test_data,
        'calib': calib_data,
        'experts': experts
    }

print("\nCaching completed successfully! Starting the sweep...\n")

# Store results
results_summary = {}

for d in DIMS:
    joint_accs = []
    for seed in SEEDS:
        run = cached_runs[seed]
        calib = run['calib']
        test = run['test']
        experts = run['experts']
        
        # Re-seed for router init/training consistency
        set_seed(seed)
        
        # Fit PCA with dimension d
        pca_proj = PCAPreprojector(n_components=d)
        pca_proj.fit(calib[0])
        
        router = BWS_Router(
            L=12,
            G=4,
            d=d,
            K=4,
            activation='Sigmoid',
            lambda_max=0.3,
            init_bias=1.0 # Default bias
        )
        
        # Train router
        train_router(
            router,
            pca_proj,
            experts,
            calib,
            epochs=100,
            lr=LR,
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
    results_summary[d] = (mean_acc, std_acc)
    print(f"PCA Dim d: {d} => Joint Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")

print("\n" + "=" * 70)
print("FINAL RESULTS TABLE IN MARKDOWN")
print("=" * 70)
print("| PCA Subspace Dimension (d) | Joint Mean Accuracy (%) |")
print("|---------------------------|-------------------------|")
for d in DIMS:
    mean_val, std_val = results_summary[d]
    print(f"| d = {d} | {mean_val:.2f}% ± {std_val:.2f}% |")
print("=" * 70)
