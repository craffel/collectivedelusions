import numpy as np
import torch
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
activations = ['Linear', 'Tanh', 'Softmax', 'Sigmoid']
lrs = [1e-2, 5e-2]

results = {act: {lr: [] for lr in lrs} for act in activations}

print("Running activation learning rate sweep across 5 seeds...")
for seed in SEEDS:
    set_seed(seed)
    train_data, test_data, calib_data = generate_synthetic_dataset(seed)
    experts = train_experts(train_data, test_data)
    pca_proj = PCAPreprojector(n_components=4)
    pca_proj.fit(calib_data[0])
    
    for act in activations:
        for lr in lrs:
            router = BWS_Router(L=12, G=4, d=4, K=4, activation=act)
            # Use same default lambda_wd=1e-3
            train_router(router, pca_proj, experts, calib_data, lr=lr, lambda_wd=1e-3)
            _, mean_acc = evaluate_router(router, pca_proj, experts, test_data, 'Homogeneous_B256')
            results[act][lr].append(mean_acc)

print("\nDual-Column Results (Mean ± Std):")
print("| Gating Activation | Joint Mean Acc at lr=1e-2 (%) | Joint Mean Acc at lr=5e-2 (%) |")
print("| :--- | :---: | :---: |")
for act in activations:
    m1, s1 = np.mean(results[act][1e-2]), np.std(results[act][1e-2])
    m2, s2 = np.mean(results[act][5e-2]), np.std(results[act][5e-2])
    print(f"| {act} | {m1:.2f} ± {s1:.2f}% | {m2:.2f} ± {s2:.2f}% |")
