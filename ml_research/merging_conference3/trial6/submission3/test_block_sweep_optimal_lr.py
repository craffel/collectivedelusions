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
M_values = [1, 2, 3, 4, 6, 12]

results = {m_val: [] for m_val in M_values}

print("Running block size sweep under optimal lr=5e-2 across 5 seeds...")
for seed in SEEDS:
    set_seed(seed)
    train_data, test_data, calib_data = generate_synthetic_dataset(seed)
    experts = train_experts(train_data, test_data)
    pca_proj = PCAPreprojector(n_components=4)
    pca_proj.fit(calib_data[0])
    
    for m_val in M_values:
        g_val = 12 // m_val
        # Match optimal settings: lr=5e-2, lambda_wd=1e-4
        router = BWS_Router(L=12, G=g_val, d=4, K=4, activation='Sigmoid')
        train_router(router, pca_proj, experts, calib_data, lr=5e-2, lambda_wd=1e-4)
        _, mean_acc = evaluate_router(router, pca_proj, experts, test_data, 'Homogeneous_B256')
        results[m_val].append(mean_acc)

print("\nOptimal LR Block Sweep Results (Mean ± Std):")
print("| Block Size (M) | Total Groups (G) | Trainable Parameters | Joint Mean Acc (%) |")
print("| :---: | :---: | :---: | :---: |")
for m_val in M_values:
    g_val = 12 // m_val
    params = g_val * 4 * 4 + g_val * 4
    m, s = np.mean(results[m_val]), np.std(results[m_val])
    print(f"| {m_val} | {g_val} | {params} | {m:.2f} ± {s:.2f}% |")
