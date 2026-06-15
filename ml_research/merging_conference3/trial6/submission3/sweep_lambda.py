import numpy as np
import torch
import torch.nn as nn
from model_routing import (
    generate_synthetic_dataset,
    train_experts,
    PCAPreprojector,
    BWS_Router,
    train_router,
    evaluate_router,
    set_seed
)

def main():
    SEEDS = [42, 43, 44, 45, 46]
    LAMBDA_VALS = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print("=" * 60)
    print("RUNNING SENSITIVITY SWEEP OVER SCALING CEILING (lambda_max)")
    print("=" * 60)
    
    # Pre-cache datasets and trained experts for each seed to speed up the sweep
    cache = {}
    for seed in SEEDS:
        print(f"Caching datasets and training experts for seed {seed}...")
        set_seed(seed)
        train_data, test_data, calib_data = generate_synthetic_dataset(seed)
        experts = train_experts(train_data, test_data)
        pca_proj = PCAPreprojector(n_components=4)
        pca_proj.fit(calib_data[0])
        cache[seed] = {
            'experts': experts,
            'test_data': test_data,
            'calib_data': calib_data,
            'pca': pca_proj
        }
    
    results = {}
    for l_val in LAMBDA_VALS:
        print(f"\nEvaluating lambda_max = {l_val}...")
        seed_accs = []
        for seed in SEEDS:
            set_seed(seed)
            data = cache[seed]
            router = BWS_Router(L=12, G=4, d=4, K=4, activation='Sigmoid', lambda_max=l_val)
            # Use optimal tuned settings: lr=5e-2, lambda_wd=1e-4
            train_router(router, data['pca'], data['experts'], data['calib_data'], lr=5e-2, lambda_wd=1e-4)
            _, mean_acc = evaluate_router(router, data['pca'], data['experts'], data['test_data'], 'Homogeneous_B256')
            seed_accs.append(mean_acc)
        
        mean_val = np.mean(seed_accs)
        std_val = np.std(seed_accs)
        results[l_val] = (mean_val, std_val)
        print(f"lambda_max = {l_val}: Joint Mean Acc = {mean_val:.2f}% +- {std_val:.2f}%")
        
    print("\n" + "=" * 60)
    print("FINAL RESULTS FOR LA_MAX SWEEP:")
    print("=" * 60)
    for l_val in LAMBDA_VALS:
        m, s = results[l_val]
        print(f"lambda_max = {l_val}: {m:.2f}% +- {s:.2f}%")

if __name__ == '__main__':
    main()
