import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from model_routing import (
    generate_synthetic_dataset,
    train_experts,
    PCAPreprojector,
    BWS_Router,
    evaluate_router,
    set_seed
)

SEEDS = [42, 43, 44, 45, 46]
CALIB_SIZES = [16, 32, 64, 128, 256, 512, 1024]

def generate_custom_calib(train_data, num_samples_per_task, noises, seed=42):
    set_seed(seed)
    # Sampling from train_data to construct larger calibration sets if needed
    K = 4
    calib_X = []
    calib_Y = []
    calib_T = []
    
    for k in range(K):
        X, Y = train_data[k]
        # X shape is [200, 192], Y is [200]
        # Take up to num_samples_per_task
        indices = np.random.choice(len(X), num_samples_per_task, replace=False)
        calib_X.append(X[indices])
        calib_Y.append(Y[indices])
        calib_T.append(np.full(num_samples_per_task, k, dtype=np.int64))
        
    calib_X = np.concatenate(calib_X, axis=0)
    calib_Y = np.concatenate(calib_Y, axis=0)
    calib_T = np.concatenate(calib_T, axis=0)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(calib_X))
    calib_X = calib_X[shuffle_idx]
    calib_Y = calib_Y[shuffle_idx]
    calib_T = calib_T[shuffle_idx]
    
    return calib_X, calib_Y, calib_T

def main():
    print("=" * 60)
    print("RUNNING CALIBRATION SAMPLE COMPLEXITY SWEEP")
    print("=" * 60)
    
    # We will track Joint Mean accuracy across seeds for:
    # 1. Unshared L3-Router (M=1, 240 params)
    # 2. BWS-Router (M=3, 80 params)
    # 3. Global Shared (M=12, 20 params)
    
    configs = {
        'L3_Unshared_M1': {'M': 1, 'G': 12},
        'BWS_Block_M3': {'M': 3, 'G': 4},
        'BWS_Global_M12': {'M': 12, 'G': 1}
    }
    
    noises = [0.001, 0.18, 0.22, 0.8]
    
    results = {cfg: {size: [] for size in CALIB_SIZES} for cfg in configs}
    
    for seed in SEEDS:
        print(f"\n>>> Running for Seed {seed}...")
        train_data, test_data, _ = generate_synthetic_dataset(seed)
        experts = train_experts(train_data, test_data)
        
        for size in CALIB_SIZES:
            samples_per_task = size // 4
            calib_X, calib_Y, calib_T = generate_custom_calib(train_data, samples_per_task, noises, seed)
            calib_data = (calib_X, calib_Y, calib_T)
            
            pca_proj = PCAPreprojector(n_components=4)
            pca_proj.fit(calib_data[0])
            
            for cfg_name, cfg in configs.items():
                router = BWS_Router(L=12, G=cfg['G'], d=4, K=4, activation='Sigmoid')
                # Use optimal tuned hyperparameters: lr=0.05, lambda_wd=1e-4
                from model_routing import train_router
                train_router(router, pca_proj, experts, calib_data, lr=0.05, lambda_wd=1e-4)
                
                _, mean_acc = evaluate_router(router, pca_proj, experts, test_data, 'Homogeneous_B256')
                results[cfg_name][size].append(mean_acc)
                
    # Compile and print results
    print("\n" + "=" * 50)
    print("SAMPLE COMPLEXITY RESULTS (MEAN ± STD)")
    print("=" * 50)
    
    for cfg_name in configs:
        print(f"\nConfiguration: {cfg_name}")
        print("| Calibration Size | Joint Mean Accuracy (%) |")
        print("| :---: | :---: |")
        for size in CALIB_SIZES:
            accs = results[cfg_name][size]
            mean = np.mean(accs)
            std = np.std(accs)
            print(f"| {size} | {mean:.2f} ± {std:.2f}% |")
            
    # Save results as json for easy plotting/loading
    output_data = {cfg_name: {str(size): results[cfg_name][size] for size in CALIB_SIZES} for cfg_name in configs}
    with open('sample_complexity_results.json', 'w') as f:
        json.dump(output_data, f, indent=4)
        
if __name__ == '__main__':
    main()