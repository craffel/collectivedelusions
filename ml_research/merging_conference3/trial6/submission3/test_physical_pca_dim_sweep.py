import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import json

# Import functions from test_physical_merging.py
from test_physical_merging import (
    set_seed,
    generate_synthetic_dataset,
    train_mlp_experts,
    PhysicalPCAPreprojector,
    PhysicalBWS_Router,
    train_physical_router,
    evaluate_physical_router
)

def fit_sequential_pcas_with_dim(experts, calib_X, d):
    K = len(experts)
    calib_X_t = torch.tensor(calib_X)
    
    # PCA 1 (Inputs)
    pca1 = PhysicalPCAPreprojector(n_components=d)
    pca1.fit(calib_X)
    
    # Propagate with uniform blending to get hidden activations for fitting PCAs
    W1 = torch.stack([experts[k].fc1.weight.data for k in range(K)], dim=0).mean(dim=0)
    B1 = torch.stack([experts[k].fc1.bias.data for k in range(K)], dim=0).mean(dim=0)
    W2 = torch.stack([experts[k].fc2.weight.data for k in range(K)], dim=0).mean(dim=0)
    B2 = torch.stack([experts[k].fc2.bias.data for k in range(K)], dim=0).mean(dim=0)
    
    relu = nn.ReLU()
    with torch.no_grad():
        h1_calib = relu(torch.matmul(calib_X_t, W1.t()) + B1)
        h2_calib = relu(torch.matmul(h1_calib, W2.t()) + B2)
        
    pca2 = PhysicalPCAPreprojector(n_components=d)
    pca2.fit(h1_calib)
    
    pca3 = PhysicalPCAPreprojector(n_components=d)
    pca3.fit(h2_calib)
    
    return [pca1, pca2, pca3]

def main():
    seeds = [42, 43, 44, 45, 46]
    dims = [2, 3, 4, 6, 8, 12, 16]
    
    print("=" * 80)
    print("RUNNING PHYSICAL SEQUENTIAL PCA DIMENSION SWEEP ACROSS 5 SEEDS")
    print("=" * 80)
    
    # Cache experts and datasets for fast execution
    cached_runs = {}
    for seed in seeds:
        print(f"Pre-generating and training experts for Seed {seed}...")
        set_seed(seed)
        train_data, test_data, calib_data = generate_synthetic_dataset(seed)
        experts = train_mlp_experts(train_data, test_data)
        cached_runs[seed] = {
            'train': train_data,
            'test': test_data,
            'calib': calib_data,
            'experts': experts
        }
        
    print("\nCaching completed successfully! Starting the PCA dimension sweep...\n")
    
    sweep_results = {}
    
    for d in dims:
        homo_means = []
        hetero_means = []
        
        for seed in seeds:
            run = cached_runs[seed]
            calib = run['calib']
            test = run['test']
            experts = run['experts']
            
            # Re-seed for router init/training consistency
            set_seed(seed)
            
            # Fit sequential PCAs with target dimension d
            pcas = fit_sequential_pcas_with_dim(experts, calib[0], d)
            
            # Initialize Shared Physical BWS Router (M=3, meaning G=1) with input dimension d
            router = PhysicalBWS_Router(L=3, G=1, d=d, K=4, activation='Sigmoid', init_bias=-2.0)
            
            # Train router
            train_physical_router(router, experts, pcas, calib, lr=0.05, lambda_wd=1e-4)
            
            # Evaluate Homogeneous
            _, homo_mean = evaluate_physical_router(router, experts, pcas, test, mode='Homogeneous_B256')
            homo_means.append(homo_mean)
            
            # Evaluate Heterogeneous
            _, hetero_mean = evaluate_physical_router(router, experts, pcas, test, mode='Heterogeneous_B256')
            hetero_means.append(hetero_mean)
            
        mean_homo = np.mean(homo_means)
        std_homo = np.std(homo_means)
        mean_hetero = np.mean(hetero_means)
        std_hetero = np.std(hetero_means)
        
        sweep_results[d] = {
            'homo': (mean_homo, std_homo),
            'hetero': (mean_hetero, std_hetero)
        }
        
        print(f"d = {d:2d} | Homogeneous Joint Mean: {mean_homo:.2f}% ± {std_homo:.2f}% | Heterogeneous Joint Mean: {mean_hetero:.2f}% ± {std_hetero:.2f}%")
        
    print("\n" + "=" * 80)
    print("FINAL RESULTS TABLE IN MARKDOWN (Physical Sequential Weight-Merging)")
    print("=" * 80)
    print("| PCA Dimension (d) | Homogeneous Joint Mean (%) | Heterogeneous Joint Mean (%) |")
    print("|-------------------|----------------------------|------------------------------|")
    for d in dims:
        homo = sweep_results[d]['homo']
        hetero = sweep_results[d]['hetero']
        print(f"| d = {d:2d} | {homo[0]:.2f}% ± {homo[1]:.2f}% | {hetero[0]:.2f}% ± {hetero[1]:.2f}% |")
    print("=" * 80)
    
    # Save sweep results to json for paper integration if needed
    with open('physical_pca_dim_sweep_results.json', 'w') as f:
        json.dump({str(k): {'homo': f"{v['homo'][0]:.2f} ± {v['homo'][1]:.2f}%", 'hetero': f"{v['hetero'][0]:.2f} ± {v['hetero'][1]:.2f}%"} for k, v in sweep_results.items()}, f, indent=4)
        
if __name__ == '__main__':
    main()
