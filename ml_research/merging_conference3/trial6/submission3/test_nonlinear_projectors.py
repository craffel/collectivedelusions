import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import KernelPCA
from model_routing import (
    generate_synthetic_dataset,
    train_experts,
    PCAPreprojector,
    BWS_Router,
    train_router,
    evaluate_router,
    set_seed
)

# Custom Kernel PCA Preprojector
class KernelPCAPreprojector:
    def __init__(self, n_components=4, kernel='rbf'):
        self.n_components = n_components
        self.kernel = kernel
        # Set eigen_solver='dense' for stable deterministic projection on small datasets
        self.kpca = KernelPCA(n_components=n_components, kernel=kernel, eigen_solver='dense')
        
    def fit(self, X_calib):
        self.kpca.fit(X_calib)
        
    def project(self, X):
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
        proj = self.kpca.transform(X_np)
        psi = torch.tensor(proj, dtype=torch.float32)
        if isinstance(X, torch.Tensor):
            psi = psi.to(X.device)
        psi = psi / (torch.norm(psi, dim=-1, keepdim=True) + 1e-8)
        return psi

SEEDS = [42, 43, 44, 45, 46]
kernels = ['linear', 'rbf', 'cosine', 'poly']

results = {k: [] for k in kernels}

print("Running Unsupervised Projection Kernel Sweep across 5 seeds...")
for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    set_seed(seed)
    train_data, test_data, calib_data = generate_synthetic_dataset(seed)
    experts = train_experts(train_data, test_data)
    
    X_calib = calib_data[0]
    
    for k in kernels:
        if k == 'linear':
            # Standard linear PCA preprojector from model_routing
            pca_proj = PCAPreprojector(n_components=4)
        else:
            # Non-linear Kernel PCA preprojector
            pca_proj = KernelPCAPreprojector(n_components=4, kernel=k)
            
        pca_proj.fit(X_calib)
        
        # Instantiate optimal BWS-Router (M=3, Sigmoid, Reg)
        router = BWS_Router(L=12, G=4, d=4, K=4, activation='Sigmoid')
        
        # Train router under optimal configurations
        train_router(router, pca_proj, experts, calib_data, lr=5e-2, lambda_wd=1e-4)
        
        # Evaluate under Homogeneous B256 mode
        _, mean_acc = evaluate_router(router, pca_proj, experts, test_data, 'Homogeneous_B256')
        results[k].append(mean_acc)
        print(f"Kernel {k:7s} | Joint Mean Acc: {mean_acc:.2f}%")

print("\n=== Kernel Projection Sweep Results (Mean ± Std across 5 seeds) ===")
print("| Projection Kernel | Joint Mean Accuracy (%) |")
print("| :--- | :--- |")
for k in kernels:
    m = np.mean(results[k])
    s = np.std(results[k])
    print(f"| {k.capitalize():10s} | {m:.2f} ± {s:.2f}% |")

# Save the results to json
import json
with open("nonlinear_projector_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results to nonlinear_projector_results.json")
