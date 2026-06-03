import torch
import torch.nn as nn
from torchvision import models
import os
import copy

def sym_matrix_power(M, power, eps=1e-8):
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    return eigenvectors @ torch.diag(torch.pow(eigenvalues, power)) @ eigenvectors.T

def bures_wasserstein_barycenter_norm(covs, weights, max_iter=30, tol=1e-6, eps=1e-8):
    d = covs[0].shape[0]
    device = covs[0].device
    dtype = covs[0].dtype
    
    # Scale normalization: compute average trace
    avg_trace = sum(torch.trace(c) for c in covs) / len(covs)
    print(f"Original average trace: {avg_trace.item():.2e}")
    
    # Normalize input covariances
    norm_covs = [c / avg_trace for c in covs]
    
    # Initialize with arithmetic mean of normalized covariances
    sigma = torch.zeros((d, d), device=device, dtype=dtype)
    for w, cov in zip(weights, norm_covs):
        sigma += w * cov
        
    for iteration in range(max_iter):
        sigma_sqrt = sym_matrix_power(sigma, 0.5, eps)
        sigma_inv_sqrt = sym_matrix_power(sigma, -0.5, eps)
        
        sum_term = torch.zeros((d, d), device=device, dtype=dtype)
        for w, cov in zip(weights, norm_covs):
            inner = sigma_sqrt @ cov @ sigma_sqrt
            inner_sqrt = sym_matrix_power(inner, 0.5, eps)
            sum_term += w * inner_sqrt
            
        next_sigma = sigma_inv_sqrt @ (sum_term @ sum_term) @ sigma_inv_sqrt
        next_sigma = next_sigma + eps * torch.eye(d, device=device, dtype=dtype)
        
        diff = torch.norm(next_sigma - sigma) / torch.norm(sigma)
        print(f"Iter {iteration+1:02d}: relative diff = {diff.item():.2e}")
        sigma = next_sigma
        if diff < tol:
            print("Converged!")
            break
            
    # Scale back
    return sigma * avg_trace

if __name__ == '__main__':
    print("Loading checkpoints and testing scale-normalized stabilization...")
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_state = base_model.state_dict()
    
    task_names = ['cifar10', 'svhn', 'fashionmnist']
    task_states = []
    for name in task_names:
        chk_path = f"checkpoint_{name}.pt"
        if os.path.exists(chk_path):
            state = torch.load(chk_path, map_location='cpu')
            task_states.append(state)
            
    if len(task_states) == 3:
        weights = [1.0/3.0] * 3
        key = "layer4.1.conv2.weight"
        orig_shape = base_state[key].shape
        d1 = orig_shape[0]
        d2 = base_state[key].numel() // d1
        
        W0 = base_state[key].float().view(d1, d2)
        Wi_list = [state[key].float().view(d1, d2) for state in task_states]
        Ti_list = [Wi - W0 for Wi in Wi_list]
        
        covs = []
        eps = 1e-8
        for Ti in Ti_list:
            cov = (Ti @ Ti.T) / d2 + eps * torch.eye(d1)
            covs.append(cov)
            
        print("--- Running scale-normalized algorithm ---")
        bures_wasserstein_barycenter_norm(covs, weights, max_iter=20, eps=eps)
    else:
        print("Checkpoints not found.")
