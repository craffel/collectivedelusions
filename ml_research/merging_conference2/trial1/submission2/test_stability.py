import torch
import torch.nn as nn
from torchvision import models
import os
import copy

# Helper function to compute matrix power of symmetric positive-definite matrices
def sym_matrix_power(M, power, eps=1e-8):
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    return eigenvectors @ torch.diag(torch.pow(eigenvalues, power)) @ eigenvectors.T

def bures_wasserstein_barycenter_stabilized(covs, weights, max_iter=20, tol=1e-6, eps=1e-8):
    d = covs[0].shape[0]
    device = covs[0].device
    dtype = covs[0].dtype
    
    # Initialize with arithmetic mean
    sigma = torch.zeros((d, d), device=device, dtype=dtype)
    for w, cov in zip(weights, covs):
        sigma += w * cov
        
    for iteration in range(max_iter):
        sigma_sqrt = sym_matrix_power(sigma, 0.5, eps)
        sigma_inv_sqrt = sym_matrix_power(sigma, -0.5, eps)
        
        sum_term = torch.zeros((d, d), device=device, dtype=dtype)
        for w, cov in zip(weights, covs):
            inner = sigma_sqrt @ cov @ sigma_sqrt
            inner_sqrt = sym_matrix_power(inner, 0.5, eps)
            sum_term += w * inner_sqrt
            
        next_sigma = sigma_inv_sqrt @ (sum_term @ sum_term) @ sigma_inv_sqrt
        # Add regularization to maintain positive-definiteness
        next_sigma = next_sigma + eps * torch.eye(d, device=device, dtype=dtype)
        
        diff = torch.norm(next_sigma - sigma) / torch.norm(sigma)
        print(f"Iter {iteration+1:02d}: relative diff = {diff.item():.2e}")
        sigma = next_sigma
        if diff < tol:
            print("Converged!")
            break
            
    return sigma

if __name__ == '__main__':
    print("Loading checkpoints and testing stabilization on layer4.1.conv2.weight...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_model = models.resnet18()
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
        # Column covariance since d2 > d1 (512 x 4608)
        eps = 1e-8
        for Ti in Ti_list:
            cov = (Ti @ Ti.T) / d2 + eps * torch.eye(d1)
            covs.append(cov)
            
        print("--- Running original (without next_sigma stabilization) ---")
        # Run original
        sigma = torch.zeros((d1, d1))
        for w, cov in zip(weights, covs):
            sigma += w * cov
        for iteration in range(10):
            sigma_sqrt = sym_matrix_power(sigma, 0.5, eps)
            sigma_inv_sqrt = sym_matrix_power(sigma, -0.5, eps)
            sum_term = torch.zeros((d1, d1))
            for w, cov in zip(weights, covs):
                inner = sigma_sqrt @ cov @ sigma_sqrt
                inner_sqrt = sym_matrix_power(inner, 0.5, eps)
                sum_term += w * inner_sqrt
            next_sigma = sigma_inv_sqrt @ (sum_term @ sum_term) @ sigma_inv_sqrt
            diff = torch.norm(next_sigma - sigma) / torch.norm(sigma)
            print(f"Iter {iteration+1:02d}: relative diff = {diff.item():.2e}")
            sigma = next_sigma
            
        print("\n--- Running STABILIZED (with next_sigma stabilization) ---")
        bures_wasserstein_barycenter_stabilized(covs, weights, max_iter=20, eps=eps)
    else:
        print("Checkpoints not found, cannot test on real weights.")
