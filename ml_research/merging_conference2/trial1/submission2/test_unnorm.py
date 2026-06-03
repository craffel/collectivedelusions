import torch
import torch.nn as nn
from torchvision import models
import os

def sym_matrix_power(M, power, eps=1e-8):
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    return eigenvectors @ torch.diag(torch.pow(eigenvalues, power)) @ eigenvectors.T

def bures_wasserstein_barycenter_unnorm(covs, weights, max_iter=30, tol=1e-6, eps=1e-8):
    d = covs[0].shape[0]
    device = covs[0].device
    dtype = covs[0].dtype
    
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
        next_sigma = next_sigma + eps * torch.eye(d, device=device, dtype=dtype)
        
        diff = torch.norm(next_sigma - sigma) / torch.norm(sigma)
        sigma = next_sigma
        
    return sigma

if __name__ == '__main__':
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
            
        cov_star = bures_wasserstein_barycenter_unnorm(covs, weights, max_iter=30, eps=eps)
        print("Trace of unnormalized cov_star:", torch.trace(cov_star).item())
        
        # Now let's see what happens to the eigenvalues of cov_star
        ev, _ = torch.linalg.eigh(cov_star)
        print("Min eigenvalue of cov_star:", torch.min(ev).item())
        print("Max eigenvalue of cov_star:", torch.max(ev).item())
        
        # Let's compare with scale-normalized cov_star
        from test_norm import bures_wasserstein_barycenter_norm
        cov_star_norm = bures_wasserstein_barycenter_norm(covs, weights, max_iter=30, eps=eps)
        print("Trace of scale-normalized cov_star:", torch.trace(cov_star_norm).item())
        ev_norm, _ = torch.linalg.eigh(cov_star_norm)
        print("Min eigenvalue of scale-normalized cov_star:", torch.min(ev_norm).item())
        print("Max eigenvalue of scale-normalized cov_star:", torch.max(ev_norm).item())
