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

def bures_wasserstein_barycenter_norm(covs, weights, max_iter=30, tol=1e-6, eps=1e-8):
    d = covs[0].shape[0]
    device = covs[0].device
    dtype = covs[0].dtype
    
    avg_trace = sum(torch.trace(c) for c in covs) / len(covs)
    norm_covs = [c / avg_trace for c in covs]
    
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
        sigma = next_sigma
        if diff < tol:
            break
            
    return sigma * avg_trace

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
            
        T_TA = sum(w * Ti for w, Ti in zip(weights, Ti_list))
        cov_TA = (T_TA @ T_TA.T) / d2 + eps * torch.eye(d1)
        
        # Unnormalized T_merged
        cov_star_unnorm = bures_wasserstein_barycenter_unnorm(covs, weights, max_iter=30, eps=eps)
        cov_TA_inv_sqrt = sym_matrix_power(cov_TA, -0.5, eps=eps)
        cov_star_sqrt_unnorm = sym_matrix_power(cov_star_unnorm, 0.5, eps=eps)
        T_merged_unnorm = cov_star_sqrt_unnorm @ cov_TA_inv_sqrt @ T_TA
        
        # Normalized T_merged
        cov_star_norm = bures_wasserstein_barycenter_norm(covs, weights, max_iter=30, eps=eps)
        cov_star_sqrt_norm = sym_matrix_power(cov_star_norm, 0.5, eps=eps)
        T_merged_norm = cov_star_sqrt_norm @ cov_TA_inv_sqrt @ T_TA
        
        # Check norms
        print("Frobenius norm of T_TA:", torch.norm(T_TA).item())
        print("Frobenius norm of T_merged_unnorm:", torch.norm(T_merged_unnorm).item())
        print("Frobenius norm of T_merged_norm:", torch.norm(T_merged_norm).item())
        
        # Correlation (cosine similarity)
        cos_sim = torch.sum(T_merged_unnorm * T_merged_norm) / (torch.norm(T_merged_unnorm) * torch.norm(T_merged_norm))
        print("Cosine similarity between unnorm and norm merged updates:", cos_sim.item())
        
        # Cosine similarity to T_TA
        print("Cosine similarity of T_merged_unnorm to T_TA:", (torch.sum(T_merged_unnorm * T_TA) / (torch.norm(T_merged_unnorm) * torch.norm(T_TA))).item())
        print("Cosine similarity of T_merged_norm to T_TA:", (torch.sum(T_merged_norm * T_TA) / (torch.norm(T_merged_norm) * torch.norm(T_TA))).item())
