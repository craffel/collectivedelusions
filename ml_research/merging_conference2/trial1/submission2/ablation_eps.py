import torch
import torch.nn as nn
from torchvision import models
import json
import os
import copy
from merge import get_test_loader, evaluate, sym_matrix_power

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED error
torch.backends.cudnn.enabled = False

def bures_wasserstein_barycenter_logged(covs, weights, max_iter=30, tol=1e-6, eps=1e-8):
    d = covs[0].shape[0]
    device = covs[0].device
    dtype = covs[0].dtype
    
    # Initialize with arithmetic mean
    sigma = torch.zeros((d, d), device=device, dtype=dtype)
    for w, cov in zip(weights, covs):
        sigma += w * cov
        
    history = []
    for iteration in range(max_iter):
        sigma_sqrt = sym_matrix_power(sigma, 0.5, eps)
        sigma_inv_sqrt = sym_matrix_power(sigma, -0.5, eps)
        
        sum_term = torch.zeros((d, d), device=device, dtype=dtype)
        for w, cov in zip(weights, covs):
            inner = sigma_sqrt @ cov @ sigma_sqrt
            inner_sqrt = sym_matrix_power(inner, 0.5, eps)
            sum_term += w * inner_sqrt
            
        next_sigma = sigma_inv_sqrt @ (sum_term @ sum_term) @ sigma_inv_sqrt
        # Add stabilization to maintain positive-definiteness and prevent zero eigenvalues
        next_sigma = next_sigma + eps * torch.eye(d, device=device, dtype=dtype)
        
        diff = torch.norm(next_sigma - sigma) / torch.norm(sigma)
        history.append(diff.item())
        sigma = next_sigma
        if diff < tol:
            break
            
    return sigma, history

def merge_models_eps(base_state, task_states, weights, eps, target_layer_key):
    merged_state = copy.deepcopy(base_state)
    target_history = []
    
    for key in base_state.keys():
        if not isinstance(base_state[key], torch.Tensor):
            continue
            
        if base_state[key].dim() < 2 or "weight" not in key:
            update = torch.zeros_like(base_state[key], dtype=torch.float32)
            for w, task_state in zip(weights, task_states):
                task_vector = task_state[key].float() - base_state[key].float()
                update += w * task_vector
            merged_state[key] = base_state[key] + update
            merged_state[key] = merged_state[key].to(base_state[key].dtype)
            continue
            
        orig_shape = base_state[key].shape
        device = base_state[key].device
        dtype = base_state[key].dtype
        
        d1 = orig_shape[0]
        d2 = base_state[key].numel() // d1
        
        W0 = base_state[key].float().view(d1, d2)
        Wi_list = [task_state[key].float().view(d1, d2) for task_state in task_states]
        Ti_list = [Wi - W0 for Wi in Wi_list]
        
        T_TA = torch.zeros_like(W0)
        for w, Ti in zip(weights, Ti_list):
            T_TA += w * Ti
            
        if d2 <= d1:
            covs = []
            for Ti in Ti_list:
                cov = (Ti.T @ Ti) / d1 + eps * torch.eye(d2, device=device)
                covs.append(cov)
            
            if key == target_layer_key:
                cov_star, hist = bures_wasserstein_barycenter_logged(covs, weights, eps=eps)
                target_history = hist
            else:
                cov_star, _ = bures_wasserstein_barycenter_logged(covs, weights, eps=eps)
                
            cov_TA = (T_TA.T @ T_TA) / d1 + eps * torch.eye(d2, device=device)
            cov_TA_inv_sqrt = sym_matrix_power(cov_TA, -0.5, eps=eps)
            cov_star_sqrt = sym_matrix_power(cov_star, 0.5, eps=eps)
            T_merged = T_TA @ cov_TA_inv_sqrt @ cov_star_sqrt
        else:
            covs = []
            for Ti in Ti_list:
                cov = (Ti @ Ti.T) / d2 + eps * torch.eye(d1, device=device)
                covs.append(cov)
                
            if key == target_layer_key:
                cov_star, hist = bures_wasserstein_barycenter_logged(covs, weights, eps=eps)
                target_history = hist
            else:
                cov_star, _ = bures_wasserstein_barycenter_logged(covs, weights, eps=eps)
                
            cov_TA = (T_TA @ T_TA.T) / d2 + eps * torch.eye(d1, device=device)
            cov_TA_inv_sqrt = sym_matrix_power(cov_TA, -0.5, eps=eps)
            cov_star_sqrt = sym_matrix_power(cov_star, 0.5, eps=eps)
            T_merged = cov_star_sqrt @ cov_TA_inv_sqrt @ T_TA
            
        W_merged = W0 + T_merged
        merged_state[key] = W_merged.view(orig_shape).to(dtype)
        
    return merged_state, target_history

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading pretrained ResNet-18 base model...")
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_state = base_model.state_dict()
    
    task_names = ['cifar10', 'svhn', 'fashionmnist']
    task_states = []
    for name in task_names:
        chk_path = f"checkpoint_{name}.pt"
        if not os.path.exists(chk_path):
            raise FileNotFoundError(f"Checkpoint not found: {chk_path}")
        state = torch.load(chk_path, map_location='cpu')
        task_states.append(state)
        
    weights = [1.0 / len(task_names)] * len(task_names)
    
    # We choose a key layer to track convergence: layer4.1.conv2.weight
    # This is a convolutional layer of shape [512, 512, 3, 3] -> 512 x 4608
    target_layer_key = "layer4.1.conv2.weight"
    
    eps_values = [1e-8, 1e-6, 1e-4, 1e-2]
    ablation_results = {}
    
    # Get test loaders
    test_loaders = {name: get_test_loader(name) for name in task_names}
    
    for eps in eps_values:
        print(f"\nEvaluating WSA with regularization eps = {eps}...")
        merged_state, hist = merge_models_eps(base_state, task_states, weights, eps, target_layer_key)
        
        # Load state dict
        merged_model = models.resnet18()
        merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)
        merged_model.load_state_dict(merged_state)
        merged_model = merged_model.to(device)
        
        results = {}
        for name in task_names:
            acc = evaluate(merged_model, test_loaders[name], device)
            results[name] = acc
            print(f"Accuracy on {name} with eps={eps}: {acc:.2f}%")
            
        avg_acc = sum(results.values()) / len(results)
        print(f"Average Multi-task Accuracy with eps={eps}: {avg_acc:.2f}%")
        
        ablation_results[str(eps)] = {
            "accuracies": results,
            "average_accuracy": avg_acc,
            "convergence_history": hist
        }
        
    with open("ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=4)
        
    print("\nAblation study completed and results saved to ablation_results.json!")

if __name__ == '__main__':
    main()
