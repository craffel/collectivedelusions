import torch
import torch.nn as nn
from torchvision import models
import os
import copy
from merge import get_test_loader, evaluate, sym_matrix_power, bures_wasserstein_barycenter

def merge_models_custom_scale(base_state, task_states, weights, wsa_scale, other_scale=1.0, eps=1e-8):
    merged_state = copy.deepcopy(base_state)
    
    for key in base_state.keys():
        if not isinstance(base_state[key], torch.Tensor):
            continue
            
        if base_state[key].dim() < 2 or "weight" not in key:
            update = torch.zeros_like(base_state[key], dtype=torch.float32)
            for w, task_state in zip(weights, task_states):
                task_vector = task_state[key].float() - base_state[key].float()
                update += w * task_vector
            merged_state[key] = base_state[key] + other_scale * update
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
            try:
                cov_star = bures_wasserstein_barycenter(covs, weights, eps=eps)
                cov_TA = (T_TA.T @ T_TA) / d1 + eps * torch.eye(d2, device=device)
                cov_TA_inv_sqrt = sym_matrix_power(cov_TA, -0.5, eps=eps)
                cov_star_sqrt = sym_matrix_power(cov_star, 0.5, eps=eps)
                T_merged = T_TA @ cov_TA_inv_sqrt @ cov_star_sqrt
            except Exception:
                T_merged = T_TA
        else:
            covs = []
            for Ti in Ti_list:
                cov = (Ti @ Ti.T) / d2 + eps * torch.eye(d1, device=device)
                covs.append(cov)
            try:
                cov_star = bures_wasserstein_barycenter(covs, weights, eps=eps)
                cov_TA = (T_TA @ T_TA.T) / d2 + eps * torch.eye(d1, device=device)
                cov_TA_inv_sqrt = sym_matrix_power(cov_TA, -0.5, eps=eps)
                cov_star_sqrt = sym_matrix_power(cov_star, 0.5, eps=eps)
                T_merged = cov_star_sqrt @ cov_TA_inv_sqrt @ T_TA
            except Exception:
                T_merged = T_TA
                
        W_merged = W0 + wsa_scale * T_merged
        merged_state[key] = W_merged.view(orig_shape).to(dtype)
        
    return merged_state

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
        test_loaders = {name: get_test_loader(name) for name in task_names}
        
        # Let's sweep the wsa_scale factor from 1.0 to 2.5
        for wsa_scale in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]:
            print(f"\n--- WSA scale: {wsa_scale} ---")
            merged_state = merge_models_custom_scale(base_state, task_states, weights, wsa_scale)
            
            merged_model = models.resnet18()
            merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)
            merged_model.load_state_dict(merged_state)
            merged_model = merged_model.to(device)
            
            results = {}
            for name in task_names:
                acc = evaluate(merged_model, test_loaders[name], device)
                results[name] = acc
                
            avg_acc = sum(results.values()) / len(results)
            print(f"WSA Scale {wsa_scale} -> CIFAR-10: {results['cifar10']:.2f}%, SVHN: {results['svhn']:.2f}%, F-MNIST: {results['fashionmnist']:.2f}%, Avg: {avg_acc:.2f}%")
