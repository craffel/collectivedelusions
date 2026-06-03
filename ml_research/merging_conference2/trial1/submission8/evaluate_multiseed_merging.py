import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# Import components from merge.py
from merge import set_seed

# Highly optimized caching layer to avoid redundant SVD and Procrustes calculations
_DECOUPLING_CACHE = {}
_SPECTRAL_WEIGHTS_CACHE = {}

def get_spectral_weights_cached(delta_weights, method="entropy", gamma=1.0, cache_key=None):
    if cache_key in _SPECTRAL_WEIGHTS_CACHE:
        return _SPECTRAL_WEIGHTS_CACHE[cache_key]
    from merge import get_spectral_weights
    weights = get_spectral_weights(delta_weights, method=method, gamma=gamma)
    _SPECTRAL_WEIGHTS_CACHE[cache_key] = weights
    return weights

def merge_models_cached(base_state_dict, task_state_dicts, method="task_arithmetic", spectral_method="uniform", gamma=1.0, reg_factor=1.0, spectral_rotation=False, seed=None, epsilon=1e-6):
    if method == "task_arithmetic":
        from merge import merge_models
        return merge_models(base_state_dict, task_state_dicts, method="task_arithmetic")
        
    merged_state_dict = {}
    N = len(task_state_dicts)
    keys_to_merge = [k for k in base_state_dict.keys() if not k.startswith("fc.")]
    
    for k in base_state_dict.keys():
        if k not in keys_to_merge:
            merged_state_dict[k] = base_state_dict[k]
            continue
            
        tensor_shape = base_state_dict[k].shape
        if len(tensor_shape) < 2:
            avg_delta = torch.zeros_like(base_state_dict[k], dtype=torch.float32)
            for t_sd in task_state_dicts:
                avg_delta += (t_sd[k].float() - base_state_dict[k].float()) / N
            merged_state_dict[k] = base_state_dict[k] + avg_delta
            continue
            
        W0 = base_state_dict[k].float()
        C_out = tensor_shape[0]
        
        # Check cache for R_list, Q_list, rho_list, delta_W_list
        decouple_key = (seed, reg_factor, k)
        if decouple_key in _DECOUPLING_CACHE:
            R_list, Q_list, rho_list, delta_W_list = _DECOUPLING_CACHE[decouple_key]
        else:
            R_list = []
            Q_list = []
            rho_list = []
            delta_W_list = []
            
            for t_sd in task_state_dicts:
                Wi = t_sd[k].float()
                delta_Wi = Wi - W0
                delta_W_list.append(delta_Wi)
                
                W0_2d = W0.view(C_out, -1)
                Wi_2d = Wi.view(C_out, -1)
                
                target = torch.matmul(Wi_2d, W0_2d.t()) + reg_factor * torch.eye(C_out, device=W0.device)
                try:
                    U, Sigma, V_t = torch.linalg.svd(target, full_matrices=False)
                    R = torch.matmul(U, V_t)
                except Exception as e:
                    R = torch.eye(C_out, device=W0.device)
                
                I = torch.eye(C_out, device=W0.device)
                try:
                    R_plus_I_inv = torch.linalg.inv(R + I + epsilon * I)
                    Q = torch.matmul(R - I, R_plus_I_inv)
                except Exception as e:
                    Q = torch.zeros_like(R)
                    
                R_W0_2d = torch.matmul(R, W0_2d)
                rho_2d = Wi_2d - R_W0_2d
                rho = rho_2d.view_as(W0)
                
                R_list.append(R)
                Q_list.append(Q)
                rho_list.append(rho)
                
            _DECOUPLING_CACHE[decouple_key] = (R_list, Q_list, rho_list, delta_W_list)
            
        if spectral_method != "uniform":
            spec_key = (seed, k, spectral_method, gamma)
            alpha_list = get_spectral_weights_cached(delta_W_list, method=spectral_method, gamma=gamma, cache_key=spec_key)
        else:
            alpha_list = [1.0 / N] * N
            
        avg_Q = torch.zeros_like(Q_list[0])
        if spectral_rotation:
            for alpha_i, Q in zip(alpha_list, Q_list):
                avg_Q += alpha_i * Q
        else:
            for Q in Q_list:
                avg_Q += Q / N
                
        sum_norm_Q = sum(Q.norm() for Q in Q_list)
        avg_Q_norm = avg_Q.norm() + 1e-12
        c_scale = sum_norm_Q / avg_Q_norm
        Q_merged = c_scale * avg_Q
        
        try:
            I = torch.eye(C_out, device=W0.device)
            R_merged = torch.matmul(I + Q_merged, torch.linalg.inv(I - Q_merged + epsilon * I))
        except Exception as e:
            R_merged = torch.zeros_like(R_list[0])
            for R in R_list:
                R_merged += R / N
                
        rho_merged = torch.zeros_like(W0)
        for alpha_i, rho_i in zip(alpha_list, rho_list):
            rho_merged += alpha_i * rho_i
            
        W0_2d = W0.view(C_out, -1)
        merged_2d = torch.matmul(R_merged, W0_2d) + rho_merged.view(C_out, -1)
        merged_state_dict[k] = merged_2d.view_as(W0)
        
    return merged_state_dict

def get_fast_dataloader(dataset_name, batch_size, data_dir="./data"):
    # Extremely lightweight CPU transforms (no resize, no grayscale conversion, no normalization)
    transform_simple = transforms.ToTensor()
    
    if dataset_name == "MNIST":
        test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=transform_simple)
    elif dataset_name == "FashionMNIST":
        test_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=transform_simple)
    elif dataset_name == "CIFAR10":
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_simple)
    elif dataset_name == "SVHN":
        test_dataset = torchvision.datasets.SVHN(root=data_dir, split='test', download=False, transform=transform_simple)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return test_loader

def evaluate_model_fast(model, test_loader, dataset_name, device):
    model.eval()
    correct = 0
    total = 0
    
    # ImageNet normalization parameters on GPU
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Grayscale datasets have shape (B, 1, 28, 28), repeat to 3 channels
            if dataset_name in ["MNIST", "FashionMNIST"]:
                images = images.repeat(1, 3, 1, 1)
                
            # Resize on GPU using bilinear interpolation
            images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Normalize on GPU
            images = (images - mean) / std
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100.0 * correct / total

def run_merge_and_eval_in_memory(method, spectral_method, gamma, reg_factor, spectral_rotation, seed, base_state_dict, checkpoints_by_seed, dataloaders, eval_model, device):
    # Prepare base state dict on device
    base_sd_device = {k: v.to(device) for k, v in base_state_dict.items()}
    
    # Prepare task state dicts and heads on device
    task_state_dicts = []
    task_heads = {}
    datasets = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]
    
    for d in datasets:
        ckpt = checkpoints_by_seed[seed][d]
        sd = ckpt['state_dict']
        # Clone to prevent modifying the pre-loaded checkpoints
        sd_device = {k: v.clone().to(device) for k, v in sd.items()}
        task_state_dicts.append(sd_device)
        task_heads[d] = {
            "weight": sd_device["fc.weight"].clone(),
            "bias": sd_device["fc.bias"].clone()
        }
        
    # Merge backbones (uses cached decomposition)
    merged_backbone_sd = merge_models_cached(
        base_sd_device, 
        task_state_dicts, 
        method=method, 
        spectral_method=spectral_method, 
        gamma=gamma,
        reg_factor=reg_factor,
        spectral_rotation=spectral_rotation,
        seed=seed
    )
    
    # Evaluate merged model on each dataset
    accuracies = {}
    for d in datasets:
        # Load merged backbone
        eval_model.load_state_dict(merged_backbone_sd)
        # Restore task-specific classification head
        eval_model.fc.weight.data.copy_(task_heads[d]["weight"])
        eval_model.fc.bias.data.copy_(task_heads[d]["bias"])
        
        acc = evaluate_model_fast(eval_model, dataloaders[d], d, device)
        accuracies[d] = acc
        
    accuracies["Average"] = np.mean([accuracies[d] for d in datasets])
    return accuracies

def run_all_seeds_in_memory(method, spectral_method, gamma, reg_factor, spectral_rotation, seeds, base_state_dict, checkpoints_by_seed, dataloaders, eval_model, device):
    seed_results = []
    for seed in seeds:
        res = run_merge_and_eval_in_memory(
            method, spectral_method, gamma, reg_factor, spectral_rotation, 
            seed, base_state_dict, checkpoints_by_seed, dataloaders, eval_model, device
        )
        if res:
            seed_results.append(res)
            
    if not seed_results:
        return None
        
    # Aggregate across seeds
    aggregated = {}
    keys = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN", "Average"]
    for key in keys:
        vals = [r[key] for r in seed_results if key in r]
        if vals:
            aggregated[f"{key}_mean"] = np.mean(vals)
            aggregated[f"{key}_std"] = np.std(vals)
            
    return aggregated

def main():
    set_seed(42)
    print("==========================================================")
    print("   EVALUATING MULTI-SEED SPECTRAL-AWARE ORTHOMERGE        ")
    print("==========================================================\n")
    
    # Handle device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for all merging and evaluations.")
    
    seeds = [42, 100, 2026]
    datasets = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]
    
    # 1. Pre-load base model
    print("Pre-loading base pre-trained ResNet-18 model...")
    base_model = resnet18(weights=None)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_state_dict = base_model.state_dict()
    
    # 2. Pre-load all task checkpoints
    print("Pre-loading all task checkpoints across all seeds...")
    checkpoints_by_seed = {}
    for seed in seeds:
        checkpoints_by_seed[seed] = {}
        for d in datasets:
            ckpt_path = os.path.join("./checkpoints", f"{d}_seed{seed}.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Please train all seeds first.")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            checkpoints_by_seed[seed][d] = ckpt
            
    # 3. Pre-load all fast dataloaders
    print("Pre-loading all test dataloaders (shared across seeds, optimized for GPU)...")
    dataloaders = {}
    for d in datasets:
        dataloaders[d] = get_fast_dataloader(d, 256)
        
    # 4. Initialize evaluation model on device
    eval_model = resnet18(weights=None)
    eval_model.fc = nn.Linear(eval_model.fc.in_features, 10)
    eval_model = eval_model.to(device)
    
    # 5. Baseline: Task Arithmetic
    print("\nRunning Task Arithmetic baseline across seeds...")
    ta_results = run_all_seeds_in_memory(
        "task_arithmetic", "uniform", 1.0, 1.0, False, 
        seeds, base_state_dict, checkpoints_by_seed, dataloaders, eval_model, device
    )
    
    reg_values = [10.0, 50.0, 100.0]
    
    # Print the table header
    print("\n" + "="*145)
    print(f"{'Method / Configuration':<45} | {'Reg':<6} | {'Gamma':<6} | {'Rot':<5} | {'MNIST':<15} | {'Fashion':<15} | {'CIFAR10':<15} | {'SVHN':<15} | {'Average':<15}")
    print("="*145)
    
    def print_row(name, reg, g, rot, res):
        if res:
            m_mnist, s_mnist = res.get('MNIST_mean', 0.0), res.get('MNIST_std', 0.0)
            m_fash, s_fash = res.get('FashionMNIST_mean', 0.0), res.get('FashionMNIST_std', 0.0)
            m_c10, s_c10 = res.get('CIFAR10_mean', 0.0), res.get('CIFAR10_std', 0.0)
            m_svhn, s_svhn = res.get('SVHN_mean', 0.0), res.get('SVHN_std', 0.0)
            m_avg, s_avg = res.get('Average_mean', 0.0), res.get('Average_std', 0.0)
            
            print(f"{name:<45} | {reg:<6} | {g:<6} | {rot:<5} | "
                  f"{m_mnist:>6.2f} ± {s_mnist:<5.2f} | "
                  f"{m_fash:>6.2f} ± {s_fash:<5.2f} | "
                  f"{m_c10:>6.2f} ± {s_c10:<5.2f} | "
                  f"{m_svhn:>6.2f} ± {s_svhn:<5.2f} | "
                  f"{m_avg:>6.2f} ± {s_avg:<5.2f}")
            
    print_row("Task Arithmetic (Euclidean)", "-", "-", "-", ta_results)
    
    # Standard OrthoMerge (Uniform)
    for reg in reg_values:
        res = run_all_seeds_in_memory("orthomerge", "uniform", 1.0, reg, False, seeds, base_state_dict, checkpoints_by_seed, dataloaders, eval_model, device)
        print_row("OrthoMerge (Standard Uniform)", str(reg), "1.0", "Unif", res)
    print("-"*145)
    
    # SEW-OrthoMerge (Residuals-Only, Dominance)
    print("Running SEW-OrthoMerge (Residuals-Only, Dominance)...")
    for reg in reg_values:
        for g in [1.0, 2.0, 5.0]:
            res = run_all_seeds_in_memory("orthomerge", "dominance", g, reg, False, seeds, base_state_dict, checkpoints_by_seed, dataloaders, eval_model, device)
            print_row("SEW-OrthoMerge (Residuals, Dominance)", str(reg), str(g), "Unif", res)
    print("-"*145)
            
    # SEW-OrthoMerge (Residuals-Only, Entropy)
    print("Running SEW-OrthoMerge (Residuals-Only, Entropy)...")
    for reg in reg_values:
        for g in [1.0, 2.0]:
            res = run_all_seeds_in_memory("orthomerge", "entropy", g, reg, False, seeds, base_state_dict, checkpoints_by_seed, dataloaders, eval_model, device)
            print_row("SEW-OrthoMerge (Residuals, Entropy)", str(reg), str(g), "Unif", res)
    print("-"*145)
            
    # SEW-OrthoMerge (Joint, Dominance)
    print("Running SEW-OrthoMerge (Joint, Dominance)...")
    for reg in reg_values:
        for g in [1.0, 2.0, 5.0]:
            res = run_all_seeds_in_memory("orthomerge", "dominance", g, reg, True, seeds, base_state_dict, checkpoints_by_seed, dataloaders, eval_model, device)
            print_row("SEW-OrthoMerge (Joint, Dominance)", str(reg), str(g), "Spec", res)
    print("-"*145)
            
    # SEW-OrthoMerge (Joint, Entropy)
    print("Running SEW-OrthoMerge (Joint, Entropy)...")
    for reg in reg_values:
        for g in [1.0, 2.0]:
            res = run_all_seeds_in_memory("orthomerge", "entropy", g, reg, True, seeds, base_state_dict, checkpoints_by_seed, dataloaders, eval_model, device)
            print_row("SEW-OrthoMerge (Joint, Entropy)", str(reg), str(g), "Spec", res)
    print("="*145)

if __name__ == "__main__":
    main()
