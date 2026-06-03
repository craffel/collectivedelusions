import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms.functional as F_t
import numpy as np

# Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED error
torch.backends.cudnn.enabled = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Helper to load dataset (same as train_experts.py)
def get_dataset(task_name, is_train=True, corruption=None, corruption_severity=0.1):
    os.makedirs('data', exist_ok=True)
    
    # Base transform
    if task_name in ['mnist', 'fmnist']:
        base_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # duplicate channels
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else: # cifar10
        base_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
    # Apply corruption if requested
    if corruption == 'noise':
        # Add gaussian noise
        transform = transforms.Compose([
            base_transform,
            transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * corruption_severity, -2.5, 2.5))
        ])
    elif corruption == 'blur':
        # Apply gaussian blur
        transform = transforms.Compose([
            base_transform,
            transforms.Lambda(lambda x: F_t.gaussian_blur(x, kernel_size=5, sigma=1.5))
        ])
    else:
        transform = base_transform

    if task_name == 'mnist':
        dataset = torchvision.datasets.MNIST('data', train=is_train, download=False, transform=transform)
    elif task_name == 'fmnist':
        dataset = torchvision.datasets.FashionMNIST('data', train=is_train, download=False, transform=transform)
    elif task_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10('data', train=is_train, download=False, transform=transform)
    else:
        raise ValueError(f"Unknown task: {task_name}")
        
    return dataset

# Expert Model Definition
class ExpertModel(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super(ExpertModel, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

# Multi-Task Merged Model Definition
class MergedModel(nn.Module):
    def __init__(self, backbone, heads):
        super(MergedModel, self).__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        
    def forward(self, x, task_name=None):
        features = self.backbone(x)
        if task_name is not None:
            return self.heads[task_name](features)
        return features

# ----------------- Merging Helpers -----------------

def average_tensors(tensors):
    """Helper to safely average tensors of both floating and integer dtypes."""
    if tensors[0].is_floating_point():
        return torch.stack(tensors, dim=0).mean(dim=0)
    else:
        return tensors[0] # Return the first one for integers/tracking buffers

# ----------------- Merging Algorithms -----------------

def merge_wa(progenitor, experts):
    # Standard Weight Averaging (equivalent to TA with lambda=1/K)
    merged = {}
    for key in progenitor.keys():
        vals = [exp[key] for exp in experts]
        merged[key] = average_tensors(vals)
    return merged

def merge_ta(progenitor, experts, lmbda=0.3):
    # Task Arithmetic: W_init + lambda * sum(W_k - W_init)
    merged = {}
    for key in progenitor.keys():
        if progenitor[key].dim() >= 2 and progenitor[key].is_floating_point(): # apply only to weight matrices
            w_init = progenitor[key]
            updates = [exp[key] - w_init for exp in experts]
            merged[key] = w_init + lmbda * torch.stack(updates, dim=0).sum(dim=0)
        else:
            vals = [exp[key] for exp in experts]
            merged[key] = average_tensors(vals)
    return merged

def merge_s_ipr(progenitor, experts):
    merged = {}
    K = len(experts)
    for key in progenitor.keys():
        if 'weight' in key and progenitor[key].dim() >= 2 and progenitor[key].is_floating_point():
            w_init = progenitor[key]
            tau_experts = [exp[key] - w_init for exp in experts]
            tau_merged = torch.stack(tau_experts, dim=0).mean(dim=0)
            
            orig_shape = tau_merged.shape
            out_dim = orig_shape[0]
            tau_merged_2d = tau_merged.view(out_dim, -1)
            tau_exps_2d = [tau_exp.view(out_dim, -1) for tau_exp in tau_experts]
            
            try:
                # Disable cuDNN just for SVD, running on CPU float to ensure stability
                U_m, S_m, V_m_h = torch.linalg.svd(tau_merged_2d.cpu().float(), full_matrices=False)
                
                S_exps = []
                for tau_exp_2d in tau_exps_2d:
                    _, S_e, _ = torch.linalg.svd(tau_exp_2d.cpu().float(), full_matrices=False)
                    S_exps.append(S_e)
                    
                S_avg = torch.stack(S_exps, dim=0).mean(dim=0)
                
                tau_sipr_2d = U_m @ torch.diag(S_avg) @ V_m_h
                tau_sipr = tau_sipr_2d.view(orig_shape).to(tau_merged.device)
                merged[key] = w_init + tau_sipr
            except Exception as e:
                print(f"SVD failed for {key}: {e}. Falling back to WA.")
                merged[key] = w_init + tau_merged
        else:
            vals = [exp[key] for exp in experts]
            merged[key] = average_tensors(vals)
    return merged

def wcpr_align_tensor(merged, experts):
    C = merged.shape[0]
    aligned = torch.zeros_like(merged)
    for c in range(C):
        m_slice = merged[c].view(-1)
        exp_slices = [exp[c].view(-1) for exp in experts]
        
        # Sort experts
        sorted_exps = [torch.sort(slice)[0] for slice in exp_slices]
        # Wasserstein-2 Barycenter (mean of sorted values)
        target_sorted = torch.stack(sorted_exps, dim=0).mean(dim=0)
        
        # Sort merged and get rank
        m_sorted, m_args = torch.sort(m_slice)
        
        # Reconstruct
        m_reconstructed = torch.zeros_like(m_slice)
        m_reconstructed[m_args] = target_sorted
        aligned[c] = m_reconstructed.view(merged[c].shape)
    return aligned

def merge_wcpr(progenitor, experts):
    merged = {}
    for key in progenitor.keys():
        if 'weight' in key and progenitor[key].dim() >= 2 and progenitor[key].is_floating_point():
            w_init = progenitor[key]
            tau_experts = [exp[key] - w_init for exp in experts]
            tau_merged = torch.stack(tau_experts, dim=0).mean(dim=0)
            
            tau_wcpr = wcpr_align_tensor(tau_merged, tau_experts)
            merged[key] = w_init + tau_wcpr
        else:
            vals = [exp[key] for exp in experts]
            merged[key] = average_tensors(vals)
    return merged

def merge_qr_ipr(progenitor, experts, gamma=2.0):
    merged = {}
    K = len(experts)
    for key in progenitor.keys():
        if 'weight' in key and progenitor[key].dim() >= 2 and progenitor[key].is_floating_point():
            w_init = progenitor[key]
            tau_experts = [exp[key] - w_init for exp in experts]
            tau_merged = torch.stack(tau_experts, dim=0).mean(dim=0)
            
            C = tau_merged.shape[0]
            s_c_list = []
            
            for c in range(C):
                norms_exp = [torch.norm(tau_exp[c].float(), p=2).item() for tau_exp in tau_experts]
                ne = sum(norms_exp) / K
                nm = torch.norm(tau_merged[c].float(), p=2).item()
                sc = ne / (nm + 1e-8)
                s_c_list.append(sc)
                
            s_tensor = torch.tensor(s_c_list, device=tau_merged.device)
            median_s = torch.median(s_tensor).item()
            mad_s = torch.median(torch.abs(s_tensor - median_s)).item()
            mad_s = max(mad_s, 1e-4)
            
            L = max(0.1, median_s - gamma * mad_s)
            U = min(4.0, median_s + gamma * mad_s)
            
            s_robust = torch.clamp(s_tensor, L, U)
            view_shape = [C] + [1] * (tau_merged.dim() - 1)
            s_robust_view = s_robust.view(view_shape).to(tau_merged.device)
            
            tau_qr_ipr = s_robust_view * tau_merged
            merged[key] = w_init + tau_qr_ipr
        else:
            vals = [exp[key] for exp in experts]
            merged[key] = average_tensors(vals)
    return merged

# ----------------- Quantization Helpers -----------------

def quantize_symmetric(tensor, bits=8, per_channel=True):
    if not per_channel or tensor.dim() <= 1:
        qmax = tensor.abs().max().item()
        if qmax == 0:
            return tensor
        scale = qmax / (2**(bits-1) - 1)
        val_min = -(2**(bits-1))
        val_max = 2**(bits-1) - 1
        q_tensor = torch.clamp(torch.round(tensor / scale), val_min, val_max)
        return q_tensor * scale
    else:
        q_tensor = torch.zeros_like(tensor)
        val_min = -(2**(bits-1))
        val_max = 2**(bits-1) - 1
        for i in range(tensor.shape[0]):
            slice_t = tensor[i]
            qmax = slice_t.abs().max().item()
            if qmax == 0:
                q_tensor[i] = slice_t
                continue
            scale = qmax / (2**(bits-1) - 1)
            q_tensor[i] = torch.clamp(torch.round(slice_t / scale), val_min, val_max) * scale
        return q_tensor

def quantize_model_weights(model, bits=8, per_channel=True):
    if bits == 32:
        return
    for name, param in model.backbone.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            q_val = quantize_symmetric(param.data, bits=bits, per_channel=per_channel)
            param.data.copy_(q_val)

# ----------------- BatchNorm Calibration -----------------

def calibrate_bn(model, train_loaders, num_samples=32):
    if num_samples == 0:
        return
        
    was_training = model.training
    model.eval()
    
    # Set only BN layers to train mode
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.train()
            m.reset_running_stats()
            
    # Run forward passes with unlabeled data
    with torch.no_grad():
        for task_name, loader in train_loaders.items():
            count = 0
            for images, _ in loader:
                images = images.to(device)
                _ = model(images, task_name=task_name)
                count += images.size(0)
                if count >= num_samples:
                    break
                    
    if not was_training:
        model.eval()

# ----------------- Evaluation Pipeline -----------------

def evaluate_model(model, test_loaders):
    model.eval()
    accs = {}
    with torch.no_grad():
        for task_name, loader in test_loaders.items():
            correct = 0
            total = 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, task_name=task_name)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            accs[task_name] = 100. * correct / total
    return accs

def run_experiments():
    # Load progenitor and experts
    print("\nLoading checkpoints...")
    try:
        progenitor_state = torch.load("checkpoints/progenitor.pt", map_location=device)
        expert_mnist = torch.load("checkpoints/expert_mnist.pt", map_location=device)
        expert_fmnist = torch.load("checkpoints/expert_fmnist.pt", map_location=device)
        expert_cifar10 = torch.load("checkpoints/expert_cifar10.pt", map_location=device)
    except FileNotFoundError:
        print("Expert model checkpoints not found. Please train them first!")
        return

    expert_states = [
        expert_mnist['backbone_state_dict'],
        expert_fmnist['backbone_state_dict'],
        expert_cifar10['backbone_state_dict']
    ]
    
    expert_heads = {
        'mnist': expert_mnist['fc_state_dict'],
        'fmnist': expert_fmnist['fc_state_dict'],
        'cifar10': expert_cifar10['fc_state_dict']
    }

    # Prepare loaders
    tasks = ['mnist', 'fmnist', 'cifar10']
    
    print("Preparing test and train loaders...")
    train_loaders_for_bn = {}
    test_loaders = {}
    test_loaders_noise = {}
    test_loaders_blur = {}
    
    for task in tasks:
        # Train dataset (for DE-BN calibration)
        train_ds = get_dataset(task, is_train=True)
        train_loaders_for_bn[task] = DataLoader(train_ds, batch_size=64, shuffle=True)
        
        # Test datasets
        test_ds = get_dataset(task, is_train=False)
        test_loaders[task] = DataLoader(test_ds, batch_size=256, shuffle=False)
        
        test_ds_noise = get_dataset(task, is_train=False, corruption='noise')
        test_loaders_noise[task] = DataLoader(test_ds_noise, batch_size=256, shuffle=False)
        
        test_ds_blur = get_dataset(task, is_train=False, corruption='blur')
        test_loaders_blur[task] = DataLoader(test_ds_blur, batch_size=256, shuffle=False)

    print("\nReady to run experiment sweep!\n")
    
    # First, let's pre-define merging states for fast evaluation
    merging_states = {}
    
    print("Merging with WA...")
    merging_states['WA'] = merge_wa(progenitor_state, expert_states)
    
    print("Grid searching best lambda for Task Arithmetic on Clean FP32 with 32-sample BN-Cal...")
    best_lambda = 0.3
    best_avg_acc = 0.0
    for lmb in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
        merged_ta_state = merge_ta(progenitor_state, expert_states, lmbda=lmb)
        
        # Instantiate model
        backbone = models.resnet18()
        backbone.fc = nn.Identity()
        backbone.load_state_dict(merged_ta_state)
        
        heads = {}
        for t in tasks:
            head = nn.Linear(512, 10)
            head.load_state_dict(expert_heads[t])
            heads[t] = head
            
        model = MergedModel(backbone, heads).to(device)
        calibrate_bn(model, train_loaders_for_bn, num_samples=32)
        accs = evaluate_model(model, test_loaders)
        avg = sum(accs.values()) / 3.0
        print(f"TA (lambda={lmb:.1f}) -> MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR: {accs['cifar10']:.2f}%, Avg: {avg:.2f}%")
        if avg > best_avg_acc:
            best_avg_acc = avg
            best_lambda = lmb
            
    print(f"Best Task Arithmetic lambda: {best_lambda} (Avg Clean FP32 Acc: {best_avg_acc:.2f}%)")
    merging_states['Tuned TA'] = merge_ta(progenitor_state, expert_states, lmbda=best_lambda)
    
    print("Merging with S-IPR...")
    merging_states['S-IPR'] = merge_s_ipr(progenitor_state, expert_states)
    
    print("Merging with WCPR (Wasserstein alignment)...")
    merging_states['WCPR'] = merge_wcpr(progenitor_state, expert_states)
    
    print("Merging with QR-IPR...")
    merging_states['QR-IPR'] = merge_qr_ipr(progenitor_state, expert_states, gamma=2.0)

    # Output headers
    print("\n" + "="*90)
    print(f"{'Method':<12} | {'Bits':<4} | {'BN-Cal':<6} | {'Env':<5} | {'MNIST':<8} | {'FMNIST':<8} | {'CIFAR10':<8} | {'Average':<8}")
    print("="*90)
    
    results = []

    # Sweep settings
    bits_options = [32, 8, 4]
    bn_options = [0, 16, 32]
    env_options = [
        ('Clean', test_loaders),
        ('Noise', test_loaders_noise),
        ('Blur', test_loaders_blur)
    ]

    for method_name, m_state in merging_states.items():
        for bits in bits_options:
            for bn_samples in bn_options:
                for env_name, env_loaders in env_options:
                    # Deep copy the merged backbone state
                    backbone = models.resnet18()
                    backbone.fc = nn.Identity()
                    backbone.load_state_dict(copy.deepcopy(m_state))
                    
                    heads = {}
                    for t in tasks:
                        head = nn.Linear(512, 10)
                        head.load_state_dict(expert_heads[t])
                        heads[t] = head
                        
                    model = MergedModel(backbone, heads).to(device)
                    
                    # Apply weight quantization
                    quantize_model_weights(model, bits=bits, per_channel=True)
                    
                    # Calibrate BN stats after quantization (crucial for Methodologist theory!)
                    calibrate_bn(model, train_loaders_for_bn, num_samples=bn_samples)
                    
                    # Evaluate
                    accs = evaluate_model(model, env_loaders)
                    avg_acc = sum(accs.values()) / 3.0
                    
                    print(f"{method_name:<12} | {bits:<4} | {bn_samples:<6} | {env_name:<5} | {accs['mnist']:7.2f}% | {accs['fmnist']:7.2f}% | {accs['cifar10']:7.2f}% | {avg_acc:7.2f}%")
                    
                    results.append({
                        'method': method_name,
                        'bits': bits,
                        'bn_samples': bn_samples,
                        'env': env_name,
                        'mnist': accs['mnist'],
                        'fmnist': accs['fmnist'],
                        'cifar10': accs['cifar10'],
                        'average': avg_acc
                    })
                    
    # Save results to json
    import json
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nAll experiments complete. Results saved to experiment_results.json")

if __name__ == "__main__":
    run_experiments()
