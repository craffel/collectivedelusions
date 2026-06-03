import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import os
import copy
import random
import numpy as np

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Disable cuDNN to bypass driver compatibility issues on the cluster
    torch.backends.cudnn.enabled = False

# Define MLP architecture
class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.out = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.out(x)

# Define ResNet18 Wrapper for Task Specific Head
class MergedResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(MergedResNet18, self).__init__()
        # Load backbone
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Save output dimension and replace fc with identity
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Linear head
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

def get_model(arch, num_classes=10):
    if arch.lower() == 'resnet18':
        return MergedResNet18(num_classes)
    elif arch.lower() == 'mlp':
        return MLP(num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

# Data Loading functions
def get_dataloaders(dataset_name, batch_size=256, root='./data'):
    os.makedirs(root, exist_ok=True)
    
    # Preprocessing transforms
    if dataset_name.lower() in ['mnist', 'fmnist']:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # Duplicate grayscale channel to 3 channels
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if dataset_name.lower() == 'mnist':
        train_set = datasets.MNIST(root, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root, train=False, download=True, transform=transform)
    elif dataset_name.lower() == 'fmnist':
        train_set = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root, train=False, download=True, transform=transform)
    elif dataset_name.lower() == 'cifar10':
        train_set = datasets.CIFAR10(root, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# Training loop
def train_expert(model, train_loader, epochs=5, lr=1e-3, weight_decay=1e-4, device='cuda'):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

# Evaluation loop
def evaluate_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return 100.0 * correct / total

# Post-Training Quantization
def quantize_model(model, num_bits=8, per_channel=False):
    if num_bits is None:
        return model
    
    q_model = copy.deepcopy(model)
    qmax = 2**(num_bits - 1) - 1
    
    with torch.no_grad():
        for name, param in q_model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                tensor = param.data
                if not per_channel:
                    # Per-tensor quantization
                    max_val = torch.max(torch.abs(tensor))
                    if max_val > 0:
                        scale = max_val / qmax
                        param.copy_(torch.clamp(torch.round(tensor / scale), -qmax, qmax) * scale)
                else:
                    # Per-channel quantization along first dim
                    quant = tensor.clone()
                    for c in range(tensor.size(0)):
                        max_val = torch.max(torch.abs(tensor[c]))
                        if max_val > 0:
                            scale = max_val / qmax
                            quant[c] = torch.clamp(torch.round(tensor[c] / scale), -qmax, qmax) * scale
                    param.copy_(quant)
    return q_model

# Data-Efficient BatchNorm Calibration (DE-BN)
def calibrate_bn(model, calibration_datasets, num_samples_per_task=16, device='cuda'):
    model.to(device)
    model.train()
    
    # Reset all running stats in BatchNorm layers
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None # Cumulative average estimation
            
    # Pool calibration samples across tasks
    calib_samples = []
    for dataset in calibration_datasets:
        # Randomly select samples
        indices = np.random.choice(len(dataset), min(num_samples_per_task, len(dataset)), replace=False)
        for idx in indices:
            img, _ = dataset[idx]
            calib_samples.append(img)
            
    # Shuffle joint samples
    random.shuffle(calib_samples)
    
    # Run forward passes in batches
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(calib_samples), batch_size):
            batch = calib_samples[i:i+batch_size]
            batch_tensor = torch.stack(batch).to(device)
            model(batch_tensor)
            
    model.eval()

# Model Merging Algorithms

# Weight Averaging
def merge_wa(progenitor, experts):
    merged = copy.deepcopy(progenitor)
    K = len(experts)
    
    with torch.no_grad():
        for name, param in merged.named_parameters():
            param_device = param.device
            expert_params = [exp.state_dict()[name].to(param_device) for exp in experts]
            param.copy_(sum(expert_params) / K)
            
    return merged

# Task Arithmetic
def merge_ta(progenitor, experts, scaling_factor=0.3):
    merged = copy.deepcopy(progenitor)
    K = len(experts)
    
    with torch.no_grad():
        for name, param in merged.named_parameters():
            param_device = param.device
            p_val = progenitor.state_dict()[name].to(param_device)
            # Get expert updates
            updates = []
            for exp in experts:
                exp_val = exp.state_dict()[name].to(param_device)
                update = exp_val - p_val
                updates.append(update)
            # Apply scaling
            merged_update = scaling_factor * (sum(updates) / K)
            param.copy_(p_val + merged_update)
            
    return merged

# Helper to get sparsification mask
def get_ties_mask(tensor, sparsity_ratio):
    if sparsity_ratio == 0:
        return torch.ones_like(tensor, dtype=torch.bool)
    
    flat = tensor.flatten()
    k = int(sparsity_ratio * flat.numel())
    if k == 0:
        return torch.ones_like(tensor, dtype=torch.bool)
    if k >= flat.numel():
        return torch.zeros_like(tensor, dtype=torch.bool)
        
    threshold = torch.kthvalue(torch.abs(flat), k).values
    return torch.abs(tensor) > threshold

# TIES-Merging and DARE-Merging
def sparsify_and_merge_task_vectors(progenitor, experts, method='ties', sparsity_ratio=0.2):
    K = len(experts)
    merged_state = copy.deepcopy(progenitor.state_dict())
    
    with torch.no_grad():
        for name in merged_state.keys():
            param_device = merged_state[name].device
            p_val = progenitor.state_dict()[name].to(param_device)
            
            if not torch.is_floating_point(p_val):
                continue
                
            # Get task vectors
            tvs = []
            for exp in experts:
                exp_val = exp.state_dict()[name].to(param_device)
                tv = exp_val - p_val
                tvs.append(tv)
                
            # Step 1: Sparsify
            sparse_tvs = []
            for tv in tvs:
                if method == 'ties':
                    mask = get_ties_mask(tv, sparsity_ratio)
                    sparse_tvs.append(tv * mask)
                elif method == 'dare':
                    # DARE drops updates randomly and rescales
                    mask = (torch.rand_like(tv) > sparsity_ratio)
                    rescaled_tv = (tv * mask) / (1.0 - sparsity_ratio)
                    sparse_tvs.append(rescaled_tv)
                else:
                    sparse_tvs.append(tv)
                    
            # Step 2: Resolve Sign Conflicts / Merge
            if method == 'ties' and K > 1:
                # Majority sign
                signs = torch.stack([torch.sign(stv) for stv in sparse_tvs])
                mag_sum_pos = torch.stack([stv * (stv > 0).float() for stv in sparse_tvs]).sum(dim=0)
                mag_sum_neg = torch.stack([torch.abs(stv) * (stv < 0).float() for stv in sparse_tvs]).sum(dim=0)
                
                majority_sign = (mag_sum_pos >= mag_sum_neg).float() * 2.0 - 1.0 # +1 or -1
                
                # Keep only values matching majority sign
                filtered_tvs = []
                for stv in sparse_tvs:
                    filtered_tvs.append(stv * ((torch.sign(stv) == majority_sign) | (stv == 0)).float())
                    
                # Count sign-compatible experts
                agree_counts = torch.stack([(ftv != 0).float() for ftv in filtered_tvs]).sum(dim=0)
                sum_values = torch.stack(filtered_tvs).sum(dim=0)
                
                # Avoid division by zero
                merged_tv = torch.where(agree_counts > 0, sum_values / agree_counts, torch.zeros_like(sum_values))
            else:
                # DARE or standard WA of updates
                merged_tv = sum(sparse_tvs) / K
                
            merged_state[name] = p_val + merged_tv
            
    merged = copy.deepcopy(progenitor)
    merged.load_state_dict(merged_state)
    return merged

# WCPR - Wasserstein-Calibrated Parameter Resonance
def merge_wcpr(progenitor, experts):
    K = len(experts)
    merged_state = copy.deepcopy(progenitor.state_dict())
    
    with torch.no_grad():
        for name in merged_state.keys():
            param_device = merged_state[name].device
            p_val = progenitor.state_dict()[name].to(param_device)
            
            if not torch.is_floating_point(p_val):
                continue
                
            # Check if this parameter is a 2D or 4D weight tensor
            param_shape = progenitor.state_dict()[name].shape
            is_weight_tensor = ('weight' in name) and (len(param_shape) >= 2)
            
            tvs = [exp.state_dict()[name].to(param_device) - p_val for exp in experts]
            tv_merged = sum(tvs) / K
            
            if is_weight_tensor:
                # Channel-wise sorting
                C_out = param_shape[0]
                tv_wcpr = torch.zeros_like(tv_merged)
                
                for c in range(C_out):
                    m_c = tv_merged[c].flatten()
                    I_c = torch.argsort(m_c)
                    
                    # Sort experts
                    s_k_list = []
                    for k in range(K):
                        s_k = torch.sort(tvs[k][c].flatten()).values
                        s_k_list.append(s_k)
                        
                    s_target_c = sum(s_k_list) / K
                    c_flat = torch.zeros_like(m_c)
                    c_flat[I_c] = s_target_c
                    tv_wcpr[c] = c_flat.view_as(tv_merged[c])
            else:
                # Fallback to U-IPR
                norm_experts = sum([torch.norm(tv, p='fro') for tv in tvs]) / K
                norm_merged = torch.norm(tv_merged, p='fro')
                if norm_merged > 0:
                    scale = norm_experts / (norm_merged + 1e-8)
                    scale = torch.clamp(scale, 0.1, 10.0)
                    tv_wcpr = scale * tv_merged
                else:
                    tv_wcpr = tv_merged
                    
            merged_state[name] = p_val + tv_wcpr
            
    merged = copy.deepcopy(progenitor)
    merged.load_state_dict(merged_state)
    return merged

# SC-WCPR - Sparsity-Compensated WCPR
def merge_sc_wcpr(progenitor, experts, method='ties', sparsity_ratio=0.2, compensation='dare'):
    K = len(experts)
    
    # 1. Sparsify and Merge using TIES or DARE first to get the sparse base merged update
    merged_base = sparsify_and_merge_task_vectors(progenitor, experts, method=method, sparsity_ratio=sparsity_ratio)
    merged_state = copy.deepcopy(progenitor.state_dict())
    
    with torch.no_grad():
        for name in merged_state.keys():
            param_device = merged_state[name].device
            p_val = progenitor.state_dict()[name].to(param_device)
            
            if not torch.is_floating_point(p_val):
                continue
                
            # Check if this parameter is a 2D or 4D weight tensor
            param_shape = progenitor.state_dict()[name].shape
            is_weight_tensor = ('weight' in name) and (len(param_shape) >= 2)
            
            # The expert dense updates
            tvs = [exp.state_dict()[name].to(param_device) - p_val for exp in experts]
            # The sparse merged update
            tv_merged = (merged_base.state_dict()[name] - progenitor.state_dict()[name]).to(param_device)
            
            if is_weight_tensor:
                # Channel-wise sorting ONLY for active elements
                C_out = param_shape[0]
                tv_sc_wcpr = torch.zeros_like(tv_merged)
                
                for c in range(C_out):
                    m_c = tv_merged[c]
                    mask = (m_c != 0)
                    N_active = mask.sum().item()
                    N = m_c.numel()
                    
                    if N_active > 0:
                        m_active = m_c[mask].flatten()
                        I_active = torch.argsort(m_active)
                        
                        # Resample expert sorted arrays to size N_active
                        s_k_list = []
                        for k in range(K):
                            s_k = torch.sort(tvs[k][c].flatten()).values
                            # Interpolate s_k of size N to size N_active
                            if N_active == N:
                                resampled_s = s_k
                            else:
                                indices = torch.linspace(0, N - 1, N_active, device=s_k.device)
                                idx_low = indices.floor().long().clamp(0, N - 1)
                                idx_high = indices.ceil().long().clamp(0, N - 1)
                                weight = indices - idx_low
                                resampled_s = (1.0 - weight) * s_k[idx_low] + weight * s_k[idx_high]
                            s_k_list.append(resampled_s)
                            
                        s_target_c = sum(s_k_list) / K
                        
                        # Apply active-ratio scaling compensation
                        p_c = N_active / N
                        if compensation == 'sqrt':
                            s_target_c = s_target_c * np.sqrt(p_c)
                        elif compensation == 'linear':
                            s_target_c = s_target_c * p_c
                        elif compensation == 'inv_sqrt':
                            s_target_c = s_target_c / (np.sqrt(p_c) + 1e-8)
                        elif compensation == 'inv_linear':
                            s_target_c = s_target_c / (p_c + 1e-8)
                        elif compensation == 'dare':
                            s_target_c = s_target_c / (1.0 - sparsity_ratio)
                        # 'none' does nothing
                        
                        c_flat = m_c.clone()
                        c_flat[mask] = 0.0 # Clear out so we map sorted target
                        # Map sorted active elements to target
                        c_flat_active = torch.zeros_like(m_active)
                        c_flat_active[I_active] = s_target_c
                        c_flat[mask] = c_flat_active
                        
                        tv_sc_wcpr[c] = c_flat
                    else:
                        # No active elements, leave at zero
                        tv_sc_wcpr[c] = tv_merged[c]
            else:
                # Fallback to U-IPR
                norm_experts = sum([torch.norm(tv, p='fro') for tv in tvs]) / K
                norm_merged = torch.norm(tv_merged, p='fro')
                if norm_merged > 0:
                    scale = norm_experts / (norm_merged + 1e-8)
                    scale = torch.clamp(scale, 0.1, 10.0)
                    tv_sc_wcpr = scale * tv_merged
                else:
                    tv_sc_wcpr = tv_merged
                    
            merged_state[name] = p_val + tv_sc_wcpr
            
    merged = copy.deepcopy(progenitor)
    merged.load_state_dict(merged_state)
    return merged
