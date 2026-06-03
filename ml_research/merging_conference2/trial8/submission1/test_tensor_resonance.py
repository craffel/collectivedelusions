import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import copy
import numpy as np

# Set seed for reproducibility
torch.manual_seed(2026)
np.random.seed(2026)

device = torch.device("cpu") # run on CPU
print(f"Using device: {device}")

# 1. Prepare Datasets & DataLoaders (using smaller subset or full test set)
print("Preparing datasets...")
mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
    transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3081, 0.3081, 0.3081))
])

fmnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
    transforms.Normalize(mean=(0.2860, 0.2860, 0.2860), std=(0.3530, 0.3530, 0.3530))
])

cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

test_datasets = {
    "mnist": datasets.MNIST(root="./data", train=False, transform=mnist_transform, download=False),
    "fmnist": datasets.FashionMNIST(root="./data", train=False, transform=fmnist_transform, download=False),
    "cifar10": datasets.CIFAR10(root="./data", train=False, transform=cifar_transform, download=False)
}

# Evaluate on full test sets
test_loaders = {k: DataLoader(v, batch_size=128, shuffle=False, num_workers=4) for k, v in test_datasets.items()}


# 2. Define Architecture
def get_resnet18_with_head():
    model = models.resnet18()
    backbone = nn.Sequential(*list(model.children())[:-1])
    head = nn.Linear(512, 10)
    return backbone, head

# Load expert models
print("Loading expert checkpoints...")
backbones = {}
heads = {}
for task in ["mnist", "fmnist", "cifar10"]:
    bb, hd = get_resnet18_with_head()
    bb.load_state_dict(torch.load(f"checkpoints/{task}_backbone.pth", map_location=device))
    hd.load_state_dict(torch.load(f"checkpoints/{task}_head.pth", map_location=device))
    backbones[task] = bb
    heads[task] = hd

progenitor_backbone, _ = get_resnet18_with_head()
# Note: Since the previous run loaded ImageNet pre-trained ResNet18, let's load it too to keep weights exactly matched
# Actually, the checkpoints were trained from IMAGENET1K_V1, so progenitor_backbone should be loaded with IMAGENET1K_V1
progenitor_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
progenitor_backbone = nn.Sequential(*list(progenitor_backbone.children())[:-1])


# 3. Evaluation Function
def evaluate_model(backbone, task, loader):
    backbone.eval()
    head = heads[task]
    head.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            feats = backbone(x).squeeze(-1).squeeze(-1)
            outputs = head(feats)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    return 100.0 * correct / total


# 4. Helper to copy backbone
def copy_backbone():
    dst = copy.deepcopy(progenitor_backbone)
    return dst


# 5. Define Merging Algorithms

# Standard Weight Averaging (WA)
def merge_wa():
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        tensors = [backbones[task].state_dict()[key].float() for task in backbones]
        merged_sd[key] = torch.stack(tensors, dim=0).mean(dim=0).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged

# Task Arithmetic (TA, lam=1.0)
def merge_ta(lam=1.0):
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    prog_sd = progenitor_backbone.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        updates = [backbones[task].state_dict()[key].float() - prog_sd[key].float() for task in backbones]
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        merged_sd[key] = (prog_sd[key].float() + lam * merged_update).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged

# Update-level Isotropic Parameter Resonance (U-IPR)
def merge_u_ipr():
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    prog_sd = progenitor_backbone.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        updates = [backbones[task].state_dict()[key].float() - prog_sd[key].float() for task in backbones]
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        
        norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
        norm_merged = torch.norm(merged_update)
        scale = norm_experts / (norm_merged + 1e-8)
        scale = torch.clamp(scale, min=0.1, max=10.0)
        
        merged_sd[key] = (prog_sd[key].float() + scale * merged_update).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged

# Spectral Parameter Resonance (S-IPR)
def merge_s_ipr():
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    prog_sd = progenitor_backbone.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        updates = [backbones[task].state_dict()[key].float() - prog_sd[key].float() for task in backbones]
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        
        if merged_update.dim() >= 2:
            orig_shape = merged_update.shape
            if merged_update.dim() == 4:
                flat_shape = (orig_shape[0], -1)
                updates_2d = [u.view(flat_shape) for u in updates]
                merged_2d = merged_update.view(flat_shape)
            else:
                updates_2d = updates
                merged_2d = merged_update
                
            try:
                expert_singular_values = []
                for u_2d in updates_2d:
                    _, S, _ = torch.linalg.svd(u_2d, full_matrices=False)
                    expert_singular_values.append(S)
                avg_spectrum = torch.stack(expert_singular_values, dim=0).mean(dim=0)
                U_m, S_m, Vh_m = torch.linalg.svd(merged_2d, full_matrices=False)
                corrected_2d = U_m @ torch.diag(avg_spectrum) @ Vh_m
                corrected_update = corrected_2d.view(orig_shape)
            except Exception as e:
                norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
                norm_merged = torch.norm(merged_update)
                scale = norm_experts / (norm_merged + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                corrected_update = scale * merged_update
        else:
            norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
            norm_merged = torch.norm(merged_update)
            scale = norm_experts / (norm_merged + 1e-8)
            scale = torch.clamp(scale, min=0.1, max=10.0)
            corrected_update = scale * merged_update
            
        merged_sd[key] = (prog_sd[key].float() + corrected_update).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged

# HOT-PR V1 (Original Element-wise Core Scaling)
def hosvd(T):
    T1 = T.permute(0, 1, 2, 3).reshape(T.shape[0], -1)
    U1, _, _ = torch.linalg.svd(T1, full_matrices=False)
    T2 = T.permute(1, 0, 2, 3).reshape(T.shape[1], -1)
    U2, _, _ = torch.linalg.svd(T2, full_matrices=False)
    T3 = T.permute(2, 0, 1, 3).reshape(T.shape[2], -1)
    U3, _, _ = torch.linalg.svd(T3, full_matrices=False)
    T4 = T.permute(3, 0, 1, 2).reshape(T.shape[3], -1)
    U4, _, _ = torch.linalg.svd(T4, full_matrices=False)
    G = torch.einsum('abcd,ai,bj,ck,dl->ijkl', T, U1, U2, U3, U4)
    return G, U1, U2, U3, U4

def reconstruct_hosvd(G, U1, U2, U3, U4):
    return torch.einsum('ijkl,ai,bj,ck,dl->abcd', G, U1, U2, U3, U4)

def merge_hot_pr_v1():
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    prog_sd = progenitor_backbone.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        updates = [backbones[task].state_dict()[key].float() - prog_sd[key].float() for task in backbones]
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        
        if merged_update.dim() == 4:
            try:
                expert_cores = []
                for u in updates:
                    G_exp, _, _, _, _ = hosvd(u)
                    expert_cores.append(G_exp)
                avg_core_mag = torch.stack([G.abs() for G in expert_cores], dim=0).mean(dim=0)
                G_m, U1_m, U2_m, U3_m, U4_m = hosvd(merged_update)
                G_corrected = torch.sign(G_m) * avg_core_mag
                corrected_update = reconstruct_hosvd(G_corrected, U1_m, U2_m, U3_m, U4_m)
            except Exception as e:
                # Fallback to U-IPR
                norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
                norm_merged = torch.norm(merged_update)
                scale = norm_experts / (norm_merged + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                corrected_update = scale * merged_update
        elif merged_update.dim() == 2:
            try:
                expert_singular_values = []
                for u in updates:
                    _, S, _ = torch.linalg.svd(u, full_matrices=False)
                    expert_singular_values.append(S)
                avg_spectrum = torch.stack(expert_singular_values, dim=0).mean(dim=0)
                U_m, S_m, Vh_m = torch.linalg.svd(merged_update, full_matrices=False)
                corrected_update = U_m @ torch.diag(avg_spectrum) @ Vh_m
            except Exception as e:
                norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
                norm_merged = torch.norm(merged_update)
                scale = norm_experts / (norm_merged + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                corrected_update = scale * merged_update
        else:
            norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
            norm_merged = torch.norm(merged_update)
            scale = norm_experts / (norm_merged + 1e-8)
            scale = torch.clamp(scale, min=0.1, max=10.0)
            corrected_update = scale * merged_update
            
        merged_sd[key] = (prog_sd[key].float() + corrected_update).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged

# NEW: Channel-wise Parameter Resonance (CPR - Unified Mode-1 Slice Scaling)
def merge_cpr():
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    prog_sd = progenitor_backbone.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        updates = [backbones[task].state_dict()[key].float() - prog_sd[key].float() for task in backbones]
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        
        if merged_update.dim() >= 2:
            try:
                # Compute Mode-1 (output channels) slice norms
                expert_m1_norms = []
                for u in updates:
                    m1_norms = torch.norm(u.view(u.size(0), -1), dim=1)
                    expert_m1_norms.append(m1_norms)
                avg_m1_norms = torch.stack(expert_m1_norms, dim=0).mean(dim=0)
                
                m1_norms_m = torch.norm(merged_update.view(merged_update.size(0), -1), dim=1)
                
                scale = avg_m1_norms / (m1_norms_m + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                
                # Reshape to broadcast
                scale_tensor = scale.view(-1, *([1] * (merged_update.dim() - 1)))
                corrected_update = scale_tensor * merged_update
            except Exception as e:
                norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
                norm_merged = torch.norm(merged_update)
                scale = norm_experts / (norm_merged + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                corrected_update = scale * merged_update
        else:
            norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
            norm_merged = torch.norm(merged_update)
            scale = norm_experts / (norm_merged + 1e-8)
            scale = torch.clamp(scale, min=0.1, max=10.0)
            corrected_update = scale * merged_update
            
        merged_sd[key] = (prog_sd[key].float() + corrected_update).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged

# NEW: Multi-linear Channel-wise Parameter Resonance (MC-PR - Scales Mode-1 and Mode-2 Slices)
def merge_mc_pr():
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    prog_sd = progenitor_backbone.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        updates = [backbones[task].state_dict()[key].float() - prog_sd[key].float() for task in backbones]
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        
        if merged_update.dim() >= 3: # Typically 3D or 4D tensors (Conv / self-attention QKV)
            try:
                # Mode-1 scaling factor
                expert_m1_norms = []
                for u in updates:
                    expert_m1_norms.append(torch.norm(u.view(u.size(0), -1), dim=1))
                avg_m1_norms = torch.stack(expert_m1_norms, dim=0).mean(dim=0)
                m1_norms_m = torch.norm(merged_update.view(merged_update.size(0), -1), dim=1)
                scale1 = avg_m1_norms / (m1_norms_m + 1e-8)
                scale1 = torch.clamp(scale1, min=0.1, max=10.0)
                
                # Mode-2 scaling factor
                expert_m2_norms = []
                for u in updates:
                    # Permute mode-2 to dim-0, then flatten rest
                    u_perm = u.permute(1, 0, *range(2, u.dim()))
                    expert_m2_norms.append(torch.norm(u_perm.reshape(u_perm.size(0), -1), dim=1))
                avg_m2_norms = torch.stack(expert_m2_norms, dim=0).mean(dim=0)
                merged_perm = merged_update.permute(1, 0, *range(2, merged_update.dim()))
                m2_norms_m = torch.norm(merged_perm.reshape(merged_perm.size(0), -1), dim=1)
                scale2 = avg_m2_norms / (m2_norms_m + 1e-8)
                scale2 = torch.clamp(scale2, min=0.1, max=10.0)
                
                # Multi-linear scaling using geometric mean
                # Apply sqrt(scale1) and sqrt(scale2)
                s1_tensor = torch.sqrt(scale1).view(-1, 1, *([1] * (merged_update.dim() - 2)))
                s2_tensor = torch.sqrt(scale2).view(1, -1, *([1] * (merged_update.dim() - 2)))
                
                corrected_update = merged_update * s1_tensor * s2_tensor
            except Exception as e:
                norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
                norm_merged = torch.norm(merged_update)
                scale = norm_experts / (norm_merged + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                corrected_update = scale * merged_update
        elif merged_update.dim() == 2: # Linear layers, standard Mode-1 / Mode-2 scaling
            try:
                expert_m1_norms = [torch.norm(u, dim=1) for u in updates]
                avg_m1_norms = torch.stack(expert_m1_norms, dim=0).mean(dim=0)
                m1_norms_m = torch.norm(merged_update, dim=1)
                scale1 = avg_m1_norms / (m1_norms_m + 1e-8)
                scale1 = torch.clamp(scale1, min=0.1, max=10.0)
                
                expert_m2_norms = [torch.norm(u, dim=0) for u in updates]
                avg_m2_norms = torch.stack(expert_m2_norms, dim=0).mean(dim=0)
                m2_norms_m = torch.norm(merged_update, dim=0)
                scale2 = avg_m2_norms / (m2_norms_m + 1e-8)
                scale2 = torch.clamp(scale2, min=0.1, max=10.0)
                
                s1_tensor = torch.sqrt(scale1).view(-1, 1)
                s2_tensor = torch.sqrt(scale2).view(1, -1)
                corrected_update = merged_update * s1_tensor * s2_tensor
            except Exception as e:
                norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
                norm_merged = torch.norm(merged_update)
                scale = norm_experts / (norm_merged + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                corrected_update = scale * merged_update
        else:
            norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
            norm_merged = torch.norm(merged_update)
            scale = norm_experts / (norm_merged + 1e-8)
            scale = torch.clamp(scale, min=0.1, max=10.0)
            corrected_update = scale * merged_update
            
        merged_sd[key] = (prog_sd[key].float() + corrected_update).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged

# NEW: Multi-linear Spectral Parameter Resonance (MS-PR - Our proposed final HOT-PR method)
def merge_ms_pr():
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    prog_sd = progenitor_backbone.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        updates = [backbones[task].state_dict()[key].float() - prog_sd[key].float() for task in backbones]
        merged_update = torch.stack(updates, dim=0).mean(dim=0)
        
        if merged_update.dim() == 4:
            try:
                # Mode-1 singular values of experts
                expert_s1 = []
                for u in updates:
                    u1 = u.permute(0, 1, 2, 3).reshape(u.shape[0], -1)
                    _, S1, _ = torch.linalg.svd(u1, full_matrices=False)
                    expert_s1.append(S1)
                avg_s1 = torch.stack(expert_s1, dim=0).mean(dim=0)
                
                # Mode-2 singular values of experts
                expert_s2 = []
                for u in updates:
                    u2 = u.permute(1, 0, 2, 3).reshape(u.shape[1], -1)
                    _, S2, _ = torch.linalg.svd(u2, full_matrices=False)
                    expert_s2.append(S2)
                avg_s2 = torch.stack(expert_s2, dim=0).mean(dim=0)
                
                # Merged SVDs
                merged_u1 = merged_update.permute(0, 1, 2, 3).reshape(merged_update.shape[0], -1)
                _, S1_m, _ = torch.linalg.svd(merged_u1, full_matrices=False)
                
                merged_u2 = merged_update.permute(1, 0, 2, 3).reshape(merged_update.shape[1], -1)
                _, S2_m, _ = torch.linalg.svd(merged_u2, full_matrices=False)
                
                # Mode-1 & Mode-2 scale factors
                scale1 = avg_s1 / (S1_m + 1e-8)
                scale1 = torch.clamp(scale1, min=0.1, max=10.0)
                
                scale2 = avg_s2 / (S2_m + 1e-8)
                scale2 = torch.clamp(scale2, min=0.1, max=10.0)
                
                # Compute HOSVD for merged update
                G_m, U1_m, U2_m, U3_m, U4_m = hosvd(merged_update)
                
                # Apply multi-linear spectral scaling to Mode-1 and Mode-2 factor matrices
                U1_scaled = U1_m @ torch.diag(torch.sqrt(scale1))
                U2_scaled = U2_m @ torch.diag(torch.sqrt(scale2))
                
                # Reconstruct
                corrected_update = reconstruct_hosvd(G_m, U1_scaled, U2_scaled, U3_m, U4_m)
                
            except Exception as e:
                # Fallback to S-IPR Mode-1
                try:
                    orig_shape = merged_update.shape
                    flat_shape = (orig_shape[0], -1)
                    updates_2d = [u.view(flat_shape) for u in updates]
                    merged_2d = merged_update.view(flat_shape)
                    expert_singular_values = [torch.linalg.svd(u_2d, full_matrices=False)[1] for u_2d in updates_2d]
                    avg_spectrum = torch.stack(expert_singular_values, dim=0).mean(dim=0)
                    U_m, S_m, Vh_m = torch.linalg.svd(merged_2d, full_matrices=False)
                    corrected_2d = U_m @ torch.diag(avg_spectrum) @ Vh_m
                    corrected_update = corrected_2d.view(orig_shape)
                except Exception as e2:
                    norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
                    norm_merged = torch.norm(merged_update)
                    scale = norm_experts / (norm_merged + 1e-8)
                    scale = torch.clamp(scale, min=0.1, max=10.0)
                    corrected_update = scale * merged_update
                    
        elif merged_update.dim() == 2:
            try:
                expert_singular_values = []
                for u in updates:
                    _, S, _ = torch.linalg.svd(u, full_matrices=False)
                    expert_singular_values.append(S)
                avg_spectrum = torch.stack(expert_singular_values, dim=0).mean(dim=0)
                U_m, S_m, Vh_m = torch.linalg.svd(merged_update, full_matrices=False)
                corrected_update = U_m @ torch.diag(avg_spectrum) @ Vh_m
            except Exception as e:
                norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
                norm_merged = torch.norm(merged_update)
                scale = norm_experts / (norm_merged + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                corrected_update = scale * merged_update
        else:
            norm_experts = torch.stack([torch.norm(u) for u in updates]).mean()
            norm_merged = torch.norm(merged_update)
            scale = norm_experts / (norm_merged + 1e-8)
            scale = torch.clamp(scale, min=0.1, max=10.0)
            corrected_update = scale * merged_update
            
        merged_sd[key] = (prog_sd[key].float() + corrected_update).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged

# NEW: TIES-Merging (Trim, Elect, and Sign)
def merge_ties(keep_ratio=0.2):
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    prog_sd = progenitor_backbone.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        updates = [backbones[task].state_dict()[key].float() - prog_sd[key].float() for task in backbones]
        
        # Step 1: Trim (keep top-k% largest magnitude values for each expert update)
        trimmed_updates = []
        for u in updates:
            if u.numel() == 0:
                trimmed_updates.append(u)
                continue
            flat_u = u.view(-1)
            k = max(1, int(keep_ratio * flat_u.numel()))
            threshold = torch.topk(flat_u.abs(), k).values[-1]
            mask = flat_u.abs() >= threshold
            trimmed_u = flat_u * mask
            trimmed_updates.append(trimmed_u.view_as(u))
            
        # Step 2: Elect Sign
        signs = torch.stack([torch.sign(tu) for tu in trimmed_updates], dim=0)
        sign_sum = signs.sum(dim=0)
        elected_sign = torch.sign(sign_sum)
        
        # Step 3: Disjoint Merge
        stacked_trimmed = torch.stack(trimmed_updates, dim=0)
        same_sign_mask = (torch.sign(stacked_trimmed) == elected_sign) & (elected_sign != 0)
        
        sum_vals = (stacked_trimmed * same_sign_mask).sum(dim=0)
        count_vals = same_sign_mask.sum(dim=0).float()
        
        merged_update = torch.where(count_vals > 0, sum_vals / count_vals, torch.zeros_like(sum_vals))
        merged_sd[key] = (prog_sd[key].float() + merged_update).to(merged_sd[key].dtype)
        
    merged.load_state_dict(merged_sd)
    return merged

# NEW: DARE-Task Arithmetic (DARE-TA)
def merge_dare(drop_rate=0.5):
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    prog_sd = progenitor_backbone.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        updates = [backbones[task].state_dict()[key].float() - prog_sd[key].float() for task in backbones]
        
        dropped_updates = []
        for u in updates:
            mask = (torch.rand_like(u) > drop_rate).float()
            dropped_u = u * mask / (1.0 - drop_rate + 1e-8)
            dropped_updates.append(dropped_u)
            
        merged_update = torch.stack(dropped_updates, dim=0).mean(dim=0)
        merged_sd[key] = (prog_sd[key].float() + merged_update).to(merged_sd[key].dtype)
        
    merged.load_state_dict(merged_sd)
    return merged


# 6. Run Evaluations
results = {}

print("\n--- Running Evaluations on CPU ---")

# Evaluate WA
print("Evaluating WA...")
wa_model = merge_wa()
results["WA"] = {task: evaluate_model(wa_model, task, test_loaders[task]) for task in backbones}

# Evaluate TA (lam=1.0)
print("Evaluating TA...")
ta_model = merge_ta(lam=1.0)
results["TA (lam=1.0)"] = {task: evaluate_model(ta_model, task, test_loaders[task]) for task in backbones}

# Evaluate TIES-Merging
print("Evaluating TIES-Merging...")
ties_model = merge_ties(keep_ratio=0.2)
results["TIES-Merging"] = {task: evaluate_model(ties_model, task, test_loaders[task]) for task in backbones}

# Evaluate DARE-TA
print("Evaluating DARE-TA...")
dare_model = merge_dare(drop_rate=0.5)
results["DARE-TA"] = {task: evaluate_model(dare_model, task, test_loaders[task]) for task in backbones}

# Evaluate U-IPR
print("Evaluating U-IPR...")
u_ipr_model = merge_u_ipr()
results["U-IPR"] = {task: evaluate_model(u_ipr_model, task, test_loaders[task]) for task in backbones}

# Evaluate S-IPR
print("Evaluating S-IPR...")
s_ipr_model = merge_s_ipr()
results["S-IPR"] = {task: evaluate_model(s_ipr_model, task, test_loaders[task]) for task in backbones}

# Evaluate HOT-PR V1
print("Evaluating HOT-PR V1...")
hot_pr_v1_model = merge_hot_pr_v1()
results["HOT-PR V1"] = {task: evaluate_model(hot_pr_v1_model, task, test_loaders[task]) for task in backbones}

# Evaluate CPR
print("Evaluating CPR...")
cpr_model = merge_cpr()
results["CPR (Channel-PR)"] = {task: evaluate_model(cpr_model, task, test_loaders[task]) for task in backbones}

# Evaluate MC-PR
print("Evaluating MC-PR...")
mc_pr_model = merge_mc_pr()
results["MC-PR (Multi-Channel-PR)"] = {task: evaluate_model(mc_pr_model, task, test_loaders[task]) for task in backbones}

# Evaluate MS-PR
print("Evaluating MS-PR...")
ms_pr_model = merge_ms_pr()
results["MS-PR (Multi-Spectral-PR Ours)"] = {task: evaluate_model(ms_pr_model, task, test_loaders[task]) for task in backbones}


# 7. Print Results Table
print("\n" + "="*95)
print(f"{'Method':<35} | {'MNIST':<10} | {'FMNIST':<10} | {'CIFAR-10':<10} | {'Average':<10}")
print("="*95)
for method, task_accs in results.items():
    mnist_acc = task_accs["mnist"]
    fmnist_acc = task_accs["fmnist"]
    cifar_acc = task_accs["cifar10"]
    avg_acc = (mnist_acc + fmnist_acc + cifar_acc) / 3.0
    print(f"{method:<35} | {mnist_acc:<10.2f}% | {fmnist_acc:<10.2f}% | {cifar_acc:<10.2f}% | {avg_acc:<10.2f}%")
print("="*95)
