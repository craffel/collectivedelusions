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

# 1. Prepare Datasets & DataLoaders
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

# Define task weights (non-uniform mixing coefficients)
task_weights = {
    "mnist": 0.5,
    "fmnist": 0.3,
    "cifar10": 0.2
}
print(f"Using non-uniform task weights: {task_weights}")

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


# 5. Define Generalized Non-Uniform Merging Algorithms

# Weighted Weight Averaging (WWA)
def merge_wwa():
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        tensors = [backbones[task].state_dict()[key].float() * task_weights[task] for task in backbones]
        merged_sd[key] = torch.stack(tensors, dim=0).sum(dim=0).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged

# Weighted Update-level IPR (WU-IPR)
def merge_wu_ipr():
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    prog_sd = progenitor_backbone.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        updates = [backbones[task].state_dict()[key].float() - prog_sd[key].float() for task in backbones]
        
        # Weighted merged update
        weighted_updates = [updates[i] * task_weights[task] for i, task in enumerate(backbones)]
        merged_update = torch.stack(weighted_updates, dim=0).sum(dim=0)
        
        # Expected norm under weighted scaling
        norm_experts = torch.stack([torch.norm(u) * task_weights[task] for u, task in zip(updates, backbones)]).sum()
        norm_merged = torch.norm(merged_update)
        scale = norm_experts / (norm_merged + 1e-8)
        scale = torch.clamp(scale, min=0.1, max=10.0)
        
        merged_sd[key] = (prog_sd[key].float() + scale * merged_update).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged

# Weighted Channel-PR (WCPR)
def merge_wcpr():
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    prog_sd = progenitor_backbone.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        updates = [backbones[task].state_dict()[key].float() - prog_sd[key].float() for task in backbones]
        weighted_updates = [u * task_weights[task] for u, task in zip(updates, backbones)]
        merged_update = torch.stack(weighted_updates, dim=0).sum(dim=0)
        
        if merged_update.dim() >= 2:
            try:
                # Compute Mode-1 (output channels) slice norms
                expert_m1_norms = []
                for u, task in zip(updates, backbones):
                    m1_norms = torch.norm(u.view(u.size(0), -1), dim=1)
                    expert_m1_norms.append(m1_norms * task_weights[task])
                avg_m1_norms = torch.stack(expert_m1_norms, dim=0).sum(dim=0)
                
                m1_norms_m = torch.norm(merged_update.view(merged_update.size(0), -1), dim=1)
                
                scale = avg_m1_norms / (m1_norms_m + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                
                scale_tensor = scale.view(-1, *([1] * (merged_update.dim() - 1)))
                corrected_update = scale_tensor * merged_update
            except Exception as e:
                norm_experts = torch.stack([torch.norm(u) * task_weights[task] for u, task in zip(updates, backbones)]).sum()
                norm_merged = torch.norm(merged_update)
                scale = norm_experts / (norm_merged + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                corrected_update = scale * merged_update
        else:
            norm_experts = torch.stack([torch.norm(u) * task_weights[task] for u, task in zip(updates, backbones)]).sum()
            norm_merged = torch.norm(merged_update)
            scale = norm_experts / (norm_merged + 1e-8)
            scale = torch.clamp(scale, min=0.1, max=10.0)
            corrected_update = scale * merged_update
            
        merged_sd[key] = (prog_sd[key].float() + corrected_update).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged

# Weighted Multi-Channel-PR (WMC-PR)
def merge_wmc_pr():
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    prog_sd = progenitor_backbone.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        updates = [backbones[task].state_dict()[key].float() - prog_sd[key].float() for task in backbones]
        weighted_updates = [u * task_weights[task] for u, task in zip(updates, backbones)]
        merged_update = torch.stack(weighted_updates, dim=0).sum(dim=0)
        
        if merged_update.dim() >= 3:
            try:
                # Mode-1 scaling factor
                expert_m1_norms = []
                for u, task in zip(updates, backbones):
                    expert_m1_norms.append(torch.norm(u.view(u.size(0), -1), dim=1) * task_weights[task])
                avg_m1_norms = torch.stack(expert_m1_norms, dim=0).sum(dim=0)
                m1_norms_m = torch.norm(merged_update.view(merged_update.size(0), -1), dim=1)
                scale1 = avg_m1_norms / (m1_norms_m + 1e-8)
                scale1 = torch.clamp(scale1, min=0.1, max=10.0)
                
                # Mode-2 scaling factor
                expert_m2_norms = []
                for u, task in zip(updates, backbones):
                    u_perm = u.permute(1, 0, *range(2, u.dim()))
                    expert_m2_norms.append(torch.norm(u_perm.reshape(u_perm.size(0), -1), dim=1) * task_weights[task])
                avg_m2_norms = torch.stack(expert_m2_norms, dim=0).sum(dim=0)
                merged_perm = merged_update.permute(1, 0, *range(2, merged_update.dim()))
                m2_norms_m = torch.norm(merged_perm.reshape(merged_perm.size(0), -1), dim=1)
                scale2 = avg_m2_norms / (m2_norms_m + 1e-8)
                scale2 = torch.clamp(scale2, min=0.1, max=10.0)
                
                s1_tensor = torch.sqrt(scale1).view(-1, 1, *([1] * (merged_update.dim() - 2)))
                s2_tensor = torch.sqrt(scale2).view(1, -1, *([1] * (merged_update.dim() - 2)))
                
                corrected_update = merged_update * s1_tensor * s2_tensor
            except Exception as e:
                norm_experts = torch.stack([torch.norm(u) * task_weights[task] for u, task in zip(updates, backbones)]).sum()
                norm_merged = torch.norm(merged_update)
                scale = norm_experts / (norm_merged + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                corrected_update = scale * merged_update
        elif merged_update.dim() == 2:
            try:
                expert_m1_norms = [torch.norm(u, dim=1) * task_weights[task] for u, task in zip(updates, backbones)]
                avg_m1_norms = torch.stack(expert_m1_norms, dim=0).sum(dim=0)
                m1_norms_m = torch.norm(merged_update, dim=1)
                scale1 = avg_m1_norms / (m1_norms_m + 1e-8)
                scale1 = torch.clamp(scale1, min=0.1, max=10.0)
                
                expert_m2_norms = [torch.norm(u, dim=0) * task_weights[task] for u, task in zip(updates, backbones)]
                avg_m2_norms = torch.stack(expert_m2_norms, dim=0).sum(dim=0)
                m2_norms_m = torch.norm(merged_update, dim=0)
                scale2 = avg_m2_norms / (m2_norms_m + 1e-8)
                scale2 = torch.clamp(scale2, min=0.1, max=10.0)
                
                s1_tensor = torch.sqrt(scale1).view(-1, 1)
                s2_tensor = torch.sqrt(scale2).view(1, -1)
                corrected_update = merged_update * s1_tensor * s2_tensor
            except Exception as e:
                norm_experts = torch.stack([torch.norm(u) * task_weights[task] for u, task in zip(updates, backbones)]).sum()
                norm_merged = torch.norm(merged_update)
                scale = norm_experts / (norm_merged + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                corrected_update = scale * merged_update
        else:
            norm_experts = torch.stack([torch.norm(u) * task_weights[task] for u, task in zip(updates, backbones)]).sum()
            norm_merged = torch.norm(merged_update)
            scale = norm_experts / (norm_merged + 1e-8)
            scale = torch.clamp(scale, min=0.1, max=10.0)
            corrected_update = scale * merged_update
            
        merged_sd[key] = (prog_sd[key].float() + corrected_update).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged

# Weighted Multi-Spectral-PR (WMS-PR)
def merge_wms_pr():
    merged = copy_backbone()
    merged_sd = merged.state_dict()
    prog_sd = progenitor_backbone.state_dict()
    for key in merged_sd:
        if "num_batches_tracked" in key:
            continue
        updates = [backbones[task].state_dict()[key].float() - prog_sd[key].float() for task in backbones]
        weighted_updates = [u * task_weights[task] for u, task in zip(updates, backbones)]
        merged_update = torch.stack(weighted_updates, dim=0).sum(dim=0)
        
        if merged_update.dim() == 4:
            try:
                # Mode-1 singular values of experts
                expert_s1 = []
                for u, task in zip(updates, backbones):
                    u1 = u.permute(0, 1, 2, 3).reshape(u.shape[0], -1)
                    _, S1, _ = torch.linalg.svd(u1, full_matrices=False)
                    expert_s1.append(S1 * task_weights[task])
                avg_s1 = torch.stack(expert_s1, dim=0).sum(dim=0)
                
                # Mode-2 singular values of experts
                expert_s2 = []
                for u, task in zip(updates, backbones):
                    u2 = u.permute(1, 0, 2, 3).reshape(u.shape[1], -1)
                    _, S2, _ = torch.linalg.svd(u2, full_matrices=False)
                    expert_s2.append(S2 * task_weights[task])
                avg_s2 = torch.stack(expert_s2, dim=0).sum(dim=0)
                
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
                try:
                    orig_shape = merged_update.shape
                    flat_shape = (orig_shape[0], -1)
                    updates_2d = [u.view(flat_shape) for u in updates]
                    merged_2d = merged_update.view(flat_shape)
                    expert_singular_values = [torch.linalg.svd(u_2d, full_matrices=False)[1] * task_weights[task] for u_2d, task in zip(updates_2d, backbones)]
                    avg_spectrum = torch.stack(expert_singular_values, dim=0).sum(dim=0)
                    U_m, S_m, Vh_m = torch.linalg.svd(merged_2d, full_matrices=False)
                    corrected_2d = U_m @ torch.diag(avg_spectrum) @ Vh_m
                    corrected_update = corrected_2d.view(orig_shape)
                except Exception as e2:
                    norm_experts = torch.stack([torch.norm(u) * task_weights[task] for u, task in zip(updates, backbones)]).sum()
                    norm_merged = torch.norm(merged_update)
                    scale = norm_experts / (norm_merged + 1e-8)
                    scale = torch.clamp(scale, min=0.1, max=10.0)
                    corrected_update = scale * merged_update
                    
        elif merged_update.dim() == 2:
            try:
                expert_singular_values = []
                for u, task in zip(updates, backbones):
                    _, S, _ = torch.linalg.svd(u, full_matrices=False)
                    expert_singular_values.append(S * task_weights[task])
                avg_spectrum = torch.stack(expert_singular_values, dim=0).sum(dim=0)
                U_m, S_m, Vh_m = torch.linalg.svd(merged_update, full_matrices=False)
                corrected_update = U_m @ torch.diag(avg_spectrum) @ Vh_m
            except Exception as e:
                norm_experts = torch.stack([torch.norm(u) * task_weights[task] for u, task in zip(updates, backbones)]).sum()
                norm_merged = torch.norm(merged_update)
                scale = norm_experts / (norm_merged + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                corrected_update = scale * merged_update
        else:
            norm_experts = torch.stack([torch.norm(u) * task_weights[task] for u, task in zip(updates, backbones)]).sum()
            norm_merged = torch.norm(merged_update)
            scale = norm_experts / (norm_merged + 1e-8)
            scale = torch.clamp(scale, min=0.1, max=10.0)
            corrected_update = scale * merged_update
            
        merged_sd[key] = (prog_sd[key].float() + corrected_update).to(merged_sd[key].dtype)
    merged.load_state_dict(merged_sd)
    return merged


# 6. Run Evaluations
results = {}

print("\n--- Running Weighted Evaluations on CPU ---")

# Evaluate WWA
print("Evaluating Weighted Weight Averaging (WWA)...")
wwa_model = merge_wwa()
results["WWA"] = {task: evaluate_model(wwa_model, task, test_loaders[task]) for task in backbones}

# Evaluate WU-IPR
print("Evaluating Weighted Update-level IPR (WU-IPR)...")
wu_ipr_model = merge_wu_ipr()
results["WU-IPR"] = {task: evaluate_model(wu_ipr_model, task, test_loaders[task]) for task in backbones}

# Evaluate WCPR
print("Evaluating Weighted Channel-PR (WCPR)...")
wcpr_model = merge_wcpr()
results["WCPR"] = {task: evaluate_model(wcpr_model, task, test_loaders[task]) for task in backbones}

# Evaluate WMC-PR
print("Evaluating Weighted Multi-Channel-PR (WMC-PR)...")
wmc_pr_model = merge_wmc_pr()
results["WMC-PR"] = {task: evaluate_model(wmc_pr_model, task, test_loaders[task]) for task in backbones}

# Evaluate WMS-PR
print("Evaluating Weighted Multi-Spectral-PR (WMS-PR)...")
wms_pr_model = merge_wms_pr()
results["WMS-PR"] = {task: evaluate_model(wms_pr_model, task, test_loaders[task]) for task in backbones}


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
