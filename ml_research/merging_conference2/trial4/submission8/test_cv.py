import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import numpy as np
import copy
import json

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ReplicateChannel(object):
    def __call__(self, tensor):
        return tensor.repeat(3, 1, 1)

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32), antialias=True),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    ReplicateChannel(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=mnist_transform)
test_fashion = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=mnist_transform)
test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

test_loaders = {
    'mnist': DataLoader(test_mnist, batch_size=256, shuffle=False, num_workers=4),
    'fashion': DataLoader(test_fashion, batch_size=256, shuffle=False, num_workers=4),
    'cifar': DataLoader(test_cifar, batch_size=256, shuffle=False, num_workers=4)
}

train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=mnist_transform)
train_fashion = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=mnist_transform)
train_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

def get_calibration_loader(N, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_per_task = N // 3
    
    mnist_idx = np.random.choice(len(train_mnist), n_per_task, replace=False)
    fashion_idx = np.random.choice(len(train_fashion), n_per_task, replace=False)
    cifar_idx = np.random.choice(len(train_cifar), n_per_task, replace=False)
    
    mnist_sub = Subset(train_mnist, mnist_idx)
    fashion_sub = Subset(train_fashion, fashion_idx)
    cifar_sub = Subset(train_cifar, cifar_idx)
    
    joint_loader = DataLoader(
        torch.utils.data.ConcatDataset([mnist_sub, fashion_sub, cifar_sub]),
        batch_size=N, shuffle=False
    )
    
    task_loaders = {
        'mnist': DataLoader(mnist_sub, batch_size=n_per_task, shuffle=False),
        'fashion': DataLoader(fashion_sub, batch_size=n_per_task, shuffle=False),
        'cifar': DataLoader(cifar_sub, batch_size=n_per_task, shuffle=False)
    }
    
    return joint_loader, task_loaders

def get_weight_averaged_backbone():
    mnist_sd = torch.load('mnist_expert.pth', map_location=device)
    fashion_sd = torch.load('fashion_expert.pth', map_location=device)
    cifar_sd = torch.load('cifar_expert.pth', map_location=device)
    
    merged_model = models.resnet18(weights=None)
    merged_model.fc = nn.Linear(merged_model.fc.in_features, 10)
    merged_model = merged_model.to(device)
    
    merged_sd = merged_model.state_dict()
    for key in merged_sd.keys():
        if 'fc' in key:
            continue
        merged_sd[key] = (mnist_sd[key] + fashion_sd[key] + cifar_sd[key]) / 3.0
        
    merged_model.load_state_dict(merged_sd)
    return merged_model

def compute_sp_taac_scales(merged_model, experts_paths, joint_loader, task_loaders):
    merged_model.eval()
    experts = {}
    for name, path in experts_paths.items():
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        experts[name] = model.to(device)
        
    bn_modules_merged = []
    for m in merged_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_modules_merged.append(m)
            
    bn_modules_experts = {name: [] for name in experts.keys()}
    for name, model in experts.items():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_modules_experts[name].append(m)
                
    num_bn = len(bn_modules_merged)
    
    merged_acts = [[] for _ in range(num_bn)]
    def get_collect_hook(storage):
        def hook(module, input, output):
            storage.append(output.detach().cpu())
        return hook
        
    hooks = []
    for idx, m in enumerate(bn_modules_merged):
        hooks.append(m.register_forward_hook(get_collect_hook(merged_acts[idx])))
        
    with torch.no_grad():
        for inputs, _ in joint_loader:
            inputs = inputs.to(device)
            _ = merged_model(inputs)
            
    for h in hooks:
        h.remove()
        
    expert_acts = {name: [[] for _ in range(num_bn)] for name in experts.keys()}
    for name, model in experts.items():
        expert_hooks = []
        for idx, m in enumerate(bn_modules_experts[name]):
            expert_hooks.append(m.register_forward_hook(get_collect_hook(expert_acts[name][idx])))
            
        with torch.no_grad():
            for inputs, _ in task_loaders[name]:
                inputs = inputs.to(device)
                _ = model(inputs)
                
        for h in expert_hooks:
            h.remove()
            
    gammas = []
    for idx in range(num_bn):
        m_act = torch.cat(merged_acts[idx], dim=0)
        sigma_merged = torch.std(m_act, dim=(0, 1, 2, 3)).item() + 1e-8
        
        sigma_experts = []
        for name in experts.keys():
            e_act = torch.cat(expert_acts[name][idx], dim=0)
            sigma_expert = torch.std(e_act, dim=(0, 1, 2, 3)).item() + 1e-8
            sigma_experts.append(sigma_expert)
            
        sigma_target = np.mean(sigma_experts)
        gamma = sigma_target / sigma_merged
        gammas.append(gamma)
        
    return gammas

def apply_wfc_folding(model, gammas):
    folded_model = copy.deepcopy(model)
    bn_modules = []
    for m in folded_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_modules.append(m)
            
    with torch.no_grad():
        for idx, m in enumerate(bn_modules):
            m.weight.mul_(gammas[idx])
            m.bias.mul_(gammas[idx])
    return folded_model

def k_fold_cv_select_reg(features, targets, W0_bias=None, reg_list=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], num_folds=5):
    M = features.size(0)
    num_folds = min(num_folds, M)
    if num_folds < 2:
        return 1000.0 if W0_bias is not None else 1.0
        
    indices = np.arange(M)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_sizes = np.full(num_folds, M // num_folds, dtype=int)
    fold_sizes[:M % num_folds] += 1
    current = 0
    folds = []
    for size in fold_sizes:
        folds.append(indices[current:current + size])
        current += size
        
    best_reg = None
    best_val_score = float('inf')
    
    D_plus_1 = features.size(1)
    I = torch.eye(D_plus_1, device=device)
    
    for reg in reg_list:
        total_val_loss = 0.0
        for f in range(num_folds):
            val_idx = folds[f]
            train_idx = np.hstack([folds[i] for i in range(num_folds) if i != f])
            
            F_train = features[train_idx]
            Y_train = targets[train_idx]
            F_val = features[val_idx]
            Y_val = targets[val_idx]
            
            lhs = F_train.t() @ F_train + reg * I
            if W0_bias is not None:
                Y_tilde = Y_train - F_train @ W0_bias.t()
                rhs = F_train.t() @ Y_tilde
                V_bias = torch.linalg.solve(lhs, rhs)
            else:
                rhs = F_train.t() @ Y_train
                V_bias = torch.linalg.solve(lhs, rhs)
                
            if W0_bias is not None:
                W_bias = W0_bias.t() + V_bias
            else:
                W_bias = V_bias
                
            pred_val = F_val @ W_bias
            val_loss = torch.mean((pred_val - Y_val) ** 2).item()
            total_val_loss += val_loss
            
        avg_val_loss = total_val_loss / num_folds
        
        if avg_val_loss < best_val_score:
            best_val_score = avg_val_loss
            best_reg = reg
            
    return best_reg

def apply_autotuned_pr_lsha(backbone, experts_paths, task_loaders, N, beta=10.0):
    backbone.eval()
    aligned_heads = {}
    chosen_regs = {}
    
    for task_name, loader in task_loaders.items():
        expert = models.resnet18(weights=None)
        expert.fc = nn.Linear(expert.fc.in_features, 10)
        expert.load_state_dict(torch.load(experts_paths[task_name], map_location=device))
        expert = expert.to(device)
        expert.eval()
        
        W0_weight = expert.fc.weight.data
        W0_bias = expert.fc.bias.data
        W0 = torch.cat([W0_weight, W0_bias.unsqueeze(1)], dim=1)
        
        target_logits_list = []
        features = []
        def feat_hook(module, input, output):
            features.append(output.flatten(1).detach())
            
        hook = backbone.avgpool.register_forward_hook(feat_hook)
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                _ = backbone(inputs)
                logits = expert(inputs)
                target_logits_list.append(logits.detach())
                
        hook.remove()
        
        F = torch.cat(features, dim=0)
        Y = torch.cat(target_logits_list, dim=0)
        F_bias = torch.cat([F, torch.ones(F.size(0), 1, device=device)], dim=1)
        
        # Calculate dimension-to-sample ratio
        D = F.size(1)
        M = F.size(0)
        ratio_bound = beta * (D / M)
        
        # Candidate set
        full_grid = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
        # Restrict candidate set to prevent overfitting under high D/M ratio
        reg_candidates = [r for r in full_grid if r >= ratio_bound]
        if not reg_candidates:
            reg_candidates = [max(full_grid)]
            
        best_reg = k_fold_cv_select_reg(F_bias, Y, W0_bias=W0, reg_list=reg_candidates, num_folds=5)
        chosen_regs[task_name] = best_reg
        
        I = torch.eye(F_bias.size(1), device=device)
        lhs = F_bias.t() @ F_bias + best_reg * I
        Y_tilde = Y - F_bias @ W0.t()
        rhs = F_bias.t() @ Y_tilde
        V_bias = torch.linalg.solve(lhs, rhs)
        W_bias = W0.t() + V_bias
        
        W_new = W_bias[:-1, :].t()
        b_new = W_bias[-1, :]
        
        aligned_heads[task_name] = {
            'weight': W_new,
            'bias': b_new
        }
        
    return aligned_heads, chosen_regs

def evaluate_multi_task_aligned_heads(backbone, aligned_heads, test_loaders):
    backbone.eval()
    accuracies = {}
    
    for task_name, test_loader in test_loaders.items():
        head_params = aligned_heads[task_name]
        
        old_fc = copy.deepcopy(backbone.fc)
        backbone.fc.weight.data.copy_(head_params['weight'])
        backbone.fc.bias.data.copy_(head_params['bias'])
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = backbone(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        accuracies[task_name] = correct / total
        backbone.fc = old_fc
        
    accuracies['avg'] = sum(accuracies.values()) / len(accuracies)
    return accuracies

if __name__ == '__main__':
    experts_paths = {
        'mnist': 'mnist_expert.pth',
        'fashion': 'fashion_expert.pth',
        'cifar': 'cifar_expert.pth'
    }
    
    wa_backbone = get_weight_averaged_backbone()
    budgets = [16, 64, 256]
    betas = [1.0, 5.0, 10.0, 20.0, 50.0]
    
    print("\nEvaluating Constrained Cross-Validation (Dim-CV) with 100% Fidelity...")
    for N in budgets:
        print(f"\n--- BUDGET N={N} ---")
        joint_loader, task_loaders_cal = get_calibration_loader(N, seed=42)
        
        gammas = compute_sp_taac_scales(wa_backbone, experts_paths, joint_loader, task_loaders_cal)
        wfc_backbone = apply_wfc_folding(wa_backbone, gammas)
        
        for beta in betas:
            aligned_heads, chosen_regs = apply_autotuned_pr_lsha(wfc_backbone, experts_paths, task_loaders_cal, N, beta=beta)
            accs = evaluate_multi_task_aligned_heads(wfc_backbone, aligned_heads, test_loaders)
            print(f"  beta={beta:4g} -> Chosen lambda: MNIST={chosen_regs['mnist']:5g}, Fashion={chosen_regs['fashion']:5g}, CIFAR={chosen_regs['cifar']:5g} | Test Acc: {accs['avg']*100:.2f}% (MNIST: {accs['mnist']*100:.2f} | Fashion: {accs['fashion']*100:.2f} | CIFAR: {accs['cifar']*100:.2f})")
