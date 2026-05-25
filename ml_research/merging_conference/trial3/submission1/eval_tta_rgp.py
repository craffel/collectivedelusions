import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure cuDNN is disabled to prevent initialization errors
torch.backends.cudnn.enabled = False

# Model architecture definitions (must match train_experts.py)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(x)

# Corruptions implementation
def add_gaussian_noise(x, severity=0.3):
    return x + severity * torch.randn_like(x)

def add_gaussian_blur(x, severity=1.5):
    kernel_size = 5
    sigma = severity
    x_coord = torch.arange(kernel_size) - (kernel_size - 1) / 2
    gaussian_1d = torch.exp(-x_coord**2 / (2 * sigma**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d.unsqueeze(1) * gaussian_1d.unsqueeze(0)
    gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0).to(x.device)
    return F.conv2d(x, gaussian_2d, padding=kernel_size//2)

def adjust_contrast(x, severity=0.3):
    mean = x.mean(dim=[-1, -2], keepdim=True)
    return (x - mean) * severity + mean

# SPOR Regularization (Relative to Expert Head)
def compute_spor(head, expert_head):
    W = head.fc2.weight
    W0 = expert_head.fc2.weight
    C = W.size(0)
    W_norm = F.normalize(W, p=2, dim=1)
    W0_norm = F.normalize(W0, p=2, dim=1)
    M = torch.matmul(W_norm, W0_norm.t())
    MMT = torch.matmul(M, M.t())
    I = torch.eye(C, device=W.device)
    spor = torch.sum((MMT - I)**2) / (C**2)
    return spor

# Relative Geometry Preservation (RGP) Regularization
def compute_rgp(head, expert_head):
    W = head.fc2.weight
    W0 = expert_head.fc2.weight
    C = W.size(0)
    W_norm = F.normalize(W, p=2, dim=1)
    W0_norm = F.normalize(W0, p=2, dim=1)
    
    gram = torch.matmul(W_norm, W_norm.t())
    gram0 = torch.matmul(W0_norm, W0_norm.t())
    
    rgp = torch.sum((gram - gram0)**2) / (C**2)
    return rgp

def get_corrupted_batch(x, corr_type):
    if corr_type == 'clean':
        return x
    elif corr_type == 'noise':
        return add_gaussian_noise(x, severity=0.4)
    elif corr_type == 'blur':
        return add_gaussian_blur(x, severity=2.0)
    elif corr_type == 'contrast':
        return adjust_contrast(x, severity=0.15)
    else:
        return x

def run_evaluation(device, corruption='clean', use_softmax_tmm=False):
    base_encoder_sd = torch.load("checkpoints/base_encoder.pt", map_location=device)
    
    tasks = ["mnist", "fashion", "kmnist"]
    expert_encoders = {}
    expert_heads = {}
    task_vectors = {}
    
    for task in tasks:
        enc_sd = torch.load(f"checkpoints/expert_{task}_encoder.pt", map_location=device)
        head_sd = torch.load(f"checkpoints/expert_{task}_head.pt", map_location=device)
        
        enc = Encoder().to(device)
        enc.load_state_dict(enc_sd)
        enc.eval()
        expert_encoders[task] = enc
        
        hd = Head().to(device)
        hd.load_state_dict(head_sd)
        hd.eval()
        expert_heads[task] = hd
        
        task_vectors[task] = {}
        for k in base_encoder_sd.keys():
            task_vectors[task][k] = enc_sd[k] - base_encoder_sd[k]
            
    # Datasets and Loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_loaders = {}
    test_datasets = {
        "mnist": datasets.MNIST(root='./data', train=False, download=True, transform=transform),
        "fashion": datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform),
        "kmnist": datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    }
    
    for task in tasks:
        subset_indices = list(range(1000))
        subset_ds = torch.utils.data.Subset(test_datasets[task], subset_indices)
        test_loaders[task] = torch.utils.data.DataLoader(subset_ds, batch_size=64, shuffle=False)

    methods = ["Standard TTA", "SATA-SBF (Ours)", "SATA-RGP (Enhanced Ours)"]
    method_accuracies = {m: {} for m in methods}
    
    for m in methods:
        # Crucial for deterministic behavior and 100% fair comparison across methods!
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize TTA parameters
        keys = list(base_encoder_sd.keys())
        lambdas = torch.full((len(keys), 3), 0.33, dtype=torch.float32, device=device, requires_grad=True)
        
        adapted_heads = {}
        for task in tasks:
            adapted_heads[task] = Head().to(device)
            adapted_heads[task].load_state_dict(expert_heads[task].state_dict())
            for p in adapted_heads[task].parameters():
                p.requires_grad = True
                
        head_params = []
        for task in tasks:
            head_params += list(adapted_heads[task].parameters())
            
        optimizer = torch.optim.Adam([
            {"params": [lambdas], "lr": 0.005},
            {"params": head_params, "lr": 0.05}
        ])
        
        active_params = [lambdas] + head_params
        
        running_fisher = {}
        for p in active_params:
            running_fisher[id(p)] = torch.zeros_like(p)
            
        fisher_momentum = 0.99
        rho = 0.05  # SAM perturbation radius
        spor_beta = 0.1 # SPOR weight
        
        task_correct = {task: 0 for task in tasks}
        task_total = {task: 0 for task in tasks}
        
        iterators = {task: iter(test_loaders[task]) for task in tasks}
        max_batches = max(len(test_loaders[task]) for task in tasks)
        
        for batch_idx in range(max_batches):
            for task_idx, task in enumerate(tasks):
                try:
                    x, y = next(iterators[task])
                except StopIteration:
                    continue
                
                x, y = x.to(device), y.to(device)
                x_corr = get_corrupted_batch(x, corruption)
                
                with torch.no_grad():
                    expert_features = expert_encoders[task](x)
                    expert_logits = expert_heads[task](expert_features)
                    expert_probs = F.softmax(expert_logits, dim=1)
                
                def forward_merged(x_input):
                    merged_sd = copy.deepcopy(base_encoder_sd)
                    # Apply softmax or unconstrained
                    if use_softmax_tmm and m != "Standard TTA":
                        lambdas_normalized = F.softmax(lambdas, dim=1)
                    else:
                        lambdas_normalized = lambdas
                        
                    for i, k in enumerate(keys):
                        merged_sd[k] = merged_sd[k] + lambdas_normalized[i, 0]*task_vectors["mnist"][k] + lambdas_normalized[i, 1]*task_vectors["fashion"][k] + lambdas_normalized[i, 2]*task_vectors["kmnist"][k]
                    
                    temp_encoder = Encoder().to(device)
                    temp_encoder.load_state_dict(merged_sd)
                    
                    features = temp_encoder(x_input)
                    logits = adapted_heads[task](features)
                    return logits
                
                # Eval
                with torch.no_grad():
                    logits_eval = forward_merged(x_corr)
                    _, predicted = logits_eval.max(1)
                    task_correct[task] += predicted.eq(y).sum().item()
                    task_total[task] += y.size(0)
                
                # Step
                optimizer.zero_grad()
                logits_merged = forward_merged(x_corr)
                probs_merged = F.softmax(logits_merged, dim=1)
                
                kl_loss = F.kl_div(probs_merged.log(), expert_probs, reduction='batchmean')
                
                # Regularization choice
                reg_loss = torch.tensor(0.0, device=device)
                if m == "SATA-SBF (Ours)":
                    entropy = -torch.sum(probs_merged * torch.log(probs_merged + 1e-8), dim=1).mean()
                    log_C = 2.302585
                    confidence = torch.clamp(1.0 - (entropy / log_C), min=0.0, max=1.0)
                    spor_beta_dynamic = (spor_beta * confidence).detach()
                    reg_loss = compute_spor(adapted_heads[task], expert_heads[task])
                    total_loss = kl_loss + (spor_beta_dynamic * reg_loss)
                elif m == "SATA-RGP (Enhanced Ours)":
                    entropy = -torch.sum(probs_merged * torch.log(probs_merged + 1e-8), dim=1).mean()
                    log_C = 2.302585
                    confidence = torch.clamp(1.0 - (entropy / log_C), min=0.0, max=1.0)
                    spor_beta_dynamic = (spor_beta * confidence).detach()
                    reg_loss = compute_rgp(adapted_heads[task], expert_heads[task])
                    total_loss = kl_loss + (spor_beta_dynamic * reg_loss)
                else:
                    total_loss = kl_loss
                
                if m == "Standard TTA":
                    total_loss.backward()
                    optimizer.step()
                    
                elif m in ["SATA-SBF (Ours)", "SATA-RGP (Enhanced Ours)"]:
                    total_loss.backward()
                    
                    # Update running Fisher
                    for p in active_params:
                        if p.grad is not None:
                            running_fisher[id(p)] = fisher_momentum * running_fisher[id(p)] + (1.0 - fisher_momentum) * (p.grad.data ** 2)
                    
                    # Compute SBF perturbation
                    grad_terms = []
                    t_factors = {}
                    for p in active_params:
                        if p.grad is not None:
                            F_tensor = running_fisher[id(p)]
                            F_mean = F_tensor.mean()
                            t = torch.exp(-F_tensor / (F_mean + 1e-8))
                            t_factors[id(p)] = t
                            grad_terms.append((t**2 * p.grad.data)**2)
                    
                    denom = torch.sqrt(sum(gt.sum() for gt in grad_terms) + 1e-8)
                    
                    # Perturb
                    for p in active_params:
                        if p.grad is not None:
                            t = t_factors[id(p)]
                            p.data += rho * (t**2 * p.grad.data) / denom
                            
                    # Second backward
                    optimizer.zero_grad()
                    logits_perturbed = forward_merged(x_corr)
                    probs_perturbed = F.softmax(logits_perturbed, dim=1)
                    loss_perturbed = F.kl_div(probs_perturbed.log(), expert_probs, reduction='batchmean')
                    
                    if m == "SATA-SBF (Ours)":
                        spor_perturbed = compute_spor(adapted_heads[task], expert_heads[task])
                    else:
                        spor_perturbed = compute_rgp(adapted_heads[task], expert_heads[task])
                        
                    total_loss_perturbed = loss_perturbed + (spor_beta_dynamic * spor_perturbed)
                    total_loss_perturbed.backward()
                    
                    # Restore and step
                    for p in active_params:
                        if p.grad is not None:
                            t = t_factors[id(p)]
                            p.data -= rho * (t**2 * p.grad.data) / denom
                            
                    optimizer.step()
        
        avg_acc = 0.0
        for task in tasks:
            acc = 100.0 * task_correct[task] / task_total[task]
            method_accuracies[m][task] = acc
            avg_acc += acc
        avg_acc /= len(tasks)
        method_accuracies[m]["average"] = avg_acc
        
    return method_accuracies

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    corruptions = ['clean', 'noise', 'blur', 'contrast']
    
    print("\n--- Running WITHOUT Softmax TMM normalization ---")
    results_unconstrained = {}
    for corr in corruptions:
        results_unconstrained[corr] = run_evaluation(device, corruption=corr, use_softmax_tmm=False)
        print(f"Corruption: {corr:8s} | Standard: {results_unconstrained[corr]['Standard TTA']['average']:.2f}% | SATA-SBF (Ours): {results_unconstrained[corr]['SATA-SBF (Ours)']['average']:.2f}% | SATA-RGP (Enhanced): {results_unconstrained[corr]['SATA-RGP (Enhanced Ours)']['average']:.2f}%")
        
    print("\n--- Running WITH Softmax TMM normalization ---")
    results_softmax = {}
    for corr in corruptions:
        results_softmax[corr] = run_evaluation(device, corruption=corr, use_softmax_tmm=True)
        print(f"Corruption: {corr:8s} | Standard: {results_softmax[corr]['Standard TTA']['average']:.2f}% | SATA-SBF (Ours): {results_softmax[corr]['SATA-SBF (Ours)']['average']:.2f}% | SATA-RGP (Enhanced): {results_softmax[corr]['SATA-RGP (Enhanced Ours)']['average']:.2f}%")
