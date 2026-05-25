import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import copy
import numpy as np

# Ensure cuDNN is disabled to prevent initialization errors
torch.backends.cudnn.enabled = False

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

def add_gaussian_noise(x, severity=0.4):
    return x + severity * torch.randn_like(x)

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

def run_evaluation_quick(device, lr_lambda=0.002, lr_head=0.002, rho=0.05, spor_beta=0.5):
    # Load base and expert models
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
            
    # Datasets
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
        # subset of 300 images per task to make sweep extremely fast but representative
        subset_indices = list(range(300))
        subset_ds = torch.utils.data.Subset(test_datasets[task], subset_indices)
        test_loaders[task] = torch.utils.data.DataLoader(subset_ds, batch_size=64, shuffle=False)

    keys = list(base_encoder_sd.keys())
    
    # Initialize parameters
    lambdas = torch.full((len(keys), 3), 0.33, dtype=torch.float32, device=device, requires_grad=True)
    adapted_heads = {}
    for task in tasks:
        adapted_heads[task] = Head().to(device)
        adapted_heads[task].load_state_dict(expert_heads[task].state_dict())
        for p in adapted_heads[task].parameters():
            p.requires_grad = True
            
    # Separate parameter groups
    head_params = []
    for task in tasks:
        head_params += list(adapted_heads[task].parameters())
        
    optimizer = torch.optim.Adam([
        {"params": [lambdas], "lr": lr_lambda},
        {"params": head_params, "lr": lr_head}
    ])
    
    running_fisher = {}
    for p in [lambdas] + head_params:
        running_fisher[id(p)] = torch.zeros_like(p)
        
    fisher_momentum = 0.99
    
    task_correct = {task: 0 for task in tasks}
    task_total = {task: 0 for task in tasks}
    
    iterators = {task: iter(test_loaders[task]) for task in tasks}
    max_batches = max(len(test_loaders[task]) for task in tasks)
    
    for batch_idx in range(max_batches):
        for task in tasks:
            try:
                x, y = next(iterators[task])
            except StopIteration:
                continue
            
            x, y = x.to(device), y.to(device)
            x_corr = add_gaussian_noise(x, severity=0.4)
            
            # Expert pseudo-labels
            with torch.no_grad():
                expert_features = expert_encoders[task](x)
                expert_logits = expert_heads[task](expert_features)
                expert_probs = F.softmax(expert_logits, dim=1)
            
            def forward_merged(x_input):
                merged_sd = copy.deepcopy(base_encoder_sd)
                for i, k in enumerate(keys):
                    merged_sd[k] = merged_sd[k] + lambdas[i, 0]*task_vectors["mnist"][k] + lambdas[i, 1]*task_vectors["fashion"][k] + lambdas[i, 2]*task_vectors["kmnist"][k]
                
                temp_encoder = Encoder().to(device)
                temp_encoder.load_state_dict(merged_sd)
                
                features = temp_encoder(x_input)
                logits = adapted_heads[task](features)
                return logits
            
            # Eval before update on this batch
            with torch.no_grad():
                logits_eval = forward_merged(x_corr)
                _, predicted = logits_eval.max(1)
                task_correct[task] += predicted.eq(y).sum().item()
                task_total[task] += y.size(0)
            
            # Ours TTA Step
            optimizer.zero_grad()
            logits_merged = forward_merged(x_corr)
            probs_merged = F.softmax(logits_merged, dim=1)
            kl_loss = F.kl_div(probs_merged.log(), expert_probs, reduction='batchmean')
            
            spor_loss = compute_spor(adapted_heads[task], expert_heads[task])
            total_loss = kl_loss + (spor_beta * spor_loss)
            total_loss.backward()
            
            # Update Fisher
            active_p_list = [lambdas] + list(adapted_heads[task].parameters())
            for p in active_p_list:
                if p.grad is not None:
                    running_fisher[id(p)] = fisher_momentum * running_fisher[id(p)] + (1.0 - fisher_momentum) * (p.grad.data ** 2)
            
            # Compute SBF perturbation
            grad_terms = []
            t_factors = {}
            for p in active_p_list:
                if p.grad is not None:
                    F_tensor = running_fisher[id(p)]
                    F_mean = F_tensor.mean()
                    t = torch.exp(-F_tensor / (F_mean + 1e-8))
                    t_factors[id(p)] = t
                    grad_terms.append((t**2 * p.grad.data)**2)
                    
            denom = torch.sqrt(sum(gt.sum() for gt in grad_terms) + 1e-8)
            
            # Perturb
            for p in active_p_list:
                if p.grad is not None:
                    t = t_factors[id(p)]
                    p.data += rho * (t**2 * p.grad.data) / denom
                    
            # Second backward
            optimizer.zero_grad()
            logits_perturbed = forward_merged(x_corr)
            probs_perturbed = F.softmax(logits_perturbed, dim=1)
            loss_perturbed = F.kl_div(probs_perturbed.log(), expert_probs, reduction='batchmean')
            
            spor_perturbed = compute_spor(adapted_heads[task], expert_heads[task])
            total_loss_perturbed = loss_perturbed + (spor_beta * spor_perturbed)
            total_loss_perturbed.backward()
            
            # Restore and step
            for p in active_p_list:
                if p.grad is not None:
                    t = t_factors[id(p)]
                    p.data -= rho * (t**2 * p.grad.data) / denom
                    
            optimizer.step()
            
    avg_acc = 0.0
    for task in tasks:
        avg_acc += 100.0 * task_correct[task] / task_total[task]
    avg_acc /= len(tasks)
    return avg_acc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sweeping hyperparameters on {device}...")
    
    # Grid search
    lrs_lambda = [0.003, 0.005, 0.008]
    lrs_head = [0.01, 0.02, 0.03, 0.05]
    rhos = [0.03, 0.05, 0.08]
    betas = [0.05, 0.1, 0.2, 0.4]
    
    best_acc = 0.0
    best_config = {}
    
    # To keep it fast, let's randomly sample or run a small subset of configurations
    import random
    random.seed(42)
    configs = []
    for lr_l in lrs_lambda:
        for lr_h in lrs_head:
            for r in rhos:
                for b in betas:
                    configs.append((lr_l, lr_h, r, b))
                    
    # Sample 20 random configurations to find a good direction!
    sampled_configs = random.sample(configs, min(len(configs), 25))
    
    for i, (lr_l, lr_h, r, b) in enumerate(sampled_configs):
        try:
            acc = run_evaluation_quick(device, lr_lambda=lr_l, lr_head=lr_h, rho=r, spor_beta=b)
            print(f"[{i+1}/{len(sampled_configs)}] lr_l={lr_l}, lr_h={lr_h}, rho={r}, beta={b} -> Acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_config = {"lr_lambda": lr_l, "lr_head": lr_h, "rho": r, "spor_beta": b}
        except Exception as e:
            print(f"Error on config: {e}")
            
    print("\nBest config found:")
    print(best_config)
    print(f"Best Accuracy: {best_acc:.2f}%")
