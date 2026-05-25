import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.func import functional_call

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Disable cuDNN to avoid cluster driver mismatches
    torch.backends.cudnn.enabled = False

# Define SimpleCNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) # Output: 26x26
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # Output: 24x24
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2) # Output: 12x12
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.50)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        feat = F.relu(self.fc1(x))
        x = self.dropout2(feat)
        out = self.fc2(x)
        return out, feat

# Gaussian noise addition
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.6):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn(tensor.size()) * self.std + self.mean, 0., 1.)

# Loader for Datasets
def get_datasets(dry_run=False):
    transform_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform_clean)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_clean)
    
    fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_clean)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_clean)
    
    kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform_clean)
    
    if dry_run:
        mnist_train = Subset(mnist_train, list(range(100)))
        mnist_test = Subset(mnist_test, list(range(50)))
        fmnist_train = Subset(fmnist_train, list(range(100)))
        fmnist_test = Subset(fmnist_test, list(range(50)))
        kmnist_test = Subset(kmnist_test, list(range(50)))
        
    return mnist_train, mnist_test, fmnist_train, fmnist_test, kmnist_test

# Train an expert model
def train_expert(model, train_dataset, device, epochs=2):
    model.train()
    loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model

# Evaluate a model's accuracy on a test dataset
def evaluate_model(model, test_dataset, device):
    model.eval()
    loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
    return correct / total

# Precompute prototypes in feature space
def precompute_prototypes(model, dataset, device, num_samples=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    data, target = next(iter(loader))
    data = data.to(device)
    
    with torch.no_grad():
        _, feats = model(data)
        
    feats = feats.cpu().numpy()
    target = target.numpy()
    
    # Precompute class-wise prototypes (centroids of features for each class 0-9)
    prototypes = []
    for c in range(10):
        class_feats = feats[target == c]
        if len(class_feats) > 0:
            prototypes.append(class_feats.mean(axis=0))
        else:
            prototypes.append(np.zeros_like(feats[0]))
            
    return torch.tensor(np.array(prototypes), device=device)

# Precompute offline diagonal Fisher sensitivities for CLW-Fisher
def precompute_fisher(model, dataset, device, num_samples=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    data, _ = next(iter(loader))
    data = data.to(device)
    
    # We will register hooks to compute diagonal Fisher on the entropy loss
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)
        
    outputs, _ = model(data)
    probs = F.softmax(outputs, dim=-1)
    log_probs = F.log_softmax(outputs, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1).mean()
    
    # Backpropagation to compute parameter sensitivities
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    entropy.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            fisher[name] = param.grad.data.clone().pow(2)
            
    return fisher

# Generate the 50 sequential batches representing our test stream
def generate_stream(mnist_test, fmnist_test, kmnist_test, dry_run=False):
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)
    fmnist_loader = DataLoader(fmnist_test, batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True)
    
    mnist_iter = iter(mnist_loader)
    fmnist_iter = iter(fmnist_loader)
    kmnist_iter = iter(kmnist_loader)
    
    batches = []
    
    # Segment 1: Clean MNIST (batches 0-9)
    for _ in range(5 if dry_run else 10):
        try:
            x, y = next(mnist_iter)
        except StopIteration:
            mnist_iter = iter(mnist_loader)
            x, y = next(mnist_iter)
        batches.append((x, y, "Clean MNIST"))
        
    # Segment 2: Noisy MNIST (batches 10-19)
    for _ in range(5 if dry_run else 10):
        try:
            x, y = next(mnist_iter)
        except StopIteration:
            mnist_iter = iter(mnist_loader)
            x, y = next(mnist_iter)
        x_noisy = torch.clamp(x + torch.randn_like(x) * 0.6, 0.0, 1.0)
        batches.append((x_noisy, y, "Noisy MNIST"))
        
    # Segment 3: Clean FashionMNIST (batches 20-29)
    for _ in range(5 if dry_run else 10):
        try:
            x, y = next(fmnist_iter)
        except StopIteration:
            fmnist_iter = iter(fmnist_loader)
            x, y = next(fmnist_iter)
        batches.append((x, y, "Clean FashionMNIST"))
        
    # Segment 4: Noisy FashionMNIST (batches 30-39)
    for _ in range(5 if dry_run else 10):
        try:
            x, y = next(fmnist_iter)
        except StopIteration:
            fmnist_iter = iter(fmnist_loader)
            x, y = next(fmnist_iter)
        x_noisy = torch.clamp(x + torch.randn_like(x) * 0.6, 0.0, 1.0)
        batches.append((x_noisy, y, "Noisy FashionMNIST"))
        
    # Segment 5: Novel KMNIST (batches 40-49)
    for _ in range(5 if dry_run else 10):
        try:
            x, y = next(kmnist_iter)
        except StopIteration:
            kmnist_iter = iter(kmnist_loader)
            x, y = next(kmnist_iter)
        batches.append((x, y, "Novel KMNIST"))
        
    return batches

# Compute distance gaps based on prototypes
def compute_prototype_distances(x_feats, prototypes_0, prototypes_1):
    feats_expanded = x_feats.unsqueeze(1) # [B, 1, D]
    dist_0 = torch.norm(feats_expanded - prototypes_0.unsqueeze(0), dim=2) # [B, 10]
    dist_1 = torch.norm(feats_expanded - prototypes_1.unsqueeze(0), dim=2) # [B, 10]
    min_dist_0 = dist_0.min(dim=1)[0] # [B]
    min_dist_1 = dist_1.min(dim=1)[0] # [B]
    return min_dist_0.mean().item(), min_dist_1.mean().item()

# Extraction of model state_dict for functional call
def get_model_params_and_buffers(model):
    params = {k: v.clone().detach().requires_grad_(True) for k, v in model.named_parameters()}
    buffers = {k: v.clone().detach() for k, v in model.named_buffers()}
    return params, buffers

# Helper to merge parameters
def merge_parameters(params0, params1, lambdas):
    merged = {}
    for name in params0.keys():
        if isinstance(lambdas, dict):
            l = lambdas[name]
        else:
            l = lambdas
        merged[name] = l * params0[name] + (1 - l) * params1[name]
    return merged

# Helper to fuse Batch Normalization buffers using moment matching
def fuse_bn_buffers(buffers0, buffers1, w):
    if isinstance(w, torch.Tensor):
        w = w.detach()
    fused = {}
    for name in buffers0.keys():
        if "running_mean" in name:
            fused[name] = w * buffers0[name] + (1 - w) * buffers1[name]
        elif "running_var" in name:
            mean_name = name.replace("running_var", "running_mean")
            mu0 = buffers0[mean_name]
            mu1 = buffers1[mean_name]
            mu_fused = w * mu0 + (1 - w) * mu1
            fused[name] = w * (buffers0[name] + (mu0 - mu_fused).pow(2)) + (1 - w) * (buffers1[name] + (mu1 - mu_fused).pow(2))
        else:
            fused[name] = buffers0[name]
    return fused


# ==========================================
# 1. Static Merging Baseline
# ==========================================
def run_static_merging(batches, model_0, model_1, base_model, device):
    params0, buffers0 = get_model_params_and_buffers(model_0)
    params1, buffers1 = get_model_params_and_buffers(model_1)
    
    merged_params = merge_parameters(params0, params1, 0.5)
    fused_buffers = fuse_bn_buffers(buffers0, buffers1, 0.5)
    
    segment_accs = {}
    overall_correct = 0
    overall_total = 0
    
    for x, y, segment in batches:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            outputs, _ = functional_call(base_model, (merged_params, fused_buffers), x)
            pred = outputs.argmax(dim=1)
            correct = pred.eq(y).sum().item()
            
        segment_accs[segment] = segment_accs.get(segment, []) + [(correct, len(x))]
        overall_correct += correct
        overall_total += len(x)
        
    return {seg: np.sum([c for c, _ in val])/np.sum([t for _, t in val]) for seg, val in segment_accs.items()}, overall_correct / overall_total


# ==========================================
# 2. Fixed TTA (Unconstrained Entropy Minimization)
# ==========================================
def run_fixed_tta(batches, model_0, model_1, base_model, device, steps=5, lr=0.01):
    params0, buffers0 = get_model_params_and_buffers(model_0)
    params1, buffers1 = get_model_params_and_buffers(model_1)
    
    segment_accs = {}
    overall_correct = 0
    overall_total = 0
    
    for x, y, segment in batches:
        x, y = x.to(device), y.to(device)
        
        w_logits = {k: torch.tensor(0.0, device=device, requires_grad=True) for k in params0.keys()}
        optimizer = torch.optim.SGD(list(w_logits.values()), lr=lr)
        
        for step in range(steps):
            optimizer.zero_grad()
            lambdas = {k: torch.sigmoid(v) for k, v in w_logits.items()}
            merged_params = merge_parameters(params0, params1, lambdas)
            
            mean_lambda = torch.mean(torch.stack([l for l in lambdas.values()]))
            fused_buffers = fuse_bn_buffers(buffers0, buffers1, mean_lambda)
            
            outputs, _ = functional_call(base_model, (merged_params, fused_buffers), x)
            probs = F.softmax(outputs, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
            
            entropy.backward()
            optimizer.step()
            
        with torch.no_grad():
            lambdas = {k: torch.sigmoid(v) for k, v in w_logits.items()}
            merged_params = merge_parameters(params0, params1, lambdas)
            mean_lambda = torch.mean(torch.stack([l for l in lambdas.values()]))
            fused_buffers = fuse_bn_buffers(buffers0, buffers1, mean_lambda)
            
            outputs, _ = functional_call(base_model, (merged_params, fused_buffers), x)
            pred = outputs.argmax(dim=1)
            correct = pred.eq(y).sum().item()
            
        segment_accs[segment] = segment_accs.get(segment, []) + [(correct, len(x))]
        overall_correct += correct
        overall_total += len(x)
        
    return {seg: np.sum([c for c, _ in val])/np.sum([t for _, t in val]) for seg, val in segment_accs.items()}, overall_correct / overall_total


# ==========================================
# 3. CLW-Fisher Baseline (using precomputed Fisher on source dataset)
# ==========================================
def run_clw_fisher(batches, model_0, model_1, base_model, device, prototypes_0, prototypes_1, source_fisher, steps=5, lr=0.05, beta=1.5, gamma=0.02, s=3.0, eps_stab=150.0):
    params0, buffers0 = get_model_params_and_buffers(model_0)
    params1, buffers1 = get_model_params_and_buffers(model_1)
    
    normalized_fisher = {}
    total_fisher_sum = sum(f.sum().item() for f in source_fisher.values())
    for k, v in source_fisher.items():
        normalized_fisher[k] = v / (total_fisher_sum + 1e-8)
        
    segment_accs = {}
    overall_correct = 0
    overall_total = 0
    
    for x, y, segment in batches:
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            merged_params = merge_parameters(params0, params1, 0.5)
            fused_buffers = fuse_bn_buffers(buffers0, buffers1, 0.5)
            _, feats = functional_call(base_model, (merged_params, fused_buffers), x)
            
            d0, d1 = compute_prototype_distances(feats, prototypes_0, prototypes_1)
            gap = abs(d0 - d1)
            tau_self = gap / s + eps_stab
            
            scores = torch.tensor([-d0, -d1], device=device)
            routing_prior = F.softmax(scores / tau_self, dim=0)
            p = routing_prior[0].item()
            
        p_clamp = np.clip(p, 1e-4, 1 - 1e-4)
        w_global_init = np.log(p_clamp / (1.0 - p_clamp))
        
        w_global = torch.tensor(w_global_init, device=device, requires_grad=True)
        delta = {k: torch.tensor(0.0, device=device, requires_grad=True) for k in params0.keys()}
        
        optimizer = torch.optim.SGD([w_global] + list(delta.values()), lr=lr)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            lambdas = {k: torch.sigmoid(w_global + d) for k, d in delta.items()}
            merged_params = merge_parameters(params0, params1, lambdas)
            
            mean_lambda = torch.mean(torch.stack([l for l in lambdas.values()]))
            fused_buffers = fuse_bn_buffers(buffers0, buffers1, mean_lambda)
            
            outputs, _ = functional_call(base_model, (merged_params, fused_buffers), x)
            probs = F.softmax(outputs, dim=-1)
            loss_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
            
            lambdas_tensor = torch.stack([l for l in lambdas.values()])
            p_target = torch.tensor([p_clamp, 1.0 - p_clamp], device=device)
            lambda_avg = torch.mean(lambdas_tensor)
            p_pred = torch.stack([lambda_avg, 1.0 - lambda_avg])
            loss_kl = F.kl_div(p_pred.log().unsqueeze(0), p_target.unsqueeze(0), reduction='batchmean')
            
            loss_coherence = gamma * sum(d.pow(2).sum() for d in delta.values())
            
            loss = loss_entropy + beta * loss_kl + loss_coherence
            loss.backward()
            
            with torch.no_grad():
                w_global.grad.copy_(w_global.grad)
                for k, d in delta.items():
                    if d.grad is not None:
                        preconditioner = 1.0 / (normalized_fisher[k].mean() + 1e-5)
                        d.grad.copy_(d.grad * preconditioner)
                        
            optimizer.step()
            
        with torch.no_grad():
            lambdas = {k: torch.sigmoid(w_global + d) for k, d in delta.items()}
            merged_params = merge_parameters(params0, params1, lambdas)
            mean_lambda = torch.mean(torch.stack([l for l in lambdas.values()]))
            fused_buffers = fuse_bn_buffers(buffers0, buffers1, mean_lambda)
            
            outputs, _ = functional_call(base_model, (merged_params, fused_buffers), x)
            pred = outputs.argmax(dim=1)
            correct = pred.eq(y).sum().item()
            
        segment_accs[segment] = segment_accs.get(segment, []) + [(correct, len(x))]
        overall_correct += correct
        overall_total += len(x)
        
    return {seg: np.sum([c for c, _ in val])/np.sum([t for _, t in val]) for seg, val in segment_accs.items()}, overall_correct / overall_total


# ==========================================
# 4. KT-Fisher Baseline (unsupervised trace-based KFAC preconditioning + hard routing)
# ==========================================
def run_kt_fisher(batches, model_0, model_1, base_model, device, steps=5, lr=0.005, beta_damping=0.5, eps_scale=1e-5):
    params0, buffers0 = get_model_params_and_buffers(model_0)
    params1, buffers1 = get_model_params_and_buffers(model_1)
    
    segment_accs = {}
    overall_correct = 0
    overall_total = 0
    
    for x, y, segment in batches:
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            outputs_0, _ = model_0(x)
            outputs_1, _ = model_1(x)
            p0 = F.softmax(outputs_0, dim=-1)
            p1 = F.softmax(outputs_1, dim=-1)
            ent0 = -torch.sum(p0 * torch.log(p0 + 1e-10), dim=-1).mean().item()
            ent1 = -torch.sum(p1 * torch.log(p1 + 1e-10), dim=-1).mean().item()
            target_expert = 0 if ent0 < ent1 else 1
            target_vector = torch.tensor([1.0, 0.0] if target_expert == 0 else [0.0, 1.0], device=device)
            
        w_logits = {k: torch.tensor(0.0, device=device, requires_grad=True) for k in params0.keys()}
        
        for step in range(steps):
            lambdas = {k: torch.sigmoid(v) for k, v in w_logits.items()}
            merged_params = merge_parameters(params0, params1, lambdas)
            mean_lambda = torch.mean(torch.stack([l for l in lambdas.values()]))
            fused_buffers = fuse_bn_buffers(buffers0, buffers1, mean_lambda)
            
            activations = {}
            def get_activation_hook(name):
                def hook(module, input, output):
                    activations[name] = input[0].pow(2).mean().item()
                return hook
                
            hooks = []
            for name, module in base_model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    hooks.append(module.register_forward_hook(get_activation_hook(name)))
                    
            outputs, _ = functional_call(base_model, (merged_params, fused_buffers), x)
            probs = F.softmax(outputs, dim=-1)
            loss_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
            
            for h in hooks:
                h.remove()
                
            pre_grads = {}
            def get_pre_grad_hook(name):
                def hook(module, grad_input, grad_output):
                    pre_grads[name] = grad_output[0].pow(2).mean().item()
                return hook
                
            hooks = []
            for name, module in base_model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    hooks.append(module.register_full_backward_hook(get_pre_grad_hook(name)))
                    
            loss_entropy.backward()
            
            for h in hooks:
                h.remove()
                
            with torch.no_grad():
                for k, v in w_logits.items():
                    if v.grad is not None:
                        layer_name = k.split('.')[0]
                        act_norm = activations.get(layer_name, 1.0)
                        grad_norm = pre_grads.get(layer_name, 1.0) if (layer_name in pre_grads) else 1.0
                        F_w = act_norm * grad_norm
                        lr_precond = lr * ((F_w + eps_scale) ** (-beta_damping))
                        v.copy_(v - lr_precond * v.grad)
                        v.grad.zero_()
                        
        with torch.no_grad():
            lambdas = {k: torch.sigmoid(v) for k, v in w_logits.items()}
            merged_params = merge_parameters(params0, params1, lambdas)
            mean_lambda = torch.mean(torch.stack([l for l in lambdas.values()]))
            fused_buffers = fuse_bn_buffers(buffers0, buffers1, mean_lambda)
            
            outputs, _ = functional_call(base_model, (merged_params, fused_buffers), x)
            pred = outputs.argmax(dim=1)
            correct = pred.eq(y).sum().item()
            
        segment_accs[segment] = segment_accs.get(segment, []) + [(correct, len(x))]
        overall_correct += correct
        overall_total += len(x)
        
    return {seg: np.sum([c for c, _ in val])/np.sum([t for _, t in val]) for seg, val in segment_accs.items()}, overall_correct / overall_total


# ==========================================
# 5. DF-Bayes-TTMM Baseline
# ==========================================
def run_df_bayes_ttmm(batches, model_0, model_1, base_model, device, steps=3, lr=0.01, beta=0.5, temp_scaling=15.0, novelty_threshold=1.2):
    params0, buffers0 = get_model_params_and_buffers(model_0)
    params1, buffers1 = get_model_params_and_buffers(model_1)
    
    segment_accs = {}
    overall_correct = 0
    overall_total = 0
    
    for x, y, segment in batches:
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            outputs_0, _ = model_0(x)
            outputs_1, _ = model_1(x)
            p0 = F.softmax(outputs_0, dim=-1)
            p1 = F.softmax(outputs_1, dim=-1)
            ent0 = -torch.sum(p0 * torch.log(p0 + 1e-10), dim=-1).mean()
            ent1 = -torch.sum(p1 * torch.log(p1 + 1e-10), dim=-1).mean()
            avg_entropy = 0.5 * (ent0 + ent1).item()
            is_novel = avg_entropy > novelty_threshold
            
            if is_novel:
                w = torch.tensor([0.5, 0.5], device=device)
            else:
                logits = torch.stack([-ent0 * temp_scaling, -ent1 * temp_scaling])
                w = F.softmax(logits, dim=0)
            p = w[0].item()
            
        p_clamp = np.clip(p, 1e-10, 1.0 - 1e-10)
        init_logit = np.log(p_clamp / (1.0 - p_clamp))
        
        w_global = torch.tensor(init_logit, device=device, requires_grad=True)
        w_logits = {k: torch.tensor(init_logit, device=device, requires_grad=True) for k in params0.keys()}
        
        optimizer = torch.optim.Adam([w_global] + list(w_logits.values()), lr=lr)
        fused_buffers = fuse_bn_buffers(buffers0, buffers1, p_clamp)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            lambdas = {k: torch.sigmoid(v) for k, v in w_logits.items()}
            merged_params = merge_parameters(params0, params1, lambdas)
            
            mean_lambda = torch.sigmoid(w_global)
            fused_buffers = fuse_bn_buffers(buffers0, buffers1, mean_lambda)
            
            outputs, _ = functional_call(base_model, (merged_params, fused_buffers), x)
            probs = F.softmax(outputs, dim=-1)
            loss_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
            
            loss_prior = 0.5 * beta * (w_global - init_logit).pow(2)
            loss_prior += 0.5 * beta * sum((v - init_logit).pow(2) for v in w_logits.values())
            
            loss = loss_entropy + loss_prior
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            lambdas = {k: torch.sigmoid(v) for k, v in w_logits.items()}
            merged_params = merge_parameters(params0, params1, lambdas)
            mean_lambda = torch.sigmoid(w_global)
            fused_buffers = fuse_bn_buffers(buffers0, buffers1, mean_lambda)
            
            outputs, _ = functional_call(base_model, (merged_params, fused_buffers), x)
            pred = outputs.argmax(dim=1)
            correct = pred.eq(y).sum().item()
            
        segment_accs[segment] = segment_accs.get(segment, []) + [(correct, len(x))]
        overall_correct += correct
        overall_total += len(x)
        
    return {seg: np.sum([c for c, _ in val])/np.sum([t for _, t in val]) for seg, val in segment_accs.items()}, overall_correct / overall_total


# ==========================================
# 6. BK-CoMerge (Ours: Flagship Unified Framework)
# ==========================================
def run_bk_co_merge(batches, model_0, model_1, base_model, device, prototypes_0, prototypes_1, steps=5, lr=0.02, beta_kl=1.5, gamma_coherence=0.05, s_scale=3.0, eps_stab=0.1, novelty_threshold=1.2, temporal_smoothing=False, smoothing_gamma=0.8, p_history=None):
    params0, buffers0 = get_model_params_and_buffers(model_0)
    params1, buffers1 = get_model_params_and_buffers(model_1)
    
    segment_accs = {}
    overall_correct = 0
    overall_total = 0
    
    ema_gap = None
    
    # Initialize running pre_grads with 1.0 for each module
    running_pre_grads = {}
    for k in params0.keys():
        layer_name = k.split('.')[0]
        running_pre_grads[layer_name] = 1.0
    
    for idx_batch, (x, y, segment) in enumerate(batches):
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            outputs_0, feats_0 = model_0(x)
            outputs_1, feats_1 = model_1(x)
            p0 = F.softmax(outputs_0, dim=-1)
            p1 = F.softmax(outputs_1, dim=-1)
            ent0 = -torch.sum(p0 * torch.log(p0 + 1e-10), dim=-1).mean()
            ent1 = -torch.sum(p1 * torch.log(p1 + 1e-10), dim=-1).mean()
            avg_entropy = 0.5 * (ent0 + ent1).item()
            is_novel = avg_entropy > novelty_threshold
            
            if is_novel:
                p = 0.5
            else:
                # Correct gap calculation: use absolute difference in expert entropies as per SCTS equation
                gap = abs(ent0 - ent1).item()
                
                if temporal_smoothing:
                    if ema_gap is None:
                        ema_gap = gap
                    else:
                        ema_gap = smoothing_gamma * ema_gap + (1 - smoothing_gamma) * gap
                    current_gap = ema_gap
                else:
                    current_gap = gap
                    
                tau_self = current_gap / s_scale + eps_stab
                scores = torch.tensor([-ent0, -ent1], device=device)
                routing_prior = F.softmax(scores / tau_self, dim=0)
                p = routing_prior[0].item()
                
            if p_history is not None:
                p_history.append(p)
                
        p_clamp = np.clip(p, 1e-4, 1.0 - 1e-4)
        w_global_init = np.log(p_clamp / (1.0 - p_clamp))
        
        w_global = torch.tensor(w_global_init, device=device, requires_grad=True)
        delta = {k: torch.tensor(0.0, device=device, requires_grad=True) for k in params0.keys()}
        
        optimizer = torch.optim.SGD([w_global] + list(delta.values()), lr=lr)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            lambdas = {k: torch.sigmoid(w_global + d) for k, d in delta.items()}
            merged_params = merge_parameters(params0, params1, lambdas)
            
            mean_lambda = torch.sigmoid(w_global)
            fused_buffers = fuse_bn_buffers(buffers0, buffers1, mean_lambda)
            
            activations = {}
            def get_activation_hook(name):
                def hook(module, input, output):
                    activations[name] = input[0].pow(2).mean().item()
                return hook
                
            hooks = []
            for name, module in base_model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    hooks.append(module.register_forward_hook(get_activation_hook(name)))
                    
            outputs, _ = functional_call(base_model, (merged_params, fused_buffers), x)
            probs = F.softmax(outputs, dim=-1)
            loss_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
            
            for h in hooks:
                h.remove()
                
            lambdas_tensor = torch.stack([l for l in lambdas.values()])
            p_target = torch.tensor([p_clamp, 1.0 - p_clamp], device=device)
            lambda_avg = torch.mean(lambdas_tensor)
            p_pred = torch.stack([lambda_avg, 1.0 - lambda_avg])
            loss_kl = F.kl_div(p_pred.log().unsqueeze(0), p_target.unsqueeze(0), reduction='batchmean')
            
            pre_grads = {}
            def get_pre_grad_hook(name):
                def hook(module, grad_input, grad_output):
                    pre_grads[name] = grad_output[0].pow(2).mean().item()
                return hook
                
            hooks = []
            for name, module in base_model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    hooks.append(module.register_full_backward_hook(get_pre_grad_hook(name)))
                    
            loss_coherence = torch.tensor(0.0, device=device)
            for k, d in delta.items():
                layer_name = k.split('.')[0]
                act_norm = activations.get(layer_name, 1.0)
                grad_norm = running_pre_grads.get(layer_name, 1.0)
                F_w = act_norm * grad_norm
                loss_coherence += gamma_coherence * F_w * d.pow(2).sum()
                
            loss = loss_entropy + beta_kl * loss_kl + loss_coherence
            loss.backward()
            
            for h in hooks:
                h.remove()
                
            # Update running_pre_grads after the backward pass
            for layer_name, val in pre_grads.items():
                running_pre_grads[layer_name] = val
                
            with torch.no_grad():
                raw_sens = {}
                for k in delta.keys():
                    layer_name = k.split('.')[0]
                    act_norm = activations.get(layer_name, 1.0)
                    grad_norm = pre_grads.get(layer_name, 1.0) if (layer_name in pre_grads) else 1.0
                    raw_sens[k] = act_norm * grad_norm
                    
                total_sens = sum(raw_sens.values()) + 1e-8
                for k, d in delta.items():
                    if d.grad is not None:
                        norm_F = raw_sens[k] / total_sens
                        preconditioner = 1.0 / (norm_F + 1e-5)
                        d.grad.copy_(d.grad * preconditioner)
                        
            optimizer.step()
            
        with torch.no_grad():
            lambdas = {k: torch.sigmoid(w_global + d) for k, d in delta.items()}
            merged_params = merge_parameters(params0, params1, lambdas)
            mean_lambda = torch.sigmoid(w_global)
            fused_buffers = fuse_bn_buffers(buffers0, buffers1, mean_lambda)
            
            outputs, _ = functional_call(base_model, (merged_params, fused_buffers), x)
            pred = outputs.argmax(dim=1)
            correct = pred.eq(y).sum().item()
            
        segment_accs[segment] = segment_accs.get(segment, []) + [(correct, len(x))]
        overall_correct += correct
        overall_total += len(x)
        
    return {seg: np.sum([c for c, _ in val])/np.sum([t for _, t in val]) for seg, val in segment_accs.items()}, overall_correct / overall_total


def benchmark_runtimes(batches, model_0, model_1, base_model, device, prototypes_0, prototypes_1):
    print("\n--- Running Wall-Clock Adaptation Time Benchmark (10 batches) ---")
    bench_batches = batches[:10]
    
    # 1. DF-Bayes-TTMM (Adam, 5 steps for fair comparison)
    t0 = time.time()
    _ = run_df_bayes_ttmm(bench_batches, model_0, model_1, base_model, device, steps=5)
    t_bayes = (time.time() - t0) / len(bench_batches)
    print(f"DF-Bayes-TTMM (Adam, 5 steps) Average Time per Batch: {t_bayes:.4f} seconds")
    
    # 2. BK-CoMerge (Ours, 5 steps)
    t0 = time.time()
    _ = run_bk_co_merge(bench_batches, model_0, model_1, base_model, device, prototypes_0, prototypes_1, steps=5, temporal_smoothing=False)
    t_bk = (time.time() - t0) / len(bench_batches)
    print(f"BK-CoMerge (Ours, 5 steps) Average Time per Batch: {t_bk:.4f} seconds")
    
    speedup = t_bayes / t_bk
    print(f"BK-CoMerge relative speedup: {speedup:.2f}x")
    return t_bayes, t_bk, speedup


# ==========================================
# Main Experiment Execution Flow
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BK-CoMerge Experiments")
    parser.add_argument('--dry_run', action='store_true', help='Execute quick dry run on CPU')
    args = parser.parse_args()
    
    # Initialize set_seed immediately to disable cuDNN before model allocations
    set_seed(42)
    
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.dry_run) else 'cpu')
    print(f"Executing experiments on device: {device}")
    
    # 1. Load data
    print("Loading datasets...")
    mnist_train, mnist_test, fmnist_train, fmnist_test, kmnist_test = get_datasets(dry_run=args.dry_run)
    
    # 2. Train/Load Expert Models
    print("Initializing SimpleCNN Experts...")
    model_0 = SimpleCNN().to(device)
    model_1 = SimpleCNN().to(device)
    
    base_model = SimpleCNN().to(device)
    base_model.eval()
    for m in base_model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.train()
    
    epochs = 1 if args.dry_run else 2
    print(f"Training Expert 0 (MNIST) for {epochs} epochs...")
    set_seed(42)
    model_0 = train_expert(model_0, mnist_train, device, epochs=epochs)
    acc0 = evaluate_model(model_0, mnist_test, device)
    print(f"Expert 0 (MNIST) Accuracy: {acc0:.4%}")
    
    print(f"Training Expert 1 (FashionMNIST) for {epochs} epochs...")
    set_seed(42)
    model_1 = train_expert(model_1, fmnist_train, device, epochs=epochs)
    acc1 = evaluate_model(model_1, fmnist_test, device)
    print(f"Expert 1 (FashionMNIST) Accuracy: {acc1:.4%}")
    
    # 3. Precompute Prototypes
    print("Precomputing expert class-wise feature prototypes...")
    prot_samples = 32 if args.dry_run else 256
    set_seed(42)
    prototypes_0 = precompute_prototypes(model_0, mnist_train, device, num_samples=prot_samples)
    set_seed(42)
    prototypes_1 = precompute_prototypes(model_1, fmnist_train, device, num_samples=prot_samples)
    
    # 4. Precompute Offline Fisher for CLW-Fisher baseline
    print("Precomputing offline source Fisher Information on calibration sets...")
    source_fisher = precompute_fisher(model_0, mnist_train, device, num_samples=prot_samples)
    
    # 5. Generate Test Stream
    print("Generating non-stationary open-world test stream...")
    # Reset random seed immediately before calling generate_stream to ensure perfect reproducibility
    set_seed(42)
    batches = generate_stream(mnist_test, fmnist_test, kmnist_test, dry_run=args.dry_run)
    print(f"Stream generated with {len(batches)} batches of size 64.")
    
    # 6. Run Evaluations
    print("\n--- Running Static Merging Baseline ---")
    set_seed(42)
    static_results, static_overall = run_static_merging(batches, model_0, model_1, base_model, device)
    print(f"Static Merging Overall Acc: {static_overall:.4%}, Segment Accs: {static_results}")
    
    print("\n--- Running Fixed TTA Baseline ---")
    set_seed(42)
    tta_results, tta_overall = run_fixed_tta(batches, model_0, model_1, base_model, device, steps=(2 if args.dry_run else 5))
    print(f"Fixed TTA Overall Acc: {tta_overall:.4%}, Segment Accs: {tta_results}")
    
    print("\n--- Running CLW-Fisher Baseline ---")
    set_seed(42)
    clw_results, clw_overall = run_clw_fisher(batches, model_0, model_1, base_model, device, prototypes_0, prototypes_1, source_fisher, steps=(2 if args.dry_run else 5))
    print(f"CLW-Fisher Overall Acc: {clw_overall:.4%}, Segment Accs: {clw_results}")
    
    print("\n--- Running KT-Fisher Baseline ---")
    set_seed(42)
    kt_results, kt_overall = run_kt_fisher(batches, model_0, model_1, base_model, device, steps=(2 if args.dry_run else 5))
    print(f"KT-Fisher Overall Acc: {kt_overall:.4%}, Segment Accs: {kt_results}")
    
    print("\n--- Running DF-Bayes-TTMM Baseline ---")
    set_seed(42)
    bayes_results, bayes_overall = run_df_bayes_ttmm(batches, model_0, model_1, base_model, device, steps=(2 if args.dry_run else 3))
    print(f"DF-Bayes-TTMM Overall Acc: {bayes_overall:.4%}, Segment Accs: {bayes_results}")
    
    print("\n--- Running BK-CoMerge (Ours: Flagship) ---")
    set_seed(42)
    bk_p_history = []
    bk_results, bk_overall = run_bk_co_merge(batches, model_0, model_1, base_model, device, prototypes_0, prototypes_1, steps=(2 if args.dry_run else 5), lr=0.05, gamma_coherence=0.02, temporal_smoothing=False, p_history=bk_p_history)
    print(f"BK-CoMerge Overall Acc: {bk_overall:.4%}, Segment Accs: {bk_results}")
    
    print("\n--- Running TS-BK-CoMerge (Ours: Temporally Smoothed) ---")
    set_seed(42)
    ts_p_history = []
    ts_results, ts_overall = run_bk_co_merge(batches, model_0, model_1, base_model, device, prototypes_0, prototypes_1, steps=(2 if args.dry_run else 2), lr=0.08, gamma_coherence=0.005, temporal_smoothing=True, p_history=ts_p_history)
    print(f"TS-BK-CoMerge Overall Acc: {ts_overall:.4%}, Segment Accs: {ts_results}")
    
    # Run Wall-Clock Adaptation Time Benchmark
    t_bayes, t_bk, speedup = benchmark_runtimes(batches, model_0, model_1, base_model, device, prototypes_0, prototypes_1)
    
    # Save the empirical results to results.txt for analysis and paper drafting
    print("\nSaving results to results.txt...")
    with open("results.txt", "w") as f:
        f.write("Method,Clean MNIST,Noisy MNIST,Clean FashionMNIST,Noisy FashionMNIST,Novel KMNIST,Overall\n")
        
        methods = [
            ("Static Merging", static_results, static_overall),
            ("Fixed TTA", tta_results, tta_overall),
            ("CLW-Fisher", clw_results, clw_overall),
            ("KT-Fisher", kt_results, kt_overall),
            ("DF-Bayes-TTMM", bayes_results, bayes_overall),
            ("BK-CoMerge (Ours)", bk_results, bk_overall),
            ("TS-BK-CoMerge (Ours)", ts_results, ts_overall)
        ]
        
        for name, seg_res, overall in methods:
            row = f"{name},{seg_res.get('Clean MNIST', 0.0):.4f},{seg_res.get('Noisy MNIST', 0.0):.4f},{seg_res.get('Clean FashionMNIST', 0.0):.4f},{seg_res.get('Noisy FashionMNIST', 0.0):.4f},{seg_res.get('Novel KMNIST', 0.0):.4f},{overall:.4f}\n"
            f.write(row)
            
        f.write("\nBenchmark results:\n")
        f.write(f"DF-Bayes-TTMM_time_per_batch_sec,{t_bayes:.6f}\n")
        f.write(f"BK-CoMerge_time_per_batch_sec,{t_bk:.6f}\n")
        f.write(f"Relative_speedup,{speedup:.2f}x\n")
        
    print("Saving routing histories to p_histories.txt...")
    with open("p_histories.txt", "w") as f:
        f.write("Batch,BK-CoMerge_p,TS-BK-CoMerge_p\n")
        for idx, (bk_p, ts_p) in enumerate(zip(bk_p_history, ts_p_history)):
            f.write(f"{idx},{bk_p:.6f},{ts_p:.6f}\n")
            
    print("All tasks completed successfully!")
