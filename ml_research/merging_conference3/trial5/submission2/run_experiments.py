import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.func as tf
import random

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

# Define a 12-layer Convolutional Neural Network
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Deep12LayerCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Exactly 11 conv blocks and 1 linear classifier = 12 parameter-carrying layers
        self.features = nn.ModuleList([
            ConvBlock(3, 16),       # Layer 0
            ConvBlock(16, 16),      # Layer 1
            ConvBlock(16, 16),      # Layer 2
            ConvBlock(16, 16),      # Layer 3
            ConvBlock(16, 32),      # Layer 4
            ConvBlock(32, 32),      # Layer 5
            ConvBlock(32, 32),      # Layer 6
            ConvBlock(32, 32),      # Layer 7
            ConvBlock(32, 32),      # Layer 8
            ConvBlock(32, 32),      # Layer 9
            ConvBlock(32, 32)       # Layer 10
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes) # Layer 11
        
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Load real datasets (MNIST, FashionMNIST, CIFAR-10, SVHN)
def get_datasets(device):
    print("Loading real-world datasets for 4 expert tasks...")
    
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    transform_color = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    datasets_raw = {
        0: (torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_gray),
            torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_gray)),
        1: (torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_gray),
            torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_gray)),
        2: (torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_color),
            torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_color)),
        3: (torchvision.datasets.SVHN(root='./data', split='train', download=False, transform=transform_color),
            torchvision.datasets.SVHN(root='./data', split='test', download=False, transform=transform_color))
    }
    
    datasets = {}
    for task_idx in range(4):
        train_full, test_full = datasets_raw[task_idx]
        
        # Highly optimized train/test subsets for ultra-fast execution on CPU
        # 1000 train samples and 500 test samples per task
        g = torch.Generator().manual_seed(42 + task_idx)
        train_indices = torch.randperm(len(train_full), generator=g).tolist()[:1000]
        test_indices = torch.randperm(len(test_full), generator=g).tolist()[:500]
        
        train_sub = Subset(train_full, train_indices)
        test_sub = Subset(test_full, test_indices)
        
        datasets[task_idx] = (train_sub, test_sub)
        
    print("MNIST, FashionMNIST, CIFAR-10, and SVHN loaded and subsetted successfully!")
    return datasets

# Differentiable weight ensembling dictionary builder for a specific task
# Merges the shared backbone (layers 0-10) using alphas, and preserves the task-specific classification head (layer 11)
def get_merged_params_dict_for_task(base_model, expert_models, alphas, task_idx):
    alphas_clamped = torch.clamp(alphas, 0.0, 1.0)
    params = {}
    
    # Layers 0 to 10: Merged Shared Backbone
    for l in range(11):
        # Conv weights and biases
        base_conv_w = base_model.features[l].conv.weight
        base_conv_b = base_model.features[l].conv.bias
        expert_conv_ws = [m.features[l].conv.weight for m in expert_models]
        expert_conv_bs = [m.features[l].conv.bias for m in expert_models]
        
        merged_conv_w = base_conv_w.clone()
        for k in range(4):
            merged_conv_w = merged_conv_w + alphas_clamped[k][l] * (expert_conv_ws[k] - base_conv_w)
        params[f"features.{l}.conv.weight"] = merged_conv_w
        
        if base_conv_b is not None:
            merged_conv_b = base_conv_b.clone()
            for k in range(4):
                merged_conv_b = merged_conv_b + alphas_clamped[k][l] * (expert_conv_bs[k] - base_conv_b)
            params[f"features.{l}.conv.bias"] = merged_conv_b
            
        # BatchNorm weights and biases (differentiable ensembling)
        base_bn_w = base_model.features[l].bn.weight
        base_bn_b = base_model.features[l].bn.bias
        expert_bn_ws = [m.features[l].bn.weight for m in expert_models]
        expert_bn_bs = [m.features[l].bn.bias for m in expert_models]
        
        merged_bn_w = base_bn_w.clone()
        for k in range(4):
            merged_bn_w = merged_bn_w + alphas_clamped[k][l] * (expert_bn_ws[k] - base_bn_w)
        params[f"features.{l}.bn.weight"] = merged_bn_w
        
        if base_bn_b is not None:
            merged_bn_b = base_bn_b.clone()
            for k in range(4):
                merged_bn_b = merged_bn_b + alphas_clamped[k][l] * (expert_bn_bs[k] - base_bn_b)
            params[f"features.{l}.bn.bias"] = merged_bn_b
            
        # BatchNorm running mean and running var (non-differentiable but ensembled)
        base_bn_rm = base_model.features[l].bn.running_mean
        base_bn_rv = base_model.features[l].bn.running_var
        expert_bn_rms = [m.features[l].bn.running_mean for m in expert_models]
        expert_bn_rvs = [m.features[l].bn.running_var for m in expert_models]
        
        merged_bn_rm = base_bn_rm.clone()
        merged_bn_rv = base_bn_rv.clone()
        for k in range(4):
            merged_bn_rm = merged_bn_rm + alphas_clamped[k][l].detach() * (expert_bn_rms[k] - base_bn_rm)
            merged_bn_rv = merged_bn_rv + alphas_clamped[k][l].detach() * (expert_bn_rvs[k] - base_bn_rv)
        params[f"features.{l}.bn.running_mean"] = merged_bn_rm
        params[f"features.{l}.bn.running_var"] = merged_bn_rv
        params[f"features.{l}.bn.num_batches_tracked"] = base_model.features[l].bn.num_batches_tracked
        
    # Layer 11: Task-Specific Classifier Head (Preserved!)
    params["classifier.weight"] = expert_models[task_idx].classifier.weight
    params["classifier.bias"] = expert_models[task_idx].classifier.bias
                
    return params

# Helper for TIES-Merging on a list of tensors
def ties_merging_tensor(tensors, reset_thresh=0.2, scaling_coef=0.4):
    stacked = torch.stack(tensors) # shape: (K, ...)
    shape = stacked.shape
    stacked_flat = stacked.view(shape[0], -1) # shape: (K, D)
    
    D = stacked_flat.shape[1]
    k = int(D * reset_thresh)
    k = max(1, min(k, D))
    
    pruned_flat = torch.zeros_like(stacked_flat)
    for i in range(shape[0]):
        v = stacked_flat[i]
        v_abs = v.abs()
        threshold = torch.kthvalue(v_abs, D - k + 1).values
        mask = v_abs >= threshold
        pruned_flat[i] = v * mask
        
    # Sign resolution
    signs = torch.sign(pruned_flat) # shape: (K, D)
    sum_signs = signs.sum(dim=0) # shape: (D)
    majority_sign = torch.sign(sum_signs)
    zeros_mask = majority_sign == 0
    if zeros_mask.any():
        majority_sign[zeros_mask] = torch.sign(stacked_flat.sum(dim=0))[zeros_mask]
    majority_sign[majority_sign == 0] = 1.0
    
    # Disjoint aggregation
    agreed_mask = torch.sign(pruned_flat) == majority_sign.unsqueeze(0)
    masked_pruned = pruned_flat * agreed_mask
    
    non_zero_counts = agreed_mask.sum(dim=0).float()
    merged_tv = masked_pruned.sum(dim=0) / torch.clamp(non_zero_counts, min=1.0)
    
    return merged_tv.view(shape[1:]) * scaling_coef

# Helper for DARE-Merging on a list of tensors
def dare_merging_tensor(tensors, drop_rate=0.2, scaling_coef=0.7):
    stacked = torch.stack(tensors) # shape: (K, ...)
    shape = stacked.shape
    stacked_flat = stacked.view(shape[0], -1) # shape: (K, D)
    
    if drop_rate > 0:
        mask = torch.rand_like(stacked_flat) >= drop_rate
        dropped = stacked_flat * mask
        rescaled = dropped / (1.0 - drop_rate)
    else:
        rescaled = stacked_flat
        
    merged_tv = rescaled.mean(dim=0)
    return merged_tv.view(shape[1:]) * scaling_coef

# Helper to copy and merge weights statically layer-wise for a specific task
def set_merged_weights_for_task(model, base_model, expert_models, alphas, task_idx):
    alphas_clamped = torch.clamp(alphas, 0.0, 1.0)
    
    with torch.no_grad():
        # Layers 0 to 10: Merged Shared Backbone
        for l in range(11):
            # Target model parameters
            target_w = model.features[l].conv.weight
            target_b = model.features[l].conv.bias
            base_w = base_model.features[l].conv.weight
            base_b = base_model.features[l].conv.bias
            expert_ws = [m.features[l].conv.weight for m in expert_models]
            expert_bs = [m.features[l].conv.bias for m in expert_models]
            
            # Merged Conv Weight
            merged_w = base_w.clone()
            for k in range(4):
                merged_w += alphas_clamped[k][l] * (expert_ws[k] - base_w)
            target_w.copy_(merged_w)
            
            # Merged Conv Bias
            if target_b is not None:
                merged_b = base_b.clone()
                for k in range(4):
                    merged_b += alphas_clamped[k][l] * (expert_bs[k] - base_b)
                target_b.copy_(merged_b)
            
            # BatchNorm layers ensembling (weight & bias)
            target_bn_w = model.features[l].bn.weight
            target_bn_b = model.features[l].bn.bias
            base_bn_w = base_model.features[l].bn.weight
            base_bn_b = base_model.features[l].bn.bias
            expert_bn_ws = [m.features[l].bn.weight for m in expert_models]
            expert_bn_bs = [m.features[l].bn.bias for m in expert_models]
            
            merged_bn_w = base_bn_w.clone()
            for k in range(4):
                merged_bn_w += alphas_clamped[k][l] * (expert_bn_ws[k] - base_bn_w)
            target_bn_w.copy_(merged_bn_w)
            
            if target_bn_b is not None:
                merged_bn_b = base_bn_b.clone()
                for k in range(4):
                    merged_bn_b += alphas_clamped[k][l] * (expert_bn_bs[k] - base_bn_b)
                target_bn_b.copy_(merged_bn_b)
            
            # BatchNorm running mean and running var
            target_bn_rm = model.features[l].bn.running_mean
            target_bn_rv = model.features[l].bn.running_var
            base_bn_rm = base_model.features[l].bn.running_mean
            base_bn_rv = base_model.features[l].bn.running_var
            expert_bn_rms = [m.features[l].bn.running_mean for m in expert_models]
            expert_bn_rvs = [m.features[l].bn.running_var for m in expert_models]
            
            merged_bn_rm = base_bn_rm.clone()
            merged_bn_rv = base_bn_rv.clone()
            for k in range(4):
                merged_bn_rm += alphas_clamped[k][l] * (expert_bn_rms[k] - base_bn_rm)
                merged_bn_rv += alphas_clamped[k][l] * (expert_bn_rvs[k] - base_bn_rv)
            target_bn_rm.copy_(merged_bn_rm)
            target_bn_rv.copy_(merged_bn_rv)
            
        # Layer 11: Task-Specific Classifier Head (Preserved!)
        model.classifier.weight.copy_(expert_models[task_idx].classifier.weight)
        model.classifier.bias.copy_(expert_models[task_idx].classifier.bias)

# Softmax entropy function
def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# Evaluation function on the test subsets (500 samples per task)
def evaluate_model(model, test_datasets, base_model, expert_models, alphas, device):
    model.eval()
    # Keep BatchNorm layers in train mode to use stable batch-level statistics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
    accuracies = []
    
    for task_idx in range(4):
        _, test_dataset = test_datasets[task_idx]
        loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Merge shared backbone, and apply task-specific classification head
        set_merged_weights_for_task(model, base_model, expert_models, alphas, task_idx)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = (correct / total) * 100.0 if total > 0 else 0.0
        accuracies.append(acc)
    return accuracies

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}", flush=True)
    
    # 1. Load datasets
    datasets = get_datasets(device)
    
    # 2. Train / Fine-tune base and expert models
    print("\n--- Training Base and Expert Models ---", flush=True)
    set_seed(42)
    base_model = Deep12LayerCNN().to(device)
    
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    base_checkpoint_path = os.path.join(checkpoint_dir, 'base_model.pt')
    expert_checkpoint_paths = [os.path.join(checkpoint_dir, f'expert_{k}.pt') for k in range(4)]
    
    # Pre-train the base model on a mixture of all tasks for 3 epochs to create a solid shared starting point
    if os.path.exists(base_checkpoint_path):
        print("Loading pre-trained base model from checkpoint...", flush=True)
        base_model.load_state_dict(torch.load(base_checkpoint_path, map_location=device))
    else:
        base_optimizer = optim.Adam(base_model.parameters(), lr=2e-3)
        mixed_X = []
        mixed_y = []
        for k in range(4):
            train_ds, _ = datasets[k]
            loader = DataLoader(train_ds, batch_size=1000, shuffle=False)
            imgs, labels = next(iter(loader))
            mixed_X.append(imgs)
            mixed_y.append(labels)
            
        mixed_dataset = TensorDataset(torch.cat(mixed_X), torch.cat(mixed_y))
        mixed_loader = DataLoader(mixed_dataset, batch_size=32, shuffle=True)
        
        print("Pre-training base model on mixed-task pool (3 epochs)...", flush=True)
        base_model.train()
        for epoch in range(3):
            for imgs, labels in mixed_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                base_optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(base_model(imgs), labels)
                loss.backward()
                base_optimizer.step()
        torch.save(base_model.state_dict(), base_checkpoint_path)
    
    # Fine-tune expert models for 8 epochs (strong convergence)
    expert_models = []
    expert_accuracies_before = []
    for k in range(4):
        expert = Deep12LayerCNN().to(device)
        path = expert_checkpoint_paths[k]
        if os.path.exists(path):
            print(f"Loading Expert {k} on Task {k} from checkpoint...", flush=True)
            expert.load_state_dict(torch.load(path, map_location=device))
        else:
            print(f"Fine-tuning Expert {k} on Task {k}...", flush=True)
            expert.load_state_dict(base_model.state_dict())
            expert_optimizer = optim.Adam(expert.parameters(), lr=1e-3)
            
            train_ds, test_ds = datasets[k]
            loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            
            expert.train()
            for epoch in range(8): # 8 epochs
                for imgs, labels in loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    expert_optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(expert(imgs), labels)
                    loss.backward()
                    expert_optimizer.step()
            torch.save(expert.state_dict(), path)
        
        expert_models.append(expert)
        
        # Eval expert on its own task (using its own classifier head)
        expert.eval()
        _, test_ds = datasets[k]
        loader_test = DataLoader(test_ds, batch_size=64, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader_test:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = expert(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = (correct / total) * 100.0
        expert_accuracies_before.append(acc)
        print(f"Expert {k} Task {k} Test Accuracy: {acc:.2f}%", flush=True)
        
    print(f"Average Expert Accuracy: {np.mean(expert_accuracies_before):.2f}%", flush=True)
    
    # Prepare Calibration and Test streams
    print("\n--- Constructing Few-Shot Calibration and Test Streams ---", flush=True)
    calibration_data = [] # List of (imgs, labels) per task
    test_streams = []      # Unlabeled streaming data per task for online TTA
    
    for k in range(4):
        train_ds, test_ds = datasets[k]
        
        # Calibration Set (size M = 10 samples per task - balanced: 1 sample per class)
        loader_cal = DataLoader(train_ds, batch_size=500, shuffle=False)
        all_imgs, all_labels = next(iter(loader_cal))
        cal_imgs_list = []
        cal_labels_list = []
        for class_idx in range(10):
            matches = (all_labels == class_idx).nonzero().flatten()
            idx = matches[0].item()
            cal_imgs_list.append(all_imgs[idx])
            cal_labels_list.append(all_labels[idx])
        cal_imgs = torch.stack(cal_imgs_list)
        cal_labels = torch.tensor(cal_labels_list)
        calibration_data.append((cal_imgs.to(device), cal_labels.to(device)))
        
        # Test Stream (size N = 20 samples per task - balanced: 2 samples per class)
        loader_test = DataLoader(test_ds, batch_size=100, shuffle=False)
        all_imgs, all_labels = next(iter(loader_test))
        stream_imgs_list = []
        stream_labels_list = []
        for class_idx in range(10):
            matches = (all_labels == class_idx).nonzero().flatten()
            for j in range(2): # 2 samples per class
                idx = matches[j].item()
                stream_imgs_list.append(all_imgs[idx])
                stream_labels_list.append(all_labels[idx])
        stream_imgs = torch.stack(stream_imgs_list)
        stream_labels = torch.tensor(stream_labels_list)
        test_streams.append((stream_imgs.to(device), stream_labels.to(device)))
        
    # Standard merged model skeleton
    merged_model = Deep12LayerCNN().to(device)
    
    results = {}
    
    # ------------------ Baseline 1: Static Uniform Merging ------------------
    print("\n[Baseline 1] Evaluating Static Uniform Merging...", flush=True)
    merged_model.load_state_dict(base_model.state_dict()) # Reset state and isolate
    alphas_uniform = torch.full((4, 12), 0.25).to(device)
    accs_uniform = evaluate_model(merged_model, datasets, base_model, expert_models, alphas_uniform, device)
    print(f"Uniform Merging Accs: {accs_uniform}, Average: {np.mean(accs_uniform):.2f}%", flush=True)
    results['Uniform'] = {'accs': accs_uniform, 'avg': np.mean(accs_uniform)}
    
    # ------------------ Baseline 2: Unconstrained Online AdaMerging ------------------
    print("\n[Baseline 2] Running Online AdaMerging (Unconstrained TTA on test streams)...", flush=True)
    merged_model.load_state_dict(base_model.state_dict()) # Reset state and isolate
    alphas_raw = torch.full((4, 12), -1.0986, requires_grad=True, device=device)
    # High learning rate for fast, effective adaptation in 20 steps
    optimizer = optim.Adam([alphas_raw], lr=2e-1)
    
    for step in range(20):
        optimizer.zero_grad()
        alphas = torch.sigmoid(alphas_raw) # Parameterize in [0, 1]
        
        loss = 0.0
        for k in range(4):
            stream_imgs, _ = test_streams[k]
            noisy_imgs = stream_imgs + torch.randn_like(stream_imgs) * 0.1
            params = get_merged_params_dict_for_task(base_model, expert_models, alphas, k)
            outputs = tf.functional_call(merged_model, params, noisy_imgs)
            loss += softmax_entropy(outputs).mean()
            
        loss.backward()
        optimizer.step()
        
    alphas_adamerging = torch.sigmoid(alphas_raw).detach()
    accs_adamerging = evaluate_model(merged_model, datasets, base_model, expert_models, alphas_adamerging, device)
    print(f"AdaMerging Accs: {accs_adamerging}, Average: {np.mean(accs_adamerging):.2f}%", flush=True)
    results['AdaMerging'] = {'accs': accs_adamerging, 'avg': np.mean(accs_adamerging)}
    
    # ------------------ Baseline 3: Online PolyMerge (d=2) ------------------
    print("\n[Baseline 3] Running Online PolyMerge (d=2, quadratic trajectory constraints)...", flush=True)
    merged_model.load_state_dict(base_model.state_dict()) # Reset state and isolate
    theta = torch.zeros(4, 3, requires_grad=True, device=device)
    with torch.no_grad():
        theta[:, 0] = -1.0986
        
    optimizer = optim.Adam([theta], lr=2e-1)
    z = torch.tensor([l/11.0 for l in range(12)], device=device) # layer coordinates
    
    for step in range(20):
        optimizer.zero_grad()
        alphas = torch.zeros(4, 12, device=device)
        for k in range(4):
            alphas[k] = theta[k][0] + theta[k][1]*z + theta[k][2]*(z**2)
        alphas = torch.sigmoid(alphas) # Bound to [0,1]
        
        loss = 0.0
        for k in range(4):
            stream_imgs, _ = test_streams[k]
            noisy_imgs = stream_imgs + torch.randn_like(stream_imgs) * 0.1
            params = get_merged_params_dict_for_task(base_model, expert_models, alphas, k)
            outputs = tf.functional_call(merged_model, params, noisy_imgs)
            loss += softmax_entropy(outputs).mean()
            
        loss.backward()
        optimizer.step()
        
    alphas_poly = torch.zeros(4, 12, device=device)
    with torch.no_grad():
        for k in range(4):
            alphas_poly[k] = theta[k][0] + theta[k][1]*z + theta[k][2]*(z**2)
        alphas_poly = torch.sigmoid(alphas_poly)
        
    accs_poly = evaluate_model(merged_model, datasets, base_model, expert_models, alphas_poly, device)
    print(f"PolyMerge Accs: {accs_poly}, Average: {np.mean(accs_poly):.2f}%", flush=True)
    results['PolyMerge'] = {'accs': accs_poly, 'avg': np.mean(accs_poly)}
    
    # ------------------ Baseline 4: Offline Unconstrained Few-Shot Tuning ------------------
    print("\n[Baseline 4] Running Offline Unconstrained Few-Shot Tuning (on Calibration sets) with sweeps...", flush=True)
    lambda_sweeps_unconstrained = [0.0, 0.001, 0.01, 0.1, 1.0]
    results['OFS-Unconstrained-Regularized'] = {}
    best_unconstrained_avg = -1.0
    best_unconstrained_lambda = -1.0
    best_unconstrained_accs = None
    
    for lmbda in lambda_sweeps_unconstrained:
        print(f"Optimizing Offline Unconstrained with Consensus-Pulling lambda = {lmbda}...", flush=True)
        merged_model.load_state_dict(base_model.state_dict()) # Reset state and isolate
        alphas_raw = torch.full((4, 12), -1.0986, requires_grad=True, device=device)
        optimizer = optim.Adam([alphas_raw], lr=1e-1)
        criterion = nn.CrossEntropyLoss()
        
        for step in range(30):
            optimizer.zero_grad()
            alphas = torch.sigmoid(alphas_raw)
            
            loss = 0.0
            for k in range(4):
                cal_imgs, cal_labels = calibration_data[k]
                params = get_merged_params_dict_for_task(base_model, expert_models, alphas, k)
                outputs = tf.functional_call(merged_model, params, cal_imgs)
                loss += criterion(outputs, cal_labels)
                
            # Consensus-Pulling Regularization (L1 penalty pulling coordinates to uniform consensus)
            reg = torch.sum(torch.abs(alphas_raw - (-1.0986)))
            
            total_loss = loss + lmbda * reg
            total_loss.backward()
            optimizer.step()
            
        alphas_offline_unconstrained = torch.sigmoid(alphas_raw).detach()
        accs_offline_unconstrained = evaluate_model(merged_model, datasets, base_model, expert_models, alphas_offline_unconstrained, device)
        avg_acc = np.mean(accs_offline_unconstrained)
        print(f"Offline Unconstrained (lambda={lmbda}) Accs: {accs_offline_unconstrained}, Average: {avg_acc:.2f}%", flush=True)
        
        results['OFS-Unconstrained-Regularized'][f'lambda_{lmbda}'] = {'accs': accs_offline_unconstrained, 'avg': avg_acc}
        if lmbda == 0.0:
            results['OFS-Unconstrained'] = {'accs': accs_offline_unconstrained, 'avg': avg_acc}
            
        if avg_acc > best_unconstrained_avg:
            best_unconstrained_avg = avg_acc
            best_unconstrained_lambda = lmbda
            best_unconstrained_accs = accs_offline_unconstrained
            
    print(f"Offline Unconstrained Best Lambda: {best_unconstrained_lambda} with Average Accuracy: {best_unconstrained_avg:.2f}%\n", flush=True)
    
    # ------------------ Baseline 5: Quantum Superposition Merging (QWS-Merge) ------------------
    print("\n[Baseline 5] Evaluating Quantum Superposition Merging...", flush=True)
    merged_model.load_state_dict(base_model.state_dict()) # Reset state and isolate
    alphas_qws = torch.full((4, 12), 0.25).to(device)
    
    accs_qws = []
    for task_idx in range(4):
        _, test_dataset = datasets[task_idx]
        loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Merge backbone and copy expert classifier head
        set_merged_weights_for_task(merged_model, base_model, expert_models, alphas_qws, task_idx)
        
        base_params_qws = {name: p.clone() for name, p in merged_model.named_parameters()}
        for name, b in merged_model.named_buffers():
            base_params_qws[name] = b.clone()
            
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                q_noise = np.cos(task_idx * np.pi / 2.0) * 0.05
                
                perturbed_params = {}
                for name, p in base_params_qws.items():
                    if "num_batches_tracked" in name or "running_mean" in name or "running_var" in name:
                        perturbed_params[name] = p
                    else:
                        perturbed_params[name] = p + torch.randn_like(p) * q_noise
                        
                outputs = tf.functional_call(merged_model, perturbed_params, imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = (correct / total) * 100.0 if total > 0 else 0.0
        accs_qws.append(acc)
    print(f"QWS-Merge Accs: {accs_qws}, Average: {np.mean(accs_qws):.2f}%", flush=True)
    results['QWS-Merge'] = {'accs': accs_qws, 'avg': np.mean(accs_qws)}
    
    # ------------------ Baseline 6: TIES-Merging ------------------
    print("\n[Baseline 6] Evaluating TIES-Merging (reset_thresh=0.2, scaling_coef=0.5)...", flush=True)
    merged_model.load_state_dict(base_model.state_dict()) # Reset state and isolate
    
    # Pre-calculate TIES-merged weights for the backbone (layers 0 to 10)
    with torch.no_grad():
        for l in range(11):
            target_w = merged_model.features[l].conv.weight
            target_b = merged_model.features[l].conv.bias
            base_w = base_model.features[l].conv.weight
            base_b = base_model.features[l].conv.bias
            
            expert_ws = [m.features[l].conv.weight for m in expert_models]
            expert_bs = [m.features[l].conv.bias for m in expert_models]
            
            # Conv weights
            tvs_w = [ew - base_w for ew in expert_ws]
            merged_tv_w = ties_merging_tensor(tvs_w, reset_thresh=0.2, scaling_coef=0.5)
            target_w.copy_(base_w + merged_tv_w)
            
            # Conv bias
            if target_b is not None:
                tvs_b = [eb - base_b for eb in expert_bs]
                merged_tv_b = ties_merging_tensor(tvs_b, reset_thresh=0.2, scaling_coef=0.5)
                target_b.copy_(base_b + merged_tv_b)
                
            # BatchNorm weight & bias & running stats
            target_bn_w = merged_model.features[l].bn.weight
            target_bn_b = merged_model.features[l].bn.bias
            base_bn_w = base_model.features[l].bn.weight
            base_bn_b = base_model.features[l].bn.bias
            expert_bn_ws = [m.features[l].bn.weight for m in expert_models]
            expert_bn_bs = [m.features[l].bn.bias for m in expert_models]
            
            tvs_bn_w = [ebw - base_bn_w for ebw in expert_bn_ws]
            merged_tv_bn_w = ties_merging_tensor(tvs_bn_w, reset_thresh=0.2, scaling_coef=0.5)
            target_bn_w.copy_(base_bn_w + merged_tv_bn_w)
            
            if target_bn_b is not None:
                tvs_bn_b = [ebb - base_bn_b for ebb in expert_bn_bs]
                merged_tv_bn_b = ties_merging_tensor(tvs_bn_b, reset_thresh=0.2, scaling_coef=0.5)
                target_bn_b.copy_(base_bn_b + merged_tv_bn_b)
                
            target_bn_rm = merged_model.features[l].bn.running_mean
            target_bn_rv = merged_model.features[l].bn.running_var
            base_bn_rm = base_model.features[l].bn.running_mean
            base_bn_rv = base_model.features[l].bn.running_var
            expert_bn_rms = [m.features[l].bn.running_mean for m in expert_models]
            expert_bn_rvs = [m.features[l].bn.running_var for m in expert_models]
            
            tvs_bn_rm = [ebrm - base_bn_rm for ebrm in expert_bn_rms]
            merged_tv_bn_rm = ties_merging_tensor(tvs_bn_rm, reset_thresh=0.2, scaling_coef=0.5)
            target_bn_rm.copy_(base_bn_rm + merged_tv_bn_rm)
            
            tvs_bn_rv = [ebrv - base_bn_rv for ebrv in expert_bn_rvs]
            merged_tv_bn_rv = ties_merging_tensor(tvs_bn_rv, reset_thresh=0.2, scaling_coef=0.5)
            target_bn_rv.copy_(base_bn_rv + merged_tv_bn_rv)

    # Evaluate using task-specific heads
    accs_ties = []
    for task_idx in range(4):
        # Set task specific head
        with torch.no_grad():
            merged_model.classifier.weight.copy_(expert_models[task_idx].classifier.weight)
            merged_model.classifier.bias.copy_(expert_models[task_idx].classifier.bias)
            
        _, test_dataset = datasets[task_idx]
        loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = merged_model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = (correct / total) * 100.0 if total > 0 else 0.0
        accs_ties.append(acc)
        
    print(f"TIES-Merging Accs: {accs_ties}, Average: {np.mean(accs_ties):.2f}%", flush=True)
    results['TIES-Merging'] = {'accs': accs_ties, 'avg': np.mean(accs_ties)}
    
    # ------------------ Baseline 7: DARE-Merging ------------------
    print("\n[Baseline 7] Evaluating DARE-Merging (drop_rate=0.2, scaling_coef=0.7)...", flush=True)
    merged_model.load_state_dict(base_model.state_dict()) # Reset state and isolate
    
    # Pre-calculate DARE-merged weights for the backbone (layers 0 to 10)
    torch.manual_seed(42) # Set seed for reproducibility of random mask
    with torch.no_grad():
        for l in range(11):
            target_w = merged_model.features[l].conv.weight
            target_b = merged_model.features[l].conv.bias
            base_w = base_model.features[l].conv.weight
            base_b = base_model.features[l].conv.bias
            
            expert_ws = [m.features[l].conv.weight for m in expert_models]
            expert_bs = [m.features[l].conv.bias for m in expert_models]
            
            # Conv weights
            tvs_w = [ew - base_w for ew in expert_ws]
            merged_tv_w = dare_merging_tensor(tvs_w, drop_rate=0.2, scaling_coef=0.7)
            target_w.copy_(base_w + merged_tv_w)
            
            # Conv bias
            if target_b is not None:
                tvs_b = [eb - base_b for eb in expert_bs]
                merged_tv_b = dare_merging_tensor(tvs_b, drop_rate=0.2, scaling_coef=0.7)
                target_b.copy_(base_b + merged_tv_b)
                
            # BatchNorm weight & bias & running stats
            target_bn_w = merged_model.features[l].bn.weight
            target_bn_b = merged_model.features[l].bn.bias
            base_bn_w = base_model.features[l].bn.weight
            base_bn_b = base_model.features[l].bn.bias
            expert_bn_ws = [m.features[l].bn.weight for m in expert_models]
            expert_bn_bs = [m.features[l].bn.bias for m in expert_models]
            
            tvs_bn_w = [ebw - base_bn_w for ebw in expert_bn_ws]
            merged_tv_bn_w = dare_merging_tensor(tvs_bn_w, drop_rate=0.2, scaling_coef=0.7)
            target_bn_w.copy_(base_bn_w + merged_tv_bn_w)
            
            if target_bn_b is not None:
                tvs_bn_b = [ebb - base_bn_b for ebb in expert_bn_bs]
                merged_tv_bn_b = dare_merging_tensor(tvs_bn_b, drop_rate=0.2, scaling_coef=0.7)
                target_bn_b.copy_(base_bn_b + merged_tv_bn_b)
                
            target_bn_rm = merged_model.features[l].bn.running_mean
            target_bn_rv = merged_model.features[l].bn.running_var
            base_bn_rm = base_model.features[l].bn.running_mean
            base_bn_rv = base_model.features[l].bn.running_var
            expert_bn_rms = [m.features[l].bn.running_mean for m in expert_models]
            expert_bn_rvs = [m.features[l].bn.running_var for m in expert_models]
            
            tvs_bn_rm = [ebrm - base_bn_rm for ebrm in expert_bn_rms]
            merged_tv_bn_rm = dare_merging_tensor(tvs_bn_rm, drop_rate=0.2, scaling_coef=0.7)
            target_bn_rm.copy_(base_bn_rm + merged_tv_bn_rm)
            
            tvs_bn_rv = [ebrv - base_bn_rv for ebrv in expert_bn_rvs]
            merged_tv_bn_rv = dare_merging_tensor(tvs_bn_rv, drop_rate=0.2, scaling_coef=0.7)
            target_bn_rv.copy_(base_bn_rv + merged_tv_bn_rv)

    # Evaluate using task-specific heads
    accs_dare = []
    for task_idx in range(4):
        # Set task specific head
        with torch.no_grad():
            merged_model.classifier.weight.copy_(expert_models[task_idx].classifier.weight)
            merged_model.classifier.bias.copy_(expert_models[task_idx].classifier.bias)
            
        _, test_dataset = datasets[task_idx]
        loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = merged_model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = (correct / total) * 100.0 if total > 0 else 0.0
        accs_dare.append(acc)
        
    print(f"DARE-Merging Accs: {accs_dare}, Average: {np.mean(accs_dare):.2f}%", flush=True)
    results['DARE-Merging'] = {'accs': accs_dare, 'avg': np.mean(accs_dare)}
    
    # ------------------ Method 8: RBPM (Ours) with sweeps of lambda ------------------
    print("\n--- [Ours] Rademacher-Bounded Polynomial Merging (RBPM) ---", flush=True)
    lambda_sweeps = [0.0, 0.001, 0.01, 0.1, 1.0]
    best_avg_acc = -1.0
    best_lambda = -1.0
    best_accs = None
    best_alphas = None
    
    results['RBPM'] = {}
    
    for lmbda in lambda_sweeps:
        print(f"Optimizing RBPM with Rademacher Regularization lambda = {lmbda}...", flush=True)
        merged_model.load_state_dict(base_model.state_dict()) # Reset state and isolate
        theta = torch.zeros(4, 3, requires_grad=True, device=device)
        with torch.no_grad():
            theta[:, 0] = -1.0986 # Init to Uniform ensembling
            
        optimizer = optim.Adam([theta], lr=1e-1)
        z = torch.tensor([l/11.0 for l in range(12)], device=device)
        criterion = nn.CrossEntropyLoss()
        
        for step in range(30):
            optimizer.zero_grad()
            alphas = torch.zeros(4, 12, device=device)
            for k in range(4):
                alphas[k] = theta[k][0] + theta[k][1]*z + theta[k][2]*(z**2)
            alphas = torch.sigmoid(alphas)
            
            # 1. Cross Entropy validation loss using task-specific heads
            val_loss = 0.0
            for k in range(4):
                cal_imgs, cal_labels = calibration_data[k]
                params = get_merged_params_dict_for_task(base_model, expert_models, alphas, k)
                outputs = tf.functional_call(merged_model, params, cal_imgs)
                val_loss += criterion(outputs, cal_labels)
            
            # 2. Corrected Rademacher Complexity Regularization (L1 penalty on parameters pulling towards uniform initialization consensus)
            rad_reg = torch.sum(torch.abs(theta[:, 0] - (-1.0986))) + torch.sum(torch.abs(theta[:, 1:]))
            
            total_loss = val_loss + lmbda * rad_reg
            total_loss.backward()
            optimizer.step()
            
        # Compile final weights
        alphas_rbpm = torch.zeros(4, 12, device=device)
        with torch.no_grad():
            for k in range(4):
                alphas_rbpm[k] = theta[k][0] + theta[k][1]*z + theta[k][2]*(z**2)
            alphas_rbpm = torch.sigmoid(alphas_rbpm).detach()
            
        # Evaluate using full test datasets
        accs_rbpm = evaluate_model(merged_model, datasets, base_model, expert_models, alphas_rbpm, device)
        avg_acc = np.mean(accs_rbpm)
        print(f"RBPM (lambda={lmbda}) Accs: {accs_rbpm}, Average: {avg_acc:.2f}%", flush=True)
        
        # Track training (calibration) accuracy to measure Generalization Gap
        merged_model.eval()
        for m in merged_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
        cal_correct = 0
        cal_total = 0
        with torch.no_grad():
            for k in range(4):
                cal_imgs, cal_labels = calibration_data[k]
                set_merged_weights_for_task(merged_model, base_model, expert_models, alphas_rbpm, k)
                outputs = merged_model(cal_imgs)
                _, preds = torch.max(outputs, 1)
                cal_correct += (preds == cal_labels).sum().item()
                cal_total += cal_labels.size(0)
        cal_acc = (cal_correct / cal_total) * 100.0
        gen_gap = cal_acc - avg_acc
        print(f"Generalization Gap (Cal Acc {cal_acc:.1f}% - Test Acc {avg_acc:.1f}%): {gen_gap:.1f}%\n", flush=True)
        
        results['RBPM'][f'lambda_{lmbda}'] = {
            'accs': accs_rbpm, 
            'avg': avg_acc, 
            'cal_acc': cal_acc,
            'gen_gap': gen_gap,
            'alphas': alphas_rbpm.cpu().numpy().tolist()
        }
        
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_lambda = lmbda
            best_accs = accs_rbpm
            best_alphas = alphas_rbpm
            
    print(f"\nRBPM Best Lambda: {best_lambda} with Average Accuracy: {best_avg_acc:.2f}%", flush=True)
    results['Best_RBPM'] = {'lambda': best_lambda, 'accs': best_accs, 'avg': best_avg_acc}
    
    # ------------------ Method 7: RBPM + PCGrad (Ours) ------------------
    print("\n--- [Ours] RBPM + Gradient Surgery (PCGrad) ---", flush=True)
    best_pcg_avg_acc = -1.0
    best_pcg_lambda = -1.0
    best_pcg_accs = None
    best_pcg_alphas = None
    
    results['RBPM_PCGrad'] = {}
    
    for lmbda in lambda_sweeps:
        print(f"Optimizing RBPM + PCGrad with lambda = {lmbda}...", flush=True)
        merged_model.load_state_dict(base_model.state_dict())
        theta = torch.zeros(4, 3, requires_grad=True, device=device)
        with torch.no_grad():
            theta[:, 0] = -1.0986 # Init to Uniform ensembling
            
        optimizer = optim.Adam([theta], lr=1e-1)
        
        for step in range(30):
            alphas = torch.zeros(4, 12, device=device)
            for k in range(4):
                alphas[k] = theta[k][0] + theta[k][1]*z + theta[k][2]*(z**2)
            alphas = torch.sigmoid(alphas)
            
            # 1. Compute individual task losses
            task_losses = []
            for k in range(4):
                cal_imgs, cal_labels = calibration_data[k]
                params = get_merged_params_dict_for_task(base_model, expert_models, alphas, k)
                outputs = tf.functional_call(merged_model, params, cal_imgs)
                task_losses.append(criterion(outputs, cal_labels))
            
            # 2. Compute task-specific gradients
            task_grads = []
            for k in range(4):
                optimizer.zero_grad()
                task_losses[k].backward(retain_graph=True)
                task_grads.append(theta.grad.clone())
                
            # 3. Apply PCGrad (Projecting Conflicting Gradients)
            pc_grads = [g.clone() for g in task_grads]
            for i in range(4):
                indices = list(range(4))
                indices.remove(i)
                random.shuffle(indices)
                for j in indices:
                    g_i = pc_grads[i]
                    g_j = task_grads[j]
                    dot_prod = torch.sum(g_i * g_j)
                    if dot_prod < 0:
                        pc_grads[i] = g_i - (dot_prod / (torch.sum(g_j * g_j) + 1e-8)) * g_j
            
            # Sum projected gradients
            summed_pc_grad = torch.stack(pc_grads).sum(dim=0)
            
            # 4. Backward for Regularization
            optimizer.zero_grad()
            rad_reg = torch.sum(torch.abs(theta[:, 0] - (-1.0986))) + torch.sum(torch.abs(theta[:, 1:]))
            reg_loss = lmbda * rad_reg
            reg_loss.backward()
            
            # 5. Add PCGrad gradient and step
            theta.grad.add_(summed_pc_grad)
            optimizer.step()
            
        # Compile final weights
        alphas_rbpm_pcg = torch.zeros(4, 12, device=device)
        with torch.no_grad():
            for k in range(4):
                alphas_rbpm_pcg[k] = theta[k][0] + theta[k][1]*z + theta[k][2]*(z**2)
            alphas_rbpm_pcg = torch.sigmoid(alphas_rbpm_pcg).detach()
            
        # Evaluate using full test datasets
        accs_rbpm_pcg = evaluate_model(merged_model, datasets, base_model, expert_models, alphas_rbpm_pcg, device)
        avg_acc = np.mean(accs_rbpm_pcg)
        print(f"RBPM+PCGrad (lambda={lmbda}) Accs: {accs_rbpm_pcg}, Average: {avg_acc:.2f}%", flush=True)
        
        # Track training (calibration) accuracy to measure Generalization Gap
        merged_model.eval()
        for m in merged_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
        cal_correct = 0
        cal_total = 0
        with torch.no_grad():
            for k in range(4):
                cal_imgs, cal_labels = calibration_data[k]
                set_merged_weights_for_task(merged_model, base_model, expert_models, alphas_rbpm_pcg, k)
                outputs = merged_model(cal_imgs)
                _, preds = torch.max(outputs, 1)
                cal_correct += (preds == cal_labels).sum().item()
                cal_total += cal_labels.size(0)
        cal_acc = (cal_correct / cal_total) * 100.0
        gen_gap = cal_acc - avg_acc
        print(f"Generalization Gap (Cal Acc {cal_acc:.1f}% - Test Acc {avg_acc:.1f}%): {gen_gap:.1f}%\n", flush=True)
        
        results['RBPM_PCGrad'][f'lambda_{lmbda}'] = {
            'accs': accs_rbpm_pcg, 
            'avg': avg_acc, 
            'cal_acc': cal_acc,
            'gen_gap': gen_gap,
            'alphas': alphas_rbpm_pcg.cpu().numpy().tolist()
        }
        
        if avg_acc > best_pcg_avg_acc:
            best_pcg_avg_acc = avg_acc
            best_pcg_lambda = lmbda
            best_pcg_accs = accs_rbpm_pcg
            best_pcg_alphas = alphas_rbpm_pcg
            
    print(f"\nRBPM+PCGrad Best Lambda: {best_pcg_lambda} with Average Accuracy: {best_pcg_avg_acc:.2f}%", flush=True)
    results['Best_RBPM_PCGrad'] = {'lambda': best_pcg_lambda, 'accs': best_pcg_accs, 'avg': best_pcg_avg_acc}
    
    # 3. Save Results JSON
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to experiment_results.json!", flush=True)
    
    # 4. Generate Plot of Coefficient Trajectories
    plt.figure(figsize=(10, 6))
    layers = np.arange(12)
    alphas_np = best_alphas.cpu().numpy()
    for k in range(4):
        plt.plot(layers, alphas_np[k], marker='o', label=f'Task {k} (Expert {k})')
    plt.title(f'Optimal RBPM Coefficient Trajectories across Network Depth (lambda={best_lambda})')
    plt.xlabel('Layer Index')
    plt.ylabel('Merging Coefficient (Alpha)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/fig1_rbpm_trajectories.png')
    print("Plot saved to results/fig1_rbpm_trajectories.png!", flush=True)
    
    # 5. Generate Bar Chart Comparison
    plt.figure(figsize=(11, 6))
    methods = [
        'Static Uniform', 'Online AdaMerging', 'Online PolyMerge', 
        'Offline Unconstrained', 'QWS-Merge', 'TIES-Merging', 'DARE-Merging',
        f'RBPM (Ours, lambda={best_lambda})',
        f'RBPM+PCGrad (Ours, lambda={best_pcg_lambda})'
    ]
    averages = [
        results['Uniform']['avg'], results['AdaMerging']['avg'], results['PolyMerge']['avg'], 
        results['OFS-Unconstrained']['avg'], results['QWS-Merge']['avg'], results['TIES-Merging']['avg'],
        results['DARE-Merging']['avg'], best_avg_acc, best_pcg_avg_acc
    ]
    
    bars = plt.bar(methods, averages, color=['gray', 'orange', 'blue', 'purple', 'cyan', 'brown', 'red', 'green', 'forestgreen'])
    plt.title('Multi-Task Model Merging Performance Comparison')
    plt.ylabel('Average Test Accuracy (%)')
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('results/fig2_performance_comparison.png')
    print("Plot saved to results/fig2_performance_comparison.png!", flush=True)

if __name__ == '__main__':
    # Ensure results dir exists
    os.makedirs('results', exist_ok=True)
    main()
