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
def get_merged_params_dict_for_task(base_model, expert_models, alphas, task_idx):
    alphas_clamped = torch.clamp(alphas, 0.0, 1.0)
    params = {}
    
    for l in range(11):
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
            
        base_bn_rm = base_model.features[l].bn.running_mean
        base_bn_rv = base_model.features[l].bn.running_var
        expert_bn_rms = [m.features[l].bn.running_mean for m in expert_models]
        expert_bn_rvs = [m.features[l].bn.running_var for m in expert_models]
        
        merged_bn_rm = base_bn_rm.clone()
        merged_bn_rv = base_bn_rv.clone()
        with torch.no_grad():
            for k in range(4):
                merged_bn_rm = merged_bn_rm + alphas_clamped[k][l].detach() * (expert_bn_rms[k] - base_bn_rm)
                merged_bn_rv = merged_bn_rv + alphas_clamped[k][l].detach() * (expert_bn_rvs[k] - base_bn_rv)
        params[f"features.{l}.bn.running_mean"] = merged_bn_rm
        params[f"features.{l}.bn.running_var"] = merged_bn_rv
        
    # Classifier head: keep task-specific expert head
    params["classifier.weight"] = expert_models[task_idx].classifier.weight
    params["classifier.bias"] = expert_models[task_idx].classifier.bias
    return params

def set_merged_weights_for_task(target_model, base_model, expert_models, alphas, task_idx):
    alphas_clamped = torch.clamp(alphas, 0.0, 1.0)
    with torch.no_grad():
        for l in range(11):
            target_conv = target_model.features[l].conv
            base_conv = base_model.features[l].conv
            target_conv.weight.copy_(base_conv.weight)
            if base_conv.bias is not None:
                target_conv.bias.copy_(base_conv.bias)
            for k in range(4):
                target_conv.weight.add_((expert_models[k].features[l].conv.weight - base_conv.weight) * alphas_clamped[k][l])
                if base_conv.bias is not None:
                    target_conv.bias.add_((expert_models[k].features[l].conv.bias - base_conv.bias) * alphas_clamped[k][l])
            
            target_bn = target_model.features[l].bn
            base_bn = base_model.features[l].bn
            target_bn.weight.copy_(base_bn.weight)
            if base_bn.bias is not None:
                target_bn.bias.copy_(base_bn.bias)
            target_bn.running_mean.copy_(base_bn.running_mean)
            target_bn.running_var.copy_(base_bn.running_var)
            for k in range(4):
                target_bn.weight.add_((expert_models[k].features[l].bn.weight - base_bn.weight) * alphas_clamped[k][l])
                if base_bn.bias is not None:
                    target_bn.bias.add_((expert_models[k].features[l].bn.bias - base_bn.bias) * alphas_clamped[k][l])
                target_bn.running_mean.add_((expert_models[k].features[l].bn.running_mean - base_bn.running_mean) * alphas_clamped[k][l])
                target_bn.running_var.add_((expert_models[k].features[l].bn.running_var - base_bn.running_var) * alphas_clamped[k][l])
                
        target_model.classifier.weight.copy_(expert_models[task_idx].classifier.weight)
        target_model.classifier.bias.copy_(expert_models[task_idx].classifier.bias)

def evaluate_model(target_model, datasets, base_model, expert_models, alphas, device):
    target_model.eval()
    # Lock batch norms to run mode for evaluation
    for m in target_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train() # use local batch statistics to resolve domain shifts cleanly
            
    accuracies = []
    for task_idx in range(4):
        _, test_dataset = datasets[task_idx]
        loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        set_merged_weights_for_task(target_model, base_model, expert_models, alphas, task_idx)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = target_model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = (correct / total) * 100.0 if total > 0 else 0.0
        accuracies.append(acc)
    return accuracies

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running sensitivity sweep on device: {device}", flush=True)
    
    datasets = get_datasets(device)
    
    # ------------------ Base and Expert Training ------------------
    set_seed(42)
    base_model = Deep12LayerCNN().to(device)
    
    # Pre-train base model
    print("\n--- Training Base and Expert Models ---", flush=True)
    print("Pre-training base model on mixed-task pool (3 epochs)...", flush=True)
    mixed_imgs, mixed_labels = [], []
    for k in range(4):
        train_ds, _ = datasets[k]
        loader = DataLoader(train_ds, batch_size=1000, shuffle=False)
        imgs, labels = next(iter(loader))
        mixed_imgs.append(imgs)
        mixed_labels.append(labels)
    mixed_imgs = torch.cat(mixed_imgs)
    mixed_labels = torch.cat(mixed_labels)
    mixed_dataset = TensorDataset(mixed_imgs, mixed_labels)
    mixed_loader = DataLoader(mixed_dataset, batch_size=32, shuffle=True)
    
    base_optimizer = optim.Adam(base_model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    base_model.train()
    for epoch in range(3):
        for imgs, labels in mixed_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            base_optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(base_model(imgs), labels)
            loss.backward()
            base_optimizer.step()
            
    # Fine-tune experts
    expert_models = []
    expert_accuracies_before = []
    for k in range(4):
        print(f"Fine-tuning Expert {k} on Task {k}...", flush=True)
        expert = Deep12LayerCNN().to(device)
        expert.load_state_dict(base_model.state_dict())
        expert_optimizer = optim.Adam(expert.parameters(), lr=1e-3)
        
        train_ds, test_ds = datasets[k]
        loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        expert.train()
        for epoch in range(8):
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                expert_optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(expert(imgs), labels)
                loss.backward()
                expert_optimizer.step()
        
        expert_models.append(expert)
        expert.eval()
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
    
    # Base Model skeleton
    merged_model = Deep12LayerCNN().to(device)
    
    # ------------------ Sensitivity Sweep over M (Samples per task) ------------------
    # M_per_task = 10 * S where S is samples per class
    # We sweep S in [1, 2, 5, 10, 20] which corresponds to M in [10, 20, 50, 100, 200]
    S_list = [1, 2, 5, 10, 20]
    M_list = [10 * s for s in S_list]
    
    results_unconstrained = []
    results_rbpm = []
    
    # Evaluate Static Uniform Baseline
    alphas_uniform = torch.full((4, 12), 0.25).to(device)
    accs_uniform = evaluate_model(merged_model, datasets, base_model, expert_models, alphas_uniform, device)
    uniform_avg = np.mean(accs_uniform)
    print(f"\nStatic Uniform Merging Average Accuracy: {uniform_avg:.2f}%", flush=True)
    
    for S in S_list:
        M = 10 * S
        print(f"\n================== CALIBRATION SWEEP: M = {M} ({S} samples per class) ==================", flush=True)
        
        # Prepare Calibration Data
        calibration_data = []
        for k in range(4):
            train_ds, _ = datasets[k]
            loader_cal = DataLoader(train_ds, batch_size=500, shuffle=False)
            all_imgs, all_labels = next(iter(loader_cal))
            cal_imgs_list = []
            cal_labels_list = []
            for class_idx in range(10):
                matches = (all_labels == class_idx).nonzero().flatten()
                for j in range(S):
                    idx = matches[j].item()
                    cal_imgs_list.append(all_imgs[idx])
                    cal_labels_list.append(all_labels[idx])
            cal_imgs = torch.stack(cal_imgs_list)
            cal_labels = torch.tensor(cal_labels_list)
            calibration_data.append((cal_imgs.to(device), cal_labels.to(device)))
            
        # --- Baseline: Unconstrained Few-Shot Tuning ---
        print(f"Running Offline Unconstrained Tuning for M = {M}...", flush=True)
        merged_model.load_state_dict(base_model.state_dict())
        alphas_raw = torch.full((4, 12), -1.0986, requires_grad=True, device=device)
        optimizer = optim.Adam([alphas_raw], lr=1e-1)
        
        for step in range(30):
            optimizer.zero_grad()
            alphas = torch.sigmoid(alphas_raw)
            
            loss = 0.0
            for k in range(4):
                cal_imgs, cal_labels = calibration_data[k]
                params = get_merged_params_dict_for_task(base_model, expert_models, alphas, k)
                outputs = tf.functional_call(merged_model, params, cal_imgs)
                loss += criterion(outputs, cal_labels)
                
            loss.backward()
            optimizer.step()
            
        alphas_uncon = torch.sigmoid(alphas_raw).detach()
        accs_uncon = evaluate_model(merged_model, datasets, base_model, expert_models, alphas_uncon, device)
        avg_uncon = np.mean(accs_uncon)
        results_unconstrained.append(avg_uncon)
        print(f"Unconstrained Tuning (M={M}) Average Accuracy: {avg_uncon:.2f}%", flush=True)
        
        # --- Method: RBPM (Ours, lambda = 0.01) ---
        print(f"Running RBPM (lambda = 0.01) for M = {M}...", flush=True)
        merged_model.load_state_dict(base_model.state_dict())
        theta = torch.zeros(4, 3, requires_grad=True, device=device)
        with torch.no_grad():
            theta[:, 0] = -1.0986
            
        optimizer = optim.Adam([theta], lr=1e-1)
        z = torch.tensor([l/11.0 for l in range(12)], device=device)
        
        for step in range(30):
            optimizer.zero_grad()
            alphas = torch.zeros(4, 12, device=device)
            for k in range(4):
                alphas[k] = theta[k][0] + theta[k][1]*z + theta[k][2]*(z**2)
            alphas = torch.sigmoid(alphas)
            
            val_loss = 0.0
            for k in range(4):
                cal_imgs, cal_labels = calibration_data[k]
                params = get_merged_params_dict_for_task(base_model, expert_models, alphas, k)
                outputs = tf.functional_call(merged_model, params, cal_imgs)
                val_loss += criterion(outputs, cal_labels)
                
            rad_reg = torch.sum(torch.abs(theta[:, 0] - (-1.0986))) + torch.sum(torch.abs(theta[:, 1:]))
            total_loss = val_loss + 0.01 * rad_reg
            total_loss.backward()
            optimizer.step()
            
        alphas_rbpm = torch.zeros(4, 12, device=device)
        with torch.no_grad():
            for k in range(4):
                alphas_rbpm[k] = theta[k][0] + theta[k][1]*z + theta[k][2]*(z**2)
            alphas_rbpm = torch.sigmoid(alphas_rbpm).detach()
            
        accs_rbpm = evaluate_model(merged_model, datasets, base_model, expert_models, alphas_rbpm, device)
        avg_rbpm = np.mean(accs_rbpm)
        results_rbpm.append(avg_rbpm)
        print(f"RBPM (M={M}) Average Accuracy: {avg_rbpm:.2f}%", flush=True)
        
    print("\n=== SWEEP RESULTS ===")
    print("M values:", M_list)
    print("Unconstrained:", results_unconstrained)
    print("RBPM (Ours):", results_rbpm)
    
    # Plot Sensitivity Analysis
    plt.figure(figsize=(10, 6))
    plt.plot(M_list, results_rbpm, marker='s', color='green', linewidth=2.5, label='RBPM (Ours, lambda = 0.01)')
    plt.plot(M_list, results_unconstrained, marker='x', color='purple', linewidth=2.0, linestyle='--', label='Offline Unconstrained Few-Shot')
    plt.axhline(y=uniform_avg, color='gray', linestyle=':', label='Static Uniform Baseline')
    
    plt.title('Few-Shot Sensitivity Analysis: Merging Accuracy vs. Calibration Set Size ($M$)')
    plt.xlabel('Calibration Dataset Size ($M$ samples per task)')
    plt.ylabel('Average Multi-Task Test Accuracy (%)')
    plt.xticks(M_list)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('results/fig3_sensitivity_sweep.png')
    print("Sensitivity plot saved to results/fig3_sensitivity_sweep.png!", flush=True)
    
    # Save sweep results to JSON
    sweep_data = {
        "M_list": M_list,
        "unconstrained": results_unconstrained,
        "rbpm": results_rbpm,
        "uniform": uniform_avg
    }
    with open('results/sweep_results.json', 'w') as f:
        json.dump(sweep_data, f, indent=2)
    print("Sweep results saved to results/sweep_results.json!", flush=True)

if __name__ == '__main__':
    main()
