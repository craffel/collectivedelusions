import os
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

def get_transforms(task):
    if task == 'fashion_mnist':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataset(task, train=True):
    transform = get_transforms(task)
    if task == 'cifar10':
        return datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif task == 'svhn':
        split = 'train' if train else 'test'
        return datasets.SVHN(root='./data', split=split, download=True, transform=transform)
    elif task == 'fashion_mnist':
        return datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown task: {task}")

# Helper to extract ResNet18 backbone features
class ResNetBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

def merge_backbones(pretrained_state, expert_states, lam):
    merged_state = copy.deepcopy(pretrained_state)
    backbone_keys = [k for k in pretrained_state.keys() if not k.startswith('fc.')]
    for k in backbone_keys:
        delta_sum = torch.zeros_like(pretrained_state[k]).float()
        for task, state in expert_states.items():
            delta = state[k].float() - pretrained_state[k].float()
            delta_sum += delta
        merged_state[k] = pretrained_state[k].float() + lam * delta_sum
        merged_state[k] = merged_state[k].to(pretrained_state[k].dtype)
    return merged_state

def get_features(model, loader):
    model.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model(images)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

def kl_loss_fn(outputs, targets, T=2.0):
    p = nn.functional.log_softmax(outputs / T, dim=1)
    q = nn.functional.softmax(targets / T, dim=1)
    return nn.functional.kl_div(p, q, reduction='batchmean') * (T ** 2)

def adapt_head_vanilla(head, cal_feats, cal_targets, epochs=50, lr=1e-2, weight_decay=1e-4):
    head = copy.deepcopy(head).to(device)
    head.train()
    optimizer = optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(cal_feats, cal_targets)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for epoch in range(epochs):
        for feats, targets in loader:
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = head(feats)
            loss = kl_loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
    return head

def adapt_head_sam(head, cal_feats, cal_targets, epochs=50, lr=1e-2, weight_decay=1e-4, rho=0.05):
    head = copy.deepcopy(head).to(device)
    optimizer = optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(cal_feats, cal_targets)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for epoch in range(epochs):
        for feats, targets in loader:
            feats, targets = feats.to(device), targets.to(device)
            head.train()
            outputs = head(feats)
            loss = kl_loss_fn(outputs, targets)
            loss.backward()
            
            params = [p for p in head.parameters() if p.requires_grad]
            grads = [p.grad.clone() for p in params if p.grad is not None]
            grad_norm = torch.sqrt(sum([torch.sum(g ** 2) for g in grads]))
            
            if grad_norm > 0:
                scale = rho / grad_norm
                for p, g in zip(params, grads):
                    p.data.add_(g * scale)
                    
            optimizer.zero_grad()
            outputs_perturbed = head(feats)
            loss_perturbed = kl_loss_fn(outputs_perturbed, targets)
            loss_perturbed.backward()
            
            if grad_norm > 0:
                for p, g in zip(params, grads):
                    p.data.sub_(g * scale)
                    
            optimizer.step()
            optimizer.zero_grad()
    return head

def main():
    print(f"Loading pre-trained ResNet18 and experts for Ablations...")
    tasks = ["cifar10", "svhn", "fashion_mnist"]
    
    pretrained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    pretrained_state = pretrained_model.state_dict()
    
    expert_states = {}
    expert_heads = {}
    for task in tasks:
        ckpt_path = f"checkpoints/expert_{task}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        expert_states[task] = ckpt['state_dict']
        head = nn.Linear(512, 10)
        head.weight.data.copy_(ckpt['state_dict']['fc.weight'])
        head.bias.data.copy_(ckpt['state_dict']['fc.bias'])
        expert_heads[task] = head

    # We use lambda = 0.3 (the best merging coefficient)
    lam = 0.3
    merged_state = merge_backbones(pretrained_state, expert_states, lam)
    
    merged_model = resnet18()
    merged_model.fc = nn.Identity()
    merged_model.load_state_dict({k: v for k, v in merged_state.items() if not k.startswith('fc.')}, strict=False)
    merged_model = merged_model.to(device)
    merged_model.eval()

    # Pre-load full datasets
    full_train_datasets = {t: get_dataset(t, train=True) for t in tasks}
    full_test_datasets = {t: get_dataset(t, train=False) for t in tasks}
    
    test_loaders = {t: DataLoader(full_test_datasets[t], batch_size=128, shuffle=False, num_workers=2) for t in tasks}
    
    print("Pre-computing test features under merged backbone...")
    test_feats_dict = {}
    test_labels_dict = {}
    for t in tasks:
        test_feats, test_labels = get_features(merged_model, test_loaders[t])
        test_feats_dict[t] = test_feats
        test_labels_dict[t] = test_labels

    # Define a function to get calibration features/targets for a given size
    def get_cal_data(task, cal_size):
        np.random.seed(42)
        cal_indices = np.random.choice(len(full_train_datasets[task]), cal_size, replace=False)
        cal_subset = Subset(full_train_datasets[task], cal_indices)
        cal_loader = DataLoader(cal_subset, batch_size=64, shuffle=False, num_workers=2)
        
        # Expert soft targets
        exp_model = resnet18()
        exp_model.fc = nn.Linear(512, 10)
        exp_model.load_state_dict(expert_states[task])
        exp_model = exp_model.to(device)
        exp_model.eval()
        
        targets = []
        with torch.no_grad():
            for images, _ in cal_loader:
                images = images.to(device)
                outputs = exp_model(images)
                targets.append(outputs.cpu())
        cal_targets = torch.cat(targets, dim=0)
        
        # Merged backbone features
        cal_feats, _ = get_features(merged_model, cal_loader)
        return cal_feats, cal_targets

    # Sweep 1: SAM perturbation radius rho
    print("=========================================")
    print("SWEEPING SAM PERTURBATION RADIUS RHO...")
    print("=========================================")
    rhos = [0.001, 0.01, 0.05, 0.1, 0.2]
    rho_results = {t: [] for t in tasks}
    
    # Pre-compute cal data for size 1000
    cal_data_1000 = {t: get_cal_data(t, 1000) for t in tasks}
    
    for rho in rhos:
        print(f"Evaluating rho = {rho}")
        for t in tasks:
            cal_feats, cal_targets = cal_data_1000[t]
            orig_head = expert_heads[t].to(device)
            
            adapted_head = adapt_head_sam(
                orig_head, cal_feats, cal_targets, 
                epochs=100, lr=1e-2, rho=rho
            )
            adapted_head.eval()
            with torch.no_grad():
                outputs = adapted_head(test_feats_dict[t].to(device))
                _, predicted = outputs.max(1)
                acc = predicted.eq(test_labels_dict[t].to(device)).sum().item() / len(test_labels_dict[t])
            rho_results[t].append(acc)
            print(f"Task: {t} | Acc: {acc*100:.2f}%")

    # Sweep 2: Calibration size N_cal
    print("=========================================")
    print("SWEEPING CALIBRATION SIZE N_CAL...")
    print("=========================================")
    cal_sizes = [100, 250, 500, 1000]
    ncal_results = {t: [] for t in tasks}
    
    for size in cal_sizes:
        print(f"Evaluating calibration size = {size}")
        for t in tasks:
            cal_feats, cal_targets = get_cal_data(t, size)
            orig_head = expert_heads[t].to(device)
            
            adapted_head = adapt_head_sam(
                orig_head, cal_feats, cal_targets, 
                epochs=100, lr=1e-2, rho=0.05
            )
            adapted_head.eval()
            with torch.no_grad():
                outputs = adapted_head(test_feats_dict[t].to(device))
                _, predicted = outputs.max(1)
                acc = predicted.eq(test_labels_dict[t].to(device)).sum().item() / len(test_labels_dict[t])
            ncal_results[t].append(acc)
            print(f"Task: {t} | Acc: {acc*100:.2f}%")

    # Save ablation results
    np.save("ablation_results.npy", {'rhos': rhos, 'rho_results': rho_results, 'cal_sizes': cal_sizes, 'ncal_results': ncal_results})

    # Plot Rho Sweep
    plt.figure(figsize=(6, 4))
    for t in tasks:
        plt.plot(rhos, [val*100 for val in rho_results[t]], label=t.upper(), marker='o')
    plt.title("Effect of SAM Perturbation Radius ($\\rho$)")
    plt.xlabel("SAM Perturbation Radius ($\\rho$)")
    plt.ylabel("Test Accuracy (%)")
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ablation_rho.png", dpi=300)
    plt.close()

    # Plot N_cal Sweep
    plt.figure(figsize=(6, 4))
    for t in tasks:
        plt.plot(cal_sizes, [val*100 for val in ncal_results[t]], label=t.upper(), marker='s')
    plt.title("Effect of Calibration Dataset Size ($N_{\\text{cal}}$)")
    plt.xlabel("Calibration Size ($N_{\\text{cal}}$)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ablation_ncal.png", dpi=300)
    plt.close()
    
    print("Ablation studies complete! Plots saved to ablation_rho.png and ablation_ncal.png")

if __name__ == "__main__":
    main()
