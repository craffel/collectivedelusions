import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import timm

import sys
sys.path.insert(0, './local_packages_310')

TASKS = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_layer_group(param_name):
    if 'patch_embed' in param_name or 'cls_token' in param_name or 'pos_embed' in param_name or 'norm_pre' in param_name:
        return 0
    elif 'blocks' in param_name:
        parts = param_name.split('.')
        block_idx = int(parts[1])
        return block_idx + 1
    elif 'norm.' in param_name:
        return 13
    else:
        return -1

def get_raw_dataset(task_name, split='train'):
    if task_name in ['MNIST', 'FashionMNIST']:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    if task_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(root='./data', train=(split=='train'), download=False, transform=transform)
    elif task_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=(split=='train'), download=False, transform=transform)
    elif task_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=(split=='train'), download=False, transform=transform)
    elif task_name == 'SVHN':
        dataset = torchvision.datasets.SVHN(root='./data', split=('train' if split=='train' else 'test'), download=False, transform=transform)
    return dataset

def get_cached_dataset(task_name, split, size, seed):
    print(f"  Pre-loading and caching raw {task_name} dataset...")
    dataset = get_raw_dataset(task_name, split)
    if size is not None and size < len(dataset):
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=g)[:size].tolist()
        dataset = Subset(dataset, indices)
    
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    x_all, y_all = next(iter(loader))
    return TensorDataset(x_all, y_all)

def main():
    set_seed(42)
    print(f"Using device: {DEVICE}")
    print("Loading test datasets (size=16 for low-latency tracing)...")
    test_datasets = {}
    for task in TASKS:
        test_datasets[task] = get_cached_dataset(task, split='test', size=16, seed=999)
    print("Datasets loaded successfully!")
    
    # Instantiate models
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    model.head = torch.nn.Linear(model.num_features, 10)
    model = model.to(DEVICE)
    
    radii = [0.0, 0.01, 0.05, 0.1, 0.2]
    # Noise scale for weight perturbation
    sigmas = [0.001, 0.002, 0.005]
    num_trials = 3
    
    results = {}
    
    for rho in radii:
        results[rho] = {sig: [] for sig in sigmas}
        print(f"\nEvaluating experts pre-trained with SAM rho = {rho}")
        
        for task in TASKS:
            expert_file = f"checkpoints/expert_{task}_seed42_rho{rho}.pt"
            if not os.path.exists(expert_file):
                print(f"Warning: checkpoint {expert_file} not found. Skipping.")
                continue
                
            sd = torch.load(expert_file, map_location=DEVICE)
            model.load_state_dict(sd)
            
            # Prepare inputs & targets
            ds = test_datasets[task]
            loader = DataLoader(ds, batch_size=16, shuffle=False)
            
            # Evaluate baseline loss
            model.eval()
            total_loss_base = 0.0
            criterion = torch.nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)
                    total_loss_base += criterion(outputs, targets).item() * inputs.size(0)
            loss_base = total_loss_base / len(ds)
            
            # Evaluate perturbed loss across trials and sigmas
            for sig in sigmas:
                trial_loss_diffs = []
                for trial in range(num_trials):
                    # Load state dict afresh
                    model.load_state_dict(sd)
                    
                    # Add noise to backbone weights
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if get_layer_group(name) >= 0:
                                noise = torch.randn_like(param) * sig
                                param.add_(noise)
                                
                    # Evaluate perturbed model
                    total_loss_pert = 0.0
                    with torch.no_grad():
                        for inputs, targets in loader:
                            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                            outputs = model(inputs)
                            total_loss_pert += criterion(outputs, targets).item() * inputs.size(0)
                    loss_pert = total_loss_pert / len(ds)
                    trial_loss_diffs.append(loss_pert - loss_base)
                
                mean_diff = np.mean(trial_loss_diffs)
                results[rho][sig].append(mean_diff)
                print(f"  Task: {task:<15} | Sigma: {sig:.3f} | Mean Loss Increase: {mean_diff:+.6f}")
                
    print("\n" + "="*50)
    print("SUMMARY: AVERAGE WEIGHT-SPACE LOSS INCREASE (trace proxy)")
    print("="*50)
    print(f"{'SAM Radius (rho)':<20} | " + " | ".join([f"Sigma={sig:.3f}" for sig in sigmas]))
    print("-"*50)
    for rho in radii:
        row_str = f"{rho:<20.2f} | "
        cols = []
        for sig in sigmas:
            val = np.mean(results[rho][sig])
            cols.append(f"{val:.6f}")
        row_str += " | ".join(cols)
        print(row_str)
    print("="*50)

if __name__ == '__main__':
    main()
