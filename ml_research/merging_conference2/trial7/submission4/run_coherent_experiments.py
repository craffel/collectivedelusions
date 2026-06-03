import os
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False

# ----------------------------------------------------------------------
# 1. Models and Merging
# ----------------------------------------------------------------------

def get_resnet18_progenitor():
    """Load standard pre-trained ImageNet ResNet-18 progenitor."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    return model

def create_expert_model(progenitor, num_classes=10):
    """Clone progenitor and replace classification head for a task."""
    model = models.resnet18()
    # Match the progenitor's fc shape first to avoid load_state_dict size mismatch
    model.fc = nn.Linear(512, progenitor.fc.out_features)
    model.load_state_dict(progenitor.state_dict())
    # If the desired num_classes is different, replace fc
    if progenitor.fc.out_features != num_classes:
        model.fc = nn.Linear(512, num_classes)
    return model

def merge_weights_averaging(expert_models):
    """Merge experts via Weight Averaging (WA)."""
    merged = create_expert_model(models.resnet18(), num_classes=10)
    merged_sd = merged.state_dict()
    
    expert_sds = [exp.state_dict() for exp in expert_models.values()]
    
    for key in merged_sd.keys():
        # Average all weights, including the classification head since they share the same head!
        stacked = torch.stack([sd[key].float().cpu() for sd in expert_sds], dim=0)
        merged_sd[key] = torch.mean(stacked, dim=0).to(merged_sd[key].dtype)
            
    merged.load_state_dict(merged_sd)
    return merged

def merge_task_arithmetic(progenitor, expert_models, lam=0.3):
    """Merge experts via Task Arithmetic (TA)."""
    merged = create_expert_model(progenitor, num_classes=10)
    merged_sd = merged.state_dict()
    
    prog_sd = progenitor.state_dict()
    expert_sds = {task: exp.state_dict() for task, exp in expert_models.items()}
    
    for key in merged_sd.keys():
        # Merge all weights (for shared task classification head, we also apply task arithmetic or average)
        # To be consistent with standard task arithmetic, we compute task vectors relative to progenitor.
        # Note: progenitor head is random/untrained or different, so we compute task vector relative to progenitor's fc
        task_vectors = []
        for task, sd in expert_sds.items():
            tv = sd[key].float().cpu() - prog_sd[key].float().cpu()
            task_vectors.append(tv)
        
        sum_tv = torch.sum(torch.stack(task_vectors, dim=0), dim=0)
        merged_sd[key] = (prog_sd[key].float().cpu() + lam * sum_tv).to(merged_sd[key].dtype)
            
    merged.load_state_dict(merged_sd)
    return merged

# ----------------------------------------------------------------------
# 2. Data Loading & Preprocessing
# ----------------------------------------------------------------------

def get_transforms():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_coherent_dataloaders(data_dir="./data", batch_size=128):
    """Load CIFAR-10 and split train set into 3 disjoint subsets of 5,000 samples each."""
    os.makedirs(data_dir, exist_ok=True)
    color_tf = get_transforms()
    
    cifar_train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=color_tf)
    cifar_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=color_tf)
    
    # Create three disjoint subsets of 5,000 samples
    subsets = {
        'cifar1': Subset(cifar_train_full, list(range(0, 5000))),
        'cifar2': Subset(cifar_train_full, list(range(5000, 10000))),
        'cifar3': Subset(cifar_train_full, list(range(10000, 15000)))
    }
    
    loaders = {
        'train': {
            'cifar1': DataLoader(subsets['cifar1'], batch_size=batch_size, shuffle=True, num_workers=2),
            'cifar2': DataLoader(subsets['cifar2'], batch_size=batch_size, shuffle=True, num_workers=2),
            'cifar3': DataLoader(subsets['cifar3'], batch_size=batch_size, shuffle=True, num_workers=2)
        },
        'test': DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=2)
    }
    return loaders

# ----------------------------------------------------------------------
# 3. Training & Evaluation
# ----------------------------------------------------------------------

def train_expert(model, dataloader, epochs=5, lr=1e-4, weight_decay=1e-4, device='cuda'):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
    return model

def evaluate_model(model, dataloader, device='cuda'):
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    accuracy = 100.0 * correct / total
    return accuracy

# ----------------------------------------------------------------------
# 4. Calibration Implementation
# ----------------------------------------------------------------------

class StatisticsHook:
    def __init__(self):
        self.sum_x = None
        self.sum_x2 = None
        self.count = 0

    def hook_fn(self, module, input, output):
        x = input[0]
        B, C, H, W = x.shape
        x_flat = x.transpose(0, 1).flatten(1) # [C, B*H*W]
        
        batch_sum = x_flat.sum(dim=1)
        batch_sum_sq = (x_flat ** 2).sum(dim=1)
        batch_count = x_flat.shape[1]
        
        if self.sum_x is None:
            self.sum_x = batch_sum.clone().double()
            self.sum_x2 = batch_sum_sq.clone().double()
            self.count = batch_count
        else:
            self.sum_x += batch_sum.double()
            self.sum_x2 += batch_sum_sq.double()
            self.count += batch_count

    def get_stats(self):
        mean = self.sum_x / self.count
        var = (self.sum_x2 / self.count) - (mean ** 2)
        var = torch.clamp(var, min=0.0)
        return mean.float(), var.float()


def run_federated_calibration(model, client_loaders, N=128, device='cuda'):
    model = model.to(device)
    model.eval()
    
    bn_layers = []
    hooks = []
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))
            hook = StatisticsHook()
            hooks.append(hook)
            handles.append(module.register_forward_hook(hook.hook_fn))
            
    client_stats = {}
    
    for task_name, loader in client_loaders.items():
        for hook in hooks:
            hook.sum_x = None
            hook.sum_x2 = None
            hook.count = 0
            
        samples_loaded = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                model(x)
                
                samples_loaded += x.shape[0]
                if samples_loaded >= N:
                    break
                    
        client_stats[task_name] = [hook.get_stats() for hook in hooks]
        
    for handle in handles:
        handle.remove()
        
    for idx, (name, module) in enumerate(bn_layers):
        global_mean = torch.zeros_like(module.running_mean)
        global_var = torch.zeros_like(module.running_var)
        
        for task_name in client_loaders.keys():
            mean_local, var_local = client_stats[task_name][idx]
            global_mean += mean_local / len(client_loaders)
            global_var += var_local / len(client_loaders)
            
        module.running_mean.copy_(global_mean)
        module.running_var.copy_(global_var)
        if module.num_batches_tracked is not None:
            module.num_batches_tracked.copy_(torch.tensor(1, dtype=torch.long))


def profile_activation_std(model, test_loader, device='cuda'):
    model.eval()
    target_module = None
    for name, module in model.named_modules():
        if "layer4.1.bn2" in name:
            target_module = module
            break
            
    if target_module is None:
        return 0.0
        
    activation_stds = []
    
    def diag_hook(module, input, output):
        x = input[0]
        std = torch.std(x, dim=(0, 2, 3))
        activation_stds.append(std.cpu())
        
    handle = target_module.register_forward_hook(diag_hook)
    
    for x, _ in test_loader:
        x = x.to(device)[:64]
        with torch.no_grad():
            model(x)
        break
        
    handle.remove()
    return torch.mean(activation_stds[0]).item()

# ----------------------------------------------------------------------
# Main Execution Pipeline
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coherent Model Merging Experiment")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per expert")
    parser.add_argument("--lr", type=type(1e-4), default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--force_retrain", action="store_true", help="Force retraining of experts")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = args.device
    print(f"Using device: {device}")
    
    # 1. Load data
    print("\n--- Loading Datasets ---")
    loaders = get_coherent_dataloaders(batch_size=args.batch_size)
    
    # 2. Train or Load Experts
    expert_paths = {
        'cifar1': 'expert_coherent_cifar1.pt',
        'cifar2': 'expert_coherent_cifar2.pt',
        'cifar3': 'expert_coherent_cifar3.pt'
    }
    
    expert_models = {}
    progenitor = get_resnet18_progenitor()
    
    print("\n--- Initializing & Fine-tuning Coherent Experts ---")
    for task in ['cifar1', 'cifar2', 'cifar3']:
        model_path = expert_paths[task]
        # Progenitor has Imagenet weights and 1000 fc classes.
        # We replace the head of progenitor with 10 fc classes for CIFAR-10.
        # To make sure we have a consistent progenitor fc, we can replace it once.
        prog_cifar10 = models.resnet18()
        prog_cifar10.load_state_dict(progenitor.state_dict())
        prog_cifar10.fc = nn.Linear(512, 10)
        
        expert_models[task] = create_expert_model(prog_cifar10, num_classes=10)
        
        if os.path.exists(model_path) and not args.force_retrain:
            print(f"Loading pre-trained expert for {task} from {model_path}...")
            expert_models[task].load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Training expert for {task} on split...")
            expert_models[task] = train_expert(
                expert_models[task], 
                loaders['train'][task], 
                epochs=args.epochs, 
                lr=args.lr, 
                device=device
            )
            torch.save(expert_models[task].state_dict(), model_path)
            print(f"Saved expert for {task} to {model_path}")
            
    # Evaluate individual expert baselines on the common test set
    print("\n--- Evaluating Coherent Expert Baselines on Full CIFAR-10 Test Set ---")
    expert_accuracies = {}
    for task in ['cifar1', 'cifar2', 'cifar3']:
        acc = evaluate_model(expert_models[task], loaders['test'], device=device)
        expert_accuracies[task] = acc
        print(f"Expert {task.upper()} Accuracy: {acc:.2f}%")
        
    expert_avg = sum(expert_accuracies.values()) / 3.0
    print(f"Average Expert Accuracy: {expert_avg:.2f}%")
    
    # 3. Model Merging Baselines (WA & TA)
    print("\n--- Merging Coherent Experts ---")
    # Define a common progenitor for TA
    prog_cifar10 = models.resnet18()
    prog_cifar10.load_state_dict(progenitor.state_dict())
    prog_cifar10.fc = nn.Linear(512, 10)
    
    merged_models = {
        'WA': merge_weights_averaging(expert_models),
        'TA': merge_task_arithmetic(prog_cifar10, expert_models, lam=0.25)
    }
    
    coherent_results = {
        'meta': {
            'expert_accuracies': expert_accuracies,
            'expert_average': expert_avg
        },
        'WA': {},
        'TA': {}
    }
    
    # 4. Evaluation of Baselines
    for merge_name, merged_model in merged_models.items():
        print(f"\nEvaluating Merging Paradigm: {merge_name}")
        
        # Scenario A: Uncalibrated
        acc_uncal = evaluate_model(merged_model, loaders['test'], device=device)
        std_uncal = profile_activation_std(merged_model, loaders['test'], device=device)
        print(f"  Uncalibrated Acc: {acc_uncal:.2f}% | Activation std: {std_uncal:.4f}")
        
        # Scenario B: Oracle Calibration (Using 1024 real samples per task)
        oracle_model = create_expert_model(prog_cifar10, num_classes=10)
        oracle_model.load_state_dict(merged_model.state_dict())
        run_federated_calibration(oracle_model, loaders['train'], N=1024, device=device)
        acc_oracle = evaluate_model(oracle_model, loaders['test'], device=device)
        std_oracle = profile_activation_std(oracle_model, loaders['test'], device=device)
        print(f"  Oracle Calibrated Acc (N=1024): {acc_oracle:.2f}% | Activation std: {std_oracle:.4f}")
        
        # Scenario C: Proposed Fed-BNC (N=128)
        fed_model = create_expert_model(prog_cifar10, num_classes=10)
        fed_model.load_state_dict(merged_model.state_dict())
        run_federated_calibration(fed_model, loaders['train'], N=128, device=device)
        acc_fed = evaluate_model(fed_model, loaders['test'], device=device)
        std_fed = profile_activation_std(fed_model, loaders['test'], device=device)
        print(f"  Fed-BNC Acc (N=128): {acc_fed:.2f}% | Activation std: {std_fed:.4f}")
        
        coherent_results[merge_name] = {
            'uncalibrated': {'accuracy': acc_uncal, 'activation_std': std_uncal},
            'oracle': {'accuracy': acc_oracle, 'activation_std': std_oracle},
            'fed_bnc': {'accuracy': acc_fed, 'activation_std': std_fed}
        }

    # Save results to JSON
    with open('coherent_results.json', 'w') as f:
        json.dump(coherent_results, f, indent=4)
    print("\nCoherent experiments complete! Results saved to coherent_results.json")

    # Generate Plot
    print("\n--- Generating Coherent Plot ---")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    plt.figure(figsize=(7, 5))
    categories = ['Uncalibrated', 'Fed-BNC (N=128)', 'Oracle (N=1024)']
    
    wa_accs = [
        coherent_results['WA']['uncalibrated']['accuracy'],
        coherent_results['WA']['fed_bnc']['accuracy'],
        coherent_results['WA']['oracle']['accuracy']
    ]
    ta_accs = [
        coherent_results['TA']['uncalibrated']['accuracy'],
        coherent_results['TA']['fed_bnc']['accuracy'],
        coherent_results['TA']['oracle']['accuracy']
    ]
    
    x_indices = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x_indices - width/2, wa_accs, width, label='Weight Averaging')
    plt.bar(x_indices + width/2, ta_accs, width, label='Task Arithmetic ($\lambda=0.25$)')
    plt.axhline(y=expert_avg, color='r', linestyle='--', label='Average Coherent Expert Baseline')
    plt.ylabel('CIFAR-10 Accuracy (%)', fontsize=12)
    plt.title('Coherent Model Merging: Calibration Efficacy', fontsize=14, fontweight='bold')
    plt.xticks(x_indices, categories, fontsize=11)
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig('plot_coherent_results.png', dpi=300)
    plt.close()
    print("Coherent Plot generated successfully as plot_coherent_results.png!")
