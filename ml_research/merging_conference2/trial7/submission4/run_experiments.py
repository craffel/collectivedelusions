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
        # Pragmatic robust solution: disable cuDNN if initialization is unstable on some cluster nodes
        torch.backends.cudnn.enabled = False

# ----------------------------------------------------------------------
# 1. Models and Merging
# ----------------------------------------------------------------------

def get_resnet18_progenitor():
    """Load standard pre-trained ImageNet ResNet-18 progenitor."""
    # Using ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    return model

def create_expert_model(progenitor, num_classes=10):
    """Clone progenitor and replace classification head for a task."""
    model = models.resnet18()
    model.load_state_dict(progenitor.state_dict())
    model.fc = nn.Linear(512, num_classes)
    return model

def set_task_head(merged_model, expert_model):
    """Assign the task head from the expert to the merged model."""
    merged_model.fc = expert_model.fc.to(next(merged_model.parameters()).device)

def merge_weights_averaging(expert_models):
    """Merge experts via Weight Averaging (WA)."""
    # Create standard ResNet-18 with 10-class head
    merged = create_expert_model(models.resnet18(), num_classes=10)
    merged_sd = merged.state_dict()
    
    expert_sds = [exp.state_dict() for exp in expert_models.values()]
    M = len(expert_sds)
    
    for key in merged_sd.keys():
        if "fc" not in key:  # Backbone weights only
            # Convert all tensors to CPU/float for robustness
            stacked = torch.stack([sd[key].float().cpu() for sd in expert_sds], dim=0)
            merged_sd[key] = torch.mean(stacked, dim=0).to(merged_sd[key].dtype)
            
    merged.load_state_dict(merged_sd)
    return merged

def merge_task_arithmetic(progenitor, expert_models, lam=0.3):
    """Merge experts via Task Arithmetic (TA)."""
    # Create standard ResNet-18 with 10-class head
    merged = create_expert_model(progenitor, num_classes=10)
    merged_sd = merged.state_dict()
    
    prog_sd = progenitor.state_dict()
    expert_sds = {task: exp.state_dict() for task, exp in expert_models.items()}
    
    for key in merged_sd.keys():
        if "fc" not in key:  # Backbone weights only
            task_vectors = []
            for task, sd in expert_sds.items():
                # Perform operations on CPU
                tv = sd[key].float().cpu() - prog_sd[key].float().cpu()
                task_vectors.append(tv)
            
            sum_tv = torch.sum(torch.stack(task_vectors, dim=0), dim=0)
            merged_sd[key] = (prog_sd[key].float().cpu() + lam * sum_tv).to(merged_sd[key].dtype)
            
    merged.load_state_dict(merged_sd)
    return merged

# ----------------------------------------------------------------------
# 2. Data Loading & Preprocessing
# ----------------------------------------------------------------------

def get_transforms(is_grayscale=False):
    """Get standardized preprocessing transforms."""
    if is_grayscale:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Replicate channels to 3
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(data_dir="./data", batch_size=128, sample_limit=None):
    """Load and prepare train and test dataloaders for MNIST, F-MNIST, CIFAR-10."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Transforms
    gray_tf = get_transforms(is_grayscale=True)
    color_tf = get_transforms(is_grayscale=False)
    
    # MNIST
    mnist_train_full = datasets.MNIST(data_dir, train=True, download=True, transform=gray_tf)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=gray_tf)
    # F-MNIST
    fmnist_train_full = datasets.FashionMNIST(data_dir, train=True, download=True, transform=gray_tf)
    fmnist_test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=gray_tf)
    # CIFAR-10
    cifar_train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=color_tf)
    cifar_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=color_tf)
    
    # Subsetting for fast expert fine-tuning
    if sample_limit is not None:
        # Replicating the 5,000-sample subsets from literature
        mnist_train = Subset(mnist_train_full, list(range(min(sample_limit, len(mnist_train_full)))))
        fmnist_train = Subset(fmnist_train_full, list(range(min(sample_limit, len(fmnist_train_full)))))
        cifar_train = Subset(cifar_train_full, list(range(min(sample_limit, len(cifar_train_full)))))
    else:
        mnist_train = mnist_train_full
        fmnist_train = fmnist_train_full
        cifar_train = cifar_train_full
        
    loaders = {
        'train': {
            'mnist': DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2),
            'fmnist': DataLoader(fmnist_train, batch_size=batch_size, shuffle=True, num_workers=2),
            'cifar': DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=2)
        },
        'test': {
            'mnist': DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2),
            'fmnist': DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, num_workers=2),
            'cifar': DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=2)
        }
    }
    return loaders

# ----------------------------------------------------------------------
# 3. Training & Evaluation
# ----------------------------------------------------------------------

def train_expert(model, dataloader, epochs=5, lr=1e-4, weight_decay=1e-4, device='cuda'):
    """Fine-tune an expert model on a specific task."""
    model = model.to(device)
    model.train()
    
    # Literature details: SGD with momentum 0.9 or AdamW
    # Let's use AdamW as it's highly robust and achieves excellent performance in fewer epochs.
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

def evaluate_model(model, dataloader, task_name, expert_model, device='cuda', corruption=None, corruption_val=0.0):
    """Evaluate model on a task by loading appropriate head and processing test sets."""
    model = model.to(device)
    model.eval()
    
    # Set the correct task-specific head on the merged model
    set_task_head(model, expert_model)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # Apply test-time corruptions if requested (The Pragmatist's robustness evaluation)
            if corruption == "gaussian_noise" and corruption_val > 0.0:
                noise = torch.randn_like(x) * corruption_val
                x = x + noise
            elif corruption == "brightness" and corruption_val != 0.0:
                x = x + corruption_val
                
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
    """Surgically measures exact channel-wise activation statistics."""
    def __init__(self):
        self.sum_x = None
        self.sum_x2 = None
        self.count = 0

    def hook_fn(self, module, input, output):
        x = input[0] # Pre-normalization activations
        B, C, H, W = x.shape
        
        # Flatten batch and spatial dimensions
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


class ClassConditionalStatisticsHook:
    """Measures class-conditional activation statistics to mitigate Non-IID class skew."""
    def __init__(self):
        self.sum_x = {}
        self.sum_x2 = {}
        self.count = {}
        self.current_labels = None

    def hook_fn(self, module, input, output):
        if self.current_labels is None:
            return
        x = input[0]
        B, C, H, W = x.shape
        labels = self.current_labels
        
        x_flat = x.flatten(2) # [B, C, H*W]
        
        for c in range(10):
            mask = (labels == c)
            if not mask.any():
                continue
            
            # Filter samples belonging to class c
            x_c = x_flat[mask] # [N_c, C, H*W]
            sample_sum = x_c.sum(dim=(0, 2)) # [C]
            sample_sum_sq = (x_c ** 2).sum(dim=(0, 2)) # [C]
            sample_count = mask.sum().item() * H * W
            
            if c not in self.sum_x:
                self.sum_x[c] = sample_sum.clone().double()
                self.sum_x2[c] = sample_sum_sq.clone().double()
                self.count[c] = sample_count
            else:
                self.sum_x[c] += sample_sum.double()
                self.sum_x2[c] += sample_sum_sq.double()
                self.count[c] += sample_count

    def get_stats_per_class(self):
        stats = {}
        for c in range(10):
            if c in self.sum_x and self.count[c] > 0:
                mean = self.sum_x[c] / self.count[c]
                var = (self.sum_x2[c] / self.count[c]) - (mean ** 2)
                var = torch.clamp(var, min=0.0)
                stats[c] = (mean.float(), var.float())
        return stats


def run_federated_calibration(model, expert_models, client_loaders, N=128, class_skew_filter=None, device='cuda'):
    """
    Simulates Federated BatchNorm Calibration (Fed-BNC).
    
    Each client runs a single forward pass of the merged model on N local, private samples,
    computes the activation statistics of the merged model, and shares ONLY the 1D statistic vectors.
    The server averages these client-side statistics to update the global model.
    """
    model = model.to(device)
    model.eval()
    
    # Identify BatchNorm layers
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
        # Reset hooks for this client
        for hook in hooks:
            hook.sum_x = None
            hook.sum_x2 = None
            hook.count = 0
            
        set_task_head(model, expert_models[task_name])
        
        samples_loaded = 0
        with torch.no_grad():
            for x, y in loader:
                # Apply class filter if non-IID class skew is simulated
                if class_skew_filter is not None:
                    mask = torch.tensor([label.item() in class_skew_filter for label in y], dtype=torch.bool)
                    if not mask.any():
                        continue
                    x, y = x[mask], y[mask]
                    
                x = x.to(device)
                model(x)
                
                samples_loaded += x.shape[0]
                if samples_loaded >= N:
                    break
                    
        # Store statistics computed on this client
        client_stats[task_name] = [hook.get_stats() for hook in hooks]
        
    # Remove hooks
    for handle in handles:
        handle.remove()
        
    # Server-side aggregation: update global BatchNorm running stats
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


def run_class_conditional_federated_calibration(model, expert_models, client_loaders, N=128, class_skew_filter=None, device='cuda'):
    """
    Simulates Class-Conditional Federated BatchNorm Calibration (CC-Fed-BNC).
    
    To combat non-IID class skew, clients compute class-conditional statistics.
    The server aggregates these class-conditional statistics uniformly to form balanced global statistics.
    """
    model = model.to(device)
    model.eval()
    
    # Identify BatchNorm layers
    bn_layers = []
    hooks = []
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))
            hook = ClassConditionalStatisticsHook()
            hooks.append(hook)
            handles.append(module.register_forward_hook(hook.hook_fn))
            
    client_stats = {}
    
    for task_name, loader in client_loaders.items():
        # Reset hooks for this client
        for hook in hooks:
            hook.sum_x = {}
            hook.sum_x2 = {}
            hook.count = {}
            hook.current_labels = None
            
        set_task_head(model, expert_models[task_name])
        
        samples_loaded = 0
        with torch.no_grad():
            for x, y in loader:
                # Apply class filter if non-IID class skew is simulated
                if class_skew_filter is not None:
                    mask = torch.tensor([label.item() in class_skew_filter for label in y], dtype=torch.bool)
                    if not mask.any():
                        continue
                    x, y = x[mask], y[mask]
                    
                x, y = x.to(device), y.to(device)
                
                # Communicate active labels to the hook
                for hook in hooks:
                    hook.current_labels = y
                    
                model(x)
                
                samples_loaded += x.shape[0]
                if samples_loaded >= N:
                    break
                    
        # Store statistics computed on this client
        client_stats[task_name] = [hook.get_stats_per_class() for hook in hooks]
        
    # Remove hooks
    for handle in handles:
        handle.remove()
        
    # Server-side Class-Balanced Aggregation
    # For each BN layer:
    for idx, (name, module) in enumerate(bn_layers):
        # We want to aggregate class-conditional stats across all classes (0 to 9)
        # First, collect class statistics across all clients
        class_means = {}
        class_vars = {}
        
        for c in range(10):
            means_for_class_c = []
            vars_for_class_c = []
            
            for task_name in client_loaders.keys():
                stats_per_class = client_stats[task_name][idx]
                if c in stats_per_class:
                    mean_c, var_c = stats_per_class[c]
                    means_for_class_c.append(mean_c)
                    vars_for_class_c.append(var_c)
            
            if len(means_for_class_c) > 0:
                # Average statistical vectors of class c across the clients that observed it
                class_means[c] = torch.mean(torch.stack(means_for_class_c), dim=0)
                class_vars[c] = torch.mean(torch.stack(vars_for_class_c), dim=0)
                
        # Now, combine the class-conditional stats into a uniform/balanced global mean and variance
        # Formula:
        # global_mean = 1/K * sum(mean_c)
        # global_var = 1/K * sum(var_c + (mean_c - global_mean)^2)
        K = len(class_means)
        if K > 0:
            global_mean = torch.zeros_like(module.running_mean, dtype=torch.double)
            # Compute global mean first
            for c in class_means.keys():
                global_mean += class_means[c].double() / K
                
            global_var = torch.zeros_like(module.running_var, dtype=torch.double)
            for c in class_means.keys():
                mean_c = class_means[c].double()
                var_c = class_vars[c].double()
                global_var += (var_c + (mean_c - global_mean) ** 2) / K
                
            module.running_mean.copy_(global_mean.float())
            module.running_var.copy_(global_var.float())
            if module.num_batches_tracked is not None:
                module.num_batches_tracked.copy_(torch.tensor(1, dtype=torch.long))

# ----------------------------------------------------------------------
# 4.5 Adaptive Blending Implementation (Ada-Fed-BNC)
# ----------------------------------------------------------------------

def extract_running_stats(model):
    """Extract and clone the current running statistics of all BN layers."""
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            stats[name] = (module.running_mean.clone(), module.running_var.clone())
    return stats

def apply_adaptive_calibration(model, expert_models, task_name, global_running_stats, alpha=1.0, device='cuda'):
    """
    Applies adaptive statistics blending for a specific task.
    
    At each BN layer, we compute the divergence D between global calibrated stats
    and the expert's pre-saved stats, compute beta = 1 - exp(-alpha * D),
    and set the running mean and variance to:
    mean = beta * mean_expert + (1 - beta) * mean_global
    var = beta * var_expert + (1 - beta) * var_global
    """
    model.eval()
    expert_model = expert_models[task_name].to(device)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            expert_module = None
            for exp_name, exp_mod in expert_model.named_modules():
                if exp_name == name:
                    expert_module = exp_mod
                    break
            
            if expert_module is not None:
                mean_global, var_global = global_running_stats[name]
                mean_expert = expert_module.running_mean.to(device)
                var_expert = expert_module.running_var.to(device)
                
                # Compute statistical divergence (normalized distance)
                # D = 1/M * sum( (mean_global - mean_expert)^2 / (var_expert + 1e-5) )
                eps = 1e-5
                diff_sq = (mean_global - mean_expert) ** 2
                div = torch.mean(diff_sq / (var_expert + eps))
                
                # Compute adaptive blending weight
                beta = 1.0 - torch.exp(-alpha * div)
                beta = torch.clamp(beta, 0.0, 1.0)
                
                # Blend statistics
                adapted_mean = beta * mean_expert + (1.0 - beta) * mean_global
                adapted_var = beta * var_expert + (1.0 - beta) * var_global
                
                module.running_mean.copy_(adapted_mean)
                module.running_var.copy_(adapted_var)

# ----------------------------------------------------------------------
# 5. Diagnostic Profiling
# ----------------------------------------------------------------------

def profile_activation_std(model, expert_models, test_loader, task_name, device='cuda'):
    """Profile the standard deviation of activations in deep layers to monitor collapse."""
    model.eval()
    set_task_head(model, expert_models[task_name])
    
    # We will hook the deepest BatchNorm layer in block layer4.1
    target_module = None
    for name, module in model.named_modules():
        if "layer4.1.bn2" in name:
            target_module = module
            break
            
    if target_module is None:
        return 0.0
        
    activation_stds = []
    
    def diag_hook(module, input, output):
        x = input[0] # Shape: [B, C, H, W]
        # Compute standard deviation channel-wise, averaged over batch and space
        # Shape after mean: [C]
        std = torch.std(x, dim=(0, 2, 3))
        activation_stds.append(std.cpu())
        
    handle = target_module.register_forward_hook(diag_hook)
    
    # Run a single batch of 64 images
    for x, _ in test_loader:
        x = x.to(device)[:64]
        with torch.no_grad():
            model(x)
        break
        
    handle.remove()
    # Return average activation standard deviation across channels
    return torch.mean(activation_stds[0]).item()

# ----------------------------------------------------------------------
# Main Execution Pipeline
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated BatchNorm Calibration (Fed-BNC) for Model Merging")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per expert")
    parser.add_argument("--sample_limit", type=int, default=5000, help="Training set size subset limit")
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
    loaders = get_dataloaders(batch_size=args.batch_size, sample_limit=args.sample_limit)
    
    # 2. Train or Load Experts
    expert_paths = {
        'mnist': 'expert_mnist.pt',
        'fmnist': 'expert_fmnist.pt',
        'cifar': 'expert_cifar.pt'
    }
    
    expert_models = {}
    progenitor = get_resnet18_progenitor()
    
    print("\n--- Initializing & Fine-tuning Experts ---")
    for task in ['mnist', 'fmnist', 'cifar']:
        model_path = expert_paths[task]
        expert_models[task] = create_expert_model(progenitor)
        
        if os.path.exists(model_path) and not args.force_retrain:
            print(f"Loading pre-trained expert for {task} from {model_path}...")
            expert_models[task].load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Training expert for {task}...")
            expert_models[task] = train_expert(
                expert_models[task], 
                loaders['train'][task], 
                epochs=args.epochs, 
                lr=args.lr, 
                device=device
            )
            torch.save(expert_models[task].state_dict(), model_path)
            print(f"Saved expert for {task} to {model_path}")
            
    # Evaluate individual expert baselines
    print("\n--- Evaluating Expert Baselines ---")
    expert_accuracies = {}
    for task in ['mnist', 'fmnist', 'cifar']:
        acc = evaluate_model(expert_models[task], loaders['test'][task], task, expert_models[task], device=device)
        expert_accuracies[task] = acc
        print(f"Expert {task.upper()} Test Accuracy: {acc:.2f}%")
        
    expert_avg = sum(expert_accuracies.values()) / 3.0
    print(f"Average Expert Accuracy: {expert_avg:.2f}%")
    
    # 3. Model Merging Baselines (WA & TA)
    print("\n--- Merging Experts ---")
    merged_models = {
        'WA': merge_weights_averaging(expert_models),
        'TA': merge_task_arithmetic(progenitor, expert_models, lam=0.25)
    }
    
    results = {
        'meta': {
            'epochs': args.epochs,
            'sample_limit': args.sample_limit,
            'expert_accuracies': expert_accuracies,
            'expert_average': expert_avg
        },
        'baselines': {},
        'sample_efficiency': {},
        'non_iid_class_skew': {},
        'robustness': {},
        'activation_diagnostics': {}
    }
    
    # 4. Standard Evaluation of Baselines (Uncalibrated vs. Oracle vs. Fed-BNC)
    for merge_name, merged_model in merged_models.items():
        print(f"\nEvaluating Merging Paradigm: {merge_name}")
        results['baselines'][merge_name] = {}
        results['activation_diagnostics'][merge_name] = {}
        
        # Scenario A: Uncalibrated
        accs_uncal = {}
        stds_uncal = {}
        for task in ['mnist', 'fmnist', 'cifar']:
            accs_uncal[task] = evaluate_model(merged_model, loaders['test'][task], task, expert_models[task], device=device)
            stds_uncal[task] = profile_activation_std(merged_model, expert_models, loaders['test'][task], task, device=device)
        avg_uncal = sum(accs_uncal.values()) / 3.0
        results['baselines'][merge_name]['uncalibrated'] = {**accs_uncal, 'average': avg_uncal}
        results['activation_diagnostics'][merge_name]['uncalibrated'] = stds_uncal
        print(f"  Uncalibrated: MNIST={accs_uncal['mnist']:.2f}%, F-MNIST={accs_uncal['fmnist']:.2f}%, CIFAR={accs_uncal['cifar']:.2f}% | Avg={avg_uncal:.2f}%")
        print(f"  Uncalibrated activation std (layer4.1.bn2): MNIST={stds_uncal['mnist']:.4f}, F-MNIST={stds_uncal['fmnist']:.4f}, CIFAR={stds_uncal['cifar']:.4f}")
        
        # Scenario B: Oracle Calibration (Using 1024 real samples per task)
        oracle_model = create_expert_model(progenitor, num_classes=10)
        oracle_model.load_state_dict(merged_model.state_dict())
        run_federated_calibration(oracle_model, expert_models, loaders['train'], N=1024, device=device)
        accs_oracle = {}
        stds_oracle = {}
        for task in ['mnist', 'fmnist', 'cifar']:
            accs_oracle[task] = evaluate_model(oracle_model, loaders['test'][task], task, expert_models[task], device=device)
            stds_oracle[task] = profile_activation_std(oracle_model, expert_models, loaders['test'][task], task, device=device)
        avg_oracle = sum(accs_oracle.values()) / 3.0
        results['baselines'][merge_name]['oracle'] = {**accs_oracle, 'average': avg_oracle}
        results['activation_diagnostics'][merge_name]['oracle'] = stds_oracle
        print(f"  Oracle Calibrated (N=1024): MNIST={accs_oracle['mnist']:.2f}%, F-MNIST={accs_oracle['fmnist']:.2f}%, CIFAR={accs_oracle['cifar']:.2f}% | Avg={avg_oracle:.2f}%")
        print(f"  Oracle activation std (layer4.1.bn2): MNIST={stds_oracle['mnist']:.4f}, F-MNIST={stds_oracle['fmnist']:.4f}, CIFAR={stds_oracle['cifar']:.4f}")
        
        # Scenario C: Proposed Fed-BNC (N=128)
        fed_model = create_expert_model(progenitor, num_classes=10)
        fed_model.load_state_dict(merged_model.state_dict())
        run_federated_calibration(fed_model, expert_models, loaders['train'], N=128, device=device)
        accs_fed = {}
        stds_fed = {}
        for task in ['mnist', 'fmnist', 'cifar']:
            accs_fed[task] = evaluate_model(fed_model, loaders['test'][task], task, expert_models[task], device=device)
            stds_fed[task] = profile_activation_std(fed_model, expert_models, loaders['test'][task], task, device=device)
        avg_fed = sum(accs_fed.values()) / 3.0
        results['baselines'][merge_name]['fed_bnc'] = {**accs_fed, 'average': avg_fed}
        results['activation_diagnostics'][merge_name]['fed_bnc'] = stds_fed
        print(f"  Fed-BNC (N=128): MNIST={accs_fed['mnist']:.2f}%, F-MNIST={accs_fed['fmnist']:.2f}%, CIFAR={accs_fed['cifar']:.2f}% | Avg={avg_fed:.2f}%")
        
        # Scenario D: Proposed Class-Conditional CC-Fed-BNC (N=128)
        cc_fed_model = create_expert_model(progenitor, num_classes=10)
        cc_fed_model.load_state_dict(merged_model.state_dict())
        run_class_conditional_federated_calibration(cc_fed_model, expert_models, loaders['train'], N=128, device=device)
        accs_cc_fed = {}
        stds_cc_fed = {}
        for task in ['mnist', 'fmnist', 'cifar']:
            accs_cc_fed[task] = evaluate_model(cc_fed_model, loaders['test'][task], task, expert_models[task], device=device)
            stds_cc_fed[task] = profile_activation_std(cc_fed_model, expert_models, loaders['test'][task], task, device=device)
        avg_cc_fed = sum(accs_cc_fed.values()) / 3.0
        results['baselines'][merge_name]['cc_fed_bnc'] = {**accs_cc_fed, 'average': avg_cc_fed}
        results['activation_diagnostics'][merge_name]['cc_fed_bnc'] = stds_cc_fed
        print(f"  CC-Fed-BNC (N=128): MNIST={accs_cc_fed['mnist']:.2f}%, F-MNIST={accs_cc_fed['fmnist']:.2f}%, CIFAR={accs_cc_fed['cifar']:.2f}% | Avg={avg_cc_fed:.2f}%")

        # Scenario E: Proposed Adaptive Ada-Fed-BNC (N=128, alpha=10.0)
        global_running_stats = extract_running_stats(fed_model)
        ada_model = create_expert_model(progenitor, num_classes=10)
        ada_model.load_state_dict(merged_model.state_dict())
        accs_ada = {}
        stds_ada = {}
        for task in ['mnist', 'fmnist', 'cifar']:
            apply_adaptive_calibration(ada_model, expert_models, task, global_running_stats, alpha=10.0, device=device)
            accs_ada[task] = evaluate_model(ada_model, loaders['test'][task], task, expert_models[task], device=device)
            stds_ada[task] = profile_activation_std(ada_model, expert_models, loaders['test'][task], task, device=device)
        avg_ada = sum(accs_ada.values()) / 3.0
        results['baselines'][merge_name]['ada_fed_bnc'] = {**accs_ada, 'average': avg_ada}
        results['activation_diagnostics'][merge_name]['ada_fed_bnc'] = stds_ada
        print(f"  Ada-Fed-BNC (N=128, alpha=10.0): MNIST={accs_ada['mnist']:.2f}%, F-MNIST={accs_ada['fmnist']:.2f}%, CIFAR={accs_ada['cifar']:.2f}% | Avg={avg_ada:.2f}%")

    # 5. Scientific Inquiry 1: Sample Efficiency Study
    print("\n--- Scientific Inquiry 1: Sample Efficiency Study ---")
    n_values = [16, 32, 64, 128, 256, 512]
    results['sample_efficiency']['WA'] = []
    results['sample_efficiency']['TA'] = []
    
    for merge_name, merged_model in merged_models.items():
        print(f"Evaluating Sample Efficiency under {merge_name}:")
        for N in n_values:
            # Fed-BNC
            fed_model = create_expert_model(progenitor, num_classes=10)
            fed_model.load_state_dict(merged_model.state_dict())
            run_federated_calibration(fed_model, expert_models, loaders['train'], N=N, device=device)
            accs_fed = {}
            for task in ['mnist', 'fmnist', 'cifar']:
                accs_fed[task] = evaluate_model(fed_model, loaders['test'][task], task, expert_models[task], device=device)
            avg_fed = sum(accs_fed.values()) / 3.0
            
            # CC-Fed-BNC
            cc_fed_model = create_expert_model(progenitor, num_classes=10)
            cc_fed_model.load_state_dict(merged_model.state_dict())
            run_class_conditional_federated_calibration(cc_fed_model, expert_models, loaders['train'], N=N, device=device)
            accs_cc_fed = {}
            for task in ['mnist', 'fmnist', 'cifar']:
                accs_cc_fed[task] = evaluate_model(cc_fed_model, loaders['test'][task], task, expert_models[task], device=device)
            avg_cc_fed = sum(accs_cc_fed.values()) / 3.0
            
            results['sample_efficiency'][merge_name].append({
                'N': N,
                'fed_bnc': avg_fed,
                'cc_fed_bnc': avg_cc_fed
            })
            print(f"  N={N:3d} | Fed-BNC Avg Acc: {avg_fed:.2f}% | CC-Fed-BNC Avg Acc: {avg_cc_fed:.2f}%")

    # 6. Scientific Inquiry 2: Non-IID Class Skew Study
    # We restrict client calibration datasets to only contain classes 0-4 (50% class skew / extreme Non-IID)
    print("\n--- Scientific Inquiry 2: Non-IID Class Skew Study (Classes 0-4 available) ---")
    skew_filter = [0, 1, 2, 3, 4]
    results['non_iid_class_skew']['WA'] = {}
    results['non_iid_class_skew']['TA'] = {}
    
    for merge_name, merged_model in merged_models.items():
        print(f"Evaluating Class Skew under {merge_name}:")
        
        # Naive Fed-BNC under Skew (N=128)
        fed_model = create_expert_model(progenitor, num_classes=10)
        fed_model.load_state_dict(merged_model.state_dict())
        run_federated_calibration(fed_model, expert_models, loaders['train'], N=128, class_skew_filter=skew_filter, device=device)
        accs_fed = {}
        for task in ['mnist', 'fmnist', 'cifar']:
            accs_fed[task] = evaluate_model(fed_model, loaders['test'][task], task, expert_models[task], device=device)
        avg_fed = sum(accs_fed.values()) / 3.0
        
        # CC-Fed-BNC under Skew (N=128)
        cc_fed_model = create_expert_model(progenitor, num_classes=10)
        cc_fed_model.load_state_dict(merged_model.state_dict())
        run_class_conditional_federated_calibration(cc_fed_model, expert_models, loaders['train'], N=128, class_skew_filter=skew_filter, device=device)
        accs_cc_fed = {}
        for task in ['mnist', 'fmnist', 'cifar']:
            accs_cc_fed[task] = evaluate_model(cc_fed_model, loaders['test'][task], task, expert_models[task], device=device)
        avg_cc_fed = sum(accs_cc_fed.values()) / 3.0
        
        results['non_iid_class_skew'][merge_name] = {
            'naive_fed_bnc': {**accs_fed, 'average': avg_fed},
            'cc_fed_bnc': {**accs_cc_fed, 'average': avg_cc_fed}
        }
        print(f"  Naive Fed-BNC: MNIST={accs_fed['mnist']:.2f}%, F-MNIST={accs_fed['fmnist']:.2f}%, CIFAR={accs_fed['cifar']:.2f}% | Avg={avg_fed:.2f}%")
        print(f"  CC-Fed-BNC   : MNIST={accs_cc_fed['mnist']:.2f}%, F-MNIST={accs_cc_fed['fmnist']:.2f}%, CIFAR={accs_cc_fed['cifar']:.2f}% | Avg={avg_cc_fed:.2f}%")

    # 7. Scientific Inquiry 3: Real-World Covariate Shift/Robustness Study
    # Evaluate under test-time Gaussian Noise on CIFAR-10 (merge paradigm: TA)
    print("\n--- Scientific Inquiry 3: Robustness to Test-Time Covariate Shift (Gaussian Noise) ---")
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    results['robustness']['TA'] = []
    
    # We compare Uncalibrated, Oracle, and Fed-BNC
    uncal_model = merged_models['TA']
    oracle_model = create_expert_model(progenitor, num_classes=10)
    oracle_model.load_state_dict(uncal_model.state_dict())
    run_federated_calibration(oracle_model, expert_models, loaders['train'], N=1024, device=device)
    
    fed_model = create_expert_model(progenitor, num_classes=10)
    fed_model.load_state_dict(uncal_model.state_dict())
    run_federated_calibration(fed_model, expert_models, loaders['train'], N=128, device=device)
    
    for noise in noise_levels:
        acc_uncal = evaluate_model(uncal_model, loaders['test']['cifar'], 'cifar', expert_models['cifar'], device=device, corruption="gaussian_noise", corruption_val=noise)
        acc_oracle = evaluate_model(oracle_model, loaders['test']['cifar'], 'cifar', expert_models['cifar'], device=device, corruption="gaussian_noise", corruption_val=noise)
        acc_fed = evaluate_model(fed_model, loaders['test']['cifar'], 'cifar', expert_models['cifar'], device=device, corruption="gaussian_noise", corruption_val=noise)
        
        results['robustness']['TA'].append({
            'noise': noise,
            'uncalibrated': acc_uncal,
            'oracle': acc_oracle,
            'fed_bnc': acc_fed
        })
        print(f"  Noise Std={noise:.2f} | Uncalibrated: {acc_uncal:5.2f}% | Oracle: {acc_oracle:5.2f}% | Fed-BNC: {acc_fed:5.2f}%")

    # 8. Servability and Compiler Optimization Study
    # Profile inference latency of standard PyTorch vs compiled PyTorch
    print("\n--- Servability & Compiler Optimization Study ---")
    profile_inputs = torch.randn(64, 3, 32, 32).to(device)
    
    # Standard Model Merging with Fed-BNC (No hooks at test-time!)
    test_cal_model = create_expert_model(progenitor, num_classes=10)
    test_cal_model.load_state_dict(merged_models['TA'].state_dict())
    run_federated_calibration(test_cal_model, expert_models, loaders['train'], N=128, device=device)
    test_cal_model.to(device).eval()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            test_cal_model(profile_inputs)
            
    # Measure native latency
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            test_cal_model(profile_inputs)
    native_latency = (time.time() - start_time) / 100.0 * 1000.0 # ms
    print(f"  Native PyTorch Eager Mode Latency: {native_latency:.3f} ms per batch (size 64)")
    
    # Test torch.compile compatibility
    print("  Compiling model with torch.compile...")
    try:
        compiled_model = torch.compile(test_cal_model)
        # Warmup compiled model
        for _ in range(10):
            with torch.no_grad():
                compiled_model(profile_inputs)
                
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                compiled_model(profile_inputs)
        compiled_latency = (time.time() - start_time) / 100.0 * 1000.0 # ms
        print(f"  torch.compile Latency: {compiled_latency:.3f} ms per batch (size 64) | Speedup: {native_latency / compiled_latency:.2f}x")
        compiled_status = "Success"
    except Exception as e:
        print(f"  torch.compile failed: {e}")
        compiled_latency = native_latency
        compiled_status = f"Failed: {e}"
        
    results['servability'] = {
        'native_latency_ms': native_latency,
        'compiled_latency_ms': compiled_latency,
        'compiled_status': compiled_status
    }

    # 8.5 Alpha Sweep Study
    print("\n--- Hyperparameter Sweep over Blending Sensitivity alpha in Ada-Fed-BNC ---")
    alpha_vals = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    results['alpha_sweep'] = {'WA': [], 'TA': []}
    
    for merge_name, merged_model in merged_models.items():
        print(f"Sweeping Alpha under {merge_name}:")
        
        # Get standard Fed-BNC global stats first
        fed_model = create_expert_model(progenitor, num_classes=10)
        fed_model.load_state_dict(merged_model.state_dict())
        run_federated_calibration(fed_model, expert_models, loaders['train'], N=128, device=device)
        global_running_stats = extract_running_stats(fed_model)
        
        for alpha in alpha_vals:
            ada_model = create_expert_model(progenitor, num_classes=10)
            ada_model.load_state_dict(merged_model.state_dict())
            accs = {}
            for task in ['mnist', 'fmnist', 'cifar']:
                apply_adaptive_calibration(ada_model, expert_models, task, global_running_stats, alpha=alpha, device=device)
                accs[task] = evaluate_model(ada_model, loaders['test'][task], task, expert_models[task], device=device)
            avg_acc = sum(accs.values()) / 3.0
            results['alpha_sweep'][merge_name].append({
                'alpha': alpha,
                'mnist': accs['mnist'],
                'fmnist': accs['fmnist'],
                'cifar': accs['cifar'],
                'average': avg_acc
            })
            print(f"  Alpha={alpha:4.1f} | Average Acc: {avg_acc:.2f}% (MNIST={accs['mnist']:.2f}%, FMNIST={accs['fmnist']:.2f}%, CIFAR={accs['cifar']:.2f}%)")

    # Save results to JSON
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nAll experiments complete! Results saved to results.json")

    # 9. Create Visualizations
    print("\n--- Generating Plots ---")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot 4: Hyperparameter Sweep over Alpha
    plt.figure(figsize=(8, 5))
    alphas = [r['alpha'] for r in results['alpha_sweep']['WA']]
    wa_accs = [r['average'] for r in results['alpha_sweep']['WA']]
    ta_accs = [r['average'] for r in results['alpha_sweep']['TA']]
    
    plt.plot(alphas, wa_accs, marker='o', linewidth=2, label='Weight Averaging (WA)')
    plt.plot(alphas, ta_accs, marker='s', linewidth=2, label='Task Arithmetic (TA, $\lambda=0.25$)')
    plt.axhline(y=results['baselines']['WA']['uncalibrated']['average'], color='blue', linestyle=':', label='WA Uncalibrated Baseline')
    plt.axhline(y=results['baselines']['TA']['uncalibrated']['average'], color='green', linestyle=':', label='TA Uncalibrated Baseline')
    plt.xlabel('Adaptive Blending Sensitivity ($\\alpha$)', fontsize=12)
    plt.ylabel('Multi-Task Average Accuracy (%)', fontsize=12)
    plt.title('Ada-Fed-BNC: Sensitivity Blending Sweep ($\\alpha$)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.xscale('symlog', linthresh=0.1)
    plt.tight_layout()
    plt.savefig('plot_alpha_sweep.png', dpi=300)
    plt.close()
    
    # Plot 1: Sample Efficiency
    plt.figure(figsize=(8, 5))
    n_vals = [r['N'] for r in results['sample_efficiency']['TA']]
    fed_accs = [r['fed_bnc'] for r in results['sample_efficiency']['TA']]
    cc_fed_accs = [r['cc_fed_bnc'] for r in results['sample_efficiency']['TA']]
    
    plt.plot(n_vals, fed_accs, marker='o', linewidth=2, label='Fed-BNC (Naive)')
    plt.plot(n_vals, cc_fed_accs, marker='s', linewidth=2, label='CC-Fed-BNC (Class-Conditional)')
    plt.axhline(y=results['baselines']['TA']['oracle']['average'], color='r', linestyle='--', label='Oracle Calibration (N=1024)')
    plt.axhline(y=results['baselines']['TA']['uncalibrated']['average'], color='gray', linestyle=':', label='Uncalibrated Baseline')
    plt.xlabel('Calibration Samples per Client ($N$)', fontsize=12)
    plt.ylabel('Multi-Task Average Accuracy (%)', fontsize=12)
    plt.title('Calibration Sample Efficiency (Task Arithmetic, $\lambda=0.25$)', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.xticks(n_vals, [str(n) for n in n_vals])
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('plot_sample_efficiency.png', dpi=300)
    plt.close()
    
    # Plot 2: Non-IID Class Skew Robustness
    plt.figure(figsize=(8, 5))
    categories = ['Uncalibrated', 'Naive Fed-BNC (IID)', 'Naive Fed-BNC (Non-IID Skew)', 'CC-Fed-BNC (Non-IID Skew)', 'Oracle']
    # WA accuracies
    wa_uncal = results['baselines']['WA']['uncalibrated']['average']
    wa_iid_naive = results['baselines']['WA']['fed_bnc']['average']
    wa_skew_naive = results['non_iid_class_skew']['WA']['naive_fed_bnc']['average']
    wa_skew_cc = results['non_iid_class_skew']['WA']['cc_fed_bnc']['average']
    wa_oracle = results['baselines']['WA']['oracle']['average']
    
    ta_uncal = results['baselines']['TA']['uncalibrated']['average']
    ta_iid_naive = results['baselines']['TA']['fed_bnc']['average']
    ta_skew_naive = results['non_iid_class_skew']['TA']['naive_fed_bnc']['average']
    ta_skew_cc = results['non_iid_class_skew']['TA']['cc_fed_bnc']['average']
    ta_oracle = results['baselines']['TA']['oracle']['average']
    
    x_indices = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x_indices - width/2, [wa_uncal, wa_iid_naive, wa_skew_naive, wa_skew_cc, wa_oracle], width, label='Weight Averaging')
    plt.bar(x_indices + width/2, [ta_uncal, ta_iid_naive, ta_skew_naive, ta_skew_cc, ta_oracle], width, label='Task Arithmetic ($\lambda=0.25$)')
    plt.ylabel('Multi-Task Average Accuracy (%)', fontsize=12)
    plt.title('Robustness to Non-IID Class Skew (N=128 Calibration Samples)', fontsize=14, fontweight='bold')
    plt.xticks(x_indices, categories, rotation=15, fontsize=10)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('plot_non_iid_robustness.png', dpi=300)
    plt.close()
    
    # Plot 3: Covariate Shift Robustness
    plt.figure(figsize=(8, 5))
    noises = [r['noise'] for r in results['robustness']['TA']]
    uncal_rob = [r['uncalibrated'] for r in results['robustness']['TA']]
    oracle_rob = [r['oracle'] for r in results['robustness']['TA']]
    fed_rob = [r['fed_bnc'] for r in results['robustness']['TA']]
    
    plt.plot(noises, uncal_rob, marker='o', linewidth=2, linestyle=':', label='Uncalibrated')
    plt.plot(noises, oracle_rob, marker='s', linewidth=2, linestyle='--', label='Oracle (N=1024)')
    plt.plot(noises, fed_rob, marker='^', linewidth=2, label='Fed-BNC (N=128)')
    plt.xlabel('Test-Time Gaussian Noise Std ($\sigma$)', fontsize=12)
    plt.ylabel('CIFAR-10 Test Accuracy (%)', fontsize=12)
    plt.title('Robustness to Test-Time Covariate Shift (Task Arithmetic)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('plot_covariate_robustness.png', dpi=300)
    plt.close()
    
    print("Plots generated successfully!")
