import os
import argparse
import json
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Disable cuDNN to bypass driver compatibility issues on the GPU cluster
    torch.backends.cudnn.enabled = False

# Define dataset loaders with proper 3-channel resizing and normalization
def get_datasets(data_dir="./data", batch_size=256, dry_run=False):
    # Grayscale to 3-channel conversion and resize to 32x32
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3081, 0.3081, 0.3081))
    ])

    transform_fmnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860, 0.2860, 0.2860), std=(0.3530, 0.3530, 0.3530))
    ])

    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    # Download and load datasets
    print("Loading MNIST...")
    mnist_train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_gray)
    mnist_test = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_gray)

    print("Loading Fashion-MNIST...")
    fmnist_train = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform_fmnist)
    fmnist_test = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform_fmnist)

    print("Loading CIFAR-10...")
    cifar_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_color)
    cifar_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_color)

    if dry_run:
        # Scale down dataset size for quick execution
        mnist_train = Subset(mnist_train, list(range(512)))
        mnist_test = Subset(mnist_test, list(range(256)))
        fmnist_train = Subset(fmnist_train, list(range(512)))
        fmnist_test = Subset(fmnist_test, list(range(256)))
        cifar_train = Subset(cifar_train, list(range(512)))
        cifar_test = Subset(cifar_test, list(range(256)))

    train_loaders = {
        "MNIST": DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2),
        "FMNIST": DataLoader(fmnist_train, batch_size=batch_size, shuffle=True, num_workers=2),
        "CIFAR10": DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=2)
    }

    test_loaders = {
        "MNIST": DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2),
        "FMNIST": DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, num_workers=2),
        "CIFAR10": DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=2)
    }

    return train_loaders, test_loaders

# Multi-task model architecture wrapper
class MultiTaskResNet(nn.Module):
    def __init__(self, heads_dict):
        super(MultiTaskResNet, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity() # Feature extractor
        self.heads = nn.ModuleDict(heads_dict)

    def forward(self, x, task_name=None):
        features = self.backbone(x)
        if task_name is not None:
            return self.heads[task_name](features)
        return features

# Train a task expert
def train_expert(model, task_name, train_loader, test_loader, epochs=5, lr=1e-4, wd=1e-4, device="cpu"):
    print(f"\n--- Training Expert for Task: {task_name} ---")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, task_name)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, task_name)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    test_acc = 100.0 * test_correct / test_total
    print(f"Finished training expert {task_name}. Test Accuracy: {test_acc:.2f}%")
    return test_acc

# Merge backbones in weight space (Weight Averaging or Task Arithmetic)
def merge_models(progenitor, experts_dict, lam=0.4):
    print(f"\nMerging expert backbones using Task Arithmetic (lambda={lam})...")
    merged_model = copy.deepcopy(progenitor)
    merged_state = merged_model.state_dict()
    prog_state = progenitor.state_dict()

    # Compute task vectors and linear combination
    task_vectors = {}
    for key, expert in experts_dict.items():
        expert_state = expert.state_dict()
        task_vectors[key] = {name: (expert_state[name].cpu() - prog_state[name].cpu()) for name in prog_state}

    # Sum task vectors with scaling lambda
    for name in prog_state:
        # Only merge backbone weights (exclude heads)
        if "backbone." in name:
            param_type = prog_state[name].dtype
            if param_type.is_floating_point:
                update = sum(task_vectors[key][name] for key in experts_dict)
                merged_state[name] = prog_state[name] + lam * update

    merged_model.load_state_dict(merged_state)
    return merged_model

def ties_merge_models(progenitor, experts_dict, lam=0.4, keep_fraction=0.2):
    print(f"\nMerging expert backbones using TIES-Merging (lambda={lam}, keep_fraction={keep_fraction})...")
    merged_model = copy.deepcopy(progenitor)
    merged_state = merged_model.state_dict()
    prog_state = progenitor.state_dict()

    # Compute task vectors
    task_vectors = {}
    for key, expert in experts_dict.items():
        expert_state = expert.state_dict()
        task_vectors[key] = {name: (expert_state[name].cpu() - prog_state[name].cpu()) for name in prog_state}

    # Sum task vectors with TIES-Merging resolution
    for name in prog_state:
        # Only merge backbone weights (exclude heads)
        if "backbone." in name:
            param_type = prog_state[name].dtype
            if param_type.is_floating_point:
                tvs = [task_vectors[key][name] for key in experts_dict]
                stacked = torch.stack(tvs, dim=0) # (K, *shape)
                K = stacked.size(0)
                
                # 1. Prune: Keep top keep_fraction values by magnitude
                flat = stacked.view(K, -1)
                num_el = flat.size(1)
                k_val = max(1, int(num_el * keep_fraction))
                
                pruned_list = []
                for t in range(K):
                    vec = flat[t]
                    mags = torch.abs(vec)
                    threshold = torch.topk(mags, k_val).values[-1]
                    mask = mags >= threshold
                    pruned_vec = vec * mask
                    pruned_list.append(pruned_vec.view(stacked.shape[1:]))
                
                stacked_pruned = torch.stack(pruned_list, dim=0)
                
                # 2. Sign Agreement
                signs = torch.sign(stacked_pruned)
                sum_signs = torch.sum(signs, dim=0)
                consensus_sign = torch.sign(sum_signs)
                
                # 3. Disagree Elimination
                matching_mask = (signs * consensus_sign) > 0
                cleared = stacked_pruned * matching_mask
                
                # 4. Average
                update = torch.sum(cleared, dim=0) / K
                merged_state[name] = prog_state[name] + lam * update

    merged_model.load_state_dict(merged_state)
    return merged_model

def dare_merge_models(progenitor, experts_dict, lam=0.4, p=0.2):
    print(f"\nMerging expert backbones using DARE-Merging (lambda={lam}, drop_rate={p})...")
    merged_model = copy.deepcopy(progenitor)
    merged_state = merged_model.state_dict()
    prog_state = progenitor.state_dict()

    # Compute task vectors
    task_vectors = {}
    for key, expert in experts_dict.items():
        expert_state = expert.state_dict()
        task_vectors[key] = {name: (expert_state[name].cpu() - prog_state[name].cpu()) for name in prog_state}

    # Sum task vectors with DARE resolution
    for name in prog_state:
        # Only merge backbone weights (exclude heads)
        if "backbone." in name:
            param_type = prog_state[name].dtype
            if param_type.is_floating_point:
                tvs = [task_vectors[key][name] for key in experts_dict]
                stacked = torch.stack(tvs, dim=0) # (K, *shape)
                K = stacked.size(0)

                # DARE-Merging:
                # Randomly drop parameters with probability p and rescale remaining by 1/(1-p)
                mask = (torch.rand_like(stacked) >= p).float()
                rescaled = (stacked * mask) / (1.0 - p) if p < 1.0 else stacked * 0.0

                # Average rescaled task vectors
                update = torch.sum(rescaled, dim=0) / K
                merged_state[name] = prog_state[name] + lam * update

    merged_model.load_state_dict(merged_state)
    return merged_model

# Post-Training Quantization (PTQ) implementation
def quantize_weights(model, bits=8, per_channel=True):
    if bits is None:
        return model
    
    quantized_model = copy.deepcopy(model)
    qmax = 2**(bits - 1) - 1
    
    with torch.no_grad():
        for name, param in quantized_model.named_parameters():
            if "weight" in name and "backbone." in name:
                W = param.data
                if per_channel:
                    # Compute per-channel scale factor (axis 0 is output channel)
                    num_channels = W.size(0)
                    W_flat = W.view(num_channels, -1)
                    max_val, _ = torch.max(torch.abs(W_flat), dim=1, keepdim=True)
                    # Avoid division by zero
                    max_val = torch.clamp(max_val, min=1e-8)
                    scale = max_val / qmax
                    
                    # Reshape scale to match W dimensions
                    scale_expanded = scale.view(num_channels, *([1] * (W.dim() - 1)))
                    W_quant = torch.clamp(torch.round(W / scale_expanded), -qmax, qmax) * scale_expanded
                    param.copy_(W_quant)
                else:
                    max_val = torch.max(torch.abs(W))
                    scale = max_val / qmax
                    W_quant = torch.clamp(torch.round(W / scale), -qmax, qmax) * scale
                    param.copy_(W_quant)
                    
    return quantized_model

# Calibration: Task-Specific DE-BN (Oracle/Routing)
def run_task_specific_de_bn(model, task_name, loader, N=64, device="cpu"):
    # Set to train mode to update batchnorm statistics
    model.to(device)
    model.train()
    
    # Save original running statistics to restore later
    orig_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            orig_stats[name] = {
                "running_mean": module.running_mean.clone() if module.running_mean is not None else None,
                "running_var": module.running_var.clone() if module.running_var is not None else None,
                "momentum": module.momentum
            }
            # Temporarily configure BatchNorm layer for direct calibration pass
            module.momentum = 1.0
            if module.running_mean is not None:
                module.running_mean.zero_()
            if module.running_var is not None:
                module.running_var.fill_(1.0)

    # Collect exactly N calibration samples
    cal_inputs = []
    collected = 0
    for inputs, _ in loader:
        cal_inputs.append(inputs)
        collected += inputs.size(0)
        if collected >= N:
            break
    
    if len(cal_inputs) > 0:
        cal_batch = torch.cat(cal_inputs, dim=0)[:N].to(device)
        # Single forward pass to update BN running statistics
        with torch.no_grad():
            _ = model(cal_batch, task_name)
            
    # Return both the model and the captured stats for this task
    captured_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            captured_stats[name] = {
                "running_mean": module.running_mean.clone() if module.running_mean is not None else None,
                "running_var": module.running_var.clone() if module.running_var is not None else None
            }
            # Restore original statistics for general usage
            if orig_stats[name]["running_mean"] is not None:
                module.running_mean.copy_(orig_stats[name]["running_mean"])
            if orig_stats[name]["running_var"] is not None:
                module.running_var.copy_(orig_stats[name]["running_var"])
            module.momentum = orig_stats[name]["momentum"]
            
    return captured_stats

# Calibration: Naive Mixed Calibration
def run_naive_mixed_calibration(model, loader_dict, N=64, device="cpu"):
    # Build a balanced mixed-task batch
    num_tasks = len(loader_dict)
    samples_per_task = N // num_tasks
    
    mixed_inputs = []
    for task_name, loader in loader_dict.items():
        task_inputs = []
        collected = 0
        for inputs, _ in loader:
            task_inputs.append(inputs)
            collected += inputs.size(0)
            if collected >= samples_per_task:
                break
        if len(task_inputs) > 0:
            mixed_inputs.append(torch.cat(task_inputs, dim=0)[:samples_per_task])
            
    cal_batch = torch.cat(mixed_inputs, dim=0).to(device)
    
    cal_model = copy.deepcopy(model)
    cal_model.to(device)
    cal_model.train()
    
    # Configure BatchNorm layers
    for name, module in cal_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 1.0
            if module.running_mean is not None:
                module.running_mean.zero_()
            if module.running_var is not None:
                module.running_var.fill_(1.0)
                
    with torch.no_grad():
        # Pass through the model (use any task head for feature extraction, features are head-independent)
        _ = cal_model(cal_batch, list(loader_dict.keys())[0])
        
    cal_model.eval()
    return cal_model

# Calibration: Proposed Centroid-Aligned Unified Calibration (CA-UC)
def run_centroid_aligned_unified_calibration(model, loader_dict, N=64, device="cpu"):
    num_tasks = len(loader_dict)
    samples_per_task = N // num_tasks
    
    # Step 1: Capture task-specific statistics on individual small batches
    task_stats = {task_name: {} for task_name in loader_dict}
    
    for task_name, loader in loader_dict.items():
        # Run standard DE-BN on the task-specific batch
        captured = run_task_specific_de_bn(model, task_name, loader, N=samples_per_task, device=device)
        task_stats[task_name] = captured
        
    # Step 2: Average the captured task-specific moments to form a unified stat set
    cal_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, module in cal_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Aggregate means and variances across tasks
                task_means = [task_stats[task][name]["running_mean"] for task in loader_dict]
                task_vars = [task_stats[task][name]["running_var"] for task in loader_dict]
                
                # Assign unified centroid-aligned statistics
                module.running_mean.copy_(torch.stack(task_means, dim=0).mean(dim=0))
                module.running_var.copy_(torch.stack(task_vars, dim=0).mean(dim=0))
                
    cal_model.eval()
    return cal_model

# Evaluation under Noise/Quantization
def evaluate_model(model, loaders_dict, calibration_type="None", calibration_data=None, noise_std=0.0, device="cpu"):
    model.eval()
    model.to(device)
    
    results = {}
    total_correct = 0
    total_samples = 0
    
    for task_name, loader in loaders_dict.items():
        # Handle Task-Specific DE-BN (where statistics are swapped task-specifically at test time)
        if calibration_type == "Task-Specific DE-BN (Oracle)":
            # Load captured task-specific stats into model
            task_stats = calibration_data[task_name]
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.running_mean.copy_(task_stats[name]["running_mean"])
                    module.running_var.copy_(task_stats[name]["running_var"])
                    
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                # Add additive Gaussian Noise if specified
                if noise_std > 0.0:
                    inputs = inputs + torch.randn_like(inputs) * noise_std
                    
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, task_name)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        acc = 100.0 * correct / total
        results[task_name] = acc
        total_correct += correct
        total_samples += total
        
    results["Average"] = 100.0 * total_correct / total_samples
    return results

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Model Merging Research")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per task expert")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--dry-run", action="store_true", help="Quick sanity check on CPU with tiny data subsets")
    parser.add_argument("--lam", type=float, default=0.4, help="Model merging interpolation lambda")
    parser.add_argument("--cal-samples", type=int, default=64, help="Total calibration samples N")
    parser.add_argument("--noise-std", type=float, default=0.1, help="Standard deviation of additive Gaussian noise")
    parser.add_argument("--merge-method", type=str, default="task_arithmetic", choices=["task_arithmetic", "ties_merging"], help="Backbone merging method")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | Dry-run: {args.dry_run}")
    set_seed(42)

    # 1. Load Data
    train_loaders, test_loaders = get_datasets(batch_size=args.batch_size, dry_run=args.dry_run)

    # 2. Setup Multi-Task heads
    heads = {
        "MNIST": nn.Linear(512, 10),
        "FMNIST": nn.Linear(512, 10),
        "CIFAR10": nn.Linear(512, 10)
    }
    
    # 3. Initialize pre-trained progenitor and train experts
    progenitor = MultiTaskResNet(heads)
    
    # Save the progenitor initial state
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not args.dry_run:
        torch.save(progenitor.state_dict(), f"{checkpoint_dir}/progenitor.pt")

    # Experts dict
    experts = {}
    expert_accuracies = {}
    
    epochs = 1 if args.dry_run else args.epochs
    for task_name in train_loaders.keys():
        expert_model = copy.deepcopy(progenitor)
        ckpt_path = f"{checkpoint_dir}/{task_name}_expert.pt"
        if os.path.exists(ckpt_path) and not args.dry_run:
            print(f"Loading pre-trained expert for {task_name} from {ckpt_path}...")
            expert_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            expert_model.to(device)
            expert_model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for inputs, targets in test_loaders[task_name]:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = expert_model(inputs, task_name)
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
            test_acc = 100.0 * test_correct / test_total
            print(f"Loaded expert {task_name}. Test Accuracy: {test_acc:.2f}%")
        else:
            test_acc = train_expert(
                expert_model, 
                task_name, 
                train_loaders[task_name], 
                test_loaders[task_name], 
                epochs=epochs, 
                device=device
            )
            if not args.dry_run:
                torch.save(expert_model.state_dict(), ckpt_path)
        experts[task_name] = expert_model
        expert_accuracies[task_name] = test_acc

    # Save expert metrics
    oracle_avg = sum(expert_accuracies.values()) / len(expert_accuracies)
    print(f"\nIndividual Expert Accuracies:")
    for task, acc in expert_accuracies.items():
        print(f"  {task}: {acc:.2f}%")
    print(f"Oracle Upper Bound (Average): {oracle_avg:.2f}%")

    # 4. Merge experts to create multi-task model
    if args.merge_method == "ties_merging":
        merged_model = ties_merge_models(progenitor, experts, lam=args.lam, keep_fraction=0.2)
    else:
        merged_model = merge_models(progenitor, experts, lam=args.lam)

    # 5. Run Calibration Algorithms
    print("\n--- Running Calibration Algorithms ---")
    
    # A. Task-Specific DE-BN (Oracle/Routing stats)
    task_specific_stats = {}
    for task_name in train_loaders.keys():
        captured = run_task_specific_de_bn(merged_model, task_name, train_loaders[task_name], N=args.cal_samples, device=device)
        task_specific_stats[task_name] = captured
        
    # B. Naive Mixed Calibration (Mixed task batch)
    naive_cal_model = run_naive_mixed_calibration(merged_model, train_loaders, N=args.cal_samples, device=device)
    
    # C. Proposed Centroid-Aligned Unified Calibration (CA-UC)
    ca_uc_model = run_centroid_aligned_unified_calibration(merged_model, train_loaders, N=args.cal_samples, device=device)

    # 6. Evaluation on diverse regimes
    print("\n--- Starting Comprehensive Evaluation ---")
    
    regimes = [
        {"name": "FP32", "bits": None, "per_channel": True, "noise": 0.0},
        {"name": "PC-INT8", "bits": 8, "per_channel": True, "noise": 0.0},
        {"name": "PT-INT8", "bits": 8, "per_channel": False, "noise": 0.0},
        {"name": "PC-INT4", "bits": 4, "per_channel": True, "noise": 0.0},
        {"name": "PT-INT4", "bits": 4, "per_channel": False, "noise": 0.0},
        {"name": "Noisy FP32", "bits": None, "per_channel": True, "noise": args.noise_std}
    ]
    
    evaluation_results = []
    
    for regime in regimes:
        print(f"\nEvaluating Regime: {regime['name']}")
        
        # Apply quantization to weights if specified
        base_merged = quantize_weights(merged_model, bits=regime["bits"], per_channel=regime["per_channel"])
        naive_model = quantize_weights(naive_cal_model, bits=regime["bits"], per_channel=regime["per_channel"])
        ca_model = quantize_weights(ca_uc_model, bits=regime["bits"], per_channel=regime["per_channel"])
        
        # Evaluate different Calibration setups
        # Setup 1: No Calibration (standard uncalibrated WA)
        res_none = evaluate_model(base_merged, test_loaders, calibration_type="None", noise_std=regime["noise"], device=device)
        
        # Setup 2: Task-Specific DE-BN (Oracle, requires routing)
        res_debn = evaluate_model(base_merged, test_loaders, calibration_type="Task-Specific DE-BN (Oracle)", calibration_data=task_specific_stats, noise_std=regime["noise"], device=device)
        
        # Setup 3: Naive Mixed Calibration
        res_naive = evaluate_model(naive_model, test_loaders, calibration_type="None", noise_std=regime["noise"], device=device)
        
        # Setup 4: Proposed Centroid-Aligned Unified Calibration (CA-UC)
        res_cauc = evaluate_model(ca_model, test_loaders, calibration_type="None", noise_std=regime["noise"], device=device)
        
        evaluation_results.append({
            "regime": regime["name"],
            "methods": {
                "Uncalibrated": res_none,
                "DE-BN (Oracle, routed)": res_debn,
                "Naive Mixed Cal": res_naive,
                "Proposed CA-UC (Task-Agnostic)": res_cauc
            }
        })

    # Print out results as markdown-style table
    print("\n================== EVALUATION RESULTS SUMMARY ==================")
    for res in evaluation_results:
        print(f"\nRegime: {res['regime']}")
        print(f"| Method | MNIST | FMNIST | CIFAR-10 | Average |")
        print(f"|---|---|---|---|---|")
        for method_name, task_res in res["methods"].items():
            print(f"| {method_name} | {task_res['MNIST']:.2f}% | {task_res['FMNIST']:.2f}% | {task_res['CIFAR10']:.2f}% | {task_res['Average']:.2f}% |")

    # Save results to JSON file
    output_data = {
        "expert_accuracies": expert_accuracies,
        "oracle_average": oracle_avg,
        "evaluation_results": evaluation_results
    }
    with open("results.json", "w") as f:
        json.dump(output_data, f, indent=4)

    # 7. Generate beautiful visualization plot
    methods_plot = ["Uncalibrated", "Naive Mixed Cal", "Proposed CA-UC (Task-Agnostic)", "DE-BN (Oracle, routed)"]
    regimes_plot = ["FP32", "PC-INT8", "PT-INT8", "PC-INT4", "PT-INT4", "Noisy FP32"]
    
    plot_data = {m: [] for m in methods_plot}
    for res in evaluation_results:
        for m in methods_plot:
            plot_data[m].append(res["methods"][m]["Average"])
            
    x = np.arange(len(regimes_plot))
    width = 0.2
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5*width, plot_data["Uncalibrated"], width, label="Uncalibrated (WA)", color="#7f8c8d")
    plt.bar(x - 0.5*width, plot_data["Naive Mixed Cal"], width, label="Naive Mixed Cal (Static)", color="#e74c3c")
    plt.bar(x + 0.5*width, plot_data["Proposed CA-UC (Task-Agnostic)"], width, label="Proposed CA-UC (Static, Task-Agnostic)", color="#2ecc71")
    plt.bar(x + 1.5*width, plot_data["DE-BN (Oracle, routed)"], width, label="DE-BN (Dynamic, Oracle routed)", color="#3498db")
    
    plt.ylabel("Multi-task Average Accuracy (%)")
    plt.title("Performance Comparison across Quantization and Noise Regimes")
    plt.xticks(x, regimes_plot)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("performance_comparison.png", dpi=300)
    print("\nSuccessfully saved performance comparison plot to performance_comparison.png!")

if __name__ == "__main__":
    main()
