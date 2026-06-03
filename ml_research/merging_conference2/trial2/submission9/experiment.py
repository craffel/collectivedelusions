import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False

class MultiTaskResNet18(nn.Module):
    def __init__(self, task_names):
        super().__init__()
        # Use ImageNet pre-trained ResNet-18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity() # Remove the default FC layer
        # Replace with task-specific heads (each mapping 512 features to 10 classes)
        self.heads = nn.ModuleDict({
            task: nn.Linear(512, 10) for task in task_names
        })
        
    def forward(self, x, task_name):
        features = self.backbone(x)
        return self.heads[task_name](features)

def get_dataloaders(batch_size=128):
    # Standard normalization for ImageNet pre-trained models
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 3-channel transform for grayscale datasets
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize
    ])
    
    # Transform for CIFAR-10
    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Load raw datasets
    train_datasets = {
        "mnist": torchvision.datasets.MNIST("./data", train=True, download=False, transform=transform_gray),
        "fashion": torchvision.datasets.FashionMNIST("./data", train=True, download=False, transform=transform_gray),
        "cifar10": torchvision.datasets.CIFAR10("./data", train=True, download=False, transform=transform_color)
    }
    
    test_datasets = {
        "mnist": torchvision.datasets.MNIST("./data", train=False, download=False, transform=transform_gray),
        "fashion": torchvision.datasets.FashionMNIST("./data", train=False, download=False, transform=transform_gray),
        "cifar10": torchvision.datasets.CIFAR10("./data", train=False, download=False, transform=transform_color)
    }
    
    # Subsampling for speed and reproducibility
    # 5,000 train images per task
    # 128 calibration images per task (subset from train)
    # 1,000 test images per task
    loaders = {}
    cal_batches = {}
    test_loaders = {}
    
    for task in ["mnist", "fashion", "cifar10"]:
        # Subsample train
        train_len = len(train_datasets[task])
        train_indices = list(range(train_len))
        random.shuffle(train_indices)
        
        train_sub_indices = train_indices[:5000]
        cal_indices = train_indices[5000:5128] # Separate 128 images for calibration
        
        train_subset = Subset(train_datasets[task], train_sub_indices)
        cal_subset = Subset(train_datasets[task], cal_indices)
        
        loaders[task] = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Calibration batch (loaded all at once)
        cal_loader = DataLoader(cal_subset, batch_size=128, shuffle=False)
        cal_batches[task] = next(iter(cal_loader))[0] # Get only images (x)
        
        # Subsample test
        test_len = len(test_datasets[task])
        test_indices = list(range(test_len))
        random.shuffle(test_indices)
        test_sub_indices = test_indices[:1000]
        test_subset = Subset(test_datasets[task], test_sub_indices)
        
        test_loaders[task] = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        
    return loaders, cal_batches, test_loaders

def train_expert(model, train_loader, test_loader, task_name, epochs=5, lr=5e-4, weight_decay=1e-4, device="cuda"):
    model.to(device)
    # Only optimize backbone and the specific head for this task
    optimizer = optim.AdamW(
        list(model.backbone.parameters()) + list(model.heads[task_name].parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    # Cosine learning rate decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, task_name)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate_model(model, test_loader, task_name, device)
        print(f"Task {task_name.upper()} | Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
    return model

def evaluate_model(model, test_loader, task_name, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, task_name)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100.0

# Collect activation statistics
def collect_stats(model, batch_x, task_name, device="cuda"):
    model.eval()
    handles = []
    activations = {}
    
    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = output.detach().clone()
        return hook_fn
        
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)
            
    with torch.no_grad():
        _ = model(batch_x.to(device), task_name)
        
    stats = {}
    for name, act in activations.items():
        # Channel-wise mean and std (shape [1, C, 1, 1])
        mu = act.mean(dim=(0, 2, 3), keepdim=True)
        var = act.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        sigma = torch.sqrt(var + 1e-5)
        
        # Layer-wise scalar mean and std (scalar)
        mu_scalar = act.mean()
        var_scalar = act.var(unbiased=False)
        sigma_scalar = torch.sqrt(var_scalar + 1e-5)
        
        stats[name] = {
            'mu': mu,
            'sigma': sigma,
            'sigma_scalar': sigma_scalar
        }
        
    for handle in handles:
        handle.remove()
        
    return stats

# Register activation correction hooks during inference
def register_inference_hooks(model, task_calibration, mode):
    handles = []
    
    def make_inference_hook(name, stats):
        def hook_fn(module, input, output):
            if mode == 'none':
                return output
                
            mu_orig = stats['mu_orig'].to(output.device)
            sigma_orig = stats['sigma_orig'].to(output.device)
            mu_merged = stats['mu_merged'].to(output.device)
            sigma_merged = stats['sigma_merged'].to(output.device)
            
            # Clamp standard deviations to prevent division-by-zero or massive scaling of noise
            sigma_orig_stable = torch.clamp(sigma_orig, min=0.05)
            sigma_merged_stable = torch.clamp(sigma_merged, min=0.05)
            
            if mode == 'tcac':
                # Channel-wise Affine (Mean shift + Std scale) with stable clamping
                calibrated = ((output - mu_merged) / sigma_merged_stable) * sigma_orig_stable + mu_orig
                return calibrated
                
            elif mode == 'sac':
                # Channel-wise Scaling-only (Std scale, no mean shift, preserves ReLU sign)
                calibrated = output * (sigma_orig_stable / sigma_merged_stable)
                return calibrated
                
            elif mode == 'lsc':
                # Layer-wise Scaling-only (Scalar std scale, no mean shift, preserves ReLU sign)
                sigma_orig_scalar = stats['sigma_orig_scalar'].to(output.device)
                sigma_merged_scalar = stats['sigma_merged_scalar'].to(output.device)
                sigma_orig_scalar_stable = torch.clamp(sigma_orig_scalar, min=0.05)
                sigma_merged_scalar_stable = torch.clamp(sigma_merged_scalar, min=0.05)
                calibrated = output * (sigma_orig_scalar_stable / sigma_merged_scalar_stable)
                return calibrated
                
            return output
        return hook_fn
        
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d) and name in task_calibration:
            stats = task_calibration[name]
            handle = module.register_forward_hook(make_inference_hook(name, stats))
            handles.append(handle)
            
    return handles

def merge_models(base_model, experts, mode="weight_average", lam=0.4):
    """
    base_model: MultiTaskResNet18 (pre-trained model)
    experts: dict of MultiTaskResNet18 expert models
    mode: "weight_average" or "task_arithmetic"
    lam: scaling coefficient for task arithmetic
    """
    merged_model = MultiTaskResNet18(list(experts.keys()))
    
    # Copy classification heads directly (heads are task-specific, so they don't conflict)
    for task_name, expert in experts.items():
        merged_model.heads[task_name].load_state_dict(expert.heads[task_name].state_dict())
        
    # Merge backbone weights
    merged_state_dict = merged_model.backbone.state_dict()
    base_state_dict = base_model.backbone.state_dict()
    expert_state_dicts = {task: expert.backbone.state_dict() for task, expert in experts.items()}
    
    for key in merged_state_dict.keys():
        # Skip non-tensor weights if any
        if not torch.is_tensor(merged_state_dict[key]):
            continue
            
        # Only merge floating point tensors. For integer/long buffers (like num_batches_tracked),
        # copy directly from the first expert.
        if not torch.is_floating_point(merged_state_dict[key]):
            first_task = list(experts.keys())[0]
            merged_state_dict[key] = expert_state_dicts[first_task][key].clone()
            continue
            
        if mode == "weight_average":
            # Average of all expert backbones
            weights = [expert_state_dicts[task][key] for task in experts.keys()]
            merged_state_dict[key] = torch.stack(weights).mean(dim=0)
            
        elif mode == "task_arithmetic":
            # W_merged = W_base + lam * sum(W_expert - W_base)
            task_vectors = []
            for task in experts.keys():
                task_vector = expert_state_dicts[task][key] - base_state_dict[key]
                task_vectors.append(task_vector)
            merged_state_dict[key] = base_state_dict[key] + lam * torch.stack(task_vectors).sum(dim=0)
            
    merged_model.backbone.load_state_dict(merged_state_dict)
    return merged_model

def run_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running experiments on: {device}")
    
    set_seed(42)
    tasks = ["mnist", "fashion", "cifar10"]
    
    # 1. Load Data Loaders
    print("Preparing dataloaders...")
    train_loaders, cal_batches, test_loaders = get_dataloaders()
    
    # 2. Get Pre-trained Base Model
    print("Loading pre-trained base model...")
    base_model = MultiTaskResNet18(tasks)
    base_model.to(device)
    
    # 3. Train or Load Experts
    experts = {}
    for task in tasks:
        ckpt_path = f"expert_{task}.pth"
        expert_model = MultiTaskResNet18(tasks)
        if os.path.exists(ckpt_path):
            print(f"Loading cached expert for {task.upper()} from {ckpt_path}...")
            expert_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            expert_model.to(device)
        else:
            print(f"Training expert for {task.upper()}...")
            expert_model = train_expert(
                expert_model, 
                train_loaders[task], 
                test_loaders[task], 
                task, 
                epochs=5, 
                lr=5e-4, 
                weight_decay=1e-4, 
                device=device
            )
            print(f"Saving checkpoint to {ckpt_path}...")
            torch.save(expert_model.state_dict(), ckpt_path)
        experts[task] = expert_model
        
    # Evaluate individual expert upper bounds
    expert_accuracies = {}
    print("\n--- Individual Expert Performance (Oracle Upper Bounds) ---")
    for task in tasks:
        acc = evaluate_model(experts[task], test_loaders[task], task, device)
        expert_accuracies[task] = acc
        print(f"Expert {task.upper()} Accuracy: {acc:.2f}%")
        
    # Define the merging configurations to evaluate
    results = {
        "expert_bounds": expert_accuracies,
        "weight_average": {},
        "task_arithmetic": {}
    }
    
    # Calibration modes to test
    modes = ['none', 'tcac', 'sac', 'lsc']
    
    # 4. Evaluate Weight Averaging (WA)
    print("\n=============================================")
    print("Evaluating Weight Averaging (WA) Merging")
    print("=============================================")
    merged_wa = merge_models(base_model, experts, mode="weight_average")
    merged_wa.to(device)
    
    # Compute calibration parameters for each task
    wa_calibration = {}
    for task in tasks:
        # Get expert stats and merged stats
        expert_stats = collect_stats(experts[task], cal_batches[task], task, device)
        merged_stats = collect_stats(merged_wa, cal_batches[task], task, device)
        
        # Combine
        task_cal = {}
        for layer in expert_stats.keys():
            task_cal[layer] = {
                'mu_orig': expert_stats[layer]['mu'],
                'sigma_orig': expert_stats[layer]['sigma'],
                'sigma_orig_scalar': expert_stats[layer]['sigma_scalar'],
                'mu_merged': merged_stats[layer]['mu'],
                'sigma_merged': merged_stats[layer]['sigma'],
                'sigma_merged_scalar': merged_stats[layer]['sigma_scalar']
            }
        wa_calibration[task] = task_cal
        
    # Evaluate for each calibration mode
    for mode in modes:
        mode_accs = {}
        for task in tasks:
            # Register hooks for this mode & task
            handles = register_inference_hooks(merged_wa, wa_calibration[task], mode)
            # Evaluate
            acc = evaluate_model(merged_wa, test_loaders[task], task, device)
            mode_accs[task] = acc
            # Remove hooks
            for handle in handles:
                handle.remove()
        avg_acc = sum(mode_accs.values()) / len(tasks)
        results["weight_average"][mode] = {**mode_accs, "average": avg_acc}
        print(f"WA Merged + {mode.upper():<4} | MNIST: {mode_accs['mnist']:.2f}% | Fashion: {mode_accs['fashion']:.2f}% | CIFAR10: {mode_accs['cifar10']:.2f}% | Average: {avg_acc:.2f}%")
        
    # 5. Evaluate Task Arithmetic (TA) for various lambda coefficients
    lambdas = [0.2, 0.4, 0.6, 0.8, 1.0]
    for lam in lambdas:
        print("\n=============================================")
        print(f"Evaluating Task Arithmetic (TA) Merging (lambda = {lam})")
        print("=============================================")
        merged_ta = merge_models(base_model, experts, mode="task_arithmetic", lam=lam)
        merged_ta.to(device)
        
        # Compute calibration parameters for each task
        ta_calibration = {}
        for task in tasks:
            expert_stats = collect_stats(experts[task], cal_batches[task], task, device)
            merged_stats = collect_stats(merged_ta, cal_batches[task], task, device)
            
            task_cal = {}
            for layer in expert_stats.keys():
                task_cal[layer] = {
                    'mu_orig': expert_stats[layer]['mu'],
                    'sigma_orig': expert_stats[layer]['sigma'],
                    'sigma_orig_scalar': expert_stats[layer]['sigma_scalar'],
                    'mu_merged': merged_stats[layer]['mu'],
                    'sigma_merged': merged_stats[layer]['sigma'],
                    'sigma_merged_scalar': merged_stats[layer]['sigma_scalar']
                }
            ta_calibration[task] = task_cal
            
        # Evaluate for each calibration mode
        results["task_arithmetic"][f"lam_{lam}"] = {}
        for mode in modes:
            mode_accs = {}
            for task in tasks:
                handles = register_inference_hooks(merged_ta, ta_calibration[task], mode)
                acc = evaluate_model(merged_ta, test_loaders[task], task, device)
                mode_accs[task] = acc
                for handle in handles:
                    handle.remove()
            avg_acc = sum(mode_accs.values()) / len(tasks)
            results["task_arithmetic"][f"lam_{lam}"][mode] = {**mode_accs, "average": avg_acc}
            print(f"TA (lambda={lam}) + {mode.upper():<4} | MNIST: {mode_accs['mnist']:.2f}% | Fashion: {mode_accs['fashion']:.2f}% | CIFAR10: {mode_accs['cifar10']:.2f}% | Average: {avg_acc:.2f}%")
            
    # Save results to JSON
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults successfully saved to results.json.")
    
    # 6. Generate Plot
    generate_plots(results, lambdas)

def generate_plots(results, lambdas):
    plt.figure(figsize=(10, 6))
    
    modes = ['none', 'tcac', 'sac', 'lsc']
    colors = {'none': 'grey', 'tcac': 'blue', 'sac': 'green', 'lsc': 'red'}
    markers = {'none': 'o', 'tcac': 's', 'sac': '^', 'lsc': 'd'}
    labels = {'none': 'Uncalibrated', 'tcac': 'TCAC (Full Affine)', 'sac': 'SAC (Scaling-only)', 'lsc': 'LSC (Layer-wise Scalar)'}
    
    for mode in modes:
        avg_accs = []
        for lam in lambdas:
            avg_accs.append(results["task_arithmetic"][f"lam_{lam}"][mode]["average"])
        plt.plot(lambdas, avg_accs, color=colors[mode], marker=markers[mode], linestyle='-', label=labels[mode], linewidth=2)
        
    # Draw oracle upper bound line
    oracle_avg = sum(results["expert_bounds"].values()) / len(results["expert_bounds"])
    plt.axhline(y=oracle_avg, color='black', linestyle='--', label='Oracle Expert Average', alpha=0.7)
    
    # Draw Weight Averaging baseline as horizontal lines for comparison
    for mode in ['none', 'tcac', 'sac', 'lsc']:
        wa_val = results["weight_average"][mode]["average"]
        plt.axhline(y=wa_val, color=colors[mode], linestyle=':', alpha=0.5, label=f"WA + {labels[mode]}")
        
    plt.title("Model Merging Multi-Task Calibration Comparison", fontsize=14, fontweight='bold')
    plt.xlabel(r"Task Arithmetic Scaling Coefficient ($\lambda$)", fontsize=12)
    plt.ylabel("Average Test Accuracy (%) across MNIST, Fashion, CIFAR10", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("calibration_comparison.png", dpi=300)
    print("Plot successfully saved to calibration_comparison.png.")

if __name__ == "__main__":
    run_experiment()
