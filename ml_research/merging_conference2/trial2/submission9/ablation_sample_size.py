import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
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
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleDict({
            task: nn.Linear(512, 10) for task in task_names
        })
        
    def forward(self, x, task_name):
        features = self.backbone(x)
        return self.heads[task_name](features)

def get_dataloaders(max_cal_size=256, batch_size=128):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize
    ])
    
    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])
    
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
    
    cal_batches = {}
    test_loaders = {}
    
    for task in ["mnist", "fashion", "cifar10"]:
        train_len = len(train_datasets[task])
        train_indices = list(range(train_len))
        rng = random.Random(42)
        rng.shuffle(train_indices)
        
        # Take up to max_cal_size separate calibration images
        cal_indices = train_indices[5000:5000+max_cal_size]
        cal_subset = Subset(train_datasets[task], cal_indices)
        
        cal_loader = DataLoader(cal_subset, batch_size=max_cal_size, shuffle=False)
        cal_batches[task] = next(iter(cal_loader))[0]
        
        test_len = len(test_datasets[task])
        test_indices = list(range(test_len))
        rng.shuffle(test_indices)
        test_sub_indices = test_indices[:1000]
        test_subset = Subset(test_datasets[task], test_sub_indices)
        
        test_loaders[task] = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        
    return cal_batches, test_loaders

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

def merge_models(base_model, experts, mode="weight_average"):
    merged_model = MultiTaskResNet18(list(experts.keys()))
    for task_name, expert in experts.items():
        merged_model.heads[task_name].load_state_dict(expert.heads[task_name].state_dict())
        
    merged_state_dict = merged_model.backbone.state_dict()
    expert_state_dicts = {task: expert.backbone.state_dict() for task, expert in experts.items()}
    
    for key in merged_state_dict.keys():
        if not torch.is_tensor(merged_state_dict[key]):
            continue
        if not torch.is_floating_point(merged_state_dict[key]):
            first_task = list(experts.keys())[0]
            merged_state_dict[key] = expert_state_dicts[first_task][key].clone()
            continue
            
        if mode == "weight_average":
            weights = [expert_state_dicts[task][key] for task in experts.keys()]
            merged_state_dict[key] = torch.stack(weights).mean(dim=0)
    merged_model.backbone.load_state_dict(merged_state_dict)
    return merged_model

# Collect statistics for a specific batch size N
def collect_stats_for_size(model, batch_x, N, device="cuda"):
    model.eval()
    handles = []
    activations = {}
    
    # Slice the batch to exactly N samples
    sliced_x = batch_x[:N].to(device)
    
    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = output.detach().clone()
        return hook_fn
        
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)
            
    with torch.no_grad():
        _ = model(sliced_x, "mnist") # Any dummy task head works for backbone activations
        
    stats = {}
    for name, act in activations.items():
        # Channel-wise stats
        mu = act.mean(dim=(0, 2, 3), keepdim=True)
        var = act.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        sigma = torch.sqrt(var + 1e-5)
        
        # Layer-wise scalar stats
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

# Register hooks during evaluation
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
            
            sigma_orig_stable = torch.clamp(sigma_orig, min=0.05)
            sigma_merged_stable = torch.clamp(sigma_merged, min=0.05)
            
            if mode == 'tcac':
                # Channel-wise Affine (Mean shift + Std scale)
                calibrated = ((output - mu_merged) / sigma_merged_stable) * sigma_orig_stable + mu_orig
                return calibrated
                
            elif mode == 'lsc':
                # Layer-wise Scaling-only (Scalar std scale)
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

def run_sample_size_ablation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Sample Size Ablation Study on: {device}")
    
    set_seed(42)
    tasks = ["mnist", "fashion", "cifar10"]
    
    # Prepare data with max calibration size of 256
    cal_batches, test_loaders = get_dataloaders(max_cal_size=256)
    
    # Load base model and experts
    base_model = MultiTaskResNet18(tasks)
    base_model.to(device)
    
    experts = {}
    for task in tasks:
        ckpt_path = f"expert_{task}.pth"
        expert_model = MultiTaskResNet18(tasks)
        expert_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        expert_model.to(device)
        experts[task] = expert_model
        
    # Weight average merging
    merged_wa = merge_models(base_model, experts, mode="weight_average")
    merged_wa.to(device)
    
    sizes = [4, 8, 16, 32, 64, 128, 256]
    results = {
        "uncalibrated": None,
        "tcac": [],
        "lsc": []
    }
    
    # First, evaluate uncalibrated baseline (constant across sizes)
    uncal_accs = {}
    for task in tasks:
        acc = evaluate_model(merged_wa, test_loaders[task], task, device)
        uncal_accs[task] = acc
    uncal_avg = sum(uncal_accs.values()) / len(tasks)
    results["uncalibrated"] = {**uncal_accs, "average": uncal_avg}
    print(f"Uncalibrated (NONE) | MNIST: {uncal_accs['mnist']:.2f}% | Fashion: {uncal_accs['fashion']:.2f}% | CIFAR10: {uncal_accs['cifar10']:.2f}% | Average: {uncal_avg:.2f}%\n")
    
    # Iterate over calibration batch sizes
    for N in sizes:
        print(f"--- Evaluating with calibration size N = {N} ---")
        
        # Collect stats for TCAC and LSC under size N
        calibration_data = {}
        for task in tasks:
            expert_stats = collect_stats_for_size(experts[task], cal_batches[task], N, device)
            merged_stats = collect_stats_for_size(merged_wa, cal_batches[task], N, device)
            
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
            calibration_data[task] = task_cal
            
        # 1. Evaluate TCAC
        tcac_accs = {}
        for task in tasks:
            handles = register_inference_hooks(merged_wa, calibration_data[task], 'tcac')
            acc = evaluate_model(merged_wa, test_loaders[task], task, device)
            tcac_accs[task] = acc
            for handle in handles:
                handle.remove()
        tcac_avg = sum(tcac_accs.values()) / len(tasks)
        results["tcac"].append({"size": N, "mnist": tcac_accs['mnist'], "fashion": tcac_accs['fashion'], "cifar10": tcac_accs['cifar10'], "average": tcac_avg})
        print(f"TCAC (N={N}) | MNIST: {tcac_accs['mnist']:.2f}% | Fashion: {tcac_accs['fashion']:.2f}% | CIFAR10: {tcac_accs['cifar10']:.2f}% | Average: {tcac_avg:.2f}%")
        
        # 2. Evaluate LSC
        lsc_accs = {}
        for task in tasks:
            handles = register_inference_hooks(merged_wa, calibration_data[task], 'lsc')
            acc = evaluate_model(merged_wa, test_loaders[task], task, device)
            lsc_accs[task] = acc
            for handle in handles:
                handle.remove()
        lsc_avg = sum(lsc_accs.values()) / len(tasks)
        results["lsc"].append({"size": N, "mnist": lsc_accs['mnist'], "fashion": lsc_accs['fashion'], "cifar10": lsc_accs['cifar10'], "average": lsc_avg})
        print(f"LSC  (N={N}) | MNIST: {lsc_accs['mnist']:.2f}% | Fashion: {lsc_accs['fashion']:.2f}% | CIFAR10: {lsc_accs['cifar10']:.2f}% | Average: {lsc_avg:.2f}%\n")
        
    # Save results to JSON
    with open("sample_size_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Ablation results successfully saved to sample_size_results.json")
    
    # Generate Plot
    plt.figure(figsize=(10, 6))
    
    tcac_sizes = [r["size"] for r in results["tcac"]]
    tcac_avgs = [r["average"] for r in results["tcac"]]
    lsc_sizes = [r["size"] for r in results["lsc"]]
    lsc_avgs = [r["average"] for r in results["lsc"]]
    
    plt.plot(lsc_sizes, lsc_avgs, color='red', marker='d', linestyle='-', label='LSC (Layer-wise Scaling, Our Proposed)', linewidth=2.5)
    plt.plot(tcac_sizes, tcac_avgs, color='blue', marker='s', linestyle='-', label='TCAC (Channel-wise Affine, Baseline)', linewidth=2)
    plt.axhline(y=results["uncalibrated"]["average"], color='grey', linestyle='--', label='Uncalibrated (NONE)', alpha=0.8)
    
    plt.title("Calibration Robustness under Varying Sample Sizes (N)", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Calibration Samples (N)", fontsize=12)
    plt.ylabel("Average Test Accuracy (%) across MNIST, Fashion, CIFAR10", fontsize=12)
    plt.xscale('log')
    plt.xticks(sizes, labels=[str(s) for s in sizes])
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig("sample_size_comparison.png", dpi=300)
    print("Plot successfully saved as sample_size_comparison.png")

if __name__ == "__main__":
    run_sample_size_ablation()
