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

def get_dataloaders(batch_size=128):
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
    
    loaders = {}
    cal_batches = {}
    test_loaders = {}
    
    for task in ["mnist", "fashion", "cifar10"]:
        train_len = len(train_datasets[task])
        train_indices = list(range(train_len))
        rng = random.Random(42)
        rng.shuffle(train_indices)
        
        cal_indices = train_indices[5000:5128]
        cal_subset = Subset(train_datasets[task], cal_indices)
        
        cal_loader = DataLoader(cal_subset, batch_size=128, shuffle=False)
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

def merge_weight_average(base_model, experts):
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
            
        weights = [expert_state_dicts[task][key] for task in experts.keys()]
        merged_state_dict[key] = torch.stack(weights).mean(dim=0)
        
    merged_model.backbone.load_state_dict(merged_state_dict)
    return merged_model

def merge_dare(base_model, experts, drop_rate=0.5, lam=1.0):
    merged_model = MultiTaskResNet18(list(experts.keys()))
    for task_name, expert in experts.items():
        merged_model.heads[task_name].load_state_dict(expert.heads[task_name].state_dict())
        
    merged_state_dict = merged_model.backbone.state_dict()
    base_state_dict = base_model.backbone.state_dict()
    expert_state_dicts = {task: expert.backbone.state_dict() for task, expert in experts.items()}
    
    for key in merged_state_dict.keys():
        if not torch.is_tensor(merged_state_dict[key]):
            continue
        if not torch.is_floating_point(merged_state_dict[key]):
            first_task = list(experts.keys())[0]
            merged_state_dict[key] = expert_state_dicts[first_task][key].clone()
            continue
            
        # For BatchNorm running statistics or trackers, do simple averaging (do NOT drop/rescale!)
        if "running_" in key or "tracked" in key:
            weights = [expert_state_dicts[task][key] for task in experts.keys()]
            merged_state_dict[key] = torch.stack(weights).mean(dim=0)
            continue
            
        task_vectors = []
        for task in experts.keys():
            tv = expert_state_dicts[task][key] - base_state_dict[key]
            mask = (torch.rand_like(tv) > drop_rate).float()
            rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 0.0
            tv_dare = tv * mask * rescale
            task_vectors.append(tv_dare)
            
        merged_state_dict[key] = base_state_dict[key] + lam * torch.stack(task_vectors).mean(dim=0)
        
    merged_model.backbone.load_state_dict(merged_state_dict)
    return merged_model

def merge_ties(base_model, experts, trim_fraction=0.2, lam=1.0):
    merged_model = MultiTaskResNet18(list(experts.keys()))
    for task_name, expert in experts.items():
        merged_model.heads[task_name].load_state_dict(expert.heads[task_name].state_dict())
        
    merged_state_dict = merged_model.backbone.state_dict()
    base_state_dict = base_model.backbone.state_dict()
    expert_state_dicts = {task: expert.backbone.state_dict() for task, expert in experts.items()}
    
    for key in merged_state_dict.keys():
        if not torch.is_tensor(merged_state_dict[key]):
            continue
        if not torch.is_floating_point(merged_state_dict[key]):
            first_task = list(experts.keys())[0]
            merged_state_dict[key] = expert_state_dicts[first_task][key].clone()
            continue
            
        # For BatchNorm running statistics or trackers, do simple averaging (do NOT trim!)
        if "running_" in key or "tracked" in key:
            weights = [expert_state_dicts[task][key] for task in experts.keys()]
            merged_state_dict[key] = torch.stack(weights).mean(dim=0)
            continue
            
        tvs = []
        for task in experts.keys():
            tv = expert_state_dicts[task][key] - base_state_dict[key]
            flat_tv = tv.view(-1)
            k = int(len(flat_tv) * trim_fraction)
            if k > 0:
                threshold = torch.topk(torch.abs(flat_tv), k).values[-1]
                mask = (torch.abs(tv) >= threshold).float()
                tv_pruned = tv * mask
            else:
                tv_pruned = torch.zeros_like(tv)
            tvs.append(tv_pruned)
            
        stacked_tvs = torch.stack(tvs, dim=0)
        signs = torch.sign(stacked_tvs)
        
        sum_votes = stacked_tvs.sum(dim=0)
        majority_sign = torch.sign(sum_votes)
        
        agreement_mask = (signs == majority_sign.unsqueeze(0)).float()
        aligned_tvs = stacked_tvs * agreement_mask
        
        non_zero_counts = (aligned_tvs != 0).float().sum(dim=0)
        sum_aligned = aligned_tvs.sum(dim=0)
        
        mean_aligned = torch.zeros_like(sum_aligned)
        valid_mask = non_zero_counts > 0
        mean_aligned[valid_mask] = sum_aligned[valid_mask] / non_zero_counts[valid_mask]
        
        merged_state_dict[key] = base_state_dict[key] + lam * mean_aligned
        
    merged_model.backbone.load_state_dict(merged_state_dict)
    return merged_model

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
        mu = act.mean(dim=(0, 2, 3), keepdim=True)
        var = act.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        sigma = torch.sqrt(var + 1e-5)
        
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

def register_inference_hooks(model, stats_dict, mode, threshold=1.30):
    handles = []
    
    def make_hook_fn(name, stats):
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
                return ((output - mu_merged) / sigma_merged_stable) * sigma_orig_stable + mu_orig
                
            elif mode == 'lsc':
                sigma_orig_scalar = stats['sigma_orig_scalar'].to(output.device)
                sigma_merged_scalar = stats['sigma_merged_scalar'].to(output.device)
                sigma_orig_scalar_stable = torch.clamp(sigma_orig_scalar, min=0.05)
                sigma_merged_scalar_stable = torch.clamp(sigma_merged_scalar, min=0.05)
                return output * (sigma_orig_scalar_stable / sigma_merged_scalar_stable)
                
            elif mode == 'tsc':
                sigma_orig_scalar = stats['sigma_orig_scalar'].to(output.device)
                sigma_merged_scalar = stats['sigma_merged_scalar'].to(output.device)
                sigma_orig_scalar_stable = torch.clamp(sigma_orig_scalar, min=0.05)
                sigma_merged_scalar_stable = torch.clamp(sigma_merged_scalar, min=0.05)
                gamma = sigma_orig_scalar_stable / sigma_merged_scalar_stable
                if gamma.item() >= threshold:
                    return output * gamma
                return output
                
            return output
        return hook_fn
        
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d) and name in stats_dict:
            handle = module.register_forward_hook(make_hook_fn(name, stats_dict[name]))
            handles.append(handle)
            
    return handles

def run_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Advanced Merging (TIES, DARE) Ablation on: {device}")
    
    set_seed(42)
    tasks = ["mnist", "fashion", "cifar10"]
    
    cal_batches, test_loaders = get_dataloaders()
    
    base_model = MultiTaskResNet18(tasks)
    base_model.to(device)
    
    experts = {}
    for task in tasks:
        ckpt_path = f"expert_{task}.pth"
        expert_model = MultiTaskResNet18(tasks)
        expert_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        expert_model.to(device)
        experts[task] = expert_model
        
    results = {}
    
    # 1. Sweep DARE Drop Rates
    print("\n=============================================")
    print("Sweeping DARE Drop Rates")
    print("=============================================")
    dare_drop_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
    results["dare_sweep"] = {}
    
    for dr in dare_drop_rates:
        print(f"\nEvaluating DARE with drop_rate={dr}")
        merged_model = merge_dare(base_model, experts, drop_rate=dr, lam=1.0)
        merged_model.to(device)
        
        stats_dict = {}
        for task in tasks:
            expert_stats = collect_stats(experts[task], cal_batches[task], task, device)
            merged_stats = collect_stats(merged_model, cal_batches[task], task, device)
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
            stats_dict[task] = task_cal
            
        dr_results = {}
        for mode in ['none', 'tcac', 'tsc']:
            mode_accs = {}
            for task in tasks:
                handles = register_inference_hooks(merged_model, stats_dict[task], mode, threshold=1.30)
                acc = evaluate_model(merged_model, test_loaders[task], task, device)
                mode_accs[task] = acc
                for handle in handles:
                    handle.remove()
            avg_acc = sum(mode_accs.values()) / len(tasks)
            dr_results[mode] = {**mode_accs, "average": avg_acc}
            print(f"DARE (dr={dr}) + {mode.upper():<4} | MNIST: {mode_accs['mnist']:.2f}% | Fashion: {mode_accs['fashion']:.2f}% | CIFAR10: {mode_accs['cifar10']:.2f}% | Average: {avg_acc:.2f}%")
        results["dare_sweep"][str(dr)] = dr_results
        
    # 2. Sweep TIES Trim Fractions
    print("\n=============================================")
    print("Sweeping TIES Trim Fractions (Fraction of Parameters Retained)")
    print("=============================================")
    ties_trim_fractions = [0.99, 0.95, 0.9, 0.8, 0.5, 0.2]
    results["ties_sweep"] = {}
    
    for tf in ties_trim_fractions:
        print(f"\nEvaluating TIES with trim_fraction={tf}")
        merged_model = merge_ties(base_model, experts, trim_fraction=tf, lam=1.0)
        merged_model.to(device)
        
        stats_dict = {}
        for task in tasks:
            expert_stats = collect_stats(experts[task], cal_batches[task], task, device)
            merged_stats = collect_stats(merged_model, cal_batches[task], task, device)
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
            stats_dict[task] = task_cal
            
        tf_results = {}
        for mode in ['none', 'tcac', 'tsc']:
            mode_accs = {}
            for task in tasks:
                handles = register_inference_hooks(merged_model, stats_dict[task], mode, threshold=1.30)
                acc = evaluate_model(merged_model, test_loaders[task], task, device)
                mode_accs[task] = acc
                for handle in handles:
                    handle.remove()
            avg_acc = sum(mode_accs.values()) / len(tasks)
            tf_results[mode] = {**mode_accs, "average": avg_acc}
            print(f"TIES (tf={tf}) + {mode.upper():<4} | MNIST: {mode_accs['mnist']:.2f}% | Fashion: {mode_accs['fashion']:.2f}% | CIFAR10: {mode_accs['cifar10']:.2f}% | Average: {avg_acc:.2f}%")
        results["ties_sweep"][str(tf)] = tf_results
        
    with open("advanced_merging_sweeps.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nAdvanced Merging sweeps successfully saved to advanced_merging_sweeps.json.")

if __name__ == "__main__":
    run_evaluation()
