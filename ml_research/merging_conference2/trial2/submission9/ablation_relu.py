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
        # Use fixed seed shuffle for reproducibility
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

def merge_models(base_model, experts, mode="weight_average", lam=0.4):
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
            
        if mode == "weight_average":
            weights = [expert_state_dicts[task][key] for task in experts.keys()]
            merged_state_dict[key] = torch.stack(weights).mean(dim=0)
    merged_model.backbone.load_state_dict(merged_state_dict)
    return merged_model

# ----------------------------------------------------
# 1. Pre-ReLU Calibration (hooking BatchNorm2d)
# ----------------------------------------------------
def collect_stats_pre_relu(model, batch_x, task_name, device="cuda"):
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
        var_scalar = act.var(unbiased=False)
        sigma_scalar = torch.sqrt(var_scalar + 1e-5)
        stats[name] = {'sigma_scalar': sigma_scalar}
        
    for handle in handles:
        handle.remove()
    return stats

def register_inference_hooks_pre_relu(model, task_calibration):
    handles = []
    def make_inference_hook(name, stats):
        def hook_fn(module, input, output):
            sigma_orig_scalar = stats['sigma_orig_scalar'].to(output.device)
            sigma_merged_scalar = stats['sigma_merged_scalar'].to(output.device)
            sigma_orig_scalar_stable = torch.clamp(sigma_orig_scalar, min=0.05)
            sigma_merged_scalar_stable = torch.clamp(sigma_merged_scalar, min=0.05)
            calibrated = output * (sigma_orig_scalar_stable / sigma_merged_scalar_stable)
            return calibrated
        return hook_fn
        
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d) and name in task_calibration:
            stats = task_calibration[name]
            handle = module.register_forward_hook(make_inference_hook(name, stats))
            handles.append(handle)
    return handles

# ----------------------------------------------------
# 2. Post-ReLU Calibration (hooking ReLU layers)
# ----------------------------------------------------
def collect_stats_post_relu(model, batch_x, task_name, device="cuda"):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
            
    handles = []
    activations = {}
    call_counts = {}
    
    def reset_hook(module, input):
        call_counts.clear()
        
    handle_pre = model.register_forward_pre_hook(reset_hook)
    handles.append(handle_pre)
    
    def make_hook(name):
        def hook_fn(module, input, output):
            call_counts[name] = call_counts.get(name, 0) + 1
            unique_name = f"{name}_call{call_counts[name]}"
            activations[unique_name] = output.detach().clone()
        return hook_fn
        
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.ReLU):
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)
            
    with torch.no_grad():
        _ = model(batch_x.to(device), task_name)
        
    stats = {}
    for name, act in activations.items():
        var_scalar = act.var(unbiased=False)
        sigma_scalar = torch.sqrt(var_scalar + 1e-5)
        stats[name] = {'sigma_scalar': sigma_scalar}
        
    for handle in handles:
        handle.remove()
    return stats

def register_inference_hooks_post_relu(model, task_calibration):
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
            
    handles = []
    call_counts = {}
    
    def reset_hook(module, input):
        call_counts.clear()
        
    handle_pre = model.register_forward_pre_hook(reset_hook)
    handles.append(handle_pre)
    
    def make_inference_hook(name, stats_dict):
        def hook_fn(module, input, output):
            call_counts[name] = call_counts.get(name, 0) + 1
            unique_name = f"{name}_call{call_counts[name]}"
            if unique_name in stats_dict:
                stats = stats_dict[unique_name]
                sigma_orig_scalar = stats['sigma_orig_scalar'].to(output.device)
                sigma_merged_scalar = stats['sigma_merged_scalar'].to(output.device)
                sigma_orig_scalar_stable = torch.clamp(sigma_orig_scalar, min=0.05)
                sigma_merged_scalar_stable = torch.clamp(sigma_merged_scalar, min=0.05)
                calibrated = output * (sigma_orig_scalar_stable / sigma_merged_scalar_stable)
                return calibrated
            return output
        return hook_fn
        
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.ReLU):
            handle = module.register_forward_hook(make_inference_hook(name, task_calibration))
            handles.append(handle)
    return handles

def run_ablation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Relu Ablation on: {device}")
    
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
        
    merged_wa = merge_models(base_model, experts, mode="weight_average")
    merged_wa.to(device)
    
    # 1. Pre-ReLU Calibration evaluation
    pre_relu_cal = {}
    for task in tasks:
        expert_stats = collect_stats_pre_relu(experts[task], cal_batches[task], task, device)
        merged_stats = collect_stats_pre_relu(merged_wa, cal_batches[task], task, device)
        task_cal = {}
        for layer in expert_stats.keys():
            task_cal[layer] = {
                'sigma_orig_scalar': expert_stats[layer]['sigma_scalar'],
                'sigma_merged_scalar': merged_stats[layer]['sigma_scalar']
            }
        pre_relu_cal[task] = task_cal
        
    pre_relu_accs = {}
    for task in tasks:
        handles = register_inference_hooks_pre_relu(merged_wa, pre_relu_cal[task])
        acc = evaluate_model(merged_wa, test_loaders[task], task, device)
        pre_relu_accs[task] = acc
        for handle in handles:
            handle.remove()
            
    pre_relu_avg = sum(pre_relu_accs.values()) / len(tasks)
    print(f"Pre-ReLU LSC  | MNIST: {pre_relu_accs['mnist']:.2f}% | Fashion: {pre_relu_accs['fashion']:.2f}% | CIFAR10: {pre_relu_accs['cifar10']:.2f}% | Average: {pre_relu_avg:.2f}%")
    
    # 2. Post-ReLU Calibration evaluation
    post_relu_cal = {}
    for task in tasks:
        expert_stats = collect_stats_post_relu(experts[task], cal_batches[task], task, device)
        merged_stats = collect_stats_post_relu(merged_wa, cal_batches[task], task, device)
        task_cal = {}
        for layer in expert_stats.keys():
            task_cal[layer] = {
                'sigma_orig_scalar': expert_stats[layer]['sigma_scalar'],
                'sigma_merged_scalar': merged_stats[layer]['sigma_scalar']
            }
        post_relu_cal[task] = task_cal
        
    post_relu_accs = {}
    for task in tasks:
        handles = register_inference_hooks_post_relu(merged_wa, post_relu_cal[task])
        acc = evaluate_model(merged_wa, test_loaders[task], task, device)
        post_relu_accs[task] = acc
        for handle in handles:
            handle.remove()
            
    post_relu_avg = sum(post_relu_accs.values()) / len(tasks)
    print(f"Post-ReLU LSC | MNIST: {post_relu_accs['mnist']:.2f}% | Fashion: {post_relu_accs['fashion']:.2f}% | CIFAR10: {post_relu_accs['cifar10']:.2f}% | Average: {post_relu_avg:.2f}%")
    
    # Save results to a json file
    results = {
        "Pre-ReLU LSC": {**pre_relu_accs, "average": pre_relu_avg},
        "Post-ReLU LSC": {**post_relu_accs, "average": post_relu_avg}
    }
    with open("relu_ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Ablation results successfully saved to relu_ablation_results.json")

if __name__ == "__main__":
    run_ablation()
