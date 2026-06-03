import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def get_calibration_batches():
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
    
    cal_batches = {}
    for task in ["mnist", "fashion", "cifar10"]:
        train_len = len(train_datasets[task])
        train_indices = list(range(train_len))
        rng = random.Random(42)
        rng.shuffle(train_indices)
        cal_indices = train_indices[5000:5128]
        cal_subset = Subset(train_datasets[task], cal_indices)
        cal_loader = DataLoader(cal_subset, batch_size=128, shuffle=False)
        cal_batches[task] = next(iter(cal_loader))[0]
    return cal_batches

def collect_layer_stats(model, batch_x, device="cuda"):
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
        _ = model.backbone(batch_x.to(device))
        
    stats = {}
    for name, act in activations.items():
        var_scalar = act.var(unbiased=False)
        sigma_scalar = torch.sqrt(var_scalar + 1e-5).item()
        stats[name] = sigma_scalar
        
    for handle in handles:
        handle.remove()
        
    return stats

def merge_models(base_model, experts):
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    tasks = ["mnist", "fashion", "cifar10"]
    
    cal_batches = get_calibration_batches()
    base_model = MultiTaskResNet18(tasks)
    base_model.to(device)
    
    experts = {}
    for task in tasks:
        ckpt_path = f"expert_{task}.pth"
        expert_model = MultiTaskResNet18(tasks)
        expert_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        expert_model.to(device)
        experts[task] = expert_model
        
    merged_wa = merge_models(base_model, experts)
    merged_wa.to(device)
    
    scaling_factors = {}
    for task in tasks:
        expert_stats = collect_layer_stats(experts[task], cal_batches[task], device)
        merged_stats = collect_layer_stats(merged_wa, cal_batches[task], device)
        
        task_factors = {}
        for layer in expert_stats.keys():
            sigma_orig = expert_stats[layer]
            sigma_merged = merged_stats[layer]
            gamma = sigma_orig / sigma_merged
            task_factors[layer] = {
                "sigma_orig": sigma_orig,
                "sigma_merged": sigma_merged,
                "gamma": gamma
            }
        scaling_factors[task] = task_factors
        
    # Save scaling factors to JSON
    with open("scaling_factors.json", "w") as f:
        json.dump(scaling_factors, f, indent=4)
        
    print("Scaling factors successfully computed and saved to scaling_factors.json")
    
    # Print a summary table of scaling factors for a few layers
    print("\nSummary of Scaling Factors (gamma = sigma_orig / sigma_merged):")
    layers = sorted(list(scaling_factors["mnist"].keys()))
    print(f"{'Layer':<30} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR-10':<8}")
    print("-" * 64)
    for layer in layers:
        g_mnist = scaling_factors["mnist"][layer]["gamma"]
        g_fashion = scaling_factors["fashion"][layer]["gamma"]
        g_cifar = scaling_factors["cifar10"][layer]["gamma"]
        print(f"{layer:<30} | {g_mnist:.4f}   | {g_fashion:.4f}   | {g_cifar:.4f}")

if __name__ == "__main__":
    main()
