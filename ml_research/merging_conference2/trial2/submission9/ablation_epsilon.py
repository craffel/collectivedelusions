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

def get_dataloaders(max_cal_size=128, batch_size=128):
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

def collect_stats(model, batch_x, device="cuda"):
    model.eval()
    handles = []
    activations = {}
    
    sliced_x = batch_x.to(device)
    
    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = output.detach().clone()
        return hook_fn
        
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)
            
    with torch.no_grad():
        _ = model(sliced_x, "mnist")
        
    stats = {}
    for name, act in activations.items():
        # Channel-wise stats
        mu = act.mean(dim=(0, 2, 3), keepdim=True)
        var = act.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        # Store raw variance and mean
        
        # Layer-wise scalar stats
        mu_scalar = act.mean()
        var_scalar = act.var(unbiased=False)
        
        stats[name] = {
            'mu': mu,
            'var': var,
            'mu_scalar': mu_scalar,
            'var_scalar': var_scalar
        }
        
    for handle in handles:
        handle.remove()
        
    return stats

def register_inference_hooks(model, task_calibration, mode, min_sigma):
    handles = []
    
    def make_inference_hook(name, stats):
        def hook_fn(module, input, output):
            if mode == 'none':
                return output
                
            mu_orig = stats['mu_orig'].to(output.device)
            var_orig = stats['var_orig'].to(output.device)
            mu_merged = stats['mu_merged'].to(output.device)
            var_merged = stats['var_merged'].to(output.device)
            
            # Compute standard deviation with current epsilon parameter
            sigma_orig = torch.sqrt(var_orig + 1e-5)
            sigma_merged = torch.sqrt(var_merged + 1e-5)
            
            # Clamping standard deviations to prevent Sparsity Trap
            sigma_orig_stable = torch.clamp(sigma_orig, min=min_sigma)
            sigma_merged_stable = torch.clamp(sigma_merged, min=min_sigma)
            
            if mode == 'tcac':
                calibrated = ((output - mu_merged) / sigma_merged_stable) * sigma_orig_stable + mu_orig
                return calibrated
                
            elif mode == 'lsc':
                var_orig_scalar = stats['var_orig_scalar'].to(output.device)
                var_merged_scalar = stats['var_merged_scalar'].to(output.device)
                sigma_orig_scalar = torch.sqrt(var_orig_scalar + 1e-5)
                sigma_merged_scalar = torch.sqrt(var_merged_scalar + 1e-5)
                
                sigma_orig_scalar_stable = torch.clamp(sigma_orig_scalar, min=min_sigma)
                sigma_merged_scalar_stable = torch.clamp(sigma_merged_scalar, min=min_sigma)
                
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

def run_epsilon_ablation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Stabilizer Epsilon Ablation Study on: {device}")
    
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
    
    # Define various min_sigma clamping thresholds
    min_sigmas = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1]
    
    # Collect calibration stats
    calib_stats = {}
    for task in tasks:
        calib_stats[task] = {
            'expert': collect_stats(experts[task], cal_batches[task], device),
            'merged': collect_stats(merged_wa, cal_batches[task], device)
        }
        
    task_calibrations = {}
    for task in tasks:
        task_calibrations[task] = {}
        for layer_name in calib_stats[task]['expert'].keys():
            task_calibrations[task][layer_name] = {
                'mu_orig': calib_stats[task]['expert'][layer_name]['mu'],
                'var_orig': calib_stats[task]['expert'][layer_name]['var'],
                'mu_merged': calib_stats[task]['merged'][layer_name]['mu'],
                'var_merged': calib_stats[task]['merged'][layer_name]['var'],
                'var_orig_scalar': calib_stats[task]['expert'][layer_name]['var_scalar'],
                'var_merged_scalar': calib_stats[task]['merged'][layer_name]['var_scalar']
            }
            
    results = {
        "min_sigmas": min_sigmas,
        "tcac": [],
        "lsc": []
    }
    
    # Evaluate LSC and TCAC across different values of min_sigma
    for ms in min_sigmas:
        print(f"\nEvaluating min_sigma = {ms}")
        
        # TCAC evaluation
        tcac_accs = {}
        for task in tasks:
            handles = register_inference_hooks(merged_wa, task_calibrations[task], mode='tcac', min_sigma=ms)
            acc = evaluate_model(merged_wa, test_loaders[task], task, device)
            tcac_accs[task] = acc
            for handle in handles:
                handle.remove()
        tcac_avg = sum(tcac_accs.values()) / len(tasks)
        results["tcac"].append({**tcac_accs, "average": tcac_avg})
        print(f"TCAC Average Accuracy: {tcac_avg:.2f}%")
        
        # LSC evaluation
        lsc_accs = {}
        for task in tasks:
            handles = register_inference_hooks(merged_wa, task_calibrations[task], mode='lsc', min_sigma=ms)
            acc = evaluate_model(merged_wa, test_loaders[task], task, device)
            lsc_accs[task] = acc
            for handle in handles:
                handle.remove()
        lsc_avg = sum(lsc_accs.values()) / len(tasks)
        results["lsc"].append({**lsc_accs, "average": lsc_avg})
        print(f"LSC Average Accuracy: {lsc_avg:.2f}%")
        
    # Save results to json
    with open("epsilon_ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nEpsilon Ablation Results saved to epsilon_ablation_results.json")

if __name__ == "__main__":
    run_epsilon_ablation()
