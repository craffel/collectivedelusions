import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False

# Helper function to get subset of data
def get_subset(dataset, num_samples):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return Subset(dataset, indices)

# Standard transformations
transform_color = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model Definition
class CustomResNet18(nn.Module):
    def __init__(self, num_tasks=3, num_classes=10):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(num_tasks)
        ])

    def forward(self, x, task_id):
        features = self.backbone(x)
        return self.heads[task_id](features)

# Load expert models
def load_experts():
    print("Loading saved expert weights...")
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    experts = []
    checkpoint = torch.load("./experts.pt", map_location=device)
    for i, task_name in enumerate(task_names):
        model = CustomResNet18().to(device)
        model.load_state_dict(checkpoint[f"expert_{i}"])
        experts.append(model)
    return experts

def get_base_backbone_state():
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Identity()
    return base_model.state_dict()

def merge_models(experts, base_backbone_state, lambda_val=0.3):
    ta_model = CustomResNet18().to(device)
    ta_state = ta_model.state_dict()
    
    # 1. Merge task heads
    for i in range(3):
        expert_state = experts[i].state_dict()
        for k in ta_state.keys():
            if k.startswith(f"heads.{i}."):
                ta_state[k].copy_(expert_state[k])
                
    # 2. Merge backbones
    theta_base = base_backbone_state
    for k in ta_state.keys():
        if k.startswith("backbone."):
            base_w = theta_base[k.replace("backbone.", "")].to(device)
            task_vectors = [experts[i].state_dict()[k] - base_w for i in range(3)]
            ta_weight = base_w + lambda_val * sum(task_vectors)
            ta_state[k].copy_(ta_weight)
            
    ta_model.load_state_dict(ta_state)
    return ta_model

def ties_merge(experts, base_backbone_state, lambda_val=0.3, fraction=0.2):
    ties_model = CustomResNet18().to(device)
    ties_state = ties_model.state_dict()
    
    # Copy task heads
    for i in range(3):
        expert_state = experts[i].state_dict()
        for k in ties_state.keys():
            if k.startswith(f"heads.{i}."):
                ties_state[k].copy_(expert_state[k])
                
    theta_base = base_backbone_state
    for k in ties_state.keys():
        if k.startswith("backbone."):
            base_w = theta_base[k.replace("backbone.", "")].to(device)
            task_vectors = [experts[i].state_dict()[k] - base_w for i in range(3)]
            tvs = torch.stack(task_vectors, dim=0)
            
            # Trim
            trimmed_tvs_list = []
            for i in range(3):
                tv = tvs[i]
                abs_tv = torch.abs(tv)
                if abs_tv.numel() <= 1:
                    trimmed_tvs_list.append(tv)
                else:
                    k_val = int(abs_tv.numel() * (1.0 - fraction))
                    k_val = max(1, min(k_val, abs_tv.numel() - 1))
                    thresh = torch.kthvalue(abs_tv.view(-1), k_val).values
                    mask = abs_tv >= thresh
                    trimmed_tvs_list.append(tv * mask)
            trimmed_tvs = torch.stack(trimmed_tvs_list, dim=0)
            
            # Elect Sign
            signs = torch.sign(trimmed_tvs)
            sum_signs = torch.sum(signs, dim=0)
            elected_sign = torch.sign(sum_signs)
            
            # Disjoint Merge
            match_mask = (signs == elected_sign) & (signs != 0)
            matching_tvs = trimmed_tvs * match_mask
            counts = match_mask.sum(dim=0).float()
            counts = torch.where(counts == 0, torch.ones_like(counts), counts)
            
            merged_tv = matching_tvs.sum(dim=0) / counts
            ties_weight = base_w + lambda_val * merged_tv
            ties_state[k].copy_(ties_weight)
            
    ties_model.load_state_dict(ties_state)
    return ties_model

def perform_head_only_sft(base_model, calib_datasets, num_epochs=15, lr=1e-3):
    sft_model = CustomResNet18().to(device)
    sft_model.load_state_dict(base_model.state_dict())
    
    # Freeze backbone
    for param in sft_model.backbone.parameters():
        param.requires_grad = False
        
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    criterion = nn.CrossEntropyLoss()
    
    for i, task_name in enumerate(task_names):
        calib_loader = DataLoader(calib_datasets[task_name], batch_size=min(32, len(calib_datasets[task_name])), shuffle=True)
        optimizer = optim.AdamW(sft_model.heads[i].parameters(), lr=lr, weight_decay=1e-2)
        
        for epoch in range(num_epochs):
            sft_model.backbone.eval()
            sft_model.heads[i].train()
            
            for imgs, labels in calib_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = sft_model(imgs, i)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
    return sft_model

def register_lsc_hooks(model, scale_factors, task_id):
    hooks = []
    layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    
    for l in layers:
        scale = scale_factors[task_id][l]
        def make_hook(s):
            return lambda module, input, output: output * s
        
        if l == 'conv1':
            submod = model.backbone.conv1
        else:
            submod = getattr(model.backbone, l)
            
        handle = submod.register_forward_hook(make_hook(scale))
        hooks.append(handle)
    return hooks

def compute_lsc_scale_factors(experts, base_model, calib_datasets):
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    
    scale_factors = {t_idx: {l: 1.0 for l in layers} for t_idx in range(3)}
    
    for t_idx, t_name in enumerate(task_names):
        calib_loader = DataLoader(calib_datasets[t_name], batch_size=min(32, len(calib_datasets[t_name])), shuffle=False)
        expert = experts[t_idx]
        expert.eval()
        base_model.eval()
        
        expert_acts = {l: [] for l in layers}
        base_acts = {l: [] for l in layers}
        
        handles = []
        for l in layers:
            if l == 'conv1':
                sub_exp = expert.backbone.conv1
                sub_base = base_model.backbone.conv1
            else:
                sub_exp = getattr(expert.backbone, l)
                sub_base = getattr(base_model.backbone, l)
                
            def exp_hook_fn(layer_name):
                return lambda m, i, o: expert_acts[layer_name].append(torch.abs(o).mean().item())
            def base_hook_fn(layer_name):
                return lambda m, i, o: base_acts[layer_name].append(torch.abs(o).mean().item())
                
            handles.append(sub_exp.register_forward_hook(exp_hook_fn(l)))
            handles.append(sub_base.register_forward_hook(base_hook_fn(l)))
            
        with torch.no_grad():
            for imgs, _ in calib_loader:
                imgs = imgs.to(device)
                _ = expert(imgs, t_idx)
                _ = base_model(imgs, t_idx)
                
        for h in handles:
            h.remove()
            
        for l in layers:
            exp_mean = np.mean(expert_acts[l]) if len(expert_acts[l]) > 0 else 1.0
            base_mean = np.mean(base_acts[l]) if len(base_acts[l]) > 0 else 1.0
            scale = exp_mean / (base_mean + 1e-8)
            scale = np.clip(scale, 0.1, 10.0)
            scale_factors[t_idx][l] = float(scale)
            
    return scale_factors

def evaluate_model(model, test_datasets, scale_factors=None):
    model.eval()
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    accs = {}
    for i, task_name in enumerate(task_names):
        hooks = []
        if scale_factors is not None:
            hooks = register_lsc_hooks(model, scale_factors, i)
            
        test_loader = DataLoader(test_datasets[task_name], batch_size=128, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs, i)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accs[task_name] = 100.0 * correct / total
        
        for h in hooks:
            h.remove()
    return accs

def main():
    print("Loading data splits...")
    os.makedirs("./data", exist_ok=True)
    
    mnist_train_full = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_gray)
    mnist_test_full = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_gray)
    
    fmnist_train_full = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_gray)
    fmnist_test_full = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_gray)
    
    cifar_train_full = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_color)
    cifar_test_full = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_color)
    
    test_datasets = {
        "MNIST": get_subset(mnist_test_full, 1000),
        "Fashion-MNIST": get_subset(fmnist_test_full, 1000),
        "CIFAR-10": get_subset(cifar_test_full, 1000)
    }
    
    experts = load_experts()
    base_backbone = get_base_backbone_state()
    
    ta_model = merge_models(experts, base_backbone, lambda_val=0.3)
    ties_model = ties_merge(experts, base_backbone, lambda_val=0.3, fraction=0.2)
    
    N_values = [16, 32, 64, 128, 256]
    
    results = {
        "N_values": N_values,
        "SFT_TA": [],
        "SFT_TIES": [],
        "LSC_TA": [],
        "LSC_TIES": []
    }
    
    for N in N_values:
        print(f"\nEvaluating calibration size N = {N}")
        
        # 1. Prepare calibration datasets
        calib_datasets = {
            "MNIST": get_subset(mnist_train_full, N),
            "Fashion-MNIST": get_subset(fmnist_train_full, N),
            "CIFAR-10": get_subset(cifar_train_full, N)
        }
        
        # 2. Evaluate SFT on TA
        sft_ta = perform_head_only_sft(ta_model, calib_datasets, num_epochs=15, lr=1e-3)
        accs_sft_ta = evaluate_model(sft_ta, test_datasets)
        avg_sft_ta = np.mean(list(accs_sft_ta.values()))
        results["SFT_TA"].append(float(avg_sft_ta))
        
        # 3. Evaluate SFT on TIES
        sft_ties = perform_head_only_sft(ties_model, calib_datasets, num_epochs=15, lr=1e-3)
        accs_sft_ties = evaluate_model(sft_ties, test_datasets)
        avg_sft_ties = np.mean(list(accs_sft_ties.values()))
        results["SFT_TIES"].append(float(avg_sft_ties))
        
        # 4. Evaluate LSC on TA
        scale_factors_ta = compute_lsc_scale_factors(experts, ta_model, calib_datasets)
        accs_lsc_ta = evaluate_model(ta_model, test_datasets, scale_factors_ta)
        avg_lsc_ta = np.mean(list(accs_lsc_ta.values()))
        results["LSC_TA"].append(float(avg_lsc_ta))
        
        # 5. Evaluate LSC on TIES
        scale_factors_ties = compute_lsc_scale_factors(experts, ties_model, calib_datasets)
        accs_lsc_ties = evaluate_model(ties_model, test_datasets, scale_factors_ties)
        avg_lsc_ties = np.mean(list(accs_lsc_ties.values()))
        results["LSC_TIES"].append(float(avg_lsc_ties))
        
        print(f"  SFT-TA: {avg_sft_ta:.2f}%, SFT-TIES: {avg_sft_ties:.2f}%, LSC-TA: {avg_lsc_ta:.2f}%, LSC-TIES: {avg_lsc_ties:.2f}%")
        
    # Save sweep results to JSON
    with open("ablation_calib_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Plot
    plt.figure(figsize=(8, 5))
    plt.plot(N_values, results["SFT_TIES"], marker='o', linestyle='-', color='tab:red', label='SFT (TIES-Merging)', linewidth=2.5)
    plt.plot(N_values, results["SFT_TA"], marker='s', linestyle='--', color='tab:orange', label='SFT (Task Arithmetic)', linewidth=2.0)
    plt.plot(N_values, results["LSC_TA"], marker='^', linestyle='-.', color='tab:blue', label='LSC (Task Arithmetic)', linewidth=2.0)
    plt.plot(N_values, results["LSC_TIES"], marker='v', linestyle=':', color='tab:cyan', label='LSC (TIES-Merging)', linewidth=2.0)
    
    # Draw reference expert upper bound (74.70%) and WA lower bound (26.73%)
    plt.axhline(y=74.70, color='gray', linestyle=':', label='Independent Experts (Upper Bound)', linewidth=1.5)
    plt.axhline(y=26.73, color='black', linestyle='--', label='Uncalibrated WA (Lower Bound)', linewidth=1.5)
    
    plt.xlabel('Calibration Sample Size ($N$ per task)', fontsize=12)
    plt.ylabel('Average Multi-Task Accuracy (%)', fontsize=12)
    plt.title('Ablation of Calibration Size ($N$) on Adaptation Performance\n(SFT is extremely data-efficient and robust under scarcity)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig("ablation_calib.png", dpi=300)
    print("Saved sweep plot to ablation_calib.png")

if __name__ == "__main__":
    main()
