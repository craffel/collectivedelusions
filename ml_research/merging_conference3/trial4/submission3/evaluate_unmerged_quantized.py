import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
from safetensors.torch import load_file
import numpy as np

# --- Quantization Utilities ---

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def round_ste(x):
    return RoundSTE.apply(x)

def quantize_dequantize_weight(W, bits, symmetric, per_channel):
    # Signed quantization limits
    qmin = -(2**(bits - 1))
    qmax = (2**(bits - 1)) - 1
    
    if per_channel:
        dim = 1
        if symmetric:
            max_val = torch.max(torch.abs(W), dim=dim, keepdim=True)[0]
            max_val = torch.clamp(max_val, min=1e-8)
            scale = max_val / qmax
            q = torch.clamp(round_ste(W / scale), qmin, qmax)
            W_dq = q * scale
        else:
            min_val = torch.min(W, dim=dim, keepdim=True)[0]
            max_val = torch.max(W, dim=dim, keepdim=True)[0]
            diff = torch.clamp(max_val - min_val, min=1e-8)
            scale = diff / (qmax - qmin)
            zp = round_ste(-min_val / scale) + qmin
            zp = torch.clamp(zp, qmin, qmax)
            q = torch.clamp(round_ste(W / scale) + zp, qmin, qmax)
            W_dq = scale * (q - zp)
    else:
        # Per-tensor
        if symmetric:
            max_val = torch.max(torch.abs(W))
            max_val = torch.clamp(max_val, min=1e-8)
            scale = max_val / qmax
            q = torch.clamp(round_ste(W / scale), qmin, qmax)
            W_dq = q * scale
        else:
            min_val = torch.min(W)
            max_val = torch.max(W)
            diff = torch.clamp(max_val - min_val, min=1e-8)
            scale = diff / (qmax - qmin)
            zp = round_ste(-min_val / scale) + qmin
            zp = torch.clamp(zp, qmin, qmax)
            q = torch.clamp(round_ste(W / scale) + zp, qmin, qmax)
            W_dq = scale * (q - zp)
            
    return W_dq

# --- Data Loading Utilities ---

def get_transforms(grayscale=False):
    if grayscale:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders():
    loaders = {}
    
    # We use a subsample of 500 images for test evaluation to ensure speed & coverage
    test_subsample = 500
    
    tasks = [
        ("mnist", datasets.MNIST, True, "train"),
        ("fashionmnist", datasets.FashionMNIST, True, "train"),
        ("cifar10", datasets.CIFAR10, False, "train"),
        ("svhn", datasets.SVHN, False, "train")
    ]
    
    for name, dataset_class, grayscale, split_type in tasks:
        transform = get_transforms(grayscale=grayscale)
        
        # Test loaders
        if name == "svhn":
            test_dataset = dataset_class(root="./data", split="test", download=True, transform=transform)
        else:
            test_dataset = dataset_class(root="./data", train=False, download=True, transform=transform)
            
        test_indices = torch.randperm(len(test_dataset))[:test_subsample].tolist()
        test_sub = Subset(test_dataset, test_indices)
        loaders[name] = DataLoader(test_sub, batch_size=64, shuffle=False, num_workers=2)
            
    return loaders

# --- Weight Utilities ---

def load_lora_updates():
    updates = {}
    tasks = ["mnist", "fashionmnist", "cifar10", "svhn"]
    for task in tasks:
        path = f"checkpoints/{task}_lora/adapter_model.safetensors"
        state_dict = load_file(path)
        task_updates = {}
        for l in range(12):
            key_A = f"base_model.model.blocks.{l}.attn.qkv.lora_A.weight"
            key_B = f"base_model.model.blocks.{l}.attn.qkv.lora_B.weight"
            W_A = state_dict[key_A] # [8, 192]
            W_B = state_dict[key_B] # [576, 8]
            task_updates[l] = W_B @ W_A
        updates[task] = task_updates
    return updates

def load_task_heads():
    heads = {}
    tasks = ["mnist", "fashionmnist", "cifar10", "svhn"]
    for task in tasks:
        path = f"checkpoints/{task}_lora/head.pt"
        head = nn.Linear(192, 10)
        head.load_state_dict(torch.load(path, map_location="cpu"))
        head.eval()
        heads[task] = head
    return heads

def evaluate_multi_task(model, heads, loaders, device):
    model.eval()
    results = {}
    for task_name, loader in loaders.items():
        head = heads[task_name].to(device)
        model.head = head
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc = 100. * correct / total
        results[task_name] = acc
    return results

def apply_weights(model, W_dict):
    for l in range(12):
        layer = model.blocks[l].attn.qkv
        if hasattr(layer, "weight"):
            if isinstance(layer.weight, nn.Parameter):
                del layer.weight
            layer.weight = W_dict[l]

def restore_weights(model, original_weights):
    for l in range(12):
        layer = model.blocks[l].attn.qkv
        if hasattr(layer, "weight"):
            del layer.weight
        layer.register_parameter("weight", nn.Parameter(original_weights[l]))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation running on: {device}")
    
    loaders = get_dataloaders()
    
    base_model = timm.create_model("vit_tiny_patch16_224", pretrained=True).to(device)
    base_model.eval()
    
    original_weights = {}
    for l in range(12):
        original_weights[l] = base_model.blocks[l].attn.qkv.weight.clone().to(device)
        
    expert_updates = load_lora_updates()
    task_heads = load_task_heads()
    
    for task in expert_updates:
        for l in range(12):
            expert_updates[task][l] = expert_updates[task][l].to(device)
            
    for task in task_heads:
        task_heads[task] = task_heads[task].to(device)
        
    configs = [
        (8, True, True, "INT8 Symmetric Per-Channel"),
        (4, True, True, "INT4 Symmetric Per-Channel"),
        (4, False, True, "INT4 Asymmetric Per-Channel"),
        (4, True, False, "INT4 Symmetric Per-Tensor")
    ]
    
    all_results = {}
    
    print("\nStarting unmerged quantized expert evaluations...")
    for bits, sym, pc, config_name in configs:
        print(f"Evaluating Config: {config_name}")
        config_results = {}
        for task in ["mnist", "fashionmnist", "cifar10", "svhn"]:
            # Combine base model with a single task's adapter
            task_unmerged_weights = {}
            for l in range(12):
                W_comb = original_weights[l] + expert_updates[task][l]
                # Apply target quantization schema directly to the individual expert
                task_unmerged_weights[l] = quantize_dequantize_weight(W_comb, bits, sym, pc)
                
            apply_weights(base_model, task_unmerged_weights)
            res = evaluate_multi_task(base_model, task_heads, {task: loaders[task]}, device)
            config_results[task] = res[task]
            restore_weights(base_model, original_weights)
            
        print(f"  Result: {config_results}")
        all_results[config_name] = config_results
        
    print("\n=== FINAL RESULTS: UNMERGED QUANTIZED EXPERTS ===")
    print("| Configuration | MNIST | FashionMNIST | CIFAR-10 | SVHN | Mean |")
    print("|---|---|---|---|---|---|")
    for config_name in [c[3] for c in configs]:
        res = all_results[config_name]
        vals = [res["mnist"], res["fashionmnist"], res["cifar10"], res["svhn"]]
        mean_val = sum(vals) / len(vals)
        print(f"| {config_name} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | {vals[3]:.2f}% | {mean_val:.2f}% |")

if __name__ == "__main__":
    main()
