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
import timm
from torch.func import functional_call

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ImageNet normalization transforms
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_grayscale = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataset(name, train=True):
    if name == "mnist":
        return torchvision.datasets.MNIST(root="./data", train=train, download=True, transform=transform_grayscale)
    elif name == "fmnist":
        return torchvision.datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform_grayscale)
    elif name == "cifar10":
        return torchvision.datasets.CIFAR10(root="./data", train=train, download=True, transform=transform_rgb)
    elif name == "svhn":
        split = "train" if train else "test"
        return torchvision.datasets.SVHN(root="./data", split=split, download=True, transform=transform_rgb)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# Map parameter names to layer groups for ResNet-18 (L=6 groups: conv1, layer1, layer2, layer3, layer4, fc)
def get_resnet_layer_group(name):
    if "conv1" in name or "bn1" in name:
        return 0
    elif "layer1" in name:
        return 1
    elif "layer2" in name:
        return 2
    elif "layer3" in name:
        return 3
    elif "layer4" in name:
        return 4
    else:
        return 5

# Quantization operators (reused from evaluate_merging.py)
def quantize_sym(W, num_bits, per_channel=False):
    if num_bits >= 16:
        return W
    qmin = - (2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    
    if per_channel and W.dim() > 1:
        max_val = W.abs().flatten(1).max(dim=1)[0]
        max_val = torch.clamp(max_val, min=1e-8)
        scale = max_val / qmax
        scale = scale.view(-1, *([1] * (W.dim() - 1)))
    else:
        max_val = W.abs().max()
        scale = max_val / qmax
        scale = torch.clamp(scale, min=1e-8)
        
    W_scaled = W / scale
    W_quant = W_scaled + (torch.clamp(torch.round(W_scaled), qmin, qmax) - W_scaled).detach()
    return W_quant * scale

def quantize_asym(W, num_bits, per_channel=False):
    if num_bits >= 16:
        return W
    qmin = - (2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    
    if per_channel and W.dim() > 1:
        min_val = W.flatten(1).min(dim=1)[0]
        max_val = W.flatten(1).max(dim=1)[0]
        scale = (max_val - min_val) / (2**num_bits - 1)
        scale = torch.clamp(scale, min=1e-8)
        
        zp = torch.round(-min_val / scale) - (2**(num_bits - 1))
        zp = torch.clamp(zp, qmin, qmax)
        
        scale = scale.view(-1, *([1] * (W.dim() - 1)))
        zp = zp.view(-1, *([1] * (W.dim() - 1)))
    else:
        min_val = W.min()
        max_val = W.max()
        scale = (max_val - min_val) / (2**num_bits - 1)
        scale = torch.clamp(scale, min=1e-8)
        zp = torch.round(-min_val / scale) - (2**(num_bits - 1))
        zp = torch.clamp(zp, qmin, qmax)
        
    W_scaled = W / scale + zp
    W_quant = W_scaled + (torch.clamp(torch.round(W_scaled), qmin, qmax) - W_scaled).detach()
    return (W_quant - zp) * scale

def quantize_tensor(W, schema, num_bits):
    if schema == 'none' or num_bits >= 16:
        return W
    elif schema == 'sym_tensor':
        return quantize_sym(W, num_bits, per_channel=False)
    elif schema == 'sym_channel':
        return quantize_sym(W, num_bits, per_channel=True)
    elif schema == 'asym_tensor':
        return quantize_asym(W, num_bits, per_channel=False)
    elif schema == 'asym_channel':
        return quantize_asym(W, num_bits, per_channel=True)
    else:
        raise ValueError(f"Unknown schema: {schema}")

print("Preparing datasets...")
datasets = {
    "mnist": get_dataset("mnist", train=True),
    "fmnist": get_dataset("fmnist", train=True),
    "cifar10": get_dataset("cifar10", train=True),
    "svhn": get_dataset("svhn", train=True)
}

test_datasets = {
    "mnist": get_dataset("mnist", train=False),
    "fmnist": get_dataset("fmnist", train=False),
    "cifar10": get_dataset("cifar10", train=False),
    "svhn": get_dataset("svhn", train=False)
}

# --- Part 1: CNN (ResNet-18) Expert Training & Evaluation ---
print("\n--- PART 1: CNN (ResNet-18) EXPERT TRAINING ---")
resnet_experts = {}
resnet_base_model = timm.create_model('resnet18', pretrained=True, num_classes=10).to(device)
resnet_base_params = {k: v.cpu().clone() for k, v in resnet_base_model.state_dict().items()}

for name, full_dataset in datasets.items():
    print(f"Training ResNet-18 expert for {name} on small subset...")
    indices = torch.randperm(len(full_dataset))[:1500].tolist()  # increased training subset
    subset = Subset(full_dataset, indices)
    loader = DataLoader(subset, batch_size=128, shuffle=True)
    
    # Init expert
    model = timm.create_model('resnet18', pretrained=True, num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):  # increased epochs
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            
    resnet_experts[name] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Test accuracy
    model.eval()
    test_indices = torch.randperm(len(test_datasets[name]))[:200].tolist()
    test_subset = Subset(test_datasets[name], test_indices)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(lbls).sum().item()
            total += lbls.size(0)
    print(f"  {name.upper()} Expert Test Accuracy (subset): {100.0 * correct / total:.2f}%")

# Save base parameters
torch.save({"base": resnet_base_params, "experts": resnet_experts}, "results/resnet18_poc_checkpoints.pth")

# --- Run Model Merging for ResNet-18 ---
print("\nEvaluating ResNet-18 Model Merging & Cross-Schema Generalization...")
# Set up a merging loop on ResNet-18
K = 4
L_resnet = 6 # Layer groups
Lambda_resnet = nn.Parameter(torch.full((K, L_resnet), 0.3, device=device))

# Build standard calibration loaders
cal_loaders = {}
for name, ds in datasets.items():
    indices = torch.randperm(len(ds))[:16].tolist()
    cal_loaders[name] = DataLoader(Subset(ds, indices), batch_size=16)

def get_merged_resnet_params(Lambda, pre_dict, expert_dicts, schema='none', num_bits=16):
    merged = {}
    expert_names = list(expert_dicts.keys())
    for name in pre_dict.keys():
        l = get_resnet_layer_group(name)
        # Compute task vectors
        delta_list = []
        for k in range(K):
            exp_name = expert_names[k]
            delta_list.append(expert_dicts[exp_name][name] - pre_dict[name])
        delta = torch.stack(delta_list, dim=0).to(device)
        coeff = Lambda[:, l].view(K, *([1] * (delta.dim() - 1)))
        W_merged = pre_dict[name].to(device) + (delta * coeff).sum(dim=0)
        
        # Handle non-differentiable batch norm buffers
        if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
            W_merged = W_merged.detach()
            
        merged[name] = quantize_tensor(W_merged, schema, num_bits)
    return merged

# Run unquantized and quantized baseline
print("Optimizing ResNet-18 coefficients under sym_channel (Source)...")
optimizer = optim.Adam([Lambda_resnet], lr=0.01)
resnet_base_model_eval = timm.create_model('resnet18', pretrained=True, num_classes=10).to(device)

for step in range(10):  # Run brief steps
    optimizer.zero_grad()
    # Merge weights
    merged_params = get_merged_resnet_params(Lambda_resnet, resnet_base_params, resnet_experts, schema='sym_channel', num_bits=4)
    
    # Compute entropy loss
    total_entropy = 0.0
    for k, (name, loader) in enumerate(cal_loaders.items()):
        for imgs, _ in loader:
            imgs = imgs.to(device)
            # Use functional_call to avoid in-place state mutation issues
            outputs = functional_call(resnet_base_model_eval, merged_params, imgs)
            probs = torch.softmax(outputs, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            total_entropy += entropy
            
    total_entropy = total_entropy / K
    total_entropy.backward()
    optimizer.step()
    with torch.no_grad():
        Lambda_resnet.clamp_(0.0, 1.0)

# Evaluate learned coefficients across schemas
opt_Lambda_resnet = Lambda_resnet.detach().cpu().clone()
print("\nResNet-18 Evaluation Matrix (Subset Accuracy %):")
target_schemas = ['sym_channel', 'sym_tensor']
resnet_results = {}
for q_eval in target_schemas:
    merged_eval = get_merged_resnet_params(opt_Lambda_resnet.to(device), resnet_base_params, resnet_experts, schema=q_eval, num_bits=4)
    
    # Evaluate on subsets of all 4 test sets
    total_acc = 0.0
    for name in test_datasets.keys():
        test_indices = torch.randperm(len(test_datasets[name]))[:100].tolist()
        test_loader = DataLoader(Subset(test_datasets[name], test_indices), batch_size=100)
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = functional_call(resnet_base_model_eval, merged_eval, imgs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(lbls).sum().item()
                total += lbls.size(0)
        acc = 100.0 * correct / total
        total_acc += acc
    avg_acc = total_acc / K
    print(f"  Target Schema {q_eval}: Average Acc = {avg_acc:.2f}%")
    resnet_results[q_eval] = avg_acc

# --- Part 2: Subspace-Constrained (LoRA-like) Merging Simulation ---
print("\n--- PART 2: SUBSPACE-CONSTRAINED (LoRA) MERGING SIMULATION ---")
# Load ViT experts
vit_base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10).to(device)
vit_base_params = {k: v.cpu().clone() for k, v in vit_base_model.state_dict().items()}

vit_experts = {}
expert_datasets = ["mnist", "fmnist", "cifar10", "svhn"]
for name in expert_datasets:
    chk_path = f"checkpoints/expert_{name}.pth"
    if os.path.exists(chk_path):
        vit_experts[name] = torch.load(chk_path, map_location="cpu")
    else:
        print(f"Checkpoints path {chk_path} not found. Creating random expert parameter shifts.")
        expert_dict = {}
        for k, val in vit_base_params.items():
            expert_dict[k] = val + torch.randn_like(val) * 0.01
        vit_experts[name] = expert_dict

# Project task vectors into a low-rank subspace
print("Projecting expert task vectors into low-rank subspace (r=4) via SVD (simulating attention-only PEFT/LoRA)...")
vit_experts_lora = {}
for name in expert_datasets:
    expert_dict = vit_experts[name]
    lora_dict = {}
    for key, val in vit_base_params.items():
        if val.dim() >= 2 and "attn" in key: # Project ONLY attention weights (like real LoRA)
            orig_shape = val.shape
            delta = expert_dict[key] - val
            delta_2d = delta.flatten(1)
            # Apply SVD
            U, S, Vt = torch.linalg.svd(delta_2d, full_matrices=False)
            # Keep rank r=4
            r = min(4, len(S))
            delta_low_rank_2d = U[:, :r] @ torch.diag(S[:r]) @ Vt[:r, :]
            delta_low_rank = delta_low_rank_2d.view(orig_shape)
            lora_dict[key] = val + delta_low_rank
        else:
            # Leave non-attention parameters (including the class heads and MLPs) intact
            lora_dict[key] = expert_dict[key].clone()
    vit_experts_lora[name] = lora_dict

# Map parameter names to L=14 layer groups
def get_layer_group(name):
    if "patch_embed" in name:
        return 0
    elif "blocks" in name:
        parts = name.split(".")
        block_idx = int(parts[1])
        return block_idx + 1
    elif "norm" in name:
        return 13
    else:
        return 0

Lambda_lora = nn.Parameter(torch.full((K, 14), 0.3, device=device))
optimizer_lora = optim.Adam([Lambda_lora], lr=0.01)

# Dynamic parameter merging function
def get_merged_lora_params(Lambda, pre_dict, expert_dicts, schema='none', num_bits=16):
    merged = {}
    expert_names = list(expert_dicts.keys())
    for name in pre_dict.keys():
        l = get_layer_group(name)
        delta_list = []
        for k in range(K):
            exp_name = expert_names[k]
            delta_list.append(expert_dicts[exp_name][name] - pre_dict[name])
        delta = torch.stack(delta_list, dim=0).to(device)
        coeff = Lambda[:, l].view(K, *([1] * (delta.dim() - 1)))
        W_merged = pre_dict[name].to(device) + (delta * coeff).sum(dim=0)
        
        # Handle non-differentiable buffers if any
        if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
            W_merged = W_merged.detach()
            
        merged[name] = quantize_tensor(W_merged, schema, num_bits)
    return merged

print("Optimizing subspace-constrained (LoRA) merging coefficients under sym_channel (Source)...")
for step in range(10):
    optimizer_lora.zero_grad()
    merged_params = get_merged_lora_params(Lambda_lora, vit_base_params, vit_experts_lora, schema='sym_channel', num_bits=4)
    # Fast dummy entropy loss
    total_entropy = 0.0
    for name, loader in cal_loaders.items():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = functional_call(vit_base_model, merged_params, imgs)
            probs = torch.softmax(outputs, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            total_entropy += entropy
    total_entropy = total_entropy / K
    total_entropy.backward()
    optimizer_lora.step()
    with torch.no_grad():
        Lambda_lora.clamp_(0.0, 1.0)

opt_Lambda_lora = Lambda_lora.detach().cpu().clone()
print("\nSubspace-Constrained (LoRA) Evaluation Matrix (Average Accuracy %):")
lora_results = {}
for q_eval in target_schemas:
    merged_eval = get_merged_lora_params(opt_Lambda_lora.to(device), vit_base_params, vit_experts_lora, schema=q_eval, num_bits=4)
    
    total_acc = 0.0
    for name in expert_datasets:
        test_indices = torch.randperm(len(test_datasets[name]))[:50].tolist()
        test_loader = DataLoader(Subset(test_datasets[name], test_indices), batch_size=50)
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = functional_call(vit_base_model, merged_eval, imgs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(lbls).sum().item()
                total += lbls.size(0)
        acc = 100.0 * correct / total
        total_acc += acc
    avg_acc = total_acc / K
    print(f"  Target Schema {q_eval}: Average Acc = {avg_acc:.2f}%")
    lora_results[q_eval] = avg_acc

# Save results
summary = {
    "resnet18_results": resnet_results,
    "lora_results": lora_results
}
with open("results/cnn_peft_poc_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nProof-of-Concept Experiments Completed successfully! Metrics saved to results/cnn_peft_poc_metrics.json.")
