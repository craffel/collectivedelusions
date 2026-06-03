import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import os
import copy
import numpy as np

# Set device to CPU since we only do a few forward passes
device = torch.device('cpu')
print(f"Using device: {device}")

# Datasets & Transforms
cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

cifar_test = datasets.CIFAR10(root='data', train=False, download=True, transform=cifar_transform)
cifar_loader = DataLoader(cifar_test, batch_size=256, shuffle=False)

# Model Definitions
def get_resnet_progenitor():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Identity()
    return model

# Merging functions
def merge_weight_averaging(state_dicts):
    merged = copy.deepcopy(state_dicts[0])
    for key in merged.keys():
        if merged[key].is_floating_point():
            merged[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
    return merged

def get_bn_to_conv_map(model):
    bn_to_conv = {}
    last_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            last_conv_name = name
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if last_conv_name is not None:
                bn_to_conv[name] = last_conv_name
    return bn_to_conv

def merge_weight_averaging_crvs(state_dicts, gamma):
    merged = merge_weight_averaging(state_dicts)
    for key in merged.keys():
        if 'running_var' in key:
            merged[key] = merged[key] * gamma
    return merged

def merge_weight_averaging_cos_rvs(expert_models, state_dicts, min_gamma=0.1, max_gamma=1.0, beta=1.0):
    K = len(state_dicts)
    merged = merge_weight_averaging(state_dicts)
    bn_to_conv = get_bn_to_conv_map(expert_models[0])
    
    for bn_name, conv_name in bn_to_conv.items():
        bn_var_key = f"{bn_name}.running_var"
        var_key = None
        for k in merged.keys():
            if k.endswith(bn_var_key):
                var_key = k
                break
                
        conv_key = None
        for k in merged.keys():
            if k.endswith(f"{conv_name}.weight"):
                conv_key = k
                break
                
        if var_key is not None and conv_key is not None:
            weights = [sd[conv_key] for sd in state_dicts]
            num_channels = weights[0].shape[0]
            flattened_weights = [w.reshape(num_channels, -1) for w in weights]
            
            gammas = []
            for c in range(num_channels):
                cos_sims = []
                for i in range(K):
                    for j in range(i + 1, K):
                        w_i = flattened_weights[i][c]
                        w_j = flattened_weights[j][c]
                        norm_i = torch.norm(w_i)
                        norm_j = torch.norm(w_j)
                        if norm_i > 1e-8 and norm_j > 1e-8:
                            sim = torch.dot(w_i, w_j) / (norm_i * norm_j)
                        else:
                            sim = torch.tensor(0.0, device=w_i.device)
                        cos_sims.append(sim)
                gamma_c = 1.0 / K + (2.0 / (K * K)) * sum(cos_sims)
                gamma_c = gamma_c * beta
                gamma_c = torch.clamp(gamma_c, min_gamma, max_gamma)
                gammas.append(gamma_c)
                
            gammas = torch.stack(gammas).to(merged[var_key].device)
            merged[var_key] = merged[var_key] * gammas
    return merged

def main():
    # Load expert model states
    print("Loading expert models...")
    resnet_experts = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        backbone = get_resnet_progenitor()
        backbone.load_state_dict(torch.load(f'checkpoints/resnet_{task}_backbone.pt', map_location='cpu'))
        resnet_experts[task] = backbone
        
    expert_states = [resnet_experts[t].state_dict() for t in ['mnist', 'fmnist', 'cifar10']]
    
    # Construct Merged Models
    print("Merging models...")
    # 1. Standard WA
    wa_state = merge_weight_averaging(expert_states)
    wa_model = get_resnet_progenitor()
    wa_model.load_state_dict(wa_state)
    
    # 2. C-RVS (gamma = 0.70)
    crvs_state = merge_weight_averaging_crvs(expert_states, 0.70)
    crvs_model = get_resnet_progenitor()
    crvs_model.load_state_dict(crvs_state)
    
    # 3. S-Cos-RVS (beta = 0.80)
    scos_state = merge_weight_averaging_cos_rvs(
        [resnet_experts[t] for t in ['mnist', 'fmnist', 'cifar10']],
        expert_states,
        min_gamma=0.1,
        beta=0.80
    )
    scos_model = get_resnet_progenitor()
    scos_model.load_state_dict(scos_state)
    
    # Hook registration and stats collection
    models_to_test = {
        'MNIST Expert': resnet_experts['mnist'],
        'FMNIST Expert': resnet_experts['fmnist'],
        'CIFAR10 Expert': resnet_experts['cifar10'],
        'WA Uncalibrated': wa_model,
        'C-RVS (gamma=0.70)': crvs_model,
        'S-Cos-RVS (beta=0.80)': scos_model
    }
    
    # Get all BatchNorm2d layers in order
    dummy_model = get_resnet_progenitor()
    bn_names = []
    for name, module in dummy_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_names.append(name)
            
    print(f"Found {len(bn_names)} BatchNorm2d layers to hook.")
    
    # Load one batch of CIFAR-10 data
    iterator = iter(cifar_loader)
    x, _ = next(iterator)
    x = x.to(device)
    
    results = {m_name: {} for m_name in models_to_test.keys()}
    
    for m_name, model in models_to_test.items():
        model.to(device)
        model.eval()
        
        # Register hooks
        handles = []
        stats = {}
        
        def make_hook(layer_name):
            def hook(module, input, output):
                # Compute channel-wise standard deviation over the spatial and batch dims
                # output shape: [B, C, H, W]
                # we calculate variance for each channel across B, H, W
                var = torch.var(output, dim=(0, 2, 3), unbiased=False)
                std = torch.sqrt(var + 1e-8)
                stats[layer_name] = std.mean().item()
            return hook
            
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                handles.append(module.register_forward_hook(make_hook(name)))
                
        # Run forward pass
        with torch.no_grad():
            _ = model(x)
            
        # Remove hooks
        for h in handles:
            h.remove()
            
        results[m_name] = stats

    # Print a markdown table of the results
    print("\n================== ACTIVATION STANDARD DEVIATIONS BY LAYER ==================")
    header = f"| Layer | " + " | ".join(models_to_test.keys()) + " |"
    sep = "| :--- | " + " :---: | " * len(models_to_test)
    print(header)
    print(sep)
    
    for bn_name in bn_names:
        row = f"| {bn_name} | "
        vals = []
        for m_name in models_to_test.keys():
            val = results[m_name].get(bn_name, 0.0)
            vals.append(f"{val:.4f}")
        row += " | ".join(vals) + " |"
        print(row)

if __name__ == '__main__':
    main()
