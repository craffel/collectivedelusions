import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import timm

import sys
sys.path.insert(0, './local_packages_310')

TASKS = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
DEVICE = 'cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def get_layer_group(param_name):
    if 'patch_embed' in param_name or 'cls_token' in param_name or 'pos_embed' in param_name or 'norm_pre' in param_name:
        return 0
    elif 'blocks' in param_name:
        parts = param_name.split('.')
        block_idx = int(parts[1])
        return block_idx + 1
    elif 'norm.' in param_name:
        return 13
    else:
        return -1

def get_raw_dataset(task_name, split='train'):
    if task_name in ['MNIST', 'FashionMNIST']:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    if task_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(root='./data', train=(split=='train'), download=False, transform=transform)
    elif task_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=(split=='train'), download=False, transform=transform)
    elif task_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=(split=='train'), download=False, transform=transform)
    elif task_name == 'SVHN':
        dataset = torchvision.datasets.SVHN(root='./data', split=('train' if split=='train' else 'test'), download=False, transform=transform)
    return dataset

def get_cached_dataset(task_name, split, size, seed):
    dataset = get_raw_dataset(task_name, split)
    if size is not None and size < len(dataset):
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=g)[:size].tolist()
        dataset = Subset(dataset, indices)
    
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    x_all, y_all = next(iter(loader))
    return TensorDataset(x_all, y_all)

def softmax_entropy(x):
    return -(x.softmax(-1) * x.log_softmax(-1)).sum(-1)

def quantize_ste(w, b):
    qmin = -(2**(b-1))
    qmax = 2**(b-1) - 1
    dim = list(range(1, w.ndim))
    max_val = w.abs()
    for d in dim:
        max_val = max_val.max(dim=d, keepdim=True)[0]
    scale = max_val / qmax
    scale = torch.clamp(scale, min=1e-8)
    scaled_w = w / scale
    rounded_w = torch.clamp(torch.round(scaled_w), qmin, qmax)
    quant_w = (rounded_w - scaled_w).detach() + scaled_w
    return quant_w * scale

def update_model_weights(active_model, base_dict, task_vectors, lambdas_raw, quantize, bits):
    lambdas = torch.clamp(lambdas_raw, min=0.0, max=1.0)
    for n, p_base in base_dict.items():
        l = get_layer_group(n)
        if l >= 0:
            w_merged = p_base + sum(lambdas[l, i] * task_vectors[i][n] for i in range(4))
            if quantize and 'bias' not in n:
                w_quant = quantize_ste(w_merged, bits)
            else:
                w_quant = w_merged
            set_attr(active_model, n.split('.'), w_quant)

def evaluate_model_coefficients(active_model, base_dict, task_vectors, lambdas_raw, expert_heads, quantize, bits, test_loaders):
    active_model.eval()
    task_accs = {}
    with torch.no_grad():
        update_model_weights(active_model, base_dict, task_vectors, lambdas_raw, quantize, bits)
        for k, task_name in enumerate(test_loaders.keys()):
            active_model.head.weight.data.copy_(expert_heads[k]['weight'])
            active_model.head.bias.data.copy_(expert_heads[k]['bias'])
            
            correct = 0
            total = 0
            for x, y in test_loaders[task_name]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = active_model(x)
                preds = outputs.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            acc = correct / total * 100.0
            task_accs[task_name] = acc
    return task_accs

def evaluate_model_direct(active_model, expert_heads, test_loaders):
    active_model.eval()
    task_accs = {}
    with torch.no_grad():
        for k, task_name in enumerate(test_loaders.keys()):
            active_model.head.weight.data.copy_(expert_heads[k]['weight'])
            active_model.head.bias.data.copy_(expert_heads[k]['bias'])
            
            correct = 0
            total = 0
            for x, y in test_loaders[task_name]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = active_model(x)
                preds = outputs.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            acc = correct / total * 100.0
            task_accs[task_name] = acc
    return task_accs

def main():
    set_seed(42)
    print("Pre-loading datasets (test_size=256 for ultra-fast comparison on CPU)...")
    test_datasets = {task: get_cached_dataset(task, split='test', size=256, seed=999) for task in TASKS}
    test_loaders = {task: DataLoader(test_datasets[task], batch_size=128, shuffle=False) for task in TASKS}
    
    calib_datasets = {task: get_cached_dataset(task, split='test', size=16, seed=777) for task in TASKS}
    calib_loaders = {task: DataLoader(calib_datasets[task], batch_size=16, shuffle=False) for task in TASKS}
    
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(DEVICE)
    active_model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(DEVICE)
    active_model.head = torch.nn.Linear(192, 10).to(DEVICE)
    
    # Run make functional on active_model
    for name, _ in list(active_model.named_parameters()):
        if get_layer_group(name) >= 0:
            del_attr(active_model, name.split('.'))
            
    base_dict = {n: p.clone().detach().to(DEVICE) for n, p in base_model.named_parameters()}
    
    rho = 0.05
    seed = 42
    bits = 4
    
    print(f"\n--- Running Task Collapse Ablation (Seed {seed}, SAM Radius {rho}, 4-bit) ---")
    expert_heads = []
    expert_dicts = []
    for task in TASKS:
        expert_file = f"checkpoints/expert_{task}_seed{seed}_rho{rho}.pt"
        sd = torch.load(expert_file, map_location=DEVICE)
        expert_dicts.append(sd)
        head_w = sd['head.weight'].clone().detach()
        head_b = sd['head.bias'].clone().detach()
        expert_heads.append({'weight': head_w, 'bias': head_b})
        
    task_vectors = []
    for k in range(4):
        tv = {}
        for n in base_dict:
            if get_layer_group(n) >= 0:
                tv[n] = expert_dicts[k][n] - base_dict[n]
        task_vectors.append(tv)
        
    # 1. FlatQ-Merge (Low-dimensional adaptation, 56 parameters)
    print("\n[FlatQ-Merge] Optimizing 56 layer-wise coefficients...")
    prior = 0.3
    lambdas_raw = torch.nn.Parameter(torch.ones(14, 4, device=DEVICE) * prior)
    optimizer = torch.optim.Adam([lambdas_raw], lr=1e-3)
    
    calib_batches = {}
    for task_name, loader in calib_loaders.items():
        x, _ = next(iter(loader))
        calib_batches[task_name] = x.to(DEVICE)
        
    for step in range(40):
        optimizer.zero_grad()
        update_model_weights(active_model, base_dict, task_vectors, lambdas_raw, quantize=True, bits=bits)
        
        loss = 0.0
        for k, task_name in enumerate(calib_loaders.keys()):
            active_model.head.weight.data.copy_(expert_heads[k]['weight'])
            active_model.head.bias.data.copy_(expert_heads[k]['bias'])
            x = calib_batches[task_name]
            outputs = active_model(x)
            entropy = softmax_entropy(outputs).mean(0)
            loss += entropy
        loss.backward()
        optimizer.step()
        
    accs_flatq = evaluate_model_coefficients(active_model, base_dict, task_vectors, lambdas_raw, expert_heads, quantize=True, bits=bits, test_loaders=test_loaders)
    avg_flatq = np.mean(list(accs_flatq.values()))
    print(f"  -> FlatQ-Merge Task Accuracies: {accs_flatq}")
    print(f"  -> FlatQ-Merge Average Accuracy: {avg_flatq:.2f}%")
    
    # 2. High-Dimensional Adaptation (TENT-style, optimizing all merged model parameters)
    print("\n[TENT-style] Re-initializing merged model and optimizing ALL active model weights (5.7M parameters)...")
    # First, initialize model with uniform weights 0.3 (matching FlatQ-Merge start)
    lambdas_init = torch.ones(14, 4, device=DEVICE) * prior
    
    # Create a fresh non-functional model to optimize weights directly
    tent_model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    tent_model.head = torch.nn.Linear(192, 10)
    tent_model = tent_model.to(DEVICE)
    
    # Set weights to uniform merged weights (then quantize them)
    with torch.no_grad():
        for n, p in tent_model.named_parameters():
            if get_layer_group(n) >= 0:
                w_merged = base_dict[n] + sum(lambdas_init[get_layer_group(n), i] * task_vectors[i][n] for i in range(4))
                # PTQ
                if 'bias' not in n:
                    p.copy_(quantize_ste(w_merged, bits))
                else:
                    p.copy_(w_merged)
                    
    # Setup optimizer for all parameters in tent_model except head (which gets swapped)
    optimize_params = [p for n, p in tent_model.named_parameters() if 'head' not in n]
    tent_optimizer = torch.optim.Adam(optimize_params, lr=1e-4) # Standard TENT learning rate
    
    for step in range(40):
        tent_optimizer.zero_grad()
        loss = 0.0
        for k, task_name in enumerate(calib_loaders.keys()):
            tent_model.head.weight.data.copy_(expert_heads[k]['weight'])
            tent_model.head.bias.data.copy_(expert_heads[k]['bias'])
            x = calib_batches[task_name]
            outputs = tent_model(x)
            entropy = softmax_entropy(outputs).mean(0)
            loss += entropy
        loss.backward()
        tent_optimizer.step()
        
    accs_tent = evaluate_model_direct(tent_model, expert_heads, test_loaders)
    avg_tent = np.mean(list(accs_tent.values()))
    print(f"  -> TENT-style Task Accuracies: {accs_tent}")
    print(f"  -> TENT-style Average Accuracy: {avg_tent:.2f}%")

if __name__ == '__main__':
    main()
