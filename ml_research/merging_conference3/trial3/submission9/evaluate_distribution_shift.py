print("Trace: Import start...")
import os
import random
import json
import numpy as np
print("Trace: Importing torch...")
import torch
print("Trace: Importing torchvision...")
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
print("Trace: Importing timm...")
import timm
print("Trace: Imports complete.")

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
    print(f"DEBUG: get_cached_dataset task={task_name} split={split} size={size}")
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

def update_model_weights(active_model, base_dict, task_vectors, lambdas_raw, mode, quantize, bits):
    if mode == 'clipping':
        lambdas = torch.clamp(lambdas_raw, min=0.0, max=1.0)
    elif mode == 'softmax':
        lambdas = torch.softmax(lambdas_raw, dim=-1)
    else:
        raise ValueError(f"Unknown mode {mode}")
        
    for n, p_base in base_dict.items():
        l = get_layer_group(n)
        if l >= 0:
            w_merged = p_base + sum(lambdas[l, i] * task_vectors[i][n] for i in range(4))
            if quantize and 'bias' not in n:
                w_quant = quantize_ste(w_merged, bits)
            else:
                w_quant = w_merged
            set_attr(active_model, n.split('.'), w_quant)

def test_time_optimize_shifted(active_model, base_dict, task_vectors, expert_heads, mode, quantize, bits, calib_batch, target_task_idx, steps=40, lr=1e-3):
    prior = 0.3
    lambdas_raw = torch.nn.Parameter(torch.ones(14, 4, device=DEVICE) * prior)
    optimizer = torch.optim.Adam([lambdas_raw], lr=lr)
    
    x = calib_batch.to(DEVICE)
    active_model.head.weight.data.copy_(expert_heads[target_task_idx]['weight'])
    active_model.head.bias.data.copy_(expert_heads[target_task_idx]['bias'])
            
    for step in range(steps):
        optimizer.zero_grad()
        update_model_weights(active_model, base_dict, task_vectors, lambdas_raw, mode, quantize, bits)
        
        outputs = active_model(x)
        loss = softmax_entropy(outputs).mean(0)
            
        loss.backward()
        optimizer.step()
        
    return lambdas_raw.detach().clone()

def test_time_optimize_balanced(active_model, base_dict, task_vectors, expert_heads, mode, quantize, bits, calib_batches, steps=40, lr=1e-3):
    prior = 0.3
    lambdas_raw = torch.nn.Parameter(torch.ones(14, 4, device=DEVICE) * prior)
    optimizer = torch.optim.Adam([lambdas_raw], lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()
        update_model_weights(active_model, base_dict, task_vectors, lambdas_raw, mode, quantize, bits)
        
        loss = 0.0
        for k, task_name in enumerate(TASKS):
            active_model.head.weight.data.copy_(expert_heads[k]['weight'])
            active_model.head.bias.data.copy_(expert_heads[k]['bias'])
            
            x = calib_batches[task_name].to(DEVICE)
            outputs = active_model(x)
            entropy = softmax_entropy(outputs).mean(0)
            loss += entropy
            
        loss.backward()
        optimizer.step()
        
    return lambdas_raw.detach().clone()

def evaluate_model(active_model, base_dict, task_vectors, lambdas_raw, expert_heads, mode, quantize, bits, test_loaders):
    active_model.eval()
    task_accs = {}
    with torch.no_grad():
        update_model_weights(active_model, base_dict, task_vectors, lambdas_raw, mode, quantize, bits)
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
    print("DEBUG: Pre-loading evaluation datasets...")
    test_datasets = {task: get_cached_dataset(task, split='test', size=256, seed=999) for task in TASKS}
    test_loaders = {task: DataLoader(test_datasets[task], batch_size=128, shuffle=False) for task in TASKS}
    
    print("DEBUG: Creating base and active models...")
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(DEVICE)
    active_model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(DEVICE)
    active_model.head = torch.nn.Linear(192, 10).to(DEVICE)
    
    print("DEBUG: Deleting matching parameters in active model...")
    for name, _ in list(active_model.named_parameters()):
        if get_layer_group(name) >= 0:
            del_attr(active_model, name.split('.'))
            
    base_dict = {n: p.clone().detach().to(DEVICE) for n, p in base_model.named_parameters()}
    
    rho = 0.05
    seed = 42
    
    print(f"DEBUG: Loading expert checkpoints for seed {seed}, SAM radius {rho}...")
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
        
    # Get calibration data (total 64 images for each task)
    calib_datasets_large = {task: get_cached_dataset(task, split='test', size=64, seed=777) for task in TASKS}
    calib_batches_large = {}
    for task in TASKS:
        loader = DataLoader(calib_datasets_large[task], batch_size=64, shuffle=False)
        calib_batches_large[task] = next(iter(loader))[0]
        
    # Balanced calibration data (16 images per task, total 64)
    calib_datasets_balanced = {task: get_cached_dataset(task, split='test', size=16, seed=777) for task in TASKS}
    calib_batches_balanced = {}
    for task in TASKS:
        loader = DataLoader(calib_datasets_balanced[task], batch_size=16, shuffle=False)
        calib_batches_balanced[task] = next(iter(loader))[0]
        
    scenarios = {
        'Balanced': ('balanced', calib_batches_balanced),
        'Only MNIST': ('MNIST', calib_batches_large['MNIST']),
        'Only FashionMNIST': ('FashionMNIST', calib_batches_large['FashionMNIST']),
        'Only CIFAR10': ('CIFAR10', calib_batches_large['CIFAR10']),
        'Only SVHN': ('SVHN', calib_batches_large['SVHN'])
    }
    
    results = {}
    
    for sc_name, (sc_type, batch_data) in scenarios.items():
        print(f"\nEvaluating Scenario: {sc_name}")
        results[sc_name] = {}
        for bits in [8, 4]:
            if sc_type == 'balanced':
                lambdas = test_time_optimize_balanced(
                    active_model, base_dict, task_vectors, expert_heads, 'clipping',
                    quantize=True, bits=bits, calib_batches=batch_data
                )
            else:
                task_idx = TASKS.index(sc_type)
                lambdas = test_time_optimize_shifted(
                    active_model, base_dict, task_vectors, expert_heads, 'clipping',
                    quantize=True, bits=bits, calib_batch=batch_data, target_task_idx=task_idx
                )
            
            accs = evaluate_model(
                active_model, base_dict, task_vectors, lambdas, expert_heads, 'clipping',
                quantize=True, bits=bits, test_loaders=test_loaders
            )
            avg_acc = np.mean(list(accs.values()))
            print(f"  {bits}-bit: Task Accuracies: {accs}")
            print(f"  {bits}-bit: Average Multi-Task Accuracy: {avg_acc:.2f}%")
            
            # Show optimized lambdas stats (variance across layers)
            l_val = torch.clamp(lambdas, min=0.0, max=1.0)
            print(f"  {bits}-bit: Lambdas mean: {l_val.mean().item():.4f}, std: {l_val.std().item():.4f}")
            
            results[sc_name][bits] = {
                'accuracies': accs,
                'average': float(avg_acc),
                'lambdas_mean': float(l_val.mean().item()),
                'lambdas_std': float(l_val.std().item())
            }
            
    # Save results to json
    with open("distribution_shift_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDistribution shift sensitivity sweep finished and saved!")

if __name__ == '__main__':
    main()
