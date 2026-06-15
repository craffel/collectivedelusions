import os
import random
import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import timm

# Configure PYTHONPATH style local package path
import sys
sys.path.insert(0, './local_packages_310')

TASKS = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def train_expert_swa(task_name, seed, train_dataset, epochs=15, swa_start=10):
    set_seed(seed)
    print(f"Training SWA expert for task {task_name} (seed={seed}) on {DEVICE}...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.head = torch.nn.Linear(model.num_features, 10)
    model = model.to(DEVICE)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    swa_weights = None
    swa_count = 0
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
        if epoch >= swa_start:
            # Collect weights
            sd = {n: p.clone().detach().cpu() for n, p in model.state_dict().items()}
            if swa_weights is None:
                swa_weights = sd
            else:
                for n in swa_weights:
                    swa_weights[n] += sd[n]
            swa_count += 1
            
    # Average weights
    for n in swa_weights:
        swa_weights[n] /= swa_count
        
    save_path = f"checkpoints/expert_{task_name}_seed{seed}_swa.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(swa_weights, save_path)
    print(f"Finished SWA pre-training. Saved SWA expert to {save_path}")

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

def test_time_optimize(active_model, base_dict, task_vectors, expert_heads, quantize, bits, calib_loaders, steps=40, lr=1e-3):
    prior = 0.3
    lambdas_raw = torch.nn.Parameter(torch.ones(14, 4, device=DEVICE) * prior)
    optimizer = torch.optim.Adam([lambdas_raw], lr=lr)
    
    calib_batches = {}
    for task_name, loader in calib_loaders.items():
        x, _ = next(iter(loader))
        calib_batches[task_name] = x.to(DEVICE)
        
    for step in range(steps):
        optimizer.zero_grad()
        update_model_weights(active_model, base_dict, task_vectors, lambdas_raw, quantize, bits)
        
        loss = 0.0
        for k, task_name in enumerate(calib_loaders.keys()):
            active_model.head.weight.data.copy_(expert_heads[k]['weight'].to(DEVICE))
            active_model.head.bias.data.copy_(expert_heads[k]['bias'].to(DEVICE))
            
            x = calib_batches[task_name]
            outputs = active_model(x)
            entropy = softmax_entropy(outputs).mean(0)
            loss += entropy
            
        loss.backward()
        optimizer.step()
        
    return lambdas_raw.detach().clone()

def evaluate_model(active_model, base_dict, task_vectors, lambdas_raw, expert_heads, quantize, bits, test_loaders):
    active_model.eval()
    task_accs = {}
    with torch.no_grad():
        update_model_weights(active_model, base_dict, task_vectors, lambdas_raw, quantize, bits)
        for k, task_name in enumerate(test_loaders.keys()):
            active_model.head.weight.data.copy_(expert_heads[k]['weight'].to(DEVICE))
            active_model.head.bias.data.copy_(expert_heads[k]['bias'].to(DEVICE))
            
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
    seed = 42
    set_seed(seed)
    
    print("Pre-loading datasets (test_size=1000 for standard comparison)...")
    test_datasets = {task: get_cached_dataset(task, split='test', size=1000, seed=999) for task in TASKS}
    test_loaders = {task: DataLoader(test_datasets[task], batch_size=128, shuffle=False) for task in TASKS}
    
    calib_datasets = {task: get_cached_dataset(task, split='test', size=16, seed=777) for task in TASKS}
    calib_loaders = {task: DataLoader(calib_datasets[task], batch_size=16, shuffle=False) for task in TASKS}
    
    train_datasets = {task: get_cached_dataset(task, split='train', size=512, seed=seed) for task in TASKS}
    
    # Train SWA experts
    for task in TASKS:
        train_expert_swa(task, seed, train_datasets[task], epochs=15, swa_start=10)
        
    # Load base model structure
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(DEVICE)
    active_model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(DEVICE)
    active_model.head = torch.nn.Linear(192, 10).to(DEVICE)
    
    for name, _ in list(active_model.named_parameters()):
        if get_layer_group(name) >= 0:
            del_attr(active_model, name.split('.'))
            
    base_dict = {n: p.clone().detach().to(DEVICE) for n, p in base_model.named_parameters()}
    
    print(f"\n--- Loading SWA Experts for Evaluation ---")
    expert_heads = []
    expert_dicts = []
    for task in TASKS:
        expert_file = f"checkpoints/expert_{task}_seed{seed}_swa.pt"
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
                tv[n] = expert_dicts[k][n].to(DEVICE) - base_dict[n]
        task_vectors.append(tv)
        
    results = {}
    for bits in [8, 4]:
        print(f"\nEvaluating {bits}-bit weight quantization:")
        results[bits] = {}
        
        # 1. NaiveUniform
        lambdas_uniform = torch.ones(14, 4, device=DEVICE) * 0.3
        accs_uniform = evaluate_model(active_model, base_dict, task_vectors, lambdas_uniform, expert_heads, quantize=True, bits=bits, test_loaders=test_loaders)
        avg_uniform = np.mean(list(accs_uniform.values()))
        print(f"  [SWA + NaiveUniform] Task Accuracies: {accs_uniform}")
        print(f"  [SWA + NaiveUniform] Average Multi-Task Accuracy: {avg_uniform:.2f}%")
        results[bits]['uniform'] = {'task_accs': accs_uniform, 'avg_acc': avg_uniform}
        
        # 2. FlatQ-Merge (STE optimized)
        lambdas_opt = test_time_optimize(active_model, base_dict, task_vectors, expert_heads, quantize=True, bits=bits, calib_loaders=calib_loaders)
        accs_opt = evaluate_model(active_model, base_dict, task_vectors, lambdas_opt, expert_heads, quantize=True, bits=bits, test_loaders=test_loaders)
        avg_opt = np.mean(list(accs_opt.values()))
        print(f"  [SWA + FlatQ-Merge] Task Accuracies: {accs_opt}")
        print(f"  [SWA + FlatQ-Merge] Average Multi-Task Accuracy: {avg_opt:.2f}%")
        results[bits]['opt'] = {'task_accs': accs_opt, 'avg_acc': avg_opt}
        
        # 3. Individual-Quantized SWA
        indiv_accs = {}
        with torch.no_grad():
            for k, task_name in enumerate(test_loaders.keys()):
                # Create a model with quantized expert weights
                task_model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(DEVICE)
                task_model.head = torch.nn.Linear(192, 10).to(DEVICE)
                task_model.head.weight.data.copy_(expert_heads[k]['weight'].to(DEVICE))
                task_model.head.bias.data.copy_(expert_heads[k]['bias'].to(DEVICE))
                
                # Apply PTQ to the individual expert weights
                for n, p in list(task_model.named_parameters()):
                    if get_layer_group(n) >= 0 and 'bias' not in n:
                        p_quant = quantize_ste(expert_dicts[k][n].to(DEVICE), bits)
                        # We must delete parameter first to assign raw tensor or assign to .data
                        p.data.copy_(p_quant)
                
                task_model.eval()
                correct = 0
                total = 0
                for x, y in test_loaders[task_name]:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    outputs = task_model(x)
                    preds = outputs.argmax(dim=-1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
                acc = correct / total * 100.0
                indiv_accs[task_name] = acc
        avg_indiv = np.mean(list(indiv_accs.values()))
        print(f"  [SWA Individual Quantized] Task Accuracies: {indiv_accs}")
        print(f"  [SWA Individual Quantized] Average Accuracy: {avg_indiv:.2f}%")
        results[bits]['individual'] = {'task_accs': indiv_accs, 'avg_acc': avg_indiv}
        
    with open('swa_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSuccessfully saved SWA results to swa_results.json.")

if __name__ == '__main__':
    main()
