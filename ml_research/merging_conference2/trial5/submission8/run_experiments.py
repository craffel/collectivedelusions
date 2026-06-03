import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

# Set seed for reproducibility
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False

class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool
        )
        self.heads = nn.ModuleDict({
            'mnist': nn.Linear(512, 10),
            'fmnist': nn.Linear(512, 10),
            'cifar10': nn.Linear(512, 10)
        })
        
    def forward(self, x, task_name=None):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        if task_name is not None:
            return self.heads[task_name](features)
        return features

class GatingNetwork(nn.Module):
    def __init__(self, channels=64, num_tasks=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, 32),
            nn.ReLU(),
            nn.Linear(32, num_tasks)
        )
        
    def forward(self, x):
        s_spatial = x.mean(dim=(2, 3))
        logits = self.fc(s_spatial)
        probs = F.softmax(logits, dim=-1)
        alpha = probs.mean(dim=0)
        return alpha

def get_dataloaders(batch_size=128):
    # Standard transform for CIFAR-10 (RGB)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Grayscale conversion and 3-channel replica for MNIST & FashionMNIST
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download datasets
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
    
    fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=mnist_transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=mnist_transform)
    
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Subsets (as per the papers)
    mnist_ft = Subset(mnist_train, list(range(3000)))  # paper says 3000 images
    mnist_cal = Subset(mnist_train, list(range(5000, 5128)))  # 128 samples calibration
    
    fmnist_ft = Subset(fmnist_train, list(range(3000)))
    fmnist_cal = Subset(fmnist_train, list(range(5000, 5128)))
    
    cifar_ft = Subset(cifar_train, list(range(3000)))
    cifar_cal = Subset(cifar_train, list(range(5000, 5128)))
    
    train_loaders = {
        'mnist': DataLoader(mnist_ft, batch_size=batch_size, shuffle=True, num_workers=2),
        'fmnist': DataLoader(fmnist_ft, batch_size=batch_size, shuffle=True, num_workers=2),
        'cifar10': DataLoader(cifar_ft, batch_size=batch_size, shuffle=True, num_workers=2)
    }
    
    cal_loaders = {
        'mnist': DataLoader(mnist_cal, batch_size=batch_size, shuffle=False),
        'fmnist': DataLoader(fmnist_cal, batch_size=batch_size, shuffle=False),
        'cifar10': DataLoader(cifar_cal, batch_size=batch_size, shuffle=False)
    }
    
    test_loaders = {
        'mnist': DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2),
        'fmnist': DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, num_workers=2),
        'cifar10': DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=2)
    }
    
    return train_loaders, cal_loaders, test_loaders

def evaluate_model(model, loader, task_name, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, task_name)
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100. * correct / total

def train_expert(task_name, train_loader, val_loader, epochs=5, device='cuda'):
    print(f"--- Training expert for task: {task_name} ---")
    model = MultiTaskModel().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, task_name)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        scheduler.step()
        train_acc = 100. * correct / total
        train_loss = total_loss / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
    test_acc = evaluate_model(model, val_loader, task_name, device)
    print(f"Final Test Acc for {task_name}: {test_acc:.2f}%\n")
    return model

def merge_models(expert_models, method='WA', lambda_val=0.3):
    device = next(expert_models[0].parameters()).device
    merged = MultiTaskModel().to(device)
    base = MultiTaskModel().to(device)
    
    state_dicts = [m.state_dict() for m in expert_models]
    base_dict = base.state_dict()
    merged_dict = merged.state_dict()
    
    for key in merged_dict.keys():
        if 'heads.' in key:
            if 'mnist' in key:
                merged_dict[key] = state_dicts[0][key].clone()
            elif 'fmnist' in key:
                merged_dict[key] = state_dicts[1][key].clone()
            elif 'cifar10' in key:
                merged_dict[key] = state_dicts[2][key].clone()
            continue
            
        if method == 'WA':
            merged_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
        elif method == 'TA':
            if state_dicts[0][key].is_floating_point():
                task_vectors = [sd[key] - base_dict[key] for sd in state_dicts]
                merged_dict[key] = base_dict[key] + lambda_val * sum(task_vectors)
            else:
                merged_dict[key] = state_dicts[0][key].clone()
                
    merged.load_state_dict(merged_dict)
    return merged

def get_bn_layers(model):
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))
    return bn_layers

def collect_expert_statistics(expert_models, cal_loaders, device='cuda'):
    tasks = ['mnist', 'fmnist', 'cifar10']
    profiles = {}
    
    for idx, (task, model) in enumerate(zip(tasks, expert_models)):
        model.eval()
        loader = cal_loaders[task]
        bn_layers = get_bn_layers(model)
        layer_activations = {name: [] for name, _ in bn_layers}
        
        handles = []
        def make_hook(name):
            def hook_fn(module, inputs, outputs):
                layer_activations[name].append(outputs.detach().cpu())
            return hook_fn
            
        for name, module in bn_layers:
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)
            
        with torch.no_grad():
            for x, _ in loader:
                _ = model(x.to(device))
                
        for handle in handles:
            handle.remove()
            
        for name, _ in bn_layers:
            acts = torch.cat(layer_activations[name], dim=0) # (128, C, H, W)
            fft_acts = torch.fft.fft2(acts)
            mags = torch.abs(fft_acts)
            profile = mags.mean(dim=(0, 1)) # (H, W)
            
            if name not in profiles:
                profiles[name] = {}
            profiles[name][idx] = profile
            
    return profiles

def calibrate_merged_model(merged_model, expert_profiles, cal_loaders, device='cuda', gamma_max=5.0):
    merged_model.eval()
    bn_layers = get_bn_layers(merged_model)
    layer_activations = {name: [] for name, _ in bn_layers}
    
    handles = []
    def make_hook(name):
        def hook_fn(module, inputs, outputs):
            layer_activations[name].append(outputs.detach().cpu())
        return hook_fn
        
    for name, module in bn_layers:
        handle = module.register_forward_hook(make_hook(name))
        handles.append(handle)
        
    tasks = ['mnist', 'fmnist', 'cifar10']
    with torch.no_grad():
        for task in tasks:
            for x, _ in cal_loaders[task]:
                _ = merged_model(x.to(device))
                
    for handle in handles:
        handle.remove()
        
    cal_filters = {}
    for name, _ in bn_layers:
        acts = torch.cat(layer_activations[name], dim=0)
        fft_acts = torch.fft.fft2(acts)
        mags = torch.abs(fft_acts)
        profile_merged = mags.mean(dim=(0, 1)) # (H, W)
        
        profile_target = sum(expert_profiles[name][i] for i in range(3)) / 3.0
        
        gamma = profile_target / (profile_merged + 1e-5)
        gamma_star = torch.clamp(gamma, 1.0 / gamma_max, gamma_max)
        
        # Compute spatial filter K via IDFT
        gamma_complex = torch.complex(gamma_star, torch.zeros_like(gamma_star))
        K = torch.real(torch.fft.ifft2(gamma_complex)) # (H, W)
        
        cal_filters[name] = {
            'gamma': gamma_star,
            'K': K
        }
        
    return cal_filters

def calibrate_tc_merged_model(merged_model, expert_profiles, cal_loaders, device='cuda', gamma_max=5.0):
    tasks = ['mnist', 'fmnist', 'cifar10']
    bn_layers = get_bn_layers(merged_model)
    cal_filters_tc = {task: {} for task in tasks}
    
    for idx, task in enumerate(tasks):
        merged_model.eval()
        layer_activations = {name: [] for name, _ in bn_layers}
        
        handles = []
        def make_hook(name):
            def hook_fn(module, inputs, outputs):
                layer_activations[name].append(outputs.detach().cpu())
            return hook_fn
            
        for name, module in bn_layers:
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)
            
        with torch.no_grad():
            for x, _ in cal_loaders[task]:
                _ = merged_model(x.to(device))
                
        for handle in handles:
            handle.remove()
            
        for name, _ in bn_layers:
            acts = torch.cat(layer_activations[name], dim=0)
            fft_acts = torch.fft.fft2(acts)
            mags = torch.abs(fft_acts)
            profile_merged = mags.mean(dim=(0, 1)) # (H, W)
            
            profile_target = expert_profiles[name][idx]
            
            gamma = profile_target / (profile_merged + 1e-5)
            gamma_star = torch.clamp(gamma, 1.0 / gamma_max, gamma_max)
            
            gamma_complex = torch.complex(gamma_star, torch.zeros_like(gamma_star))
            K = torch.real(torch.fft.ifft2(gamma_complex)) # (H, W)
            
            cal_filters_tc[task][name] = {
                'gamma': gamma_star,
                'K': K
            }
            
    return cal_filters_tc

def apply_fdsa_calibration(model, cal_filters):
    hooks = []
    bn_layers = get_bn_layers(model)
    
    for name, module in bn_layers:
        gamma = cal_filters[name]['gamma'].to(next(model.parameters()).device)
        
        def make_hook(g):
            def hook_fn(module, inputs, outputs):
                fft_out = torch.fft.fft2(outputs)
                fft_cal = fft_out * g
                return torch.real(torch.fft.ifft2(fft_cal))
            return hook_fn
            
        handle = module.register_forward_hook(make_hook(gamma))
        hooks.append(handle)
        
    return hooks

def apply_zosf_calibration(model, cal_filters):
    hooks = []
    bn_layers = get_bn_layers(model)
    
    for name, module in bn_layers:
        K = cal_filters[name]['K'].to(next(model.parameters()).device)
        
        def make_hook(k_f):
            def hook_fn(module, inputs, outputs):
                B, C, H, W = outputs.shape
                padded = F.pad(outputs, (0, W-1, 0, H-1), mode='circular')
                K_conv = k_f.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
                return F.conv2d(padded, K_conv, groups=C)
            return hook_fn
            
        handle = module.register_forward_hook(make_hook(K))
        hooks.append(handle)
        
    return hooks

def train_gating_network(merged_model, cal_loaders, device='cuda'):
    print("--- Training Gating Network for IA-ZOSF ---")
    merged_model.eval()
    bn_layers = get_bn_layers(merged_model)
    first_bn_name, first_bn_module = bn_layers[0]
    
    collected_activations = []
    collected_labels = []
    
    layer_activations = []
    def make_hook():
        def hook_fn(module, inputs, outputs):
            layer_activations.append(outputs.detach().cpu())
        return hook_fn
        
    tasks = ['mnist', 'fmnist', 'cifar10']
    for idx, task in enumerate(tasks):
        layer_activations = []
        handle = first_bn_module.register_forward_hook(make_hook())
        
        with torch.no_grad():
            for x, _ in cal_loaders[task]:
                _ = merged_model(x.to(device))
                
        handle.remove()
        
        acts = torch.cat(layer_activations, dim=0)
        s = acts.mean(dim=(2, 3))
        collected_activations.append(s)
        collected_labels.append(torch.full((s.size(0),), idx, dtype=torch.long))
        
    X_train = torch.cat(collected_activations, dim=0).to(device)
    y_train = torch.cat(collected_labels, dim=0).to(device)
    
    gating_net = GatingNetwork(channels=X_train.size(1), num_tasks=3).to(device)
    optimizer = torch.optim.Adam(gating_net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    gating_net.train()
    for epoch in range(100):
        optimizer.zero_grad()
        logits = gating_net.fc(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            preds = logits.argmax(dim=-1)
            acc = (preds == y_train).float().mean().item() * 100.0
            print(f"Gating Epoch {epoch+1}/100 | Loss: {loss.item():.4f} | Accuracy: {acc:.2f}%")
            
    gating_net.eval()
    return gating_net

def apply_iazosf_calibration(model, cal_filters_tc, gating_net):
    hooks = []
    bn_layers = get_bn_layers(model)
    first_bn_name = bn_layers[0][0]
    
    dynamic_alpha = {}
    tasks = ['mnist', 'fmnist', 'cifar10']
    
    for name, module in bn_layers:
        K_tasks = [cal_filters_tc[task][name]['K'].to(next(model.parameters()).device) for task in tasks]
        
        if name == first_bn_name:
            def make_first_hook(k_ts):
                def hook_fn(module, inputs, outputs):
                    alpha = gating_net(outputs)
                    dynamic_alpha['alpha'] = alpha
                    
                    K_blended = torch.stack(k_ts, dim=0)
                    K_fused = torch.sum(K_blended * alpha.view(3, 1, 1), dim=0)
                    
                    B, C, H, W = outputs.shape
                    padded = F.pad(outputs, (0, W-1, 0, H-1), mode='circular')
                    K_conv = K_fused.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
                    return F.conv2d(padded, K_conv, groups=C)
                return hook_fn
            handle = module.register_forward_hook(make_first_hook(K_tasks))
        else:
            def make_subsequent_hook(k_ts):
                def hook_fn(module, inputs, outputs):
                    alpha = dynamic_alpha.get('alpha', torch.tensor([1/3, 1/3, 1/3], device=outputs.device))
                    
                    K_blended = torch.stack(k_ts, dim=0)
                    K_fused = torch.sum(K_blended * alpha.view(3, 1, 1), dim=0)
                    
                    B, C, H, W = outputs.shape
                    padded = F.pad(outputs, (0, W-1, 0, H-1), mode='circular')
                    K_conv = K_fused.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
                    return F.conv2d(padded, K_conv, groups=C)
                return hook_fn
            handle = module.register_forward_hook(make_subsequent_hook(K_tasks))
            
        hooks.append(handle)
        
    return hooks

def calibrate_sptaac(merged_model, expert_models, cal_loaders, device='cuda'):
    bn_layers = get_bn_layers(merged_model)
    expert_stds = {name: [] for name, _ in bn_layers}
    merged_std = {name: 0.0 for name, _ in bn_layers}
    
    tasks = ['mnist', 'fmnist', 'cifar10']
    for idx, (task, model) in enumerate(zip(tasks, expert_models)):
        model.eval()
        layer_activations = {name: [] for name, _ in bn_layers}
        handles = []
        def make_hook(name):
            def hook_fn(module, inputs, outputs):
                layer_activations[name].append(outputs.detach().cpu())
            return hook_fn
        # Register hooks on the corresponding layers of the expert model
        expert_bn_layers = get_bn_layers(model)
        for (name, module), (exp_name, exp_module) in zip(bn_layers, expert_bn_layers):
            handle = exp_module.register_forward_hook(make_hook(name))
            handles.append(handle)
        with torch.no_grad():
            for x, _ in cal_loaders[task]:
                _ = model(x.to(device))
        for handle in handles:
            handle.remove()
        for name, _ in bn_layers:
            acts = torch.cat(layer_activations[name], dim=0)
            std = torch.std(acts).item()
            expert_stds[name].append(std)
            
    merged_model.eval()
    layer_activations = {name: [] for name, _ in bn_layers}
    handles = []
    def make_hook(name):
        def hook_fn(module, inputs, outputs):
            layer_activations[name].append(outputs.detach().cpu())
        return hook_fn
    for name, module in bn_layers:
        handle = module.register_forward_hook(make_hook(name))
        handles.append(handle)
    with torch.no_grad():
        for task in tasks:
            for x, _ in cal_loaders[task]:
                _ = merged_model(x.to(device))
    for handle in handles:
        handle.remove()
    for name, _ in bn_layers:
        acts = torch.cat(layer_activations[name], dim=0)
        std = torch.std(acts).item()
        merged_std[name] = std
        
    sptaac_scales = {}
    for name, _ in bn_layers:
        target_std = sum(expert_stds[name]) / 3.0
        scale = target_std / (merged_std[name] + 1e-5)
        sptaac_scales[name] = scale
        
    return sptaac_scales

def apply_sptaac_calibration(model, sptaac_scales):
    hooks = []
    bn_layers = get_bn_layers(model)
    for name, module in bn_layers:
        scale = sptaac_scales[name]
        def make_hook(s):
            def hook_fn(module, inputs, outputs):
                return outputs * s
            return hook_fn
        handle = module.register_forward_hook(make_hook(scale))
        hooks.append(handle)
    return hooks

def remove_hooks(handles):
    for handle in handles:
        handle.remove()

def profile_latency(model, device='cuda', compiled=False, name='Model'):
    model.eval()
    x = torch.randn(128, 3, 32, 32).to(device)
    
    if compiled:
        print(f"Compiling {name} with torch.compile...")
        model_compiled = torch.compile(model)
        # Warmup
        with torch.no_grad():
            _ = model_compiled(x, 'mnist')
        
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(50):
                _ = model_compiled(x, 'mnist')
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
    else:
        # Warmup
        with torch.no_grad():
            _ = model(x, 'mnist')
            
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(50):
                _ = model(x, 'mnist')
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        
    return (end - start) * 1000.0 / 50.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lambda_val', type=float, default=0.3)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataloaders
    train_loaders, cal_loaders, test_loaders = get_dataloaders()
    
    # Train/load experts
    tasks = ['mnist', 'fmnist', 'cifar10']
    expert_models = []
    
    for task in tasks:
        ckpt_path = f"{task}_expert.pth"
        if os.path.exists(ckpt_path):
            print(f"Loading expert for {task} from {ckpt_path}")
            model = MultiTaskModel().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            expert_models.append(model)
        else:
            model = train_expert(task, train_loaders[task], test_loaders[task], epochs=args.epochs, device=device)
            torch.save(model.state_dict(), ckpt_path)
            expert_models.append(model)
            
    # Record expert individual accuracies
    oracle_accs = {}
    for task, model in zip(tasks, expert_models):
        acc = evaluate_model(model, test_loaders[task], task, device)
        oracle_accs[task] = acc
    oracle_avg = np.mean(list(oracle_accs.values()))
    print(f"Oracle Experts Average Accuracy: {oracle_avg:.2f}% ({oracle_accs})\n")
    
    # Collect expert statistics
    print("Collecting expert activation statistics in the frequency domain...")
    expert_profiles = collect_expert_statistics(expert_models, cal_loaders, device)
    
    # We will evaluate two merging methods: Weight Averaging (WA) and Task Arithmetic (TA)
    merging_methods = ['WA', 'TA']
    results = {
        'Oracle': {**oracle_accs, 'Average': oracle_avg}
    }
    
    for method in merging_methods:
        print(f"=== Evaluating Merging Method: {method} ===")
        merged_model = merge_models(expert_models, method=method, lambda_val=args.lambda_val).to(device)
        
        # 1. Uncalibrated
        uncal_accs = {}
        for task in tasks:
            acc = evaluate_model(merged_model, test_loaders[task], task, device)
            uncal_accs[task] = acc
        uncal_avg = np.mean(list(uncal_accs.values()))
        print(f"Uncalibrated {method} Average: {uncal_avg:.2f}% ({uncal_accs})")
        results[f'{method}_Uncalibrated'] = {**uncal_accs, 'Average': uncal_avg}
        
        # 2. SP-TAAC Calibration
        print("Calibrating via SP-TAAC...")
        sptaac_scales = calibrate_sptaac(merged_model, expert_models, cal_loaders, device)
        sptaac_handles = apply_sptaac_calibration(merged_model, sptaac_scales)
        
        sptaac_accs = {}
        for task in tasks:
            acc = evaluate_model(merged_model, test_loaders[task], task, device)
            sptaac_accs[task] = acc
        sptaac_avg = np.mean(list(sptaac_accs.values()))
        print(f"SP-TAAC Calibrated {method} Average: {sptaac_avg:.2f}% ({sptaac_accs})")
        results[f'{method}_SP-TAAC'] = {**sptaac_accs, 'Average': sptaac_avg}
        remove_hooks(sptaac_handles)
        
        # Calibrate FDSA (frequency filters)
        print("Calibrating via FDSA/ZOSF...")
        cal_filters = calibrate_merged_model(merged_model, expert_profiles, cal_loaders, device)
        
        # 3. FDSA Calibration (FFT-based)
        fdsa_handles = apply_fdsa_calibration(merged_model, cal_filters)
        fdsa_accs = {}
        for task in tasks:
            acc = evaluate_model(merged_model, test_loaders[task], task, device)
            fdsa_accs[task] = acc
        fdsa_avg = np.mean(list(fdsa_accs.values()))
        print(f"FDSA Calibrated {method} Average: {fdsa_avg:.2f}% ({fdsa_accs})")
        results[f'{method}_FDSA'] = {**fdsa_accs, 'Average': fdsa_avg}
        remove_hooks(fdsa_handles)
        
        # 4. ZOSF Calibration (Our spatial Circular Conv Fusion)
        zosf_handles = apply_zosf_calibration(merged_model, cal_filters)
        zosf_accs = {}
        for task in tasks:
            acc = evaluate_model(merged_model, test_loaders[task], task, device)
            zosf_accs[task] = acc
        zosf_avg = np.mean(list(zosf_accs.values()))
        print(f"ZOSF (Ours, Circular Conv) {method} Average: {zosf_avg:.2f}% ({zosf_accs})")
        results[f'{method}_ZOSF'] = {**zosf_accs, 'Average': zosf_avg}
        
        # Verify Exact Mathematical Parity
        max_diff = 0.0
        for task in tasks:
            diff = abs(fdsa_accs[task] - zosf_accs[task])
            max_diff = max(max_diff, diff)
        print(f"Exact Mathematical Parity Verification: Max accuracy discrepancy = {max_diff:.6f}%")
        results[f'{method}_ParityVerified'] = (max_diff < 1e-5)
        remove_hooks(zosf_handles)

        # 5. TC-FDSA Calibration (Task-Conditional)
        print("Calibrating via TC-FDSA (Task-Conditional)...")
        cal_filters_tc = calibrate_tc_merged_model(merged_model, expert_profiles, cal_loaders, device)
        tc_fdsa_accs = {}
        for task in tasks:
            tc_fdsa_handles = apply_fdsa_calibration(merged_model, cal_filters_tc[task])
            acc = evaluate_model(merged_model, test_loaders[task], task, device)
            tc_fdsa_accs[task] = acc
            remove_hooks(tc_fdsa_handles)
        tc_fdsa_avg = np.mean(list(tc_fdsa_accs.values()))
        print(f"TC-FDSA Calibrated {method} Average: {tc_fdsa_avg:.2f}% ({tc_fdsa_accs})")
        results[f'{method}_TC-FDSA'] = {**tc_fdsa_accs, 'Average': tc_fdsa_avg}

        # 6. TC-ZOSF Calibration (Task-Conditional spatial circular conv)
        print("Calibrating via TC-ZOSF (Task-Conditional spatial circular conv)...")
        tc_zosf_accs = {}
        for task in tasks:
            tc_zosf_handles = apply_zosf_calibration(merged_model, cal_filters_tc[task])
            acc = evaluate_model(merged_model, test_loaders[task], task, device)
            tc_zosf_accs[task] = acc
            remove_hooks(tc_zosf_handles)
        tc_zosf_avg = np.mean(list(tc_zosf_accs.values()))
        print(f"TC-ZOSF (Ours Fused) {method} Average: {tc_zosf_avg:.2f}% ({tc_zosf_accs})")
        results[f'{method}_TC-ZOSF'] = {**tc_zosf_accs, 'Average': tc_zosf_avg}

        # Verify TC Exact Mathematical Parity
        max_diff_tc = 0.0
        for task in tasks:
            diff = abs(tc_fdsa_accs[task] - tc_zosf_accs[task])
            max_diff_tc = max(max_diff_tc, diff)
        print(f"TC Exact Mathematical Parity Verification: Max discrepancy = {max_diff_tc:.6f}%")
        results[f'{method}_TC_ParityVerified'] = (max_diff_tc < 1e-5)

        # 7. IA-ZOSF Calibration (Input-Adaptive Spectral Filter Fusion - Ours)
        gating_net = train_gating_network(merged_model, cal_loaders, device=device)
        print("Calibrating via IA-ZOSF (Input-Adaptive spatial circular conv)...")
        iazosf_handles = apply_iazosf_calibration(merged_model, cal_filters_tc, gating_net)
        iazosf_accs = {}
        for task in tasks:
            acc = evaluate_model(merged_model, test_loaders[task], task, device)
            iazosf_accs[task] = acc
        iazosf_avg = np.mean(list(iazosf_accs.values()))
        print(f"IA-ZOSF (Ours, Input-Adaptive) {method} Average: {iazosf_avg:.2f}% ({iazosf_accs})")
        results[f'{method}_IA-ZOSF'] = {**iazosf_accs, 'Average': iazosf_avg}
        remove_hooks(iazosf_handles)
        print()
        
    # Latency Profiling
    print("=== Profiling Inference Latencies ===")
    merged_model = merge_models(expert_models, method='WA').to(device)
    
    # Profile uncalibrated base
    lat_base = profile_latency(merged_model, device=device, compiled=False, name='Uncalibrated Base')
    lat_base_comp = profile_latency(merged_model, device=device, compiled=True, name='Uncalibrated Base')
    
    # Profile online FDSA (FFT hooks)
    cal_filters = calibrate_merged_model(merged_model, expert_profiles, cal_loaders, device)
    fdsa_handles = apply_fdsa_calibration(merged_model, cal_filters)
    lat_fdsa = profile_latency(merged_model, device=device, compiled=False, name='FDSA Hooked')
    # Try compiling FDSA (expecting failure, graph breaks, or high compilation overhead if it runs)
    try:
        lat_fdsa_comp = profile_latency(merged_model, device=device, compiled=True, name='FDSA Hooked')
    except Exception as e:
        print(f"FDSA compilation failed or was skipped due to: {e}")
        lat_fdsa_comp = -1.0 # Failed
    remove_hooks(fdsa_handles)
    
    # Profile ZOSF (fused circular conv hooks)
    zosf_handles = apply_zosf_calibration(merged_model, cal_filters)
    lat_zosf = profile_latency(merged_model, device=device, compiled=False, name='ZOSF Fused')
    lat_zosf_comp = profile_latency(merged_model, device=device, compiled=True, name='ZOSF Fused')
    remove_hooks(zosf_handles)

    # Profile TC-ZOSF (using task-specific filter for mnist)
    cal_filters_tc = calibrate_tc_merged_model(merged_model, expert_profiles, cal_loaders, device)
    tc_zosf_handles = apply_zosf_calibration(merged_model, cal_filters_tc['mnist'])
    lat_tc_zosf = profile_latency(merged_model, device=device, compiled=False, name='TC-ZOSF Fused')
    lat_tc_zosf_comp = profile_latency(merged_model, device=device, compiled=True, name='TC-ZOSF Fused')
    remove_hooks(tc_zosf_handles)

    # Profile IA-ZOSF (Input-Adaptive)
    gating_net = train_gating_network(merged_model, cal_loaders, device=device)
    iazosf_handles = apply_iazosf_calibration(merged_model, cal_filters_tc, gating_net)
    lat_iazosf = profile_latency(merged_model, device=device, compiled=False, name='IA-ZOSF Fused')
    try:
        lat_iazosf_comp = profile_latency(merged_model, device=device, compiled=True, name='IA-ZOSF Fused')
    except Exception as e:
        print(f"IA-ZOSF compilation failed or was skipped due to: {e}")
        lat_iazosf_comp = -1.0
    remove_hooks(iazosf_handles)
    
    print(f"\nLatency Profiles (ms per batch, batch size = 128):")
    print(f"Uncalibrated Base: Uncompiled = {lat_base:.2f} ms | Compiled = {lat_base_comp:.2f} ms")
    print(f"FDSA (FFT Hooked): Uncompiled = {lat_fdsa:.2f} ms | Compiled = {lat_fdsa_comp:.2f} ms")
    print(f"ZOSF (Ours Fused): Uncompiled = {lat_zosf:.2f} ms | Compiled = {lat_zosf_comp:.2f} ms")
    print(f"TC-ZOSF (TC Fused): Uncompiled = {lat_tc_zosf:.2f} ms | Compiled = {lat_tc_zosf_comp:.2f} ms")
    print(f"IA-ZOSF (IA Fused): Uncompiled = {lat_iazosf:.2f} ms | Compiled = {lat_iazosf_comp:.2f} ms")
    
    results['Latency'] = {
        'Uncalibrated': {'uncompiled': lat_base, 'compiled': lat_base_comp},
        'FDSA': {'uncompiled': lat_fdsa, 'compiled': lat_fdsa_comp},
        'ZOSF': {'uncompiled': lat_zosf, 'compiled': lat_zosf_comp},
        'TC-ZOSF': {'uncompiled': lat_tc_zosf, 'compiled': lat_tc_zosf_comp},
        'IA-ZOSF': {'uncompiled': lat_iazosf, 'compiled': lat_iazosf_comp}
    }
    
    # Save results to JSON
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved results to results.json.")
    
    # Generate Plots
    print("Generating performance plots...")
    
    # Plot 1: Multi-Task Model Merging Performance
    methods_plot = ['Uncalibrated', 'SP-TAAC', 'FDSA (Global)', 'ZOSF (Global)', 'TC-ZOSF (Ours)', 'IA-ZOSF (Ours)']
    wa_vals = [
        results['WA_Uncalibrated']['Average'],
        results['WA_SP-TAAC']['Average'],
        results['WA_FDSA']['Average'],
        results['WA_ZOSF']['Average'],
        results['WA_TC-ZOSF']['Average'],
        results['WA_IA-ZOSF']['Average']
    ]
    ta_vals = [
        results['TA_Uncalibrated']['Average'],
        results['TA_SP-TAAC']['Average'],
        results['TA_FDSA']['Average'],
        results['TA_ZOSF']['Average'],
        results['TA_TC-ZOSF']['Average'],
        results['TA_IA-ZOSF']['Average']
    ]
    
    x_indices = np.arange(len(methods_plot))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x_indices - width/2, wa_vals, width, label='Weight Averaging (WA)', color='#1f77b4')
    rects2 = ax.bar(x_indices + width/2, ta_vals, width, label='Task Arithmetic (TA)', color='#ff7f0e')
    
    # Add a dashed line for Oracle Upper Bound
    ax.axhline(y=oracle_avg, color='r', linestyle='--', label=f'Oracle Expert Avg ({oracle_avg:.2f}%)')
    
    ax.set_ylabel('Average Test Accuracy (%)')
    ax.set_title('Multi-Task Model Merging Performance Comparison')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(methods_plot)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    
    # Label accuracy values on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('merging_performance.png', dpi=300)
    plt.close()
    
    # Plot 2: Inference Latency Profiles
    models_lat = ['Uncalibrated Base', 'FDSA (FFT Hooked)', 'ZOSF (Ours Fused)', 'TC-ZOSF (Ours Fused)', 'IA-ZOSF (Ours Fused)']
    uncomp_lats = [lat_base, lat_fdsa, lat_zosf, lat_tc_zosf, lat_iazosf]
    comp_lats = [lat_base_comp, lat_fdsa_comp, lat_zosf_comp, lat_tc_zosf_comp, lat_iazosf_comp]
    
    x_indices_lat = np.arange(len(models_lat))
    fig, ax = plt.subplots(figsize=(9, 5))
    rects_uncomp = ax.bar(x_indices_lat - width/2, uncomp_lats, width, label='Uncompiled', color='#aec7e8')
    # Filter out failed compilation for FDSA if any
    clean_comp_lats = [l if l > 0 else 0 for l in comp_lats]
    rects_comp = ax.bar(x_indices_lat + width/2, clean_comp_lats, width, label='torch.compile', color='#1f77b4')
    
    ax.set_ylabel('Latency (ms per batch of 128)')
    ax.set_title('Inference Latency Profile Comparison')
    ax.set_xticks(x_indices_lat)
    ax.set_xticklabels(models_lat)
    ax.legend()
    
    # Label latencies on bars
    for rect in rects_uncomp:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}ms',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
        
    for rect in rects_comp:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}ms',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        else:
            ax.annotate('N/A',
                        xy=(rect.get_x() + rect.get_width() / 2, 5.0),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', color='red', weight='bold')
            
    plt.tight_layout()
    plt.savefig('inference_latency_profile.png', dpi=300)
    plt.close()
    
    print("Done! Performance plot saved as 'merging_performance.png' and latency plot saved as 'inference_latency_profile.png'.")

if __name__ == '__main__':
    main()
