import os
import time
import json
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_dataloaders(batch_size=128):
    class RepeatChannels:
        def __call__(self, x):
            return x.repeat(3, 1, 1)

    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        RepeatChannels(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=transform_gray)
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=False, transform=transform_gray)

    fmnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=False, transform=transform_gray)
    fmnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=False, transform=transform_gray)

    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_color)
    cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_color)

    loaders = {
        "mnist": (
            DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        ),
        "fmnist": (
            DataLoader(fmnist_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        ),
        "cifar10": (
            DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        ),
    }
    return loaders

def check_and_train_experts(loaders, device):
    tasks = ["mnist", "fmnist", "cifar10"]
    expert_states = {}
    
    for t in tasks:
        checkpoint_path = f"checkpoints/expert_{t}.pth"
        if not os.path.exists(checkpoint_path):
            epochs = 2 if t in ["mnist", "fmnist"] else 5
            # Import train_expert inline to avoid circular issues
            from train_experts import train_expert
            train_expert(t, loaders[t][0], loaders[t][1], epochs=epochs, device=device)
            
        expert_states[t] = torch.load(checkpoint_path, map_location="cpu")
        
    return expert_states

def merge_models(progenitor_state, expert_states, merge_method="wa", lambda_val=0.5):
    merged_state = {}
    keys = list(progenitor_state.keys())
    
    for key in keys:
        if key.startswith("fc."):
            continue
            
        if merge_method == "wa":
            stacked = torch.stack([expert_states[t][key].float() for t in expert_states])
            merged_state[key] = torch.mean(stacked, dim=0)
        elif merge_method == "ta":
            update_sum = torch.zeros_like(progenitor_state[key].float())
            for t in expert_states:
                update_sum += (expert_states[t][key].float() - progenitor_state[key].float())
            merged_state[key] = progenitor_state[key].float() + lambda_val * update_sum
            
    return merged_state

def get_eval_model(merged_backbone_state, expert_state, device):
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    
    eval_state = {}
    for k, v in merged_backbone_state.items():
        eval_state[k] = v
    for k, v in expert_state.items():
        if k.startswith("fc."):
            eval_state[k] = v
            
    model.load_state_dict(eval_state)
    model = model.to(device)
    model.eval()
    return model

def apply_sp_ttbc(model, alpha=0.9):
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            original_forward = m.forward
            def make_custom_forward(bn_module, orig_forward):
                def custom_forward(x):
                    if x.size(0) > 1 and not bn_module.training:
                        # Compute batch statistics
                        batch_mean = x.mean(dim=(0, 2, 3))
                        batch_var = x.var(dim=(0, 2, 3), unbiased=False)
                        
                        # Blend statistics
                        blended_mean = alpha * bn_module.running_mean + (1 - alpha) * batch_mean
                        blended_var = alpha * bn_module.running_var + (1 - alpha) * batch_var
                        
                        return torch.nn.functional.batch_norm(
                            x, blended_mean, blended_var, bn_module.weight, bn_module.bias,
                            training=False, momentum=0.0, eps=bn_module.eps
                        )
                    else:
                        return orig_forward(x)
                return custom_forward
            
            m.forward = make_custom_forward(m, original_forward)

def generate_procedural_inputs(method, num_samples, device):
    if method == "wn":
        return torch.randn(num_samples, 3, 32, 32, device=device)
    elif method == "pn":
        z_small = torch.randn(num_samples, 3, 8, 8, device=device)
        z_up = torch.nn.functional.interpolate(z_small, size=(32, 32), mode='bilinear', align_corners=False)
        epsilon = torch.randn(num_samples, 3, 32, 32, device=device) * 0.1
        noise = z_up + epsilon
        noise = (noise - noise.mean()) / (noise.std() + 1e-5) * 0.5
        return noise
    elif method == "fractal":
        noise = torch.zeros(num_samples, 3, 32, 32, device=device)
        scales = [4, 8, 16, 32]
        weights = [1.0, 0.5, 0.25, 0.125]
        for scale, weight in zip(scales, weights):
            z = torch.randn(num_samples, 3, scale, scale, device=device)
            z_up = torch.nn.functional.interpolate(z, size=(32, 32), mode='bilinear', align_corners=False)
            noise += weight * z_up
        noise = (noise - noise.mean()) / (noise.std() + 1e-5) * 0.5
        return noise
    elif method == "geometry":
        # Procedural Geometry-Aware Calibration (PGAC)
        xs = torch.linspace(-1, 1, 32, device=device)
        ys = torch.linspace(-1, 1, 32, device=device)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        X = X.unsqueeze(0).unsqueeze(1).repeat(num_samples, 1, 1, 1)
        Y = Y.unsqueeze(0).unsqueeze(1).repeat(num_samples, 1, 1, 1)
        
        imgs = torch.zeros(num_samples, 3, 32, 32, device=device)
        for i in range(num_samples):
            theta_bg = np.random.uniform(0, 2 * np.pi)
            bg_grad = X[i] * np.cos(theta_bg) + Y[i] * np.sin(theta_bg)
            bg_grad = (bg_grad - bg_grad.min()) / (bg_grad.max() - bg_grad.min() + 1e-5) * 2.0 - 1.0
            
            # Multi-scale background noise
            bg_noise = torch.zeros(3, 32, 32, device=device)
            scales = [4, 8, 16]
            weights = [1.0, 0.5, 0.25]
            for scale, weight in zip(scales, weights):
                z = torch.randn(3, scale, scale, device=device)
                z_up = torch.nn.functional.interpolate(z.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)
                bg_noise += weight * z_up
            bg_noise = (bg_noise - bg_noise.mean()) / (bg_noise.std() + 1e-5)
            
            color_weights = torch.randn(3, 1, 1, device=device) * 0.5 + 0.5
            img = bg_grad.repeat(3, 1, 1) * color_weights + 0.2 * bg_noise
            
            num_shapes = np.random.randint(3, 8)
            for _ in range(num_shapes):
                shape_type = np.random.choice(["circle", "line", "grating", "box"])
                color = torch.randn(3, 1, 1, device=device) * 0.8
                
                if shape_type == "circle":
                    cx, cy = np.random.uniform(-0.8, 0.8, 2)
                    r = np.random.uniform(0.15, 0.6)
                    t = np.random.uniform(0.02, 0.1)
                    dist = torch.sqrt((X[i] - cx)**2 + (Y[i] - cy)**2)
                    mask = torch.sigmoid((r - dist) / t)
                    img = img * (1 - mask) + color * mask
                elif shape_type == "box":
                    cx, cy = np.random.uniform(-0.5, 0.5, 2)
                    w, h = np.random.uniform(0.15, 0.6, 2)
                    t = np.random.uniform(0.02, 0.1)
                    mask_x = torch.sigmoid((X[i] - (cx - w/2)) / t) * torch.sigmoid(((cx + w/2) - X[i]) / t)
                    mask_y = torch.sigmoid((Y[i] - (cy - h/2)) / t) * torch.sigmoid(((cy + h/2) - Y[i]) / t)
                    mask = mask_x * mask_y
                    img = img * (1 - mask) + color * mask
                elif shape_type == "line":
                    theta = np.random.uniform(0, 2 * np.pi)
                    offset = np.random.uniform(-0.5, 0.5)
                    t = np.random.uniform(0.02, 0.1)
                    dist = X[i] * np.cos(theta) + Y[i] * np.sin(theta) + offset
                    mask = torch.sigmoid(dist / t)
                    img = img * (1 - mask) + color * mask
                elif shape_type == "grating":
                    theta = np.random.uniform(0, 2 * np.pi)
                    freq = np.random.uniform(3.0, 12.0)
                    phase = np.random.uniform(0, 2 * np.pi)
                    cx, cy = np.random.uniform(-0.5, 0.5, 2)
                    sigma = np.random.uniform(0.2, 0.6)
                    
                    xr = (X[i] - cx) * np.cos(theta) + (Y[i] - cy) * np.sin(theta)
                    yr = -(X[i] - cx) * np.sin(theta) + (Y[i] - cy) * np.cos(theta)
                    
                    sinusoid = torch.cos(freq * xr + phase)
                    gaussian = torch.exp(-(xr**2 + yr**2) / (2 * sigma**2))
                    mask = (sinusoid * gaussian + 1.0) / 2.0
                    
                    img = img * (1 - mask) + color * mask
                    
            img = (img - img.mean()) / (img.std() + 1e-5) * 0.5
            imgs[i] = img
        return imgs
    else:
        raise ValueError(f"Unknown procedural method: {method}")

def generate_task_specific_procedural_inputs(task, num_samples, device):
    if task == "mnist":
        imgs = torch.zeros(num_samples, 3, 32, 32, device=device)
        for i in range(num_samples):
            img = torch.zeros(32, 32, device=device)
            num_lines = np.random.randint(2, 6)
            for _ in range(num_lines):
                x1, y1 = np.random.randint(4, 28, 2)
                x2, y2 = np.random.randint(4, 28, 2)
                steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
                xs = np.linspace(x1, x2, steps).astype(int)
                ys = np.linspace(y1, y2, steps).astype(int)
                intensity = np.random.uniform(0.5, 1.0)
                img[ys, xs] = intensity
                
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if np.random.rand() > 0.4:
                        img[np.clip(ys + dy, 0, 31), np.clip(xs + dx, 0, 31)] = intensity * 0.4
            
            img = img.unsqueeze(0).repeat(3, 1, 1)
            img = (img - 0.5) / 0.5
            imgs[i] = img
        return imgs
        
    elif task == "fmnist":
        imgs = torch.zeros(num_samples, 3, 32, 32, device=device)
        xs = torch.linspace(-1, 1, 32, device=device)
        ys = torch.linspace(-1, 1, 32, device=device)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        X = X.unsqueeze(0).repeat(num_samples, 1, 1)
        Y = Y.unsqueeze(0).repeat(num_samples, 1, 1)
        
        for i in range(num_samples):
            shape_type = np.random.choice(["rect", "ellipse", "triangle"])
            w = np.random.uniform(0.3, 0.7)
            h = np.random.uniform(0.4, 0.8)
            t = np.random.uniform(0.05, 0.15)
            intensity = np.random.uniform(0.4, 0.9)
            
            if shape_type == "rect":
                mask_x = torch.sigmoid((X[i] + w/2) / t) * torch.sigmoid((w/2 - X[i]) / t)
                mask_y = torch.sigmoid((Y[i] + h/2) / t) * torch.sigmoid((h/2 - Y[i]) / t)
                mask = mask_x * mask_y
            elif shape_type == "ellipse":
                dist = (X[i] / w)**2 + (Y[i] / h)**2
                mask = torch.sigmoid((1.0 - dist) / t)
            elif shape_type == "triangle":
                mask = torch.sigmoid((Y[i] + h/2) / t) * torch.sigmoid((X[i]*2 + w - Y[i]) / t) * torch.sigmoid((-X[i]*2 + w - Y[i]) / t)
                
            img = mask * intensity
            img = torch.clip(img + mask * (torch.randn(32, 32, device=device) * 0.1), 0, 1)
            
            img = img.unsqueeze(0).repeat(3, 1, 1)
            img = (img - 0.5) / 0.5
            imgs[i] = img
        return imgs
        
    elif task == "cifar10":
        return generate_procedural_inputs("geometry", num_samples, device)
    else:
        raise ValueError(f"Unknown task: {task}")

def calibrate_model_offline_tspc(merged_backbone_state, task, device, num_samples=2560, batch_size=128):
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(merged_backbone_state, strict=False)
    model = model.to(device)
    
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            m.reset_running_stats()
            
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for step in range(num_batches):
            momentum = 1.0 / (step + 1)
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = momentum
            inputs = generate_task_specific_procedural_inputs(task, batch_size, device)
            _ = model(inputs)
            
    calibrated_state = {}
    model_state = model.state_dict()
    for k, v in model_state.items():
        if not k.startswith("fc."):
            calibrated_state[k] = v.cpu()
            
    return calibrated_state

def calibrate_model_offline(merged_backbone_state, calib_method, device, num_samples=2560, batch_size=128):
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(merged_backbone_state, strict=False)
    model = model.to(device)
    
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            m.reset_running_stats()
            
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for step in range(num_batches):
            momentum = 1.0 / (step + 1)
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = momentum
            inputs = generate_procedural_inputs(calib_method, batch_size, device)
            _ = model(inputs)
            
    calibrated_state = {}
    model_state = model.state_dict()
    for k, v in model_state.items():
        if not k.startswith("fc."):
            calibrated_state[k] = v.cpu()
            
    return calibrated_state

def calibrate_model_offline_real_data(merged_backbone_state, calib_loader, device, num_samples=2560):
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(merged_backbone_state, strict=False)
    model = model.to(device)
    
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            m.reset_running_stats()
            
    samples_accumulated = 0
    step = 0
    with torch.no_grad():
        for inputs, _ in calib_loader:
            if samples_accumulated >= num_samples:
                break
            momentum = 1.0 / (step + 1)
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = momentum
            inputs = inputs.to(device)
            _ = model(inputs)
            samples_accumulated += inputs.size(0)
            step += 1
            
    calibrated_state = {}
    model_state = model.state_dict()
    for k, v in model_state.items():
        if not k.startswith("fc."):
            calibrated_state[k] = v.cpu()
            
    return calibrated_state

def evaluate_backbone(merged_backbone, expert_states, loaders, device, calib_method="none", alpha=0.9):
    results = {}
    
    for task_name, loader in loaders.items():
        model = get_eval_model(merged_backbone, expert_states[task_name], device)
        
        if calib_method == "sp_ttbc":
            apply_sp_ttbc(model, alpha=alpha)
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        acc = 100.0 * correct / total
        results[task_name] = acc
        
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loaders = get_dataloaders()
    expert_states = check_and_train_experts(loaders, device)
    
    # Load Progenitor W_0 state dict
    progenitor = resnet18(weights=ResNet18_Weights.DEFAULT)
    progenitor_state = progenitor.state_dict()
    
    # We evaluate Weight Averaging and Task Arithmetic (with lambda=0.3)
    merge_methods = ["wa", "ta"]
    lambda_val = 0.3
    
    # Define experimental evaluation settings
    # 1. Standard Batch Size (64)
    # 2. Single Query Batch Size (1)
    # 3. Label Shift (Batch Size 64, sorted by label)
    
    print("\n--- Initializing Evaluation Loaders ---")
    
    # Setting 1: Batch Size 64
    eval_loaders_64 = {
        "mnist": DataLoader(torchvision.datasets.MNIST(root="./data", train=False, download=False, 
                            transform=loaders["mnist"][1].dataset.transform), batch_size=64, shuffle=False, num_workers=4),
        "fmnist": DataLoader(torchvision.datasets.FashionMNIST(root="./data", train=False, download=False,
                            transform=loaders["fmnist"][1].dataset.transform), batch_size=64, shuffle=False, num_workers=4),
        "cifar10": DataLoader(torchvision.datasets.CIFAR10(root="./data", train=False, download=False,
                            transform=loaders["cifar10"][1].dataset.transform), batch_size=64, shuffle=False, num_workers=4)
    }
    
    # Setting 2: Batch Size 1 (Single Query)
    eval_loaders_1 = {
        "mnist": DataLoader(torchvision.datasets.MNIST(root="./data", train=False, download=False, 
                            transform=loaders["mnist"][1].dataset.transform), batch_size=1, shuffle=False, num_workers=4),
        "fmnist": DataLoader(torchvision.datasets.FashionMNIST(root="./data", train=False, download=False,
                            transform=loaders["fmnist"][1].dataset.transform), batch_size=1, shuffle=False, num_workers=4),
        "cifar10": DataLoader(torchvision.datasets.CIFAR10(root="./data", train=False, download=False,
                            transform=loaders["cifar10"][1].dataset.transform), batch_size=1, shuffle=False, num_workers=4)
    }
    
    # Setting 3: Label Shift (Batch Size 64, sorted by class)
    mnist_sorted = torchvision.datasets.MNIST(root="./data", train=False, download=False, transform=loaders["mnist"][1].dataset.transform)
    mnist_sorted_indices = np.argsort(mnist_sorted.targets.numpy())
    mnist_sorted = Subset(mnist_sorted, mnist_sorted_indices)
    
    fmnist_sorted = torchvision.datasets.FashionMNIST(root="./data", train=False, download=False, transform=loaders["fmnist"][1].dataset.transform)
    fmnist_sorted_indices = np.argsort(fmnist_sorted.targets.numpy())
    fmnist_sorted = Subset(fmnist_sorted, fmnist_sorted_indices)
    
    cifar_sorted = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=loaders["cifar10"][1].dataset.transform)
    cifar_sorted_indices = np.argsort(cifar_sorted.targets)
    cifar_sorted = Subset(cifar_sorted, cifar_sorted_indices)
    
    eval_loaders_shift = {
        "mnist": DataLoader(mnist_sorted, batch_size=64, shuffle=False, num_workers=4),
        "fmnist": DataLoader(fmnist_sorted, batch_size=64, shuffle=False, num_workers=4),
        "cifar10": DataLoader(cifar_sorted, batch_size=64, shuffle=False, num_workers=4)
    }
    
    # Track results
    results_all = {}
    
    # Calibrations:
    # - none: Uncalibrated
    # - sp_ttbc: SP-TTBC (blending alpha=0.9)
    # - wn: White-Noise Calibration
    # - pn: Pink-Noise Calibration
    # - fractal: Fractal-Noise Calibration (FAST-Calib)
    # - geometry: Procedural Geometry-Aware Calibration (PGAC, Ours)
    # - tspc: Task-Specific Procedural Calibration (Ours, Data-Free)
    # - oracle: Real-Data / Oracle Joint Calibration
    calib_methods = ["none", "sp_ttbc", "wn", "pn", "fractal", "geometry", "tspc", "oracle"]
    
    for mm in merge_methods:
        print(f"\n==========================================")
        print(f"EVALUATING MERGE METHOD: {mm.upper()}")
        print(f"==========================================")
        
        merged_backbone = merge_models(progenitor_state, expert_states, merge_method=mm, lambda_val=lambda_val)
        
        results_all[mm] = {}
        
        # We pre-calibrate the backbones offline for our static methods
        calibrated_backbones = {}
        
        # Benchmark calibration time (pragmatist metrics!)
        print("\n--- Benchmarking Offline Calibration Time ---")
        for cm in ["wn", "pn", "fractal", "geometry"]:
            start_time = time.time()
            calibrated_backbones[cm] = calibrate_model_offline(merged_backbone, cm, device, num_samples=2560)
            elapsed = time.time() - start_time
            print(f"Offline Calibration [{cm.upper()}]: {elapsed:.4f} seconds")
            
        # TSPC Calibration: Calibrate task-specific backbones using task-specific procedural synthetic data
        start_time = time.time()
        tspc_backbones = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            tspc_backbones[task] = calibrate_model_offline_tspc(merged_backbone, task, device, num_samples=2560)
        elapsed = time.time() - start_time
        print(f"TSPC Offline Calibration: {elapsed:.4f} seconds")
            
        # Oracle Calibration: Calibrate on mixed real data
        start_time = time.time()
        # Mix train loaders
        mixed_train_loader = loaders["mnist"][0] # We can use a combination or just run sequentially
        # For simplicity, we calibrate sequentially on each real task's training set to simulate an Oracle REPAIR setup
        oracle_backbones = {}
        for task in ["mnist", "fmnist", "cifar10"]:
            oracle_backbones[task] = calibrate_model_offline_real_data(merged_backbone, loaders[task][0], device, num_samples=2560)
        elapsed = time.time() - start_time
        print(f"Oracle Real-Data Calibration: {elapsed:.4f} seconds")
        
        for cm in calib_methods:
            results_all[mm][cm] = {}
            print(f"\n--- Running Evaluation: Calibration = {cm.upper()} ---")
            
            # Decide which backbone to use
            if cm in ["wn", "pn", "fractal", "geometry"]:
                backbone_to_use = calibrated_backbones[cm]
            elif cm in ["oracle", "tspc"]:
                # For oracle and tspc, we have task-specific backbones
                backbone_to_use = None
            else:
                backbone_to_use = merged_backbone
                
            # 1. Evaluate Setting 1 (Batch Size 64)
            if cm == "oracle":
                task_res_64 = {}
                for task in ["mnist", "fmnist", "cifar10"]:
                    res = evaluate_backbone(oracle_backbones[task], expert_states, {task: eval_loaders_64[task]}, device)
                    task_res_64[task] = res[task]
            elif cm == "tspc":
                task_res_64 = {}
                for task in ["mnist", "fmnist", "cifar10"]:
                    res = evaluate_backbone(tspc_backbones[task], expert_states, {task: eval_loaders_64[task]}, device)
                    task_res_64[task] = res[task]
            else:
                task_res_64 = evaluate_backbone(backbone_to_use, expert_states, eval_loaders_64, device, calib_method=cm)
            avg_64 = np.mean(list(task_res_64.values()))
            results_all[mm][cm]["bs64"] = {**task_res_64, "avg": avg_64}
            print(f"[BS 64] MNIST: {task_res_64['mnist']:.2f}% | F-MNIST: {task_res_64['fmnist']:.2f}% | CIFAR10: {task_res_64['cifar10']:.2f}% | Average: {avg_64:.2f}%")
            
            # 2. Evaluate Setting 2 (Batch Size 1)
            if cm == "oracle":
                task_res_1 = {}
                for task in ["mnist", "fmnist", "cifar10"]:
                    res = evaluate_backbone(oracle_backbones[task], expert_states, {task: eval_loaders_1[task]}, device)
                    task_res_1[task] = res[task]
            elif cm == "tspc":
                task_res_1 = {}
                for task in ["mnist", "fmnist", "cifar10"]:
                    res = evaluate_backbone(tspc_backbones[task], expert_states, {task: eval_loaders_1[task]}, device)
                    task_res_1[task] = res[task]
            else:
                task_res_1 = evaluate_backbone(backbone_to_use, expert_states, eval_loaders_1, device, calib_method=cm)
            avg_1 = np.mean(list(task_res_1.values()))
            results_all[mm][cm]["bs1"] = {**task_res_1, "avg": avg_1}
            print(f"[BS  1] MNIST: {task_res_1['mnist']:.2f}% | F-MNIST: {task_res_1['fmnist']:.2f}% | CIFAR10: {task_res_1['cifar10']:.2f}% | Average: {avg_1:.2f}%")
            
            # 3. Evaluate Setting 3 (Label Shift, BS 64)
            if cm == "oracle":
                task_res_shift = {}
                for task in ["mnist", "fmnist", "cifar10"]:
                    res = evaluate_backbone(oracle_backbones[task], expert_states, {task: eval_loaders_shift[task]}, device)
                    task_res_shift[task] = res[task]
            elif cm == "tspc":
                task_res_shift = {}
                for task in ["mnist", "fmnist", "cifar10"]:
                    res = evaluate_backbone(tspc_backbones[task], expert_states, {task: eval_loaders_shift[task]}, device)
                    task_res_shift[task] = res[task]
            else:
                task_res_shift = evaluate_backbone(backbone_to_use, expert_states, eval_loaders_shift, device, calib_method=cm)
            avg_shift = np.mean(list(task_res_shift.values()))
            results_all[mm][cm]["shift"] = {**task_res_shift, "avg": avg_shift}
            print(f"[SHIFT] MNIST: {task_res_shift['mnist']:.2f}% | F-MNIST: {task_res_shift['fmnist']:.2f}% | CIFAR10: {task_res_shift['cifar10']:.2f}% | Average: {avg_shift:.2f}%")

    # Save results to JSON
    with open("results.json", "w") as f:
        json.dump(results_all, f, indent=4)
    print("\nSaved all results to results.json")
    
    # Print Markdown Summary Table
    print("\n\n" + "="*50)
    print("FINAL EXPERIMENTAL RESULTS TABLE (MARKDOWN)")
    print("="*50)
    
    for mm in merge_methods:
        print(f"\n### Merging Method: {mm.upper()}")
        print(f"| Calibration Method | BS 64 Avg Acc | BS 1 Avg Acc | Label Shift Avg Acc | MNIST (BS64) | F-MNIST (BS64) | CIFAR-10 (BS64) |")
        print(f"| --- | --- | --- | --- | --- | --- | --- |")
        for cm in calib_methods:
            res_64 = results_all[mm][cm]["bs64"]
            res_1 = results_all[mm][cm]["bs1"]
            res_shift = results_all[mm][cm]["shift"]
            print(f"| {cm.upper()} | {res_64['avg']:.2f}% | {res_1['avg']:.2f}% | {res_shift['avg']:.2f}% | {res_64['mnist']:.2f}% | {res_64['fmnist']:.2f}% | {res_64['cifar10']:.2f}% |")

    # Generate Figures
    generate_plots(results_all)

def generate_plots(results):
    # Plot 1: Batch Size Robustness (BS 64 vs BS 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ["none", "sp_ttbc", "wn", "pn", "fractal", "geometry", "tspc", "oracle"]
    method_labels = ["Uncalibrated", "SP-TTBC (Online)", "White-Noise", "Pink-Noise", "FAST-Calib", "PGAC (Ours)", "TSPC (Ours, DF)", "Oracle (Real)"]
    colors = ["#7f7f7f", "#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#e377c2", "#bcbd22"]
    
    for i, mm in enumerate(["wa", "ta"]):
        ax = axes[i]
        bs64_accs = [results[mm][cm]["bs64"]["avg"] for cm in methods]
        bs1_accs = [results[mm][cm]["bs1"]["avg"] for cm in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, bs64_accs, width, label='BS 64 (Standard)', color='#3182bd')
        rects2 = ax.bar(x + width/2, bs1_accs, width, label='BS 1 (Single Query)', color='#de2d26')
        
        ax.set_ylabel('Average Accuracy (%)')
        ax.set_title(f'Batch Size Robustness ({mm.upper()})')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=30, ha="right")
        ax.set_ylim(0, 100)
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig("batch_size_robustness.png", dpi=300)
    plt.close()
    
    # Plot 2: Label Shift Robustness
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, mm in enumerate(["wa", "ta"]):
        ax = axes[i]
        bs64_accs = [results[mm][cm]["bs64"]["avg"] for cm in methods]
        shift_accs = [results[mm][cm]["shift"]["avg"] for cm in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, bs64_accs, width, label='I.I.D. Stream', color='#31a354')
        rects2 = ax.bar(x + width/2, shift_accs, width, label='Label Shift (Sorted)', color='#fdae6b')
        
        ax.set_ylabel('Average Accuracy (%)')
        ax.set_title(f'Label Shift Robustness ({mm.upper()})')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=30, ha="right")
        ax.set_ylim(0, 100)
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig("label_shift_robustness.png", dpi=300)
    plt.close()
    
    print("\nGenerated performance comparison plots saved to batch_size_robustness.png and label_shift_robustness.png")

if __name__ == "__main__":
    main()
