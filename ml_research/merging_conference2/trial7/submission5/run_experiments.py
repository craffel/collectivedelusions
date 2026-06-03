import os
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False  # Bypasses cuDNN issues on the cluster

class MultiTaskResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load backbone with ImageNet pretrained weights
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        # Replace fc with identity to extract features
        self.backbone.fc = nn.Identity()
        
        # Create task-specific heads
        self.heads = nn.ModuleDict({
            'mnist': nn.Linear(in_features, num_classes),
            'fmnist': nn.Linear(in_features, num_classes),
            'cifar10': nn.Linear(in_features, num_classes)
        })
        
    def forward(self, x, task_name=None):
        features = self.backbone(x)
        if task_name is not None:
            return self.heads[task_name](features)
        return features

def get_datasets(data_dir='./data', batch_size=256, num_samples_train=5000):
    # Transforms
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # repeat grayscale to 3 channels
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_fmnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # repeat grayscale to 3 channels
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load
    train_mnist = datasets.MNIST(data_dir, train=True, download=True, transform=transform_mnist)
    test_mnist = datasets.MNIST(data_dir, train=False, download=True, transform=transform_mnist)
    
    train_fmnist = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform_fmnist)
    test_fmnist = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform_fmnist)
    
    train_cifar10 = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_cifar10)
    test_cifar10 = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_cifar10)
    
    # Generate subsets for training experts
    if num_samples_train is not None:
        # Task 1: MNIST
        mnist_indices = np.random.choice(len(train_mnist), num_samples_train, replace=False)
        train_mnist_sub = Subset(train_mnist, mnist_indices)
        
        # Task 2: FMNIST
        fmnist_indices = np.random.choice(len(train_fmnist), num_samples_train, replace=False)
        train_fmnist_sub = Subset(train_fmnist, fmnist_indices)
        
        # Task 3: CIFAR-10
        cifar_indices = np.random.choice(len(train_cifar10), num_samples_train, replace=False)
        train_cifar10_sub = Subset(train_cifar10, cifar_indices)
    else:
        train_mnist_sub = train_mnist
        train_fmnist_sub = train_fmnist
        train_cifar10_sub = train_cifar10
        
    return {
        'mnist': (train_mnist_sub, test_mnist),
        'fmnist': (train_fmnist_sub, test_fmnist),
        'cifar10': (train_cifar10_sub, test_cifar10)
    }

def train_expert(task_name, train_dataset, test_dataset, epochs=5, lr=1e-4, batch_size=256, device='cpu'):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    model = MultiTaskResNet18().to(device)
    
    # We only train the backbone and the task-specific head
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': lr},
        {'params': model.heads[task_name].parameters(), 'lr': lr * 10.0} # higher learning rate for linear head
    ], lr=lr, weight_decay=1e-4)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x, task_name)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        train_acc = 100.0 * correct / total
        avg_loss = total_loss / total
        
        # Eval
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x, task_name)
                _, predicted = outputs.max(1)
                test_total += y.size(0)
                test_correct += predicted.eq(y).sum().item()
        test_acc = 100.0 * test_correct / test_total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
    return model.state_dict(), test_acc

def evaluate_merged_model(merged_state_dict, heads_state_dicts, test_datasets, task_names, device='cpu'):
    # Reconstruct merged model with appropriate head
    model = MultiTaskResNet18().to(device)
    model.load_state_dict(merged_state_dict, strict=False)
    
    # Load task specific heads
    for task in task_names:
        model.heads[task].load_state_dict(heads_state_dicts[task])
        
    model.eval()
    results = {}
    with torch.no_grad():
        for task in task_names:
            test_dataset = test_datasets[task]
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
            correct = 0
            total = 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x, task)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
            acc = 100.0 * correct / total
            results[task] = acc
    results['average'] = sum(results.values()) / len(results)
    return results

def merge_expert_models(expert_state_dicts, progenitor_state_dict, merge_type='WA', lam=0.5):
    tasks = list(expert_state_dicts.keys())
    first_task = tasks[0]
    merged_backbone = {}
    
    # Filter state_dict to keep only backbone parameters
    backbone_keys = [k for k in expert_state_dicts[first_task].keys() if k.startswith('backbone.') and 'num_batches_tracked' not in k]
    
    if merge_type == 'WA':
        # Weight Averaging: element-wise average of backbones
        for key in backbone_keys:
            tensors = [expert_state_dicts[task][key] for task in tasks]
            merged_backbone[key] = torch.stack(tensors, dim=0).mean(dim=0)
            
    elif merge_type == 'TA':
        # Task Arithmetic: W_init + lambda * sum(W_expert - W_init)
        for key in backbone_keys:
            init_val = progenitor_state_dict[key]
            task_vectors = []
            for task in tasks:
                task_vectors.append(expert_state_dicts[task][key] - init_val)
            merged_backbone[key] = init_val + lam * torch.stack(task_vectors, dim=0).sum(dim=0)
            
    # Also average the BatchNorm running stats to form the standard 'uncalibrated' statistics
    # This matches equation 3 in the papers where pre-saved stats are usually the average of experts' stats
    for key in expert_state_dicts[first_task].keys():
        if any(substring in key for substring in ['running_mean', 'running_var', 'num_batches_tracked']):
            tensors = [expert_state_dicts[task][key].float() for task in tasks]
            merged_backbone[key] = torch.stack(tensors, dim=0).mean(dim=0)
            if 'num_batches_tracked' in key:
                merged_backbone[key] = merged_backbone[key].long()
                
    return merged_backbone

# Helper function to reset and calibrate running statistics of a merged model
def calibrate_running_stats(model, calibration_data, epochs=10, device='cpu'):
    # Put model in train mode to trigger BatchNorm running stats updates
    model.train()
    
    # 1. Reset all BN running stats
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            # Set momentum to 0.1 for rolling EMA updates
            m.momentum = 0.1
            
    # 2. Run forward passes
    # Create loader for calibration data
    loader = DataLoader(calibration_data, batch_size=64, shuffle=True)
    for epoch in range(epochs):
        for x in loader:
            x = x.to(device)
            # Forward pass through the backbone (to update all BN layers)
            _ = model.backbone(x)
            
    # Put back to eval mode
    model.eval()

# Generator for 2D spatial harmonic frequency sweep gratings (HarmonicCalib)
def generate_harmonic_patterns(num_samples=256, img_size=32, device='cpu'):
    x = torch.linspace(0, 1, img_size, device=device)
    y = torch.linspace(0, 1, img_size, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    patterns = []
    # Rich selection of spatial frequencies, orientations, and phases
    frequencies = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    orientations = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0] # degrees
    phases = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0] # degrees
    
    # Seed local random generator for pattern selection
    local_rng = random.Random(42)
    
    while len(patterns) < num_samples * 3: # generate excess to allow diverse sampling
        f = local_rng.choice(frequencies)
        theta_deg = local_rng.choice(orientations)
        theta = theta_deg * np.pi / 180.0
        phi_deg = local_rng.choice(phases)
        phi = phi_deg * np.pi / 180.0
        
        # 2D projection variables
        proj = grid_x * np.cos(theta) + grid_y * np.sin(theta)
        
        # Choice of pattern type
        pattern_type = local_rng.choice(['sine', 'cosine', 'grid', 'radial', 'gabor'])
        
        if pattern_type == 'sine':
            grating = torch.sin(2 * np.pi * f * proj + phi)
        elif pattern_type == 'cosine':
            grating = torch.cos(2 * np.pi * f * proj + phi)
        elif pattern_type == 'grid':
            proj_ortho = -grid_x * np.sin(theta) + grid_y * np.cos(theta)
            grating = 0.5 * torch.sin(2 * np.pi * f * proj + phi) + 0.5 * torch.sin(2 * np.pi * f * proj_ortho + phi)
        elif pattern_type == 'radial':
            r = torch.sqrt((grid_x - 0.5)**2 + (grid_y - 0.5)**2)
            grating = torch.sin(2 * np.pi * f * r + phi)
        elif pattern_type == 'gabor':
            grating = torch.sin(2 * np.pi * f * proj + phi)
            # Gaussian envelope
            g_envelope = torch.exp(-((grid_x - 0.5)**2 + (grid_y - 0.5)**2) / (2 * 0.15**2))
            grating = grating * g_envelope
            
        # Add diverse color channel representations:
        # Replicated grayscale, or phase-shifted color (R, G, B channels represent phase-shifted waves)
        color_type = local_rng.choice(['gray', 'phase_shift'])
        if color_type == 'gray':
            pattern = grating.unsqueeze(0).repeat(3, 1, 1)
        else:
            phi2 = phi + 2 * np.pi / 3
            phi3 = phi + 4 * np.pi / 3
            grating2 = torch.sin(2 * np.pi * f * proj + phi2)
            grating3 = torch.sin(2 * np.pi * f * proj + phi3)
            pattern = torch.stack([grating, grating2, grating3], dim=0)
            
        # Add random amplitude scaling and offset
        amp = local_rng.uniform(0.5, 1.0)
        offset = local_rng.uniform(-0.2, 0.2)
        pattern = pattern * amp + offset
        
        # Clamp to reasonable ranges
        pattern = torch.clamp(pattern, -1.0, 1.0)
        patterns.append(pattern)
        
    local_rng.shuffle(patterns)
    patterns_tensor = torch.stack(patterns[:num_samples], dim=0)
    
    # Print statistics of the generated harmonic sweeps
    print(f"Generated {num_samples} Harmonic Sweep samples. Tensor shape: {patterns_tensor.shape}, Range: [{patterns_tensor.min().item():.3f}, {patterns_tensor.max().item():.3f}]")
    return patterns_tensor

# Generator for Pink Noise
def generate_pink_noise(num_samples=256, img_size=32, device='cpu'):
    # Generate low resolution noise
    z_small = torch.randn(num_samples, 3, 8, 8, device=device)
    # Upsample to 32x32 bilinearly
    z_upsampled = F.interpolate(z_small, size=(img_size, img_size), mode='bilinear', align_corners=False)
    # Add high-frequency Gaussian white noise
    epsilon = 0.1 * torch.randn(num_samples, 3, img_size, img_size, device=device)
    pink_noise = z_upsampled + epsilon
    
    # Normalize to [-1, 1] range to match normalized image distributions
    pink_noise = (pink_noise - pink_noise.min()) / (pink_noise.max() - pink_noise.min() + 1e-8)
    pink_noise = pink_noise * 2.0 - 1.0
    return pink_noise

# Generative BatchNorm-Matching Calibration (DF-Calib-Gen)
def optimize_df_calib_gen(expert_state_dicts, heads_state_dicts, task_names, num_samples=256, epochs=150, device='cpu'):
    print("\n--- Running Generative BatchNorm-Matching (DF-Calib-Gen) Optimization ---")
    synthetic_datasets = {}
    
    # Standard values from the paper: alpha = 10^-3, beta = 10^-4
    alpha = 1e-3 # TV regularizer
    beta = 1e-4  # L2 regularizer
    
    for task in task_names:
        print(f"Optimizing synthetic dataset for task: {task.upper()}")
        
        # Load independent expert model
        expert = MultiTaskResNet18().to(device)
        expert.load_state_dict(expert_state_dicts[task], strict=False)
        expert.heads[task].load_state_dict(heads_state_dicts[task])
        expert.eval()
        
        # Initialize synthetic images as Gaussian noise
        X_syn = torch.randn(num_samples, 3, 32, 32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([X_syn], lr=0.1)
        
        # Register hooks to capture pre-activation stats from BatchNorm layers
        inputs_dict = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                # Save input to BatchNorm2d layer
                inputs_dict[name] = input[0]
            return hook
            
        bn_modules = {}
        for name, m in expert.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_modules[name] = m
                hooks.append(m.register_forward_hook(make_hook(name)))
                
        # Jitter function
        def apply_jitter(x, jitter=2):
            x_padded = F.pad(x, (jitter, jitter, jitter, jitter), mode='reflect')
            h, w = x.shape[-2:]
            dh = torch.randint(0, 2 * jitter + 1, (1,)).item()
            dw = torch.randint(0, 2 * jitter + 1, (1,)).item()
            return x_padded[:, :, dh:dh+h, dw:dw+w]
            
        for ep in range(epochs):
            optimizer.zero_grad()
            inputs_dict.clear()
            
            # Apply Jitter
            X_jittered = apply_jitter(X_syn)
            
            # Run forward pass (just backbone to capture stats)
            _ = expert.backbone(X_jittered)
            
            # Compute BatchNorm matching loss
            loss_stats = 0.0
            for name, x_act in inputs_dict.items():
                m = bn_modules[name]
                
                # Compute batch statistics
                mean = x_act.mean(dim=(0, 2, 3))
                var = x_act.var(dim=(0, 2, 3), unbiased=False)
                
                # Stored running stats in expert
                target_mean = m.running_mean
                target_var = m.running_var
                
                std = torch.sqrt(var + 1e-5)
                target_std = torch.sqrt(target_var + 1e-5)
                
                loss_stats += torch.sum((mean - target_mean) ** 2) + torch.sum((std - target_std) ** 2)
                
            # Regularization
            loss_tv = torch.sum((X_syn[:, :, 1:, :] - X_syn[:, :, :-1, :]) ** 2) + torch.sum((X_syn[:, :, :, 1:] - X_syn[:, :, :, :-1]) ** 2)
            loss_l2 = torch.sum(X_syn ** 2) / X_syn.numel()
            
            total_loss = loss_stats + alpha * loss_tv + beta * loss_l2
            total_loss.backward()
            optimizer.step()
            
            if (ep + 1) % 50 == 0:
                print(f"  Epoch {ep+1}/{epochs} | Stats Loss: {loss_stats.item():.4f} | TV Loss: {loss_tv.item():.4f} | Total: {total_loss.item():.4f}")
                
        # Remove hooks
        for h in hooks:
            h.remove()
            
        # Save optimized synthetic data
        synthetic_datasets[task] = X_syn.detach().clone()
        
    return synthetic_datasets

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory for checkpoints
    os.makedirs('./checkpoints', exist_ok=True)
    
    # 1. Load datasets
    datasets_dict = get_datasets(data_dir='./data', batch_size=256, num_samples_train=5000)
    task_names = ['mnist', 'fmnist', 'cifar10']
    
    # Save progenitor backbone weights (ImageNet pretrained starting weights)
    progenitor = MultiTaskResNet18().to(device)
    progenitor_state_dict = {f"backbone.{k}": v.cpu().clone() for k, v in progenitor.backbone.state_dict().items()}
    
    expert_state_dicts = {}
    heads_state_dicts = {}
    oracle_accuracies = {}
    
    # 2. Check if expert checkpoints exist, otherwise train them
    all_checkpoints_exist = True
    for task in task_names:
        chk_path = f"./checkpoints/expert_{task}.pt"
        if not os.path.exists(chk_path):
            all_checkpoints_exist = False
            break
            
    if all_checkpoints_exist:
        print("\n--- Loading Trained Experts from Checkpoints ---")
        for task in task_names:
            chk_path = f"./checkpoints/expert_{task}.pt"
            checkpoint = torch.load(chk_path, map_location='cpu')
            expert_state_dicts[task] = checkpoint['state_dict']
            heads_state_dicts[task] = checkpoint['head_state_dict']
            oracle_accuracies[task] = checkpoint['accuracy']
            print(f"Loaded {task.upper()} expert checkpoint. Test Accuracy: {oracle_accuracies[task]:.2f}%")
    else:
        print("\n--- Training Experts from Scratch ---")
        for task in task_names:
            train_dataset, test_dataset = datasets_dict[task]
            state_dict, accuracy = train_expert(task, train_dataset, test_dataset, epochs=5, batch_size=256, device=device)
            oracle_accuracies[task] = accuracy
            
            # Save state dicts
            expert_state_dicts[task] = state_dict
            # Extract head state dict
            head_dict = {k.replace(f"heads.{task}.", ""): v.cpu().clone() for k, v in state_dict.items() if k.startswith(f"heads.{task}.")}
            heads_state_dicts[task] = head_dict
            
            # Save checkpoint
            torch.save({
                'state_dict': state_dict,
                'head_state_dict': head_dict,
                'accuracy': accuracy
            }, f"./checkpoints/expert_{task}.pt")
            
    print(f"\nOracle Experts Performance summary:")
    for task in task_names:
        print(f"  {task.upper()}: {oracle_accuracies[task]:.2f}%")
    print(f"  Average: {sum(oracle_accuracies.values())/3:.2f}%")
    
    # 3. Model Merging and Calibration Evaluation
    # We will evaluate Weight Averaging (WA) and Task Arithmetic (TA, lambda=0.3, 0.5, 0.7)
    configurations = [
        ('WA', 0.5),
        ('TA', 0.3),
        ('TA', 0.5),
        ('TA', 0.7)
    ]
    
    results_matrix = {}
    
    # Prepare test datasets dictionary
    test_datasets_dict = {task: datasets_dict[task][1] for task in task_names}
    
    for merge_type, lam in configurations:
        config_name = f"{merge_type}_{lam}"
        print(f"\n=======================================================")
        print(f"Evaluating Config: {merge_type} (lambda={lam})")
        print(f"=======================================================")
        
        results_matrix[config_name] = {}
        
        # Merge backbone
        merged_backbone_state = merge_expert_models(expert_state_dicts, progenitor_state_dict, merge_type=merge_type, lam=lam)
        
        # Complete state dict of model
        merged_model_state_dict = {}
        for k, v in merged_backbone_state.items():
            merged_model_state_dict[k] = v
            
        # Run 1: Uncalibrated baseline
        print("\nMethod: Uncalibrated (No Cal)")
        uncal_results = evaluate_merged_model(merged_model_state_dict, heads_state_dicts, test_datasets_dict, task_names, device=device)
        print(f"  Results: MNIST: {uncal_results['mnist']:.2f}% | FMNIST: {uncal_results['fmnist']:.2f}% | CIFAR10: {uncal_results['cifar10']:.2f}% | Avg: {uncal_results['average']:.2f}%")
        results_matrix[config_name]['Uncalibrated'] = uncal_results
        
        # Run 2: White Noise Calibration
        print("\nMethod: White Noise Calibration (DF-Calib-WN)")
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(merged_model_state_dict, strict=False)
        wn_data = torch.randn(256, 3, 32, 32, device=device)
        # Normalize to [-1, 1] to match dataset normalization
        wn_data = (wn_data - wn_data.min()) / (wn_data.max() - wn_data.min() + 1e-8) * 2.0 - 1.0
        calibrate_running_stats(model, wn_data, epochs=10, device=device)
        wn_results = evaluate_merged_model(model.state_dict(), heads_state_dicts, test_datasets_dict, task_names, device=device)
        print(f"  Results: MNIST: {wn_results['mnist']:.2f}% | FMNIST: {wn_results['fmnist']:.2f}% | CIFAR10: {wn_results['cifar10']:.2f}% | Avg: {wn_results['average']:.2f}%")
        results_matrix[config_name]['White_Noise'] = wn_results
        
        # Run 3: Pink Noise Calibration
        print("\nMethod: Pink Noise Calibration (DF-Calib-PN)")
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(merged_model_state_dict, strict=False)
        pn_data = generate_pink_noise(num_samples=256, img_size=32, device=device)
        calibrate_running_stats(model, pn_data, epochs=10, device=device)
        pn_results = evaluate_merged_model(model.state_dict(), heads_state_dicts, test_datasets_dict, task_names, device=device)
        print(f"  Results: MNIST: {pn_results['mnist']:.2f}% | FMNIST: {pn_results['fmnist']:.2f}% | CIFAR10: {pn_results['cifar10']:.2f}% | Avg: {pn_results['average']:.2f}%")
        results_matrix[config_name]['Pink_Noise'] = pn_results
        
        # Run 4: Real-Data (Oracle) Calibration
        # We construct a small dataset of 256 samples from the real training set
        print("\nMethod: Real-Data (Oracle) Joint Calibration")
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(merged_model_state_dict, strict=False)
        # Combine 85 samples from MNIST, 85 from FMNIST, 86 from CIFAR-10 training subsets
        real_samples = []
        for i, task in enumerate(task_names):
            train_sub, _ = datasets_dict[task]
            num_req = 86 if i == 2 else 85
            indices = np.random.choice(len(train_sub), num_req, replace=False)
            for idx in indices:
                # Append only the image
                img, _ = train_sub[idx]
                real_samples.append(img)
        real_data = torch.stack(real_samples, dim=0)
        calibrate_running_stats(model, real_data, epochs=10, device=device)
        real_results = evaluate_merged_model(model.state_dict(), heads_state_dicts, test_datasets_dict, task_names, device=device)
        print(f"  Results: MNIST: {real_results['mnist']:.2f}% | FMNIST: {real_results['fmnist']:.2f}% | CIFAR10: {real_results['cifar10']:.2f}% | Avg: {real_results['average']:.2f}%")
        results_matrix[config_name]['Real_Data_Oracle'] = real_results
        
        # Run 5: DF-Calib-Gen (Optimized synthetic data)
        print("\nMethod: DF-Calib-Gen (Generative Match)")
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(merged_model_state_dict, strict=False)
        # Optimize or load optimized synthetic data
        syn_chk_path = f"./checkpoints/synthetic_data.pt"
        if os.path.exists(syn_chk_path):
            syn_datasets = torch.load(syn_chk_path, map_location=device)
        else:
            syn_datasets = optimize_df_calib_gen(expert_state_dicts, heads_state_dicts, task_names, num_samples=256, epochs=150, device=device)
            torch.save(syn_datasets, syn_chk_path)
            
        # Combine synthetic images (85 + 85 + 86)
        syn_combined = torch.cat([
            syn_datasets['mnist'][:85],
            syn_datasets['fmnist'][:85],
            syn_datasets['cifar10'][:86]
        ], dim=0)
        calibrate_running_stats(model, syn_combined, epochs=10, device=device)
        gen_results = evaluate_merged_model(model.state_dict(), heads_state_dicts, test_datasets_dict, task_names, device=device)
        print(f"  Results: MNIST: {gen_results['mnist']:.2f}% | FMNIST: {gen_results['fmnist']:.2f}% | CIFAR10: {gen_results['cifar10']:.2f}% | Avg: {gen_results['average']:.2f}%")
        results_matrix[config_name]['DF_Calib_Gen'] = gen_results
        
        # Run 6: Harmonic Resonance Calibration (HarmonicCalib - Ours)
        print("\nMethod: Harmonic Resonance Calibration (HarmonicCalib - Ours)")
        model = MultiTaskResNet18().to(device)
        model.load_state_dict(merged_model_state_dict, strict=False)
        # Generate harmonic patterns analytically (runs instantly!)
        harmonic_data = generate_harmonic_patterns(num_samples=256, img_size=32, device=device)
        calibrate_running_stats(model, harmonic_data, epochs=10, device=device)
        harmonic_results = evaluate_merged_model(model.state_dict(), heads_state_dicts, test_datasets_dict, task_names, device=device)
        print(f"  Results: MNIST: {harmonic_results['mnist']:.2f}% | FMNIST: {harmonic_results['fmnist']:.2f}% | CIFAR10: {harmonic_results['cifar10']:.2f}% | Avg: {harmonic_results['average']:.2f}%")
        results_matrix[config_name]['HarmonicCalib'] = harmonic_results
        
    # Save the results
    with open('results.json', 'w') as f:
        json.dump(results_matrix, f, indent=4)
        
    print("\n\n=======================================================")
    print("ALL EXPERIMENTS COMPLETE. WRITING SUMMARY REPORT.")
    print("=======================================================")
    
    # 4. Generate beautiful comparison plots
    # We will plot average accuracy across different configurations for each calibration method
    methods = ['Uncalibrated', 'White_Noise', 'Pink_Noise', 'HarmonicCalib', 'DF_Calib_Gen', 'Real_Data_Oracle']
    configs = [f"{m}_{l}" for m, l in configurations]
    
    plt.figure(figsize=(10, 6))
    for m in methods:
        accs = [results_matrix[cfg][m]['average'] for cfg in configs]
        plt.plot(configs, accs, marker='o', label=m, linewidth=2)
        
    # Add a horizontal line for Oracle Expert average performance
    oracle_avg = sum(oracle_accuracies.values()) / 3
    plt.axhline(y=oracle_avg, color='r', linestyle='--', label='Oracle Experts (No Merge)', alpha=0.7)
    
    plt.title('Multi-Task Merged Model Average Accuracy across Calibration Methods', fontsize=12)
    plt.xlabel('Merging Configuration (Method_Lambda)', fontsize=11)
    plt.ylabel('Average Classification Accuracy (%)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('calibration_comparison.png', dpi=300)
    print("Saved comparison plot to 'calibration_comparison.png'")

if __name__ == '__main__':
    main()
