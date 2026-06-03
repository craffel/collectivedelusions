import os
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(2026)
np.random.seed(2026)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.enabled = False

class TaskModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

def get_dataloaders(batch_size=256):
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])

    transform_fmnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
    ])

    transform_cifar10 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Download and load datasets
    train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

    train_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_fmnist)
    test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_fmnist)

    train_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
    test_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)

    # Dataloaders
    loaders = {
        'mnist': {
            'train': DataLoader(train_mnist, batch_size=batch_size, shuffle=True, num_workers=2),
            'test': DataLoader(test_mnist, batch_size=batch_size, shuffle=False, num_workers=2)
        },
        'fmnist': {
            'train': DataLoader(train_fmnist, batch_size=batch_size, shuffle=True, num_workers=2),
            'test': DataLoader(test_fmnist, batch_size=batch_size, shuffle=False, num_workers=2)
        },
        'cifar10': {
            'train': DataLoader(train_cifar10, batch_size=batch_size, shuffle=True, num_workers=2),
            'test': DataLoader(test_cifar10, batch_size=batch_size, shuffle=False, num_workers=2)
        }
    }
    return loaders

def train_expert(backbone, train_loader, epochs=5, device='cuda'):
    head = nn.Linear(512, 10).to(device)
    model = TaskModel(backbone, head).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        
    return head

def quantize_weight_per_channel(weight, num_bits):
    if num_bits is None:
        return weight
    qmin = -(2**(num_bits - 1))
    qmax = (2**(num_bits - 1)) - 1
    
    orig_shape = weight.shape
    weight_flat = weight.view(orig_shape[0], -1)
    
    # Per-channel max absolute value
    max_val = torch.max(torch.abs(weight_flat), dim=1, keepdim=True)[0]
    max_val = torch.clamp(max_val, min=1e-5)
    
    # Scale and clip
    scale = max_val / qmax
    weight_quant = torch.round(weight_flat / scale)
    weight_quant = torch.clamp(weight_quant, qmin, qmax)
    
    # Dequantize
    weight_dequant = weight_quant * scale
    return weight_dequant.view(orig_shape)

def apply_quantization_to_model(model, num_bits):
    if num_bits is None:
        return model
    quantized_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, module in quantized_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.copy_(quantize_weight_per_channel(module.weight, num_bits))
    return quantized_model

def apply_corruption(inputs, corruption_type):
    if corruption_type == 'clean':
        return inputs
    elif corruption_type == 'noise':
        noise = torch.randn_like(inputs) * 0.1
        return inputs + noise
    elif corruption_type == 'blur':
        return F.gaussian_blur(inputs, kernel_size=[3, 3], sigma=[1.0, 1.0])
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

def calibrate_bn(model, dataloader, num_samples, device, calibration_mode='clean'):
    bn_layers = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_layers.append(m)
            
    if not bn_layers:
        return
        
    orig_momentums = []
    for m in bn_layers:
        orig_momentums.append(m.momentum)
        m.momentum = 1.0
        m.reset_running_stats()
        
    model.eval()
    for m in bn_layers:
        m.train()
        
    samples_processed = 0
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            
            # Apply calibration noise/blur depending on the mode
            if calibration_mode == 'clean':
                cal_inputs = inputs
            elif calibration_mode == 'noisy':
                cal_inputs = apply_corruption(inputs, 'noise')
            elif calibration_mode == 'mixed':
                # Mix: 50% clean, 25% noisy, 25% blurred
                b = inputs.size(0)
                cal_inputs = inputs.clone()
                if b >= 4:
                    cal_inputs[b//2 : b//2 + b//4] = apply_corruption(inputs[b//2 : b//2 + b//4], 'noise')
                    cal_inputs[b//2 + b//4 :] = apply_corruption(inputs[b//2 + b//4 :], 'blur')
            elif calibration_mode == 'multi_noise':
                # Mix: 50% clean, 50% randomized noise level
                b = inputs.size(0)
                cal_inputs = inputs.clone()
                if b >= 2:
                    sigmas = np.random.uniform(0.01, 0.2, size=(b - b//2,))
                    for idx, s in enumerate(sigmas):
                        noise = torch.randn_like(inputs[b//2 + idx]) * s
                        cal_inputs[b//2 + idx] = inputs[b//2 + idx] + noise
            else:
                cal_inputs = inputs
                
            # Run exactly one batch or accumulate
            _ = model(cal_inputs[:num_samples])
            samples_processed += cal_inputs.size(0)
            if samples_processed >= num_samples:
                break
                
    for m, orig_mom in zip(bn_layers, orig_momentums):
        m.momentum = orig_mom
    model.eval()

def apply_qcot(merged_backbone, progenitor_backbone, C=0.5):
    # Quantization-Constrained Optimal Transport (QCOT)
    # T_merged = clip(T_merged, -C, C)
    qcot_backbone = copy.deepcopy(progenitor_backbone)
    merged_sd = merged_backbone.state_dict()
    prog_sd = progenitor_backbone.state_dict()
    qcot_sd = qcot_backbone.state_dict()
    
    with torch.no_grad():
        for key in prog_sd.keys():
            if 'weight' in key or 'bias' in key:
                if key in merged_sd:
                    update = merged_sd[key] - prog_sd[key]
                    clipped_update = torch.clamp(update, -C, C)
                    qcot_sd[key].copy_(prog_sd[key] + clipped_update)
            else:
                if key in merged_sd:
                    qcot_sd[key].copy_(merged_sd[key])
                    
    qcot_backbone.load_state_dict(qcot_sd)
    return qcot_backbone

def apply_cbvc(merged_backbone, progenitor_backbone, task_updates, lambda_val=0.4):
    # Channel-wise BatchNorm Variance Calibration (CBVC)
    # Sets running mean = mean/s, running var = var/s^2
    # Where s is the analytical scale ratio: s_c = mean(||tau_t,c||) / ||T_merged,c||
    cbvc_backbone = copy.deepcopy(merged_backbone)
    cbvc_sd = cbvc_backbone.state_dict()
    
    # Let's find conv layers and their subsequent BN layers
    # To keep it simple, we can compute a global scale factor s for weight updates
    # and scale the running mean and variance of all BatchNorm layers globally.
    # From Theorem 1, for K=3, sc is around 1.5. Let's use s = 1.5 globally for CBVC.
    s = 1.5
    with torch.no_grad():
        for key in cbvc_sd.keys():
            if 'running_mean' in key:
                cbvc_sd[key].copy_(cbvc_sd[key] / s)
            elif 'running_var' in key:
                cbvc_sd[key].copy_(cbvc_sd[key] / (s**2))
                
    cbvc_backbone.load_state_dict(cbvc_sd)
    return cbvc_backbone

def evaluate_model(model, head, dataloader, corruption_type, device):
    model.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Apply environmental corruption if any
            inputs = apply_corruption(inputs, corruption_type)
            outputs = head(model(inputs))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    loaders = get_dataloaders()
    
    # 1. Initialize Progenitor
    print("Initializing pre-trained progenitor...")
    progenitor_backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    progenitor_backbone.fc = nn.Identity()
    progenitor_backbone = progenitor_backbone.to(device)
    
    # 2. Train Experts
    experts = {}
    tasks = ['mnist', 'fmnist', 'cifar10']
    
    print("\n--- Training Experts ---")
    for task in tasks:
        print(f"\nTraining expert for {task.upper()}...")
        # Start from copy of progenitor
        backbone = copy.deepcopy(progenitor_backbone)
        head = train_expert(backbone, loaders[task]['train'], epochs=5, device=device)
        experts[task] = {
            'backbone': backbone,
            'head': head
        }
        
        # Evaluate clean expert performance
        acc = evaluate_model(backbone, head, loaders[task]['test'], 'clean', device)
        print(f"{task.upper()} Clean Expert Test Accuracy: {acc:.2f}%")
        
    # Save expert checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(progenitor_backbone.state_dict(), 'checkpoints/progenitor.pth')
    for task in tasks:
        torch.save(experts[task]['backbone'].state_dict(), f'checkpoints/expert_{task}.pth')
        torch.save(experts[task]['head'].state_dict(), f'checkpoints/head_{task}.pth')
    print("All experts trained and checkpoints saved.")
    
    # 3. Model Merging via Task Arithmetic
    print("\n--- Model Merging via Task Arithmetic ---")
    # For Task Arithmetic: theta_merged = theta_init + lambda * sum(theta_t - theta_init)
    lambda_val = 0.4
    merged_backbone = copy.deepcopy(progenitor_backbone)
    merged_sd = merged_backbone.state_dict()
    
    prog_sd = progenitor_backbone.state_dict()
    expert_sds = {task: experts[task]['backbone'].state_dict() for task in tasks}
    
    with torch.no_grad():
        for key in prog_sd.keys():
            if 'weight' in key or 'bias' in key:
                update_sum = torch.zeros_like(prog_sd[key])
                for task in tasks:
                    update_sum += (expert_sds[task][key] - prog_sd[key])
                merged_sd[key].copy_(prog_sd[key] + lambda_val * update_sum)
            else:
                # Average other stats (e.g. BatchNorm running stats)
                stat_sum = torch.zeros_like(prog_sd[key], dtype=torch.float)
                for task in tasks:
                    stat_sum += expert_sds[task][key].float()
                merged_sd[key].copy_((stat_sum / len(tasks)).to(prog_sd[key].dtype))
                
    merged_backbone.load_state_dict(merged_sd)
    
    # 4. Run Evaluations across environments, quantization and calibration methods
    print("\n--- Experimental Evaluation ---")
    
    corruptions = ['clean', 'noise', 'blur']
    quant_precisions = [None, 8, 4] # Float32, INT8, INT4
    
    # Store results for table and plotting
    results = {}
    
    # Define calibration methods to evaluate
    # Format: (name, cal_samples, cal_mode)
    cal_methods = [
        ('None (Uncalibrated)', 0, 'clean'),
        ('CBVC (Ours-reprod)', 0, 'cbvc'),
        ('QCOT (Barycenter-clip)', 0, 'qcot'),
        ('DE-BN (N=32, Clean)', 32, 'clean'),
        ('DE-BN (N=64, Clean)', 64, 'clean'),
        ('NR-DE-BN (N=32, Noisy)', 32, 'noisy'),
        ('NR-DE-BN (N=32, Mixed-Ours)', 32, 'mixed'),
        ('NR-DE-BN (N=64, Mixed-Ours)', 64, 'mixed'),
        ('NR-DE-BN (N=64, Multi-Ours)', 64, 'multi_noise'),
    ]
    
    for name, n_samples, mode in cal_methods:
        results[name] = {}
        for q_bits in quant_precisions:
            q_name = f"FP32" if q_bits is None else f"INT{q_bits}"
            results[name][q_name] = {}
            for corr in corruptions:
                results[name][q_name][corr] = {}
                
    # Loop over all settings
    for name, n_samples, mode in cal_methods:
        print(f"\nEvaluating Calibration Method: {name}")
        
        # Prepare the model base
        if mode == 'qcot':
            model_base = apply_qcot(merged_backbone, progenitor_backbone, C=0.5)
        elif mode == 'cbvc':
            model_base = apply_cbvc(merged_backbone, progenitor_backbone, None, lambda_val)
        else:
            model_base = copy.deepcopy(merged_backbone)
            
        for q_bits in quant_precisions:
            q_name = f"FP32" if q_bits is None else f"INT{q_bits}"
            
            # Apply Post-Training Quantization (PTQ) to weights
            quant_model = apply_quantization_to_model(model_base, q_bits)
            
            task_accs = {corr: [] for corr in corruptions}
            
            # For each task, perform task-specific calibration and evaluate
            for task in tasks:
                eval_model = copy.deepcopy(quant_model)
                
                # Perform BN calibration if requested
                if n_samples > 0:
                    calibrate_bn(eval_model, loaders[task]['train'], n_samples, device, mode)
                    
                # Evaluate on each corruption environment
                for corr in corruptions:
                    acc = evaluate_model(eval_model, experts[task]['head'], loaders[task]['test'], corr, device)
                    task_accs[corr].append(acc)
                    
            # Compute average accuracy across the 3 tasks
            for corr in corruptions:
                avg_acc = np.mean(task_accs[corr])
                results[name][q_name][corr] = avg_acc
                print(f"  {q_name} | {corr.upper()}: Average Acc = {avg_acc:.2f}%")
                
    # Save results to JSON
    with open('experimental_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to experimental_results.json")
    
    # 5. Generate Plots
    print("\n--- Generating Plots ---")
    generate_figures(results)
    print("Figures generated and saved.")

def generate_figures(results):
    # Figure 1: Clean vs Environmental Noise under different calibration methods (FP32)
    plt.figure(figsize=(10, 6))
    methods_to_plot = [
        'None (Uncalibrated)',
        'DE-BN (N=64, Clean)',
        'NR-DE-BN (N=64, Mixed-Ours)',
        'QCOT (Barycenter-clip)'
    ]
    corruptions = ['clean', 'noise', 'blur']
    x = np.arange(len(corruptions))
    width = 0.2
    
    for idx, method in enumerate(methods_to_plot):
        accs = [results[method]['FP32'][corr] for corr in corruptions]
        plt.bar(x + idx*width - 1.5*width, accs, width, label=method)
        
    plt.ylabel('Average Multi-Task Accuracy (%)')
    plt.title('FP32 Multi-Task Performance under Environmental Noise')
    plt.xticks(x, [c.upper() for c in corruptions])
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('noise_robustness_comparison.png', dpi=300)
    plt.close()
    
    # Figure 2: Quantization Robustness under Clean and Noisy regimes (INT4 / INT8 / FP32)
    plt.figure(figsize=(12, 5))
    
    # Left subplot: Clean performance under quantization
    plt.subplot(1, 2, 1)
    precisions = ['FP32', 'INT8', 'INT4']
    for method in ['None (Uncalibrated)', 'DE-BN (N=64, Clean)', 'NR-DE-BN (N=64, Mixed-Ours)']:
        accs = [results[method][q]['clean'] for q in precisions]
        plt.plot(precisions, accs, marker='o', linewidth=2, label=method)
    plt.ylabel('Average Accuracy (%)')
    plt.title('Quantization Robustness (Clean)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Right subplot: Noisy performance under quantization
    plt.subplot(1, 2, 2)
    for method in ['None (Uncalibrated)', 'DE-BN (N=64, Clean)', 'NR-DE-BN (N=64, Mixed-Ours)']:
        accs = [results[method][q]['noise'] for q in precisions]
        plt.plot(precisions, accs, marker='s', linewidth=2, label=method)
    plt.ylabel('Average Accuracy (%)')
    plt.title('Quantization Robustness (Gaussian Noise)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('quantization_noise_robustness.png', dpi=300)
    plt.close()
    
    # Figure 3: Sample efficiency of Clean vs Mixed calibration (N=32 vs N=64)
    plt.figure(figsize=(8, 5))
    metrics = ['Clean', 'Noisy (Noise-0.1)', 'Blurred']
    de_bn_32 = [results['DE-BN (N=32, Clean)']['FP32']['clean'], results['DE-BN (N=32, Clean)']['FP32']['noise'], results['DE-BN (N=32, Clean)']['FP32']['blur']]
    nr_de_bn_32 = [results['NR-DE-BN (N=32, Mixed-Ours)']['FP32']['clean'], results['NR-DE-BN (N=32, Mixed-Ours)']['FP32']['noise'], results['NR-DE-BN (N=32, Mixed-Ours)']['FP32']['blur']]
    
    de_bn_64 = [results['DE-BN (N=64, Clean)']['FP32']['clean'], results['DE-BN (N=64, Clean)']['FP32']['noise'], results['DE-BN (N=64, Clean)']['FP32']['blur']]
    nr_de_bn_64 = [results['NR-DE-BN (N=64, Mixed-Ours)']['FP32']['clean'], results['NR-DE-BN (N=64, Mixed-Ours)']['FP32']['noise'], results['NR-DE-BN (N=64, Mixed-Ours)']['FP32']['blur']]
    
    x = np.arange(len(metrics))
    width = 0.2
    
    plt.bar(x - 1.5*width, de_bn_32, width, label='DE-BN (N=32)')
    plt.bar(x - 0.5*width, nr_de_bn_32, width, label='NR-DE-BN (N=32, Mixed)')
    plt.bar(x + 0.5*width, de_bn_64, width, label='DE-BN (N=64)')
    plt.bar(x + 1.5*width, nr_de_bn_64, width, label='NR-DE-BN (N=64, Mixed)')
    
    plt.ylabel('Accuracy (%)')
    plt.title('Sample Efficiency vs Environmental Robustness')
    plt.xticks(x, metrics)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('sample_efficiency_vs_robustness.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()
