import os
import copy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

    # Load datasets (reusing local cache)
    train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_mnist)
    test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_mnist)

    train_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_fmnist)
    test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_fmnist)

    train_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_cifar10)
    test_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_cifar10)

    # Dataloaders
    loaders = {
        'mnist': {
            'train': DataLoader(train_mnist, batch_size=batch_size, shuffle=True, num_workers=0),
            'test': DataLoader(test_mnist, batch_size=batch_size, shuffle=False, num_workers=0)
        },
        'fmnist': {
            'train': DataLoader(train_fmnist, batch_size=batch_size, shuffle=True, num_workers=0),
            'test': DataLoader(test_fmnist, batch_size=batch_size, shuffle=False, num_workers=0)
        },
        'cifar10': {
            'train': DataLoader(train_cifar10, batch_size=batch_size, shuffle=True, num_workers=0),
            'test': DataLoader(test_cifar10, batch_size=batch_size, shuffle=False, num_workers=0)
        }
    }
    return loaders

def apply_corruption(inputs, corruption_type, level=0.1):
    if corruption_type == 'clean':
        return inputs
    elif corruption_type == 'noise':
        noise = torch.randn_like(inputs) * level
        return inputs + noise
    elif corruption_type == 'blur':
        return F.gaussian_blur(inputs, kernel_size=[3, 3], sigma=[1.0, 1.0])
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

def calibrate_bn_custom(model, dataloader, num_samples, device, clean_ratio=0.5, noise_level=0.1):
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
            b = inputs.size(0)
            
            # Construct mixed batch
            cal_inputs = inputs.clone()
            split_idx = int(b * clean_ratio)
            if split_idx < b:
                cal_inputs[split_idx:] = apply_corruption(inputs[split_idx:], 'noise', level=noise_level)
                
            _ = model(cal_inputs[:num_samples])
            samples_processed += cal_inputs.size(0)
            if samples_processed >= num_samples:
                break
                
    for m, orig_mom in zip(bn_layers, orig_momentums):
        m.momentum = orig_mom
    model.eval()

def evaluate_model(model, head, dataloader, corruption_type, device, noise_level=0.1):
    model.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = apply_corruption(inputs, corruption_type, level=noise_level)
            outputs = head(model(inputs))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    loaders = get_dataloaders()
    tasks = ['mnist', 'fmnist', 'cifar10']
    
    # Reload Progenitor
    print("Loading pre-trained progenitor...")
    progenitor_backbone = torchvision.models.resnet18()
    progenitor_backbone.fc = nn.Identity()
    progenitor_backbone.load_state_dict(torch.load('checkpoints/progenitor.pth', map_location=device))
    progenitor_backbone = progenitor_backbone.to(device)
    
    # Reload Experts
    experts = {}
    for task in tasks:
        backbone = torchvision.models.resnet18()
        backbone.fc = nn.Identity()
        backbone.load_state_dict(torch.load(f'checkpoints/expert_{task}.pth', map_location=device))
        backbone = backbone.to(device)
        
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(torch.load(f'checkpoints/head_{task}.pth', map_location=device))
        
        experts[task] = {
            'backbone': backbone,
            'head': head
        }
    
    # Perform Merge via Task Arithmetic
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
                stat_sum = torch.zeros_like(prog_sd[key], dtype=torch.float)
                for task in tasks:
                    stat_sum += expert_sds[task][key].float()
                merged_sd[key].copy_((stat_sum / len(tasks)).to(prog_sd[key].dtype))
    merged_backbone.load_state_dict(merged_sd)
    
    # --- SWEEP 1: Calibration Sample Size N vs Clean/Noise Acc ---
    print("\n--- Running Sweep 1: Sample Size N ---")
    N_sizes = [8, 16, 32, 64, 128]
    sweep1_results = {'clean_cal': {'clean_eval': [], 'noise_eval': []},
                      'mixed_cal': {'clean_eval': [], 'noise_eval': []}}
    
    for N in N_sizes:
        print(f"Evaluating sample size N = {N}")
        
        # 1. Clean Calibration
        task_clean_accs = []
        task_noise_accs = []
        for task in tasks:
            eval_model = copy.deepcopy(merged_backbone)
            calibrate_bn_custom(eval_model, loaders[task]['train'], N, device, clean_ratio=1.0)
            task_clean_accs.append(evaluate_model(eval_model, experts[task]['head'], loaders[task]['test'], 'clean', device))
            task_noise_accs.append(evaluate_model(eval_model, experts[task]['head'], loaders[task]['test'], 'noise', device))
        sweep1_results['clean_cal']['clean_eval'].append(float(np.mean(task_clean_accs)))
        sweep1_results['clean_cal']['noise_eval'].append(float(np.mean(task_noise_accs)))
        
        # 2. Mixed Calibration (50% clean, 50% noise-0.1)
        task_clean_accs = []
        task_noise_accs = []
        for task in tasks:
            eval_model = copy.deepcopy(merged_backbone)
            calibrate_bn_custom(eval_model, loaders[task]['train'], N, device, clean_ratio=0.5, noise_level=0.1)
            task_clean_accs.append(evaluate_model(eval_model, experts[task]['head'], loaders[task]['test'], 'clean', device))
            task_noise_accs.append(evaluate_model(eval_model, experts[task]['head'], loaders[task]['test'], 'noise', device))
        sweep1_results['mixed_cal']['clean_eval'].append(float(np.mean(task_clean_accs)))
        sweep1_results['mixed_cal']['noise_eval'].append(float(np.mean(task_noise_accs)))

    # --- SWEEP 2: Calibration Noise Intensity vs Evaluation Accuracy ---
    print("\n--- Running Sweep 2: Noise Intensity ---")
    noise_intensities = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]
    sweep2_results = {'clean_eval': [], 'noise_eval': []}
    
    for s_level in noise_intensities:
        print(f"Evaluating calibration noise level sigma = {s_level}")
        task_clean_accs = []
        task_noise_accs = []
        for task in tasks:
            eval_model = copy.deepcopy(merged_backbone)
            # Mixed calibration with specific noise level, N=64
            calibrate_bn_custom(eval_model, loaders[task]['train'], 64, device, clean_ratio=0.5, noise_level=s_level)
            task_clean_accs.append(evaluate_model(eval_model, experts[task]['head'], loaders[task]['test'], 'clean', device))
            task_noise_accs.append(evaluate_model(eval_model, experts[task]['head'], loaders[task]['test'], 'noise', device, noise_level=0.1))
        sweep2_results['clean_eval'].append(float(np.mean(task_clean_accs)))
        sweep2_results['noise_eval'].append(float(np.mean(task_noise_accs)))

    # --- SWEEP 3: Clean-to-Noise Ratio Sweep ---
    print("\n--- Running Sweep 3: Clean-to-Noise Ratio ---")
    ratios = [0.0, 0.25, 0.5, 0.75, 1.0] # clean ratio
    sweep3_results = {'clean_eval': [], 'noise_eval': []}
    
    for r in ratios:
        print(f"Evaluating Clean ratio = {r} (Noise ratio = {1.0 - r})")
        task_clean_accs = []
        task_noise_accs = []
        for task in tasks:
            eval_model = copy.deepcopy(merged_backbone)
            calibrate_bn_custom(eval_model, loaders[task]['train'], 64, device, clean_ratio=r, noise_level=0.1)
            task_clean_accs.append(evaluate_model(eval_model, experts[task]['head'], loaders[task]['test'], 'clean', device))
            task_noise_accs.append(evaluate_model(eval_model, experts[task]['head'], loaders[task]['test'], 'noise', device, noise_level=0.1))
        sweep3_results['clean_eval'].append(float(np.mean(task_clean_accs)))
        sweep3_results['noise_eval'].append(float(np.mean(task_noise_accs)))

    # --- SWEEP 4: Model Merging Coefficient lambda ---
    print("\n--- Running Sweep 4: Merging Coefficient lambda ---")
    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    sweep4_results = {
        'clean_cal': {'clean_eval': [], 'noise_eval': []},
        'mixed_cal': {'clean_eval': [], 'noise_eval': []}
    }
    
    for l_val in lambdas:
        print(f"Evaluating merging coefficient lambda = {l_val}")
        lambda_merged = copy.deepcopy(progenitor_backbone)
        lambda_sd = lambda_merged.state_dict()
        with torch.no_grad():
            for key in prog_sd.keys():
                if 'weight' in key or 'bias' in key:
                    update_sum = torch.zeros_like(prog_sd[key])
                    for task in tasks:
                        update_sum += (expert_sds[task][key] - prog_sd[key])
                    lambda_sd[key].copy_(prog_sd[key] + l_val * update_sum)
                else:
                    stat_sum = torch.zeros_like(prog_sd[key], dtype=torch.float)
                    for task in tasks:
                        stat_sum += expert_sds[task][key].float()
                    lambda_sd[key].copy_((stat_sum / len(tasks)).to(prog_sd[key].dtype))
        lambda_merged.load_state_dict(lambda_sd)
        
        # 1. Clean Calibration (N=64)
        task_clean_accs = []
        task_noise_accs = []
        for task in tasks:
            eval_model = copy.deepcopy(lambda_merged)
            calibrate_bn_custom(eval_model, loaders[task]['train'], 64, device, clean_ratio=1.0)
            task_clean_accs.append(evaluate_model(eval_model, experts[task]['head'], loaders[task]['test'], 'clean', device))
            task_noise_accs.append(evaluate_model(eval_model, experts[task]['head'], loaders[task]['test'], 'noise', device, noise_level=0.1))
        sweep4_results['clean_cal']['clean_eval'].append(float(np.mean(task_clean_accs)))
        sweep4_results['clean_cal']['noise_eval'].append(float(np.mean(task_noise_accs)))
        
        # 2. Mixed Calibration (N=64, 50/50 clean/noise-0.1)
        task_clean_accs = []
        task_noise_accs = []
        for task in tasks:
            eval_model = copy.deepcopy(lambda_merged)
            calibrate_bn_custom(eval_model, loaders[task]['train'], 64, device, clean_ratio=0.5, noise_level=0.1)
            task_clean_accs.append(evaluate_model(eval_model, experts[task]['head'], loaders[task]['test'], 'clean', device))
            task_noise_accs.append(evaluate_model(eval_model, experts[task]['head'], loaders[task]['test'], 'noise', device, noise_level=0.1))
        sweep4_results['mixed_cal']['clean_eval'].append(float(np.mean(task_clean_accs)))
        sweep4_results['mixed_cal']['noise_eval'].append(float(np.mean(task_noise_accs)))

    # Save all sweep results
    sweeps_data = {
        'sweep1_N': {
            'N_sizes': N_sizes,
            'results': sweep1_results
        },
        'sweep2_noise_intensity': {
            'intensities': noise_intensities,
            'results': sweep2_results
        },
        'sweep3_ratio': {
            'ratios': ratios,
            'results': sweep3_results
        },
        'sweep4_lambda': {
            'lambdas': lambdas,
            'results': sweep4_results
        }
    }
    with open('sweeps_results.json', 'w') as f:
        json.dump(sweeps_data, f, indent=4)
    print("\nSweep results saved to sweeps_results.json")
    
    # Plotting Refined Figures (1x4 grid)
    generate_refined_plots(N_sizes, sweep1_results, noise_intensities, sweep2_results, ratios, sweep3_results, lambdas, sweep4_results)
    print("Refined plots generated and saved.")

def generate_refined_plots(N_sizes, sweep1, intensities, sweep2, ratios, sweep3, lambdas, sweep4):
    plt.figure(figsize=(20, 4.5))
    
    # Subplot 1: N_sizes sweep
    plt.subplot(1, 4, 1)
    plt.plot(N_sizes, sweep1['clean_cal']['clean_eval'], 'o-', label='Clean Cal, Clean Eval', linewidth=2)
    plt.plot(N_sizes, sweep1['clean_cal']['noise_eval'], 's--', label='Clean Cal, Noise Eval', linewidth=2)
    plt.plot(N_sizes, sweep1['mixed_cal']['clean_eval'], '^-', label='Mixed Cal, Clean Eval', linewidth=2)
    plt.plot(N_sizes, sweep1['mixed_cal']['noise_eval'], 'd-', label='Mixed Cal, Noise Eval', linewidth=2)
    plt.xlabel('Calibration Batch Size N')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Impact of Calibration Size N')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 2: Noise Intensity sweep
    plt.subplot(1, 4, 2)
    plt.plot(intensities, sweep2['clean_eval'], 'o-', color='darkgreen', linewidth=2, label='Clean Eval')
    plt.plot(intensities, sweep2['noise_eval'], 's-', color='crimson', linewidth=2, label='Noise Eval (0.1)')
    plt.xlabel('Calibration Noise Intensity (sigma)')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Impact of Calibration Noise Level')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 3: Ratio sweep
    plt.subplot(1, 4, 3)
    plt.plot(ratios, sweep3['clean_eval'], 'o-', color='blue', linewidth=2, label='Clean Eval')
    plt.plot(ratios, sweep3['noise_eval'], 's-', color='orange', linewidth=2, label='Noise Eval')
    plt.xlabel('Clean Ratio in Calibration Batch')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Impact of Clean-to-Noise Ratio')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Subplot 4: Lambda sweep
    plt.subplot(1, 4, 4)
    plt.plot(lambdas, sweep4['clean_cal']['clean_eval'], 'o-', color='blue', linewidth=2, label='Clean Cal, Clean Eval')
    plt.plot(lambdas, sweep4['clean_cal']['noise_eval'], 's--', color='cyan', linewidth=2, label='Clean Cal, Noise Eval')
    plt.plot(lambdas, sweep4['mixed_cal']['clean_eval'], '^-', color='red', linewidth=2, label='Mixed Cal, Clean Eval')
    plt.plot(lambdas, sweep4['mixed_cal']['noise_eval'], 'd-', color='darkorange', linewidth=2, label='Mixed Cal, Noise Eval')
    plt.xlabel('Merging Coefficient lambda')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Impact of Merging Coefficient lambda')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('refined_ablation_sweeps.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()
