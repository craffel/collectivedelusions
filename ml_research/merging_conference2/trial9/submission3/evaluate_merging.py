import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import json

from models import ResNet18Backbone, MLPBackbone, CompleteModel
import merging

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False
print(f"Evaluation running on device: {device}")

# Create output directories
os.makedirs('results', exist_ok=True)

def get_eval_loaders(batch_size=256, subset_size=1000):
    # Grayscale transforms (Resize to 32x32, duplicate channels, normalize)
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # CIFAR-10 transforms (Resize to 32x32, normalize)
    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Datasets
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_gray)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_gray)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_cifar)
    
    # Take a stable subset of 1000 samples for fast evaluation
    mnist_subset = Subset(mnist_test, range(min(subset_size, len(mnist_test))))
    fmnist_subset = Subset(fmnist_test, range(min(subset_size, len(fmnist_test))))
    cifar_subset = Subset(cifar_test, range(min(subset_size, len(cifar_test))))
    
    # For BN Calibration, we also load a subset of training data
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_gray)
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_gray)
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_cifar)
    
    loaders = {
        'mnist': {
            'test': DataLoader(mnist_subset, batch_size=batch_size, shuffle=False, num_workers=2),
            'train': DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
        },
        'fmnist': {
            'test': DataLoader(fmnist_subset, batch_size=batch_size, shuffle=False, num_workers=2),
            'train': DataLoader(fmnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
        },
        'cifar10': {
            'test': DataLoader(cifar_subset, batch_size=batch_size, shuffle=False, num_workers=2),
            'train': DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=2)
        }
    }
    return loaders

def gaussian_blur(x, kernel_size=3, sigma=1.0):
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    g = g / g.sum()
    # Create 2D Gaussian kernel
    g2d = g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)
    g2d = g2d.repeat(3, 1, 1, 1).to(x.device)
    return F.conv2d(x, g2d, padding=kernel_size//2, groups=3)

def apply_de_bn(model, loaders, N=16):
    """
    Data-Efficient BatchNorm calibration: re-estimating mean and variance physically
    by passing N unlabeled samples from each task.
    """
    model.train()
    # Freeze weights
    for p in model.parameters():
        p.requires_grad = False
        
    # Reset BN running stats
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = None # Cumulative running stats
            
    # Perform forward passes on N samples per task
    with torch.no_grad():
        for task_name in ['mnist', 'fmnist', 'cifar10']:
            train_loader = loaders[task_name]['train']
            samples_drawn = 0
            for x, _ in train_loader:
                x = x.to(device)
                batch_size = x.size(0)
                if samples_drawn + batch_size > N:
                    x = x[:N - samples_drawn]
                model(x)
                samples_drawn += x.size(0)
                if samples_drawn >= N:
                    break
                    
    model.eval()
    for p in model.parameters():
        p.requires_grad = True

def evaluate_task(model, test_loader, corruption=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Apply corruptions if specified
            if corruption == 'noise':
                x = x + torch.randn_like(x) * 0.1
            elif corruption == 'blur':
                x = gaussian_blur(x, kernel_size=3, sigma=1.0)
                
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100.0 * correct / total

def run_evaluation_suite(arch='resnet18'):
    print(f"\n==============================================")
    print(f"Running Evaluation Suite for: {arch.upper()}")
    print(f"==============================================")
    
    loaders = get_eval_loaders()
    
    # 1. Load progenitors and experts
    if arch == 'resnet18':
        progenitor = ResNet18Backbone().to(device)
        progenitor.load_state_dict(torch.load('checkpoints/resnet18_progenitor.pt', map_location=device))
        
        experts = []
        heads = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            exp = ResNet18Backbone().to(device)
            exp.load_state_dict(torch.load(f'checkpoints/resnet18_{task}_backbone.pt', map_location=device))
            experts.append(exp)
            
            head = nn.Linear(512, 10).to(device)
            head.load_state_dict(torch.load(f'checkpoints/resnet18_{task}_head.pt', map_location=device))
            heads[task] = head
    else: # mlp
        progenitor = MLPBackbone().to(device)
        progenitor.load_state_dict(torch.load('checkpoints/mlp_progenitor.pt', map_location=device))
        
        experts = []
        heads = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            exp = MLPBackbone().to(device)
            exp.load_state_dict(torch.load(f'checkpoints/mlp_{task}_backbone.pt', map_location=device))
            experts.append(exp)
            
            head = nn.Linear(512, 10).to(device)
            head.load_state_dict(torch.load(f'checkpoints/mlp_{task}_head.pt', map_location=device))
            heads[task] = head
            
    # Evaluation dictionary to store results
    results = {}
    
    # Expert Oracles (to establish upper bound)
    oracle_scores = {}
    for i, task in enumerate(['mnist', 'fmnist', 'cifar10']):
        model = CompleteModel(experts[i], heads[task])
        oracle_scores[task] = evaluate_task(model, loaders[task]['test'])
    oracle_avg = sum(oracle_scores.values()) / len(oracle_scores)
    print(f"Expert Oracles: {oracle_scores} | Avg: {oracle_avg:.2f}%")
    results['Oracle'] = {task: oracle_scores[task] for task in ['mnist', 'fmnist', 'cifar10']}
    results['Oracle']['avg'] = oracle_avg
    
    # Tuned Task Arithmetic Grid Search to find best lambda
    print("Performing grid search for Tuned Task Arithmetic...")
    best_ta_lam = 0.5
    best_ta_avg = 0.0
    for lam in np.linspace(0.1, 1.5, 15):
        merged_state = merging.task_arithmetic(experts, progenitor, lam=lam)
        test_backbone = ResNet18Backbone().to(device) if arch == 'resnet18' else MLPBackbone().to(device)
        test_backbone.load_state_dict(merged_state)
        
        task_scores = []
        for task in ['mnist', 'fmnist', 'cifar10']:
            model = CompleteModel(test_backbone, heads[task])
            score = evaluate_task(model, loaders[task]['test'])
            task_scores.append(score)
        avg = sum(task_scores) / len(task_scores)
        if avg > best_ta_avg:
            best_ta_avg = avg
            best_ta_lam = lam
    print(f"Best Tuned TA lambda: {best_ta_lam:.2f} with average accuracy: {best_ta_avg:.2f}%")
    
    # Define Merge configurations
    # Format: { Name: state_dict }
    merges = {
        'Weight Averaging (WA)': merging.weight_averaging(experts),
        f'Tuned TA (lam={best_ta_lam:.2f})': merging.task_arithmetic(experts, progenitor, lam=best_ta_lam),
        'TIES-Merging (fraction=0.2)': merging.ties_merging(experts, progenitor, fraction=0.2),
        'DARE-Merging (fraction=0.2)': merging.dare_merging(experts, progenitor, fraction=0.2),
        'WCPR (mode=unified)': merging.wcpr_merging(experts, progenitor, mode='unified'),
        'QR-IPR (gamma=2.0)': merging.qr_ipr_merging(experts, progenitor, gamma=2.0),
        'QR-SP-WCPR (Ours)': merging.qr_sp_wcpr_merging(experts, progenitor, sign_merger='ties', fraction=0.2, gamma=2.0, scale_compensation=True)
    }
    
    eval_conditions = [
        ('FP32_Clean', None, None, False),
        ('INT8_Tensor_Clean', 8, False, False),
        ('INT8_Channel_Clean', 8, True, False),
        ('FP32_Noise', None, None, 'noise'),
        ('FP32_Blur', None, None, 'blur'),
    ]
    
    # Evaluate each merge across conditions
    for merge_name, state_dict in merges.items():
        results[merge_name] = {}
        print(f"\nEvaluating Merge: {merge_name}")
        
        for cond_name, num_bits, per_channel, corruption in eval_conditions:
            # We want to check with and without BN calibration for ResNet-18
            calibration_protocols = [False]
            if arch == 'resnet18':
                calibration_protocols = [False, True]
                
            for use_de_bn in calibration_protocols:
                suffix = "_DEBN" if use_de_bn else ""
                full_cond_name = cond_name + suffix
                
                # Load backbone
                backbone = ResNet18Backbone().to(device) if arch == 'resnet18' else MLPBackbone().to(device)
                backbone.load_state_dict(state_dict)
                
                # Apply BN calibration if requested
                if use_de_bn and arch == 'resnet18':
                    apply_de_bn(backbone, loaders, N=16)
                    
                # Apply Quantization if requested
                if num_bits is not None:
                    merging.apply_quantization_to_model(backbone, num_bits=num_bits, per_channel=per_channel)
                    
                # Evaluate on each task
                task_scores = {}
                for task in ['mnist', 'fmnist', 'cifar10']:
                    model = CompleteModel(backbone, heads[task])
                    task_scores[task] = evaluate_task(model, loaders[task]['test'], corruption=corruption)
                    
                avg = sum(task_scores.values()) / len(task_scores)
                results[merge_name][full_cond_name] = {
                    'scores': task_scores,
                    'avg': avg
                }
                print(f"  {full_cond_name:<25} | MNIST: {task_scores['mnist']:.2f}% | FMNIST: {task_scores['fmnist']:.2f}% | CIFAR10: {task_scores['cifar10']:.2f}% | Avg: {avg:.2f}%")
                
    # Save results to json
    with open(f'results/results_{arch}.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

def main():
    resnet_results = run_evaluation_suite('resnet18')
    mlp_results = run_evaluation_suite('mlp')
    
    print("\nEvaluation suite completed! Results saved in the results/ directory.")

if __name__ == '__main__':
    main()
