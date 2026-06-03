import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import copy
import json
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_dataset(name, batch_size=256):  # Increased batch size for faster evaluation
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if name == 'mnist':
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            norm
        ])
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        
    elif name == 'fmnist':
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            norm
        ])
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
        
    elif name == 'cifar10':
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            norm
        ])
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader

def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)
    return model

def evaluate_model(model, loader):
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
    return 100.0 * correct / total

def merge_experts_piecewise(progenitor_state, experts_states, scale_map):
    p_state = {k: v.cpu() for k, v in progenitor_state.items()}
    e_states = {name: {k: v.cpu() for k, v in state.items()} for name, state in experts_states.items()}
    
    merged_state = copy.deepcopy(p_state)
    keys = list(p_state.keys())
    expert_names = list(e_states.keys())
    
    # Merge BatchNorm stats first
    for key in keys:
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            stats = [e_states[name][key].float() for name in expert_names]
            merged_state[key] = torch.mean(torch.stack(stats), dim=0).to(p_state[key].dtype)
            
    # Merge weights with piecewise scales
    for key in keys:
        if 'fc' in key:
            continue
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            continue
            
        W_init = p_state[key].float()
        W_experts = [e_states[name][key].float() for name in expert_names]
        T_experts = [W_exp - W_init for W_exp in W_experts]
        T_merged = torch.stack(T_experts).mean(dim=0)
        
        # Determine scaling factor c based on the key
        c_val = 1.732  # default
        for pattern, scale in scale_map.items():
            if pattern in key:
                c_val = scale
                break
                
        merged_state[key] = (W_init + c_val * T_merged).to(p_state[key].dtype)
        
    return merged_state

def main():
    print("Loading datasets...")
    test_loaders = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        test_loaders[task] = get_dataset(task)
        
    print("Loading progenitor...")
    progenitor = get_model()
    progenitor_state = copy.deepcopy(progenitor.state_dict())
    
    print("Loading experts...")
    experts_states = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        ckpt_path = f"checkpoints/resnet18_{task}.pt"
        expert_data = torch.load(ckpt_path, map_location=device)
        experts_states[task] = expert_data['state_dict']
        
    # Define sweep ranges
    c_early_vals = [1.3, 1.5, 1.7]
    c_mid_vals = [1.6, 1.8, 2.0]
    c_deep_vals = [1.8, 2.0, 2.2, 2.4]
    
    sweep_results = []
    model = get_model().to(device)
    
    total_runs = len(c_early_vals) * len(c_mid_vals) * len(c_deep_vals)
    current_run = 0
    
    print(f"\nStarting sweep of {total_runs} combinations...")
    for c_early in c_early_vals:
        for c_mid in c_mid_vals:
            for c_deep in c_deep_vals:
                current_run += 1
                
                # Check that early <= mid <= deep to enforce monotonic increasing
                if c_early > c_mid or c_mid > c_deep:
                    continue
                
                scale_map = {
                    'conv1': c_early,
                    'layer1': c_early,
                    'layer2': c_mid,
                    'layer3': c_deep,
                    'layer4': c_deep
                }
                
                print(f"[{current_run}/{total_runs}] Evaluating: early={c_early}, mid={c_mid}, deep={c_deep}")
                merged_state = merge_experts_piecewise(progenitor_state, experts_states, scale_map)
                
                accuracies = {}
                for task in ['mnist', 'fmnist', 'cifar10']:
                    task_state = copy.deepcopy(merged_state)
                    task_state['fc.weight'] = copy.deepcopy(experts_states[task]['fc.weight'])
                    task_state['fc.bias'] = copy.deepcopy(experts_states[task]['fc.bias'])
                    
                    model.load_state_dict(task_state)
                    acc = evaluate_model(model, test_loaders[task])
                    accuracies[task] = acc
                
                avg_acc = sum(accuracies.values()) / 3.0
                print(f"  MNIST: {accuracies['mnist']:.2f}% | FMNIST: {accuracies['fmnist']:.2f}% | CIFAR10: {accuracies['cifar10']:.2f}% | Avg: {avg_acc:.2f}%")
                
                sweep_results.append({
                    'c_early': c_early,
                    'c_mid': c_mid,
                    'c_deep': c_deep,
                    'mnist': accuracies['mnist'],
                    'fmnist': accuracies['fmnist'],
                    'cifar10': accuracies['cifar10'],
                    'avg': avg_acc
                })
                
    # Sort results by average accuracy descending
    sweep_results.sort(key=lambda x: x['avg'], reverse=True)
    
    # Save sweep results
    with open('results_pcpr_sweep.json', 'w') as f:
        json.dump(sweep_results, f, indent=4)
        
    print("\n" + "="*50)
    print("TOP 10 PIECEWISE SCHEDULES")
    print("="*50)
    for i, r in enumerate(sweep_results[:10]):
        print(f"#{i+1:<2} | early={r['c_early']:.2f}, mid={r['c_mid']:.2f}, deep={r['c_deep']:.2f} | MNIST: {r['mnist']:.2f}% | FMNIST: {r['fmnist']:.2f}% | CIFAR10: {r['cifar10']:.2f}% | Avg: {r['avg']:.2f}%")

if __name__ == '__main__':
    main()
