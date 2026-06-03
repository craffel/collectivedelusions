import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import copy
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# We load the dataset prep from train_and_merge.py
def get_dataset(name, batch_size=128):
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
        
    # Standard baseline: Uniform CPR (c = 1.732)
    uniform_1732 = {'conv1': 1.732, 'layer1': 1.732, 'layer2': 1.732, 'layer3': 1.732, 'layer4': 1.732}
    
    # Standard baseline: Uniform CPR (c = 2.00)
    uniform_2000 = {'conv1': 2.0, 'layer1': 2.0, 'layer2': 2.0, 'layer3': 2.0, 'layer4': 2.0}
    
    # 1. Increasing schedule (smaller scaling for early layers, larger for deep layers)
    increasing_sch = {
        'conv1': 1.2,
        'layer1': 1.4,
        'layer2': 1.7,
        'layer3': 2.0,
        'layer4': 2.2
    }
    
    # 2. Decreasing schedule (larger scaling for early, smaller for deep)
    decreasing_sch = {
        'conv1': 2.2,
        'layer1': 2.0,
        'layer2': 1.7,
        'layer3': 1.4,
        'layer4': 1.2
    }
    
    # 3. U-IPR-inspired scaling schedule (approximate average layer scale factors)
    # Layer-wise analysis showed conv1 is 1.70, layer1 is ~1.70, layer2 is ~1.70, layer3 is ~1.68, layer4 is ~1.69
    # Let's see what happens if we assign exact layer averages or standard deviations
    uipr_inspired = {
        'conv1': 1.70,
        'layer1': 1.70,
        'layer2': 1.70,
        'layer3': 1.69,
        'layer4': 1.70
    }
    
    # 4. Step-wise increase (late layers boosted)
    stepwise_sch = {
        'conv1': 1.5,
        'layer1': 1.5,
        'layer2': 1.8,
        'layer3': 2.0,
        'layer4': 2.2
    }
    
    # 5. Low early scaling, standard late scaling
    low_early_sch = {
        'conv1': 1.0,
        'layer1': 1.2,
        'layer2': 1.5,
        'layer3': 1.732,
        'layer4': 1.732
    }

    schedules = {
        "Uniform CPR (c=1.732)": uniform_1732,
        "Uniform CPR (c=2.0)": uniform_2000,
        "Increasing Schedule": increasing_sch,
        "Decreasing Schedule": decreasing_sch,
        "U-IPR-inspired Schedule": uipr_inspired,
        "Stepwise Schedule": stepwise_sch,
        "Low Early Scaling": low_early_sch
    }
    
    results = {}
    model = get_model().to(device)
    
    for name, scale_map in schedules.items():
        print(f"\nEvaluating schedule: {name} | Scales: {scale_map}")
        merged_state = merge_experts_piecewise(progenitor_state, experts_states, scale_map)
        
        accuracies = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            task_state = copy.deepcopy(merged_state)
            task_state['fc.weight'] = copy.deepcopy(experts_states[task]['fc.weight'])
            task_state['fc.bias'] = copy.deepcopy(experts_states[task]['fc.bias'])
            
            model.load_state_dict(task_state)
            acc = evaluate_model(model, test_loaders[task])
            accuracies[task] = acc
            print(f"  {task.upper()}: {acc:.2f}%")
            
        avg_acc = sum(accuracies.values()) / len(accuracies)
        results[name] = {**accuracies, 'avg': avg_acc}
        print(f"  Average: {avg_acc:.2f}%")
        
    print("\n" + "="*50)
    print("FINAL RESULTS COMPARISON OF PIECEWISE SCHEDULES")
    print("="*50)
    for name, r in results.items():
        print(f"{name:<25} | MNIST: {r['mnist']:.2f}% | FMNIST: {r['fmnist']:.2f}% | CIFAR10: {r['cifar10']:.2f}% | Avg: {r['avg']:.2f}%")

if __name__ == '__main__':
    main()
