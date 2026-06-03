import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import json
import copy
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    torch.backends.cudnn.enabled = False
    print("cuDNN has been disabled to prevent initialization errors on the cluster.")

def get_dataset(name, batch_size=128):
    # ImageNet normalization parameters
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if name == 'mnist':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            norm
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            norm
        ])
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        
    elif name == 'fmnist':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            norm
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            norm
        ])
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
        
    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            norm
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            norm
        ])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def get_model():
    # Load pretrained resnet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace classification head
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

def train_expert(dataset_name, epochs=5, lr=1e-4, wd=1e-2):
    print(f"\n--- Training Expert for {dataset_name.upper()} ---")
    model = get_model().to(device)
    train_loader, test_loader = get_dataset(dataset_name)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
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
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    # Evaluate
    test_acc = evaluate_model(model, test_loader)
    print(f"Final Test Accuracy for {dataset_name.upper()}: {test_acc:.2f}%")
    return model, test_acc

def merge_experts(progenitor_state, experts_states, method='wa', **kwargs):
    """
    progenitor_state: state dict of progenitor model (ImageNet-pretrained backbone)
    experts_states: dict mapping 'mnist', 'fmnist', 'cifar10' to their respective expert state dicts
    method: 'wa', 'ta', 'u-ipr', 'cpr'
    """
    # Move all tensors to CPU for safe calculations to prevent device mismatches
    p_state = {k: v.cpu() for k, v in progenitor_state.items()}
    e_states = {name: {k: v.cpu() for k, v in state.items()} for name, state in experts_states.items()}
    
    merged_state = copy.deepcopy(p_state)
    keys = list(p_state.keys())
    expert_names = list(e_states.keys())
    K = len(expert_names)
    
    # 1. Merge the BatchNorm running statistics across all experts
    for key in keys:
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            stats = [e_states[name][key].float() for name in expert_names]
            merged_state[key] = torch.mean(torch.stack(stats), dim=0).to(p_state[key].dtype)
            
    # 2. Merge backbone parameters (everything except classification head 'fc')
    for key in keys:
        if 'fc' in key:
            continue
            
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            continue
            
        W_init = p_state[key].float()
        W_experts = [e_states[name][key].float() for name in expert_names]
        
        # Compute task vectors
        T_experts = [W_exp - W_init for W_exp in W_experts]
        
        # Compute merged task vector
        T_merged = torch.stack(T_experts).mean(dim=0)
        
        if method == 'wa':
            # Weight Averaging
            merged_state[key] = (W_init + T_merged).to(p_state[key].dtype)
            
        elif method == 'ta':
            # Task Arithmetic
            lambda_val = kwargs.get('lambda_val', 1.0)
            merged_state[key] = (W_init + lambda_val * T_merged).to(p_state[key].dtype)
            
        elif method == 'u-ipr':
            # Update-level Isotropic Parameter Resonance
            T_experts_norms = [torch.norm(T_exp) for T_exp in T_experts]
            N_experts = sum(T_experts_norms) / K
            N_merged = torch.norm(T_merged)
            
            # Compute scale factor
            Sl = N_experts / (N_merged + 1e-8)
            Sl = torch.clamp(Sl, 0.1, 10.0)
            
            merged_state[key] = (W_init + Sl * T_merged).to(p_state[key].dtype)
            
        elif method == 'cpr':
            # Constant Parameter Resonance
            c_val = kwargs.get('c_val', 1.732)
            merged_state[key] = (W_init + c_val * T_merged).to(p_state[key].dtype)
            
    return merged_state

def get_hns_state(progenitor_state, experts_states, target_task):
    """
    Reconstruct HNS merged state specifically for target_task.
    """
    # Move all tensors to CPU for safe calculations to prevent device mismatches
    p_state = {k: v.cpu() for k, v in progenitor_state.items()}
    e_states = {name: {k: v.cpu() for k, v in state.items()} for name, state in experts_states.items()}
    
    hns_state = copy.deepcopy(p_state)
    keys = list(p_state.keys())
    expert_names = list(e_states.keys())
    
    # 1. Swap in target expert's original BatchNorm stats and parameters
    for key in keys:
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key or 'bn' in key:
            hns_state[key] = copy.deepcopy(e_states[target_task][key])
            
    # 2. Channel-wise scale backbone parameters
    for key in keys:
        if 'fc' in key:
            continue
            
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key or 'bn' in key:
            continue
            
        W_init = p_state[key].float()
        W_experts = [e_states[name][key].float() for name in expert_names]
        T_experts = [W_exp - W_init for W_exp in W_experts]
        T_merged = torch.stack(T_experts).mean(dim=0)
        
        # Target update
        T_target = e_states[target_task][key].float() - W_init
        
        shape = W_init.shape
        if len(shape) > 0:
            C_out = shape[0]
            T_target_reshaped = T_target.view(C_out, -1)
            T_merged_reshaped = T_merged.view(C_out, -1)
            
            norm_target = torch.norm(T_target_reshaped, dim=1)
            norm_merged = torch.norm(T_merged_reshaped, dim=1)
            
            gamma = norm_target / (norm_merged + 1e-8)
            gamma = torch.clamp(gamma, 0.1, 10.0)
            
            reshape_dims = [C_out] + [1] * (len(shape) - 1)
            gamma_reshaped = gamma.view(*reshape_dims)
            
            W_hns = W_init + gamma_reshaped * T_merged
            hns_state[key] = W_hns.to(p_state[key].dtype)
        else:
            hns_state[key] = (W_init + T_merged).to(p_state[key].dtype)
            
    return hns_state

def evaluate_merge_method(progenitor_state, experts_states, method, test_loaders, **kwargs):
    accuracies = {}
    model = get_model().to(device)
    
    if method == 'hns':
        for task in ['mnist', 'fmnist', 'cifar10']:
            hns_state = get_hns_state(progenitor_state, experts_states, task)
            hns_state['fc.weight'] = copy.deepcopy(experts_states[task]['fc.weight'])
            hns_state['fc.bias'] = copy.deepcopy(experts_states[task]['fc.bias'])
            
            model.load_state_dict(hns_state)
            acc = evaluate_model(model, test_loaders[task])
            accuracies[task] = acc
    else:
        merged_state = merge_experts(progenitor_state, experts_states, method, **kwargs)
        for task in ['mnist', 'fmnist', 'cifar10']:
            task_state = copy.deepcopy(merged_state)
            task_state['fc.weight'] = copy.deepcopy(experts_states[task]['fc.weight'])
            task_state['fc.bias'] = copy.deepcopy(experts_states[task]['fc.bias'])
            
            model.load_state_dict(task_state)
            acc = evaluate_model(model, test_loaders[task])
            accuracies[task] = acc
            
    accuracies['avg'] = sum(accuracies.values()) / len(accuracies)
    return accuracies

def main():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # 1. Prepare test dataloaders
    test_loaders = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        _, test_loaders[task] = get_dataset(task)
        
    # 2. Instantiate progenitor (pretrained ResNet-18)
    print("Loading ImageNet pre-trained progenitor...")
    progenitor = get_model()
    progenitor_state = copy.deepcopy(progenitor.state_dict())
    
    # 3. Train or load expert models
    experts_states = {}
    expert_accs = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        ckpt_path = f"checkpoints/resnet18_{task}.pt"
        if os.path.exists(ckpt_path):
            print(f"Loading cached expert for {task.upper()} from {ckpt_path}...")
            expert_data = torch.load(ckpt_path, map_location=device)
            experts_states[task] = expert_data['state_dict']
            expert_accs[task] = expert_data['test_acc']
        else:
            model, test_acc = train_expert(task)
            torch.save({
                'state_dict': model.state_dict(),
                'test_acc': test_acc
            }, ckpt_path)
            experts_states[task] = model.state_dict()
            expert_accs[task] = test_acc
            
    print("\nIndividual Expert Accuracies:")
    for task, acc in expert_accs.items():
        print(f"  {task.upper()}: {acc:.2f}%")
        
    # 4. Evaluate baselines
    print("\n--- Evaluating Baselines ---")
    
    # Simple Weight Averaging (WA)
    wa_results = evaluate_merge_method(progenitor_state, experts_states, 'wa', test_loaders)
    print(f"Weight Averaging (WA): {wa_results}")
    
    # Update-level IPR (U-IPR)
    uipr_results = evaluate_merge_method(progenitor_state, experts_states, 'u-ipr', test_loaders)
    print(f"Update-level IPR (U-IPR): {uipr_results}")
    
    # Holographic Norm Scaling (HNS)
    hns_results = evaluate_merge_method(progenitor_state, experts_states, 'hns', test_loaders)
    print(f"Holographic Norm Scaling (HNS): {hns_results}")
    
    # 5. Sweep Task Arithmetic (TA) lambda values
    print("\n--- Sweeping Task Arithmetic Lambda ---")
    ta_lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.5]
    ta_results = []
    best_ta_avg = 0.0
    best_ta_lambda = 1.0
    for l_val in ta_lambdas:
        res = evaluate_merge_method(progenitor_state, experts_states, 'ta', test_loaders, lambda_val=l_val)
        res['lambda'] = l_val
        ta_results.append(res)
        print(f"  TA lambda={l_val:.2f}: {res['avg']:.2f}% (MNIST: {res['mnist']:.2f}%, FMNIST: {res['fmnist']:.2f}%, CIFAR: {res['cifar10']:.2f}%)")
        if res['avg'] > best_ta_avg:
            best_ta_avg = res['avg']
            best_ta_lambda = l_val
            
    # 6. Sweep Constant Parameter Resonance (CPR) c values
    print("\n--- Sweeping CPR Constant c ---")
    cpr_cs = [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.732, 1.8, 2.0, 2.2, 2.4, 2.5]
    cpr_results = []
    best_cpr_avg = 0.0
    best_cpr_c = 1.732
    for c_val in cpr_cs:
        res = evaluate_merge_method(progenitor_state, experts_states, 'cpr', test_loaders, c_val=c_val)
        res['c'] = c_val
        cpr_results.append(res)
        name_str = f"c={c_val:.3f} (theoretical sqrt(K))" if abs(c_val - 1.732) < 0.01 else f"c={c_val:.2f}"
        print(f"  CPR {name_str}: {res['avg']:.2f}% (MNIST: {res['mnist']:.2f}%, FMNIST: {res['fmnist']:.2f}%, CIFAR: {res['cifar10']:.2f}%)")
        if res['avg'] > best_cpr_avg:
            best_cpr_avg = res['avg']
            best_cpr_c = c_val
            
    # Save all results
    all_results = {
        'expert_accs': expert_accs,
        'wa_results': wa_results,
        'uipr_results': uipr_results,
        'hns_results': hns_results,
        'ta_results': ta_results,
        'cpr_results': cpr_results,
        'best_ta': {'lambda': best_ta_lambda, 'avg_acc': best_ta_avg},
        'best_cpr': {'c': best_cpr_c, 'avg_acc': best_cpr_avg}
    }
    with open('results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    print("\nResults successfully written to results.json")
    
    # 7. Generate and save plots
    print("\n--- Generating Plots ---")
    plt.figure(figsize=(10, 6))
    
    # Line for TA
    ta_x = [r['lambda'] for r in ta_results]
    ta_y = [r['avg'] for r in ta_results]
    plt.plot(ta_x, ta_y, marker='o', linestyle='-', color='blue', label='Task Arithmetic (TA)')
    
    # Line for CPR
    cpr_x = [r['c'] for r in cpr_results]
    cpr_y = [r['avg'] for r in cpr_results]
    plt.plot(cpr_x, cpr_y, marker='s', linestyle='--', color='green', label='Constant Parameter Resonance (CPR, Ours)')
    
    # Horizontal lines for other baseline models
    plt.axhline(y=wa_results['avg'], color='red', linestyle=':', label='Weight Averaging (WA)')
    plt.axhline(y=uipr_results['avg'], color='purple', linestyle='-.', label='Update-level IPR')
    plt.axhline(y=hns_results['avg'], color='orange', linestyle='-', label='Holographic Norm Scaling (HNS)')
    
    # Highlight theoretical constant sqrt(K)
    plt.axvline(x=1.732, color='darkgreen', linestyle=':', label=r'Theoretical Attractor $\sqrt{3} \approx 1.732$')
    
    plt.title('Multi-Task Model Merging Performance Comparison')
    plt.xlabel(r'Scaling Factor ($\lambda$ for TA / $c$ for CPR)')
    plt.ylabel('Average Accuracy (%)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.savefig('cpr_vs_baselines.png', dpi=300, bbox_inches='tight')
    print("Plot saved as cpr_vs_baselines.png")

if __name__ == '__main__':
    main()
