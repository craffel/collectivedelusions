import torch
import torch.nn as nn
import torchvision.models as models
import copy
import os
import json
from train_and_merge import get_model, get_dataset, evaluate_model

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def ties_merge(progenitor_state, experts_states, p_trim=20, lambda_val=1.0):
    p_state = {k: v.cpu() for k, v in progenitor_state.items()}
    e_states = {name: {k: v.cpu() for k, v in state.items()} for name, state in experts_states.items()}
    
    merged_state = copy.deepcopy(p_state)
    keys = list(p_state.keys())
    expert_names = list(e_states.keys())
    K = len(expert_names)
    
    # 1. Merge the BatchNorm running statistics across all experts uniformly
    for key in keys:
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            stats = [e_states[name][key].float() for name in expert_names]
            merged_state[key] = torch.mean(torch.stack(stats), dim=0).to(p_state[key].dtype)
            
    # 2. Merge backbone parameters via TIES-Merging
    for key in keys:
        if 'fc' in key:
            continue
            
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            continue
            
        W_init = p_state[key].float()
        W_experts = [e_states[name][key].float() for name in expert_names]
        T_experts = [W_exp - W_init for W_exp in W_experts]
        
        # Step 2a: Trimming
        T_trimmed = []
        for T_k in T_experts:
            # Flatten to find the threshold
            flat_T_k = T_k.view(-1)
            num_elements = flat_T_k.numel()
            k_keep = int(num_elements * (p_trim / 100.0))
            
            if k_keep <= 0:
                T_trimmed.append(torch.zeros_like(T_k))
            elif k_keep >= num_elements:
                T_trimmed.append(T_k)
            else:
                # Top-k largest by magnitude
                values, indices = torch.topk(torch.abs(flat_T_k), k_keep)
                threshold = values[-1]
                mask = torch.abs(T_k) >= threshold
                T_trimmed.append(T_k * mask)
                
        # Step 2b: Sign Election (using sum of updates)
        sum_T = torch.stack(T_trimmed).sum(dim=0)
        majority_sign = torch.sign(sum_T)
        
        # Step 2c: Sign Agreement & Disagreement-free Merging
        T_agreed = []
        for T_k_trimmed in T_trimmed:
            sign_mask = (torch.sign(T_k_trimmed) == majority_sign) & (majority_sign != 0)
            T_agreed.append(T_k_trimmed * sign_mask)
            
        # Count non-zero entries for each parameter across experts
        # We add a small eps in denominator, but only where non_zero_count > 0
        non_zero_count = torch.stack([(T_exp != 0).float() for T_exp in T_agreed]).sum(dim=0)
        sum_agreed = torch.stack(T_agreed).sum(dim=0)
        
        T_merged = torch.where(non_zero_count > 0, sum_agreed / (non_zero_count + 1e-8), torch.zeros_like(sum_agreed))
        
        # Apply scaling
        merged_state[key] = (W_init + lambda_val * T_merged).to(p_state[key].dtype)
        
    return merged_state

def dare_merge(progenitor_state, experts_states, p_drop=90, lambda_val=1.0, seed=42):
    # Set seed for reproducibility of random mask
    torch.manual_seed(seed)
    
    p_state = {k: v.cpu() for k, v in progenitor_state.items()}
    e_states = {name: {k: v.cpu() for k, v in state.items()} for name, state in experts_states.items()}
    
    merged_state = copy.deepcopy(p_state)
    keys = list(p_state.keys())
    expert_names = list(e_states.keys())
    K = len(expert_names)
    
    # 1. Merge BatchNorm stats uniformly
    for key in keys:
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            stats = [e_states[name][key].float() for name in expert_names]
            merged_state[key] = torch.mean(torch.stack(stats), dim=0).to(p_state[key].dtype)
            
    # 2. Merge backbone parameters via DARE-Merging
    keep_prob = 1.0 - (p_drop / 100.0)
    
    for key in keys:
        if 'fc' in key:
            continue
            
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            continue
            
        W_init = p_state[key].float()
        W_experts = [e_states[name][key].float() for name in expert_names]
        T_experts = [W_exp - W_init for W_exp in W_experts]
        
        # Step 2a: Random Drop and Rescale
        T_dropped = []
        for T_k in T_experts:
            if keep_prob >= 1.0:
                T_dropped.append(T_k)
            elif keep_prob <= 0.0:
                T_dropped.append(torch.zeros_like(T_k))
            else:
                # Generate random keep mask
                mask = (torch.rand_like(T_k) < keep_prob).float()
                # Rescale to maintain expected value
                T_dropped.append(T_k * mask / keep_prob)
                
        # Step 2b: Simple average of dropped updates
        T_merged = torch.stack(T_dropped).mean(dim=0)
        
        # Apply scaling
        merged_state[key] = (W_init + lambda_val * T_merged).to(p_state[key].dtype)
        
    return merged_state

def evaluate_extended():
    # 1. Prepare test dataloaders
    test_loaders = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        _, test_loaders[task] = get_dataset(task)
        
    # 2. Instantiate progenitor (pretrained ResNet-18)
    print("Loading ImageNet pre-trained progenitor...")
    progenitor = get_model()
    progenitor_state = copy.deepcopy(progenitor.state_dict())
    
    # 3. Load expert models
    experts_states = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        ckpt_path = f"checkpoints/resnet18_{task}.pt"
        if not os.path.exists(ckpt_path):
            print(f"Error: checkpoint {ckpt_path} not found.")
            return
        expert_data = torch.load(ckpt_path, map_location=device)
        experts_states[task] = expert_data['state_dict']
        
    print("\n--- Sweeping TIES-Merging ---")
    ties_sweeps = [
        # (p_trim, lambda_val)
        (20, 1.0), (20, 1.5), (20, 1.732), (20, 2.0),
        (50, 1.0), (50, 1.5), (50, 1.732), (50, 2.0),
        (80, 1.0), (80, 1.5), (80, 1.732), (80, 2.0),
    ]
    
    ties_results = []
    best_ties_avg = 0.0
    best_ties_config = None
    best_ties_metrics = None
    
    model = get_model().to(device)
    
    for p_trim, l_val in ties_sweeps:
        merged_state = ties_merge(progenitor_state, experts_states, p_trim=p_trim, lambda_val=l_val)
        accuracies = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            task_state = copy.deepcopy(merged_state)
            task_state['fc.weight'] = copy.deepcopy(experts_states[task]['fc.weight'])
            task_state['fc.bias'] = copy.deepcopy(experts_states[task]['fc.bias'])
            model.load_state_dict(task_state)
            acc = evaluate_model(model, test_loaders[task])
            accuracies[task] = acc
            
        avg_acc = sum(accuracies.values()) / len(accuracies)
        config = {'p_trim': p_trim, 'lambda': l_val, 'avg': avg_acc, 'mnist': accuracies['mnist'], 'fmnist': accuracies['fmnist'], 'cifar10': accuracies['cifar10']}
        ties_results.append(config)
        print(f"  TIES p={p_trim}%, lambda={l_val:.3f}: {avg_acc:.2f}% (MNIST: {accuracies['mnist']:.2f}%, FMNIST: {accuracies['fmnist']:.2f}%, CIFAR: {accuracies['cifar10']:.2f}%)")
        
        if avg_acc > best_ties_avg:
            best_ties_avg = avg_acc
            best_ties_config = (p_trim, l_val)
            best_ties_metrics = accuracies
            
    print(f"\nBest TIES Config: p={best_ties_config[0]}%, lambda={best_ties_config[1]:.3f} with Average Acc: {best_ties_avg:.2f}%")
    
    print("\n--- Sweeping DARE-Merging ---")
    dare_sweeps = [
        # (p_drop, lambda_val)
        (90, 1.0), (90, 1.5), (90, 1.732), (90, 2.0),
        (50, 1.0), (50, 1.5), (50, 1.732), (50, 2.0),
        (20, 1.0), (20, 1.5), (20, 1.732), (20, 2.0),
    ]
    
    dare_results = []
    best_dare_avg = 0.0
    best_dare_config = None
    best_dare_metrics = None
    
    for p_drop, l_val in dare_sweeps:
        merged_state = dare_merge(progenitor_state, experts_states, p_drop=p_drop, lambda_val=l_val)
        accuracies = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            task_state = copy.deepcopy(merged_state)
            task_state['fc.weight'] = copy.deepcopy(experts_states[task]['fc.weight'])
            task_state['fc.bias'] = copy.deepcopy(experts_states[task]['fc.bias'])
            model.load_state_dict(task_state)
            acc = evaluate_model(model, test_loaders[task])
            accuracies[task] = acc
            
        avg_acc = sum(accuracies.values()) / len(accuracies)
        config = {'p_drop': p_drop, 'lambda': l_val, 'avg': avg_acc, 'mnist': accuracies['mnist'], 'fmnist': accuracies['fmnist'], 'cifar10': accuracies['cifar10']}
        dare_results.append(config)
        print(f"  DARE drop={p_drop}%, lambda={l_val:.3f}: {avg_acc:.2f}% (MNIST: {accuracies['mnist']:.2f}%, FMNIST: {accuracies['fmnist']:.2f}%, CIFAR: {accuracies['cifar10']:.2f}%)")
        
        if avg_acc > best_dare_avg:
            best_dare_avg = avg_acc
            best_dare_config = (p_drop, l_val)
            best_dare_metrics = accuracies
            
    print(f"\nBest DARE Config: drop={best_dare_config[0]}%, lambda={best_dare_config[1]:.3f} with Average Acc: {best_dare_avg:.2f}%")
    
    # Save extra results back to results.json
    if os.path.exists('results.json'):
        with open('results.json', 'r') as f:
            res = json.load(f)
    else:
        res = {}
        
    res['ties_results'] = ties_results
    res['best_ties'] = {
        'p_trim': best_ties_config[0],
        'lambda': best_ties_config[1],
        'avg_acc': best_ties_avg,
        'mnist': best_ties_metrics['mnist'],
        'fmnist': best_ties_metrics['fmnist'],
        'cifar10': best_ties_metrics['cifar10']
    }
    
    res['dare_results'] = dare_results
    res['best_dare'] = {
        'p_drop': best_dare_config[0],
        'lambda': best_dare_config[1],
        'avg_acc': best_dare_avg,
        'mnist': best_dare_metrics['mnist'],
        'fmnist': best_dare_metrics['fmnist'],
        'cifar10': best_dare_metrics['cifar10']
    }
    
    with open('results.json', 'w') as f:
        json.dump(res, f, indent=4)
    print("\nSuccessfully updated results.json with TIES and DARE results!")

if __name__ == '__main__':
    evaluate_extended()
