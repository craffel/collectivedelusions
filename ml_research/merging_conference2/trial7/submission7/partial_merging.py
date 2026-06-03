import os
import copy
import torch
from run_benchmark import get_resnet18_model, get_dataloaders, evaluate_model, copy_bn_and_fc, apply_hns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def merge_partial(experts, progenitor, merge_keys_fn):
    merged_model = get_resnet18_model().to(DEVICE)
    merged_state = merged_model.state_dict()
    prog_state = progenitor.state_dict()
    
    all_keys = list(merged_state.keys())
    
    for key in all_keys:
        if 'fc' in key:
            continue
        
        # Determine whether to merge this key or keep it expert-specific (will copy at eval time)
        should_merge = merge_keys_fn(key)
        
        if should_merge:
            temp = torch.zeros_like(merged_state[key], dtype=torch.float32)
            for name, m in experts.items():
                temp += m.state_dict()[key].cpu().float()
            merged_state[key].copy_(temp / 3.0)
        else:
            # If not merged, we can initialize with progenitor or anything; 
            # we will copy the exact expert weights at evaluation time.
            merged_state[key].copy_(prog_state[key].cpu().float())
            
    merged_model.load_state_dict(merged_state)
    return merged_model

def evaluate_partial_merge(merged_model, experts, loaders, merge_keys_fn, progenitor, use_hns=False):
    tasks = ['mnist', 'fmnist', 'cifar']
    accs = {}
    
    for task in tasks:
        # For each task, we copy the task-specific parts:
        # 1. fc and bn layers
        # 2. Any layers that were NOT merged (kept expert-specific)
        temp_model = copy.deepcopy(merged_model)
        
        # Copy expert-specific layers
        target_state = temp_model.state_dict()
        expert_state = experts[task].state_dict()
        for key in target_state.keys():
            if 'fc' in key or 'bn' in key or 'downsample.1' in key:
                target_state[key].copy_(expert_state[key].to(target_state[key].device))
            elif not merge_keys_fn(key):
                # Copy expert-specific layer
                target_state[key].copy_(expert_state[key].to(target_state[key].device))
                
        temp_model.load_state_dict(target_state)
        
        # Apply HNS to merged layers if specified
        if use_hns:
            # Apply HNS specifically scaled to the current expert's weights
            apply_hns_partial(temp_model, experts[task], progenitor, merge_keys_fn)
            
        _, acc = evaluate_model(temp_model, loaders[task]['test'])
        accs[task] = acc
        
    return accs

def apply_hns_partial(merged_model, expert_model, progenitor_model, merge_keys_fn):
    merged_state = merged_model.state_dict()
    expert_state = expert_model.state_dict()
    prog_state = progenitor_model.state_dict()
    
    with torch.no_grad():
        for key in merged_state.keys():
            if 'fc' in key or 'classifier' in key or not merge_keys_fn(key):
                continue
            param_m = merged_state[key]
            param_e = expert_state[key]
            param_p = prog_state[key]
            
            if len(param_m.shape) >= 2:
                device = param_m.device
                param_e_dev = param_e.to(device).float()
                param_p_dev = param_p.to(device).float()
                param_m_float = param_m.float()
                
                tv_e = param_e_dev - param_p_dev
                tv_m = param_m_float - param_p_dev
                
                dim = tuple(range(1, len(param_m.shape)))
                norm_e = torch.norm(tv_e, p=2, dim=dim, keepdim=True)
                norm_m = torch.norm(tv_m, p=2, dim=dim, keepdim=True)
                
                scale = norm_e / (norm_m + 1e-8)
                scale = torch.clamp(scale, min=0.1, max=10.0)
                
                tv_m_scaled = tv_m * scale.view(-1, *([1]*(len(param_m.shape)-1)))
                param_m.copy_((param_p_dev + tv_m_scaled).to(param_m.dtype))

def main():
    print("Loading dataloaders...")
    loaders = get_dataloaders()
    
    progenitor = get_resnet18_model().to(DEVICE)
    progenitor.load_state_dict(torch.load('checkpoints/progenitor.pt', map_location=DEVICE))
    
    tasks = ['mnist', 'fmnist', 'cifar']
    experts = {}
    for task in tasks:
        model = get_resnet18_model().to(DEVICE)
        model.load_state_dict(torch.load(f'checkpoints/{task}_expert.pt', map_location=DEVICE))
        experts[task] = model
        
    strategies = [
        ("Merge All", lambda k: True),
        ("Merge Only Layer 1 & 2", lambda k: 'layer1' in k or 'layer2' in k or 'conv1' in k),
        ("Merge Only Layer 3 & 4", lambda k: 'layer3' in k or 'layer4' in k),
        ("Merge All except Layer 4", lambda k: 'layer4' not in k)
    ]
    
    for name, fn in strategies:
        print(f"\n=========================================")
        print(f"Strategy: {name}")
        print(f"=========================================")
        
        # 1. Baseline WA
        merged_model = merge_partial(experts, progenitor, fn)
        accs_wa = evaluate_partial_merge(merged_model, experts, loaders, fn, progenitor, use_hns=False)
        print("WA (uncalibrated):")
        for task in tasks:
            print(f"  {task.upper()}: {accs_wa[task]:.2f}%")
        print(f"  Average: {sum(accs_wa.values())/3:.2f}%")
        
        # 2. HNS Calibrated
        accs_hns = evaluate_partial_merge(merged_model, experts, loaders, fn, progenitor, use_hns=True)
        print("HNS (ours):")
        for task in tasks:
            print(f"  {task.upper()}: {accs_hns[task]:.2f}%")
        print(f"  Average: {sum(accs_hns.values())/3:.2f}%")

if __name__ == '__main__':
    main()
