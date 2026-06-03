import os
import copy
import argparse
import torch
import torch.nn as nn
from run_benchmark import get_resnet18_model, get_dataloaders, evaluate_model, copy_bn_and_fc, apply_hns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def merge_ties(models_dict, progenitor_model, fraction=0.2, lam=0.5):
    # fraction: fraction of parameters to keep (e.g. 20% / 0.2)
    merged_model = get_resnet18_model()
    merged_state = merged_model.state_dict()
    prog_state = progenitor_model.state_dict()
    
    keys = [k for k in merged_state.keys() if 'fc' not in k]
    
    for key in keys:
        init_val = prog_state[key].float().cpu()
        tvs = []
        for name, m in models_dict.items():
            tv = m.state_dict()[key].float().cpu() - init_val
            tvs.append(tv)
        
        # Stack task vectors
        tvs_stack = torch.stack(tvs, dim=0) # (K, ...)
        
        # 1. Trimming (keep top fraction)
        flat_tvs = tvs_stack.view(len(tvs), -1)
        k = max(1, int(flat_tvs.shape[1] * fraction))
        
        # For each task, keep top k elements by magnitude
        thresholds = torch.topk(flat_tvs.abs(), k, dim=1).values[:, -1].unsqueeze(1)
        mask = flat_tvs.abs() >= thresholds
        trimmed_flat = flat_tvs * mask
        trimmed_tvs = trimmed_flat.view_as(tvs_stack)
        
        # 2. Sign Agreement
        signs = torch.sign(trimmed_tvs)
        sum_signs = signs.sum(dim=0)
        majority_sign = torch.sign(sum_signs)
        
        # 3. Disjoint Merge
        agree_mask = (signs == majority_sign.unsqueeze(0)) & (trimmed_tvs != 0)
        
        # Average the agreed values
        sum_val = (trimmed_tvs * agree_mask).sum(dim=0)
        count_val = agree_mask.sum(dim=0)
        
        merged_tv = sum_val / (count_val + 1e-8)
        merged_tv = torch.where(count_val > 0, merged_tv, torch.zeros_like(merged_tv))
        
        merged_state[key].copy_(init_val + lam * merged_tv)
        
    merged_model.load_state_dict(merged_state)
    return merged_model

def merge_dare(models_dict, progenitor_model, drop_rate=0.5, lam=0.5):
    merged_model = get_resnet18_model()
    merged_state = merged_model.state_dict()
    prog_state = progenitor_model.state_dict()
    
    keys = [k for k in merged_state.keys() if 'fc' not in k]
    
    for key in keys:
        init_val = prog_state[key].float().cpu()
        tvs = []
        for name, m in models_dict.items():
            tv = m.state_dict()[key].float().cpu() - init_val
            
            # Mask (drop parameters with probability drop_rate)
            mask = (torch.rand_like(tv) >= drop_rate).float()
            # Scale remaining by 1 / (1 - drop_rate)
            tv_masked = tv * mask / (1.0 - drop_rate)
            tvs.append(tv_masked)
            
        # Average the task vectors
        tv_merged = torch.stack(tvs, dim=0).mean(dim=0)
        merged_state[key].copy_(init_val + lam * tv_merged)
        
    merged_model.load_state_dict(merged_state)
    return merged_model

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
        
    # Standard Weight Averaging of the backbone (as reference)
    wa_model = get_resnet18_model().to(DEVICE)
    wa_state = wa_model.state_dict()
    keys = [k for k in wa_state.keys() if 'fc' not in k]
    for key in keys:
        temp = torch.zeros_like(wa_state[key], dtype=torch.float32)
        for name, m in experts.items():
            temp += m.state_dict()[key].cpu().float()
        wa_state[key].copy_(temp / 3.0)
    wa_model.load_state_dict(wa_state)

    print("\n=== Evaluating TIES-Merging Baseline ===")
    ties_model = merge_ties(experts, progenitor, fraction=0.2, lam=0.5)
    ties_accs = {}
    for task in tasks:
        temp_model = copy.deepcopy(ties_model)
        copy_bn_and_fc(temp_model, experts[task])
        _, acc = evaluate_model(temp_model, loaders[task]['test'])
        ties_accs[task] = acc
        print(f"TIES-Merging Accuracy on {task.upper()}: {acc:.2f}%")
    print(f"Average TIES-Merging Accuracy: {sum(ties_accs.values())/3:.2f}%")

    print("\n=== Evaluating HNS on TIES-Merging (HNS-TIES) ===")
    hns_ties_accs = {}
    for task in tasks:
        temp_model = copy.deepcopy(ties_model)
        apply_hns(temp_model, experts[task], progenitor)
        copy_bn_and_fc(temp_model, experts[task])
        _, acc = evaluate_model(temp_model, loaders[task]['test'])
        hns_ties_accs[task] = acc
        print(f"HNS-TIES Accuracy on {task.upper()}: {acc:.2f}%")
    print(f"Average HNS-TIES Accuracy: {sum(hns_ties_accs.values())/3:.2f}%")

    print("\n=== Evaluating DARE Baseline ===")
    dare_model = merge_dare(experts, progenitor, drop_rate=0.5, lam=0.5)
    dare_accs = {}
    for task in tasks:
        temp_model = copy.deepcopy(dare_model)
        copy_bn_and_fc(temp_model, experts[task])
        _, acc = evaluate_model(temp_model, loaders[task]['test'])
        dare_accs[task] = acc
        print(f"DARE Accuracy on {task.upper()}: {acc:.2f}%")
    print(f"Average DARE Accuracy: {sum(dare_accs.values())/3:.2f}%")

    print("\n=== Evaluating HNS on DARE (HNS-DARE) ===")
    hns_dare_accs = {}
    for task in tasks:
        temp_model = copy.deepcopy(dare_model)
        apply_hns(temp_model, experts[task], progenitor)
        copy_bn_and_fc(temp_model, experts[task])
        _, acc = evaluate_model(temp_model, loaders[task]['test'])
        hns_dare_accs[task] = acc
        print(f"HNS-DARE Accuracy on {task.upper()}: {acc:.2f}%")
    print(f"Average HNS-DARE Accuracy: {sum(hns_dare_accs.values())/3:.2f}%")

if __name__ == '__main__':
    main()
