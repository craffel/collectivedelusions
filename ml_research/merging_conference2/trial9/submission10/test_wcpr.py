import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from train_and_merge import get_dataloaders, get_progenitor, ExpertModel, evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def apply_wcpr(merged_backbone, expert_backbones):
    """
    Applies Wasserstein-Calibrated Parameter Resonance (WCPR) to the merged model's weights.
    """
    with torch.no_grad():
        merged_state = merged_backbone.state_dict()
        expert_states = [expert.state_dict() for expert in expert_backbones]
        
        for name, param in merged_backbone.named_parameters():
            if 'fc' in name or 'head' in name or param.ndim < 2:
                continue
            
            Cout = param.shape[0]
            reduce_dims = list(range(1, param.ndim))
            
            for c in range(Cout):
                mc = param[c].flatten()
                Ic = torch.argsort(mc)
                
                expert_sorts = []
                for state_k in expert_states:
                    p_expert = state_k[name]
                    sk = torch.sort(p_expert[c].flatten())[0]
                    expert_sorts.append(sk)
                
                starget = torch.stack(expert_sorts).mean(dim=0)
                
                cflat = torch.zeros_like(mc)
                cflat[Ic] = starget
                
                param[c].copy_(cflat.view_as(param[c]))

def apply_cmva(merged_backbone, expert_backbones, mode='both'):
    """
    Applies Channel-wise Mean-Variance Alignment (CMVA) to the merged model's weights.
    """
    with torch.no_grad():
        merged_state = merged_backbone.state_dict()
        expert_states = [expert.state_dict() for expert in expert_backbones]
        
        for name, param in merged_backbone.named_parameters():
            if 'fc' in name or 'head' in name or param.ndim < 2:
                continue
            
            reduce_dims = list(range(1, param.ndim))
            
            expert_means = []
            expert_stds = []
            for state_k in expert_states:
                p_expert = state_k[name]
                m = p_expert.mean(dim=reduce_dims, keepdim=True)
                s = p_expert.std(dim=reduce_dims, keepdim=True)
                expert_means.append(m)
                expert_stds.append(s)
            
            target_mean = torch.stack(expert_means).mean(dim=0)
            target_std = torch.stack(expert_stds).mean(dim=0)
            
            source_mean = param.mean(dim=reduce_dims, keepdim=True)
            source_std = param.std(dim=reduce_dims, keepdim=True)
            
            if mode == 'both':
                calibrated = target_mean + (target_std / (source_std + 1e-8)) * (param - source_mean)
            elif mode == 'std_only':
                calibrated = (target_std / (source_std + 1e-8)) * param
            elif mode == 'mean_only':
                calibrated = target_mean + (param - source_mean)
            else:
                calibrated = param
            
            param.copy_(calibrated)

def run_evaluation():
    loaders = get_dataloaders()
    tasks = ['mnist', 'fmnist', 'cifar10']
    
    for arch_type in ['mlp', 'resnet18']:
        print(f"\n====================================================")
        print(f"Evaluating {arch_type.upper()}")
        print(f"====================================================")
        
        progenitor_backbone = get_progenitor(arch_type).to(device)
        progenitor_backbone.load_state_dict(torch.load(f'checkpoints/{arch_type}_progenitor.pth', map_location=device))
        
        expert_backbones = []
        expert_heads = {}
        for task in tasks:
            exp_back = get_progenitor(arch_type).to(device)
            exp_back.load_state_dict(torch.load(f'checkpoints/{arch_type}_{task}_backbone.pth', map_location=device))
            expert_backbones.append(exp_back)
            
            head = nn.Linear(512, 10).to(device)
            head.load_state_dict(torch.load(f'checkpoints/{arch_type}_{task}_head.pth', map_location=device))
            expert_heads[task] = head
            
        lambda_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        
        # We will run Task Arithmetic baseline, CMVA (Both, Std Only, Mean Only), and WCPR
        ta_results = []
        cmva_both_results = []
        cmva_std_results = []
        cmva_mean_results = []
        wcpr_results = []
        
        init_state = progenitor_backbone.state_dict()
        
        task_vectors = []
        for back in expert_backbones:
            tv = {}
            for name, param in init_state.items():
                if torch.is_floating_point(param) or torch.is_complex(param):
                    tv[name] = back.state_dict()[name] - param
                else:
                    tv[name] = torch.zeros_like(param)
            task_vectors.append(tv)
            
        for lmbda in lambda_list:
            # Task Arithmetic
            merged_ta = get_progenitor(arch_type).to(device)
            ta_state = {}
            for name, param in init_state.items():
                if torch.is_floating_point(param) or torch.is_complex(param):
                    ta_state[name] = param + lmbda * torch.stack([tv[name] for tv in task_vectors]).sum(dim=0)
                else:
                    ta_state[name] = param.clone()
            merged_ta.load_state_dict(ta_state)
            
            # WCPR
            merged_wcpr = get_progenitor(arch_type).to(device)
            merged_wcpr.load_state_dict(ta_state)
            apply_wcpr(merged_wcpr, expert_backbones)
            
            # CMVA
            merged_both = get_progenitor(arch_type).to(device)
            merged_both.load_state_dict(ta_state)
            apply_cmva(merged_both, expert_backbones, mode='both')
            
            merged_std = get_progenitor(arch_type).to(device)
            merged_std.load_state_dict(ta_state)
            apply_cmva(merged_std, expert_backbones, mode='std_only')
            
            merged_mean = get_progenitor(arch_type).to(device)
            merged_mean.load_state_dict(ta_state)
            apply_cmva(merged_mean, expert_backbones, mode='mean_only')
            
            # Eval
            acc_ta, acc_wcpr, acc_both, acc_std, acc_mean = {}, {}, {}, {}, {}
            for task in tasks:
                acc_ta[task] = evaluate_model(merged_ta, expert_heads[task], loaders[task]['test'])
                acc_wcpr[task] = evaluate_model(merged_wcpr, expert_heads[task], loaders[task]['test'])
                acc_both[task] = evaluate_model(merged_both, expert_heads[task], loaders[task]['test'])
                acc_std[task] = evaluate_model(merged_std, expert_heads[task], loaders[task]['test'])
                acc_mean[task] = evaluate_model(merged_mean, expert_heads[task], loaders[task]['test'])
                
            ta_results.append(np.mean(list(acc_ta.values())))
            wcpr_results.append(np.mean(list(acc_wcpr.values())))
            cmva_both_results.append(np.mean(list(acc_both.values())))
            cmva_std_results.append(np.mean(list(acc_std.values())))
            cmva_mean_results.append(np.mean(list(acc_mean.values())))
            
            print(f"λ={lmbda:.1f} | TA: {ta_results[-1]:.2f}% | WCPR: {wcpr_results[-1]:.2f}% | CMVA (Both): {cmva_both_results[-1]:.2f}% | CMVA (Std): {cmva_std_results[-1]:.2f}% | CMVA (Mean): {cmva_mean_results[-1]:.2f}%")
            
        print(f"\nSummary for {arch_type.upper()}:")
        print(f"Max TA: {max(ta_results):.2f}% at λ={lambda_list[np.argmax(ta_results)]:.1f}")
        print(f"Max WCPR: {max(wcpr_results):.2f}% at λ={lambda_list[np.argmax(wcpr_results)]:.1f}")
        print(f"Max CMVA (Both): {max(cmva_both_results):.2f}% at λ={lambda_list[np.argmax(cmva_both_results)]:.1f}")
        print(f"Max CMVA (Std): {max(cmva_std_results):.2f}% at λ={lambda_list[np.argmax(cmva_std_results)]:.1f}")
        print(f"Max CMVA (Mean): {max(cmva_mean_results):.2f}% at λ={lambda_list[np.argmax(cmva_mean_results)]:.1f}")

if __name__ == '__main__':
    run_evaluation()
