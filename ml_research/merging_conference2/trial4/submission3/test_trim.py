import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from experiment import (
    set_seed,
    get_datasets,
    evaluate_model,
    get_bn_modules,
    merge_models,
    get_calibration_sets,
    resnet18
)
from robust_experiments import CorruptedDataset

def run_slf_calibration(base_model, expert_models, merged_model, calib_sets, N=128, trim_prob=0.15, device="cuda"):
    expert_names = list(calib_sets.keys())
    M = len(expert_models)
    
    fused_model = copy.deepcopy(merged_model).to(device)
    fused_model.eval()
    
    bn_fused = get_bn_modules(fused_model)
    dev_experts = [copy.deepcopy(m).to(device).eval() for m in expert_models]
    
    cal_loaders = {name: DataLoader(subset, batch_size=N, shuffle=False) for name, subset in calib_sets.items()}
    joint_dataset = torch.utils.data.ConcatDataset([calib_sets[name] for name in expert_names])
    joint_loader = DataLoader(joint_dataset, batch_size=N * M, shuffle=False)
    
    epsilon = 1e-5
    
    for i in range(len(bn_fused)):
        name, bn_layer_fused = bn_fused[i]
        
        expert_stds = []
        expert_means = []
        
        activation_store = {}
        def get_hook(key):
            def hook(module, input, output):
                activation_store[key] = output.detach()
            return hook
            
        expert_hook_handles = []
        for m_idx, exp_model in enumerate(dev_experts):
            exp_bn = get_bn_modules(exp_model)[i][1]
            h = exp_bn.register_forward_hook(get_hook(f"expert_{m_idx}"))
            expert_hook_handles.append(h)
            
        for m_idx, name_exp in enumerate(expert_names):
            exp_loader = cal_loaders[name_exp]
            bx, _ = next(iter(exp_loader))
            bx = bx.to(device)
            dev_experts[m_idx].fc = expert_models[m_idx].fc.to(device)
            with torch.no_grad():
                _ = dev_experts[m_idx](bx)
                
        for h in expert_hook_handles:
            h.remove()
            
        for m_idx in range(M):
            act_exp = activation_store[f"expert_{m_idx}"]
            B = act_exp.size(0)
            
            # Score each sample by its mean squared activation (L2 norm)
            sample_scores = torch.mean(act_exp ** 2, dim=(1, 2, 3))  # [B]
            _, sorted_indices = torch.sort(sample_scores)
            
            # Discard top trim_prob fraction of samples
            k = int(B * trim_prob)
            k = max(0, min(k, B - 1))
            
            clean_indices = sorted_indices[: B - k]
            clean_act_exp = act_exp[clean_indices]
            
            mean = torch.mean(clean_act_exp, dim=(0, 2, 3))
            var = torch.var(clean_act_exp, dim=(0, 2, 3), unbiased=False)
            std = torch.sqrt(var + epsilon)
            
            expert_means.append(mean)
            expert_stds.append(std)
            
        target_mean = torch.stack(expert_means).mean(dim=0)
        target_std = torch.stack(expert_stds).mean(dim=0)
        
        fused_hook_handle = bn_layer_fused.register_forward_hook(get_hook("merged"))
        bx_joint, _ = next(iter(joint_loader))
        bx_joint = bx_joint.to(device)
        with torch.no_grad():
            _ = fused_model(bx_joint)
        fused_hook_handle.remove()
        act_merged = activation_store["merged"]
        
        # Merged statistics - also apply Sample-Level Filtering
        B = act_merged.size(0)
        sample_scores_merged = torch.mean(act_merged ** 2, dim=(1, 2, 3))
        _, sorted_indices_merged = torch.sort(sample_scores_merged)
        
        # Discard top trim_prob fraction of samples from the joint calibration set
        # Since the joint set contains M * N samples, we trim k = int(B * trim_prob)
        k = int(B * trim_prob)
        k = max(0, min(k, B - 1))
        
        clean_indices_merged = sorted_indices_merged[: B - k]
        clean_act_merged = act_merged[clean_indices_merged]
        
        merged_mean = torch.mean(clean_act_merged, dim=(0, 2, 3))
        merged_var = torch.var(clean_act_merged, dim=(0, 2, 3), unbiased=False)
        merged_std = torch.sqrt(merged_var + epsilon)
        
        scale = target_std / merged_std
        shift = target_mean - scale * merged_mean
        
        with torch.no_grad():
            bn_layer_fused.weight.copy_(scale * bn_layer_fused.weight)
            bn_layer_fused.bias.copy_(scale * bn_layer_fused.bias + shift)
            
    return fused_model

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = get_datasets()
    task_names = ['mnist', 'fmnist', 'cifar10']
    
    expert_models = []
    for task in task_names:
        ckpt_path = f"./checkpoints/expert_{task}.pt"
        model = resnet18()
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
        expert_models.append(model)
        
    from torchvision.models import ResNet18_Weights
    base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(512, 10)
    
    merged_base = merge_models(base_model, expert_models, mode="WA").to(device)
    train_datasets_dict = {task: datasets[task][0] for task in task_names}
    
    # Test for p = 0.0 and p = 0.2
    for p in [0.0, 0.2]:
        print(f"\nEvaluating Corruption p = {p}")
        clean_calib_sets = get_calibration_sets(train_datasets_dict, N=128)
        corrupted_calib_sets = {t: CorruptedDataset(s, corruption_prob=p, seed=42) for t, s in clean_calib_sets.items()}
        
        for trim in [0.0, 0.1, 0.15, 0.2, 0.25]:
            fused = run_slf_calibration(base_model, expert_models, merged_base, corrupted_calib_sets, N=128, trim_prob=trim, device=device)
            accs = []
            for idx, task in enumerate(task_names):
                _, test_ds = datasets[task]
                test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
                fused.fc = expert_models[idx].fc.to(device)
                acc = evaluate_model(fused, test_loader, device)
                accs.append(acc)
            print(f"  Trim={trim:.2f} | Acc: {np.mean(accs):.2f}% (MNIST: {accs[0]:.1f}%, FMNIST: {accs[1]:.1f}%, CIFAR: {accs[2]:.1f}%)")

if __name__ == "__main__":
    main()
