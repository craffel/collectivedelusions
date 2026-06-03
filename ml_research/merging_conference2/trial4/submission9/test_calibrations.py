import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from run_experiments import (
    get_datasets, create_base_resnet, ExpertModel, merge_backbones,
    collect_activations, apply_n_taac, eval_model_simple
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_detailed_calibration_data(expert_paths, merged_backbone, subsets, epsilon=1e-5):
    calibration_data = {
        'experts': {},
        'merged': {}
    }
    
    # 1. Collect expert activations on their respective calibration sets
    for path in expert_paths:
        name = path.replace("expert_", "").replace(".pth", "")
        ckpt = torch.load(path, map_location=device)
        
        expert_backbone = create_base_resnet().to(device)
        expert_backbone.load_state_dict(ckpt['backbone_state_dict'])
        expert_backbone.eval()
        
        cal_loader = DataLoader(subsets[name]['cal'], batch_size=128, shuffle=False)
        expert_acts = collect_activations(expert_backbone, cal_loader)
        
        calibration_data['experts'][name] = {}
        for layer_name, act in expert_acts.items():
            mean = act.mean(dim=[0, 2, 3])  # [C]
            var = act.var(dim=[0, 2, 3], unbiased=False)    # [C]
            std = torch.sqrt(var + epsilon) # [C]
            
            # Global layer-wise standard deviation
            global_std = torch.sqrt(act.var(dim=[0, 1, 2, 3], unbiased=False) + epsilon)
            
            calibration_data['experts'][name][layer_name] = {
                'mean': mean.to(device),
                'std': std.to(device),
                'global_std': global_std.to(device)
            }
            
    # 2. Collect merged model activations on each task's calibration set
    for name in subsets.keys():
        cal_loader = DataLoader(subsets[name]['cal'], batch_size=128, shuffle=False)
        merged_acts = collect_activations(merged_backbone, cal_loader)
        
        calibration_data['merged'][name] = {}
        for layer_name, act in merged_acts.items():
            mean = act.mean(dim=[0, 2, 3])  # [C]
            var = act.var(dim=[0, 2, 3], unbiased=False)    # [C]
            std = torch.sqrt(var + epsilon) # [C]
            
            global_std = torch.sqrt(act.var(dim=[0, 1, 2, 3], unbiased=False) + epsilon)
            
            calibration_data['merged'][name][layer_name] = {
                'mean': mean.to(device),
                'std': std.to(device),
                'global_std': global_std.to(device)
            }
            
    return calibration_data

def run_srac_variants_evaluation(backbone, heads, subsets, cal_data, mode="safe_channel", clamp_val=0.1, beta=15.0):
    sorted_tasks = ['mnist', 'fmnist', 'cifar']
    backbone.eval()
    for h in heads.values():
        h.eval()
        
    # 1. Extract Task Prototypes at Anchor Layer `layer2`
    prototypes = {}
    anchor_act = None
    def anchor_hook(module, input, output):
        nonlocal anchor_act
        anchor_act = output.detach()
        
    hook_handle = backbone.layer2.register_forward_hook(anchor_hook)
    
    for task_name in sorted_tasks:
        cal_loader = DataLoader(subsets[task_name]['cal'], batch_size=128, shuffle=False)
        with torch.no_grad():
            for x, _ in cal_loader:
                backbone(x.to(device))
                break
        pooled = anchor_act.mean(dim=[2, 3]) # [B, 128]
        proto = pooled.mean(dim=0) # [128]
        proto = proto / (proto.norm(p=2) + 1e-8)
        prototypes[task_name] = proto
        
    hook_handle.remove()
    
    # 2. Setup Calibration Parameters based on Mode
    cal_layers = []
    for name, module in backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d) and ('layer3' in name or 'layer4' in name):
            cal_layers.append(name)
            
    srac_params = {}
    for task_name in sorted_tasks:
        srac_params[task_name] = {}
        for l_name in cal_layers:
            if mode == "safe_channel":
                target_mean = cal_data['experts'][task_name][l_name]['mean']
                target_std = cal_data['experts'][task_name][l_name]['std']
                merged_mean = cal_data['merged'][task_name][l_name]['mean']
                merged_std = cal_data['merged'][task_name][l_name]['std']
                
                safe_merged_std = torch.clamp(merged_std, min=clamp_val)
                scale = target_std / safe_merged_std
                bias = target_mean - scale * merged_mean
                
                srac_params[task_name][l_name] = {
                    'scale': scale,
                    'bias': bias
                }
            elif mode == "layer_wise":
                target_global_std = cal_data['experts'][task_name][l_name]['global_std']
                merged_global_std = cal_data['merged'][task_name][l_name]['global_std']
                gamma = target_global_std / merged_global_std
                
                srac_params[task_name][l_name] = {
                    'gamma': gamma
                }
                
    # 3. Register Inference Dynamic Routing Hooks
    routing_weights = None
    inference_hooks = []
    
    def srac_anchor_hook(module, input, output):
        nonlocal routing_weights
        B = output.shape[0]
        pooled = output.mean(dim=[2, 3]) # [B, 128]
        pooled_norm = pooled / (pooled.norm(p=2, dim=1, keepdim=True) + 1e-8)
        
        sims = []
        for task_name in sorted_tasks:
            proto = prototypes[task_name]
            sim = torch.sum(pooled_norm * proto.unsqueeze(0), dim=1)
            sims.append(sim)
        sims = torch.stack(sims, dim=1)
        routing_weights = torch.softmax(beta * sims, dim=1)
        
    def srac_calib_hook(l_name):
        def hook(module, input, output):
            nonlocal routing_weights
            B, C, H, W = output.shape
            
            if mode == "safe_channel":
                scale_k = torch.stack([srac_params[task_name][l_name]['scale'] for task_name in sorted_tasks], dim=0)
                bias_k = torch.stack([srac_params[task_name][l_name]['bias'] for task_name in sorted_tasks], dim=0)
                
                s_interp = torch.matmul(routing_weights, scale_k).view(B, C, 1, 1)
                b_interp = torch.matmul(routing_weights, bias_k).view(B, C, 1, 1)
                return output * s_interp + b_interp
                
            elif mode == "layer_wise":
                gamma_k = torch.stack([srac_params[task_name][l_name]['gamma'] for task_name in sorted_tasks], dim=0) # [3]
                # routing_weights is [B, 3], gamma_k is [3] -> [B]
                g_interp = torch.matmul(routing_weights, gamma_k).view(B, 1, 1, 1)
                return output * g_interp
        return hook
        
    inference_hooks.append(backbone.layer2.register_forward_hook(srac_anchor_hook))
    for l_name in cal_layers:
        submodules = l_name.split('.')
        module = backbone
        for subm in submodules:
            module = getattr(module, subm)
        inference_hooks.append(module.register_forward_hook(srac_calib_hook(l_name)))
        
    # Evaluate
    accuracies = {}
    for name in sorted_tasks:
        test_loader = DataLoader(subsets[name]['test'], batch_size=256, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                features = backbone(x)
                
                logits_mnist = heads['mnist'](features)
                logits_fmnist = heads['fmnist'](features)
                logits_cifar = heads['cifar'](features)
                logits_all = torch.stack([logits_mnist, logits_fmnist, logits_cifar], dim=1)
                
                logits = torch.sum(routing_weights.unsqueeze(-1) * logits_all, dim=1)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        accuracies[name] = (correct / total) * 100.0
        
    for hook in inference_hooks:
        hook.remove()
        
    accuracies['average'] = np.mean([accuracies[k] for k in sorted_tasks])
    return accuracies

def main():
    subsets = get_datasets()
    expert_names = ['mnist', 'fmnist', 'cifar']
    expert_paths = [f"expert_{name}.pth" for name in expert_names]
    
    # Load experts
    experts = {}
    for name in expert_names:
        ckpt = torch.load(f"expert_{name}.pth", map_location=device)
        backbone = create_base_resnet().to(device)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(ckpt['head_state_dict'])
        experts[name] = ExpertModel(backbone, head).to(device)
        
    heads = {name: experts[name].head for name in expert_names}
    
    # Base merged backbones
    merged_backbone = merge_backbones(expert_paths)
    n_taac_backbone = apply_n_taac(merged_backbone, subsets)
    
    # Detailed calibration data
    cal_data_merged = compute_detailed_calibration_data(expert_paths, merged_backbone, subsets)
    cal_data_ntaac = compute_detailed_calibration_data(expert_paths, n_taac_backbone, subsets)
    
    print("\n" + "="*60)
    print("               EVALUATING SRAC VARIANTS")
    print("="*60)
    
    # 1. Base Merged Model + SRAC variants
    for mode in ["safe_channel", "layer_wise"]:
        for clamp_val in ([0.05, 0.1, 0.2] if mode == "safe_channel" else [0.0]):
            for beta in [5.0, 15.0, 30.0]:
                accs = run_srac_variants_evaluation(
                    merged_backbone, heads, subsets, cal_data_merged,
                    mode=mode, clamp_val=clamp_val, beta=beta
                )
                print(f"Base Merged | Mode: {mode:<12} | Clamp: {clamp_val:<4} | Beta: {beta:<4} | Avg Acc: {accs['average']:.2f}% (M:{accs['mnist']:.1f} F:{accs['fmnist']:.1f} C:{accs['cifar']:.1f})")
                
    # 2. N-TAAC Model + SRAC variants
    for mode in ["safe_channel", "layer_wise"]:
        for clamp_val in ([0.05, 0.1, 0.2] if mode == "safe_channel" else [0.0]):
            for beta in [5.0, 15.0, 30.0]:
                accs = run_srac_variants_evaluation(
                    n_taac_backbone, heads, subsets, cal_data_ntaac,
                    mode=mode, clamp_val=clamp_val, beta=beta
                )
                print(f"N-TAAC Base | Mode: {mode:<12} | Clamp: {clamp_val:<4} | Beta: {beta:<4} | Avg Acc: {accs['average']:.2f}% (M:{accs['mnist']:.1f} F:{accs['fmnist']:.1f} C:{accs['cifar']:.1f})")

if __name__ == "__main__":
    main()
