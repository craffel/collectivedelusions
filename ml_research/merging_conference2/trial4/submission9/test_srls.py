import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from run_experiments import get_datasets, create_base_resnet, ExpertModel, merge_backbones, collect_activations

device = torch.device("cpu")

def extract_prototypes(backbone, subsets, sorted_tasks, anchor_layer_name="layer2"):
    backbone.eval()
    prototypes = {}
    anchor_act = None
    
    def anchor_hook(module, input, output):
        nonlocal anchor_act
        anchor_act = output.detach()
        
    anchor_layer = getattr(backbone, anchor_layer_name)
    hook_handle = anchor_layer.register_forward_hook(anchor_hook)
    
    for name in sorted_tasks:
        cal_loader = DataLoader(subsets[name]['cal'], batch_size=128, shuffle=False)
        with torch.no_grad():
            for x, _ in cal_loader:
                x = x.to(device)
                backbone(x)
                break
        pooled = anchor_act.mean(dim=[2, 3])
        proto = pooled.mean(dim=0)
        proto = proto / (proto.norm(p=2) + 1e-8)
        prototypes[name] = proto
        
    hook_handle.remove()
    return prototypes

def compute_srls_gammas(expert_paths, merged_backbone, subsets, sorted_tasks, epsilon=1e-5):
    print("\nComputing SRLS Gammas (Global Layer-wise Scaling Factors)...")
    gammas = {} # task_name -> layer_name -> gamma (scalar)
    
    # 1. Collect expert activations on their respective calibration sets
    expert_global_stds = {}
    for path in expert_paths:
        name = path.replace("expert_", "").replace(".pth", "")
        ckpt = torch.load(path, map_location=device)
        expert_backbone = create_base_resnet().to(device)
        expert_backbone.load_state_dict(ckpt['backbone_state_dict'])
        expert_backbone.eval()
        
        cal_loader = DataLoader(subsets[name]['cal'], batch_size=128, shuffle=False)
        expert_acts = collect_activations(expert_backbone, cal_loader)
        expert_global_stds[name] = {}
        for layer_name, act in expert_acts.items():
            global_std = torch.sqrt(act.var(dim=[0, 1, 2, 3], unbiased=False) + epsilon)
            expert_global_stds[name][layer_name] = global_std.to(device)
            
    # 2. Collect merged activations on each task's calibration set
    merged_backbone.eval()
    for name in sorted_tasks:
        gammas[name] = {}
        cal_loader = DataLoader(subsets[name]['cal'], batch_size=128, shuffle=False)
        merged_acts = collect_activations(merged_backbone, cal_loader)
        for layer_name, act in merged_acts.items():
            global_std_merged = torch.sqrt(act.var(dim=[0, 1, 2, 3], unbiased=False) + epsilon)
            global_std_expert = expert_global_stds[name][layer_name]
            
            # Gamma is expert global std / merged global std
            gammas[name][layer_name] = (global_std_expert / global_std_merged).to(device)
            
    return gammas

def run_srls_evaluation(merged_backbone, heads, subsets, gammas, sorted_tasks, beta=30.0, anchor_layer_name="layer2", target_bn_layers=None):
    routing_container = {"weights": None}
    prototypes = extract_prototypes(merged_backbone, subsets, sorted_tasks, anchor_layer_name)
    
    eval_backbone = create_base_resnet().to(device)
    eval_backbone.load_state_dict(merged_backbone.state_dict())
    eval_backbone.eval()
    
    hooks = []
    
    # 1. Pre-forward hook
    def reset_pre_hook(module, input):
        routing_container["weights"] = None
    hooks.append(eval_backbone.register_forward_pre_hook(reset_pre_hook))
    
    # 2. Anchor Hook
    def srls_anchor_hook(module, input, output):
        B = output.shape[0]
        pooled = output.mean(dim=[2, 3])
        pooled_norm = pooled / (pooled.norm(p=2, dim=1, keepdim=True) + 1e-8)
        
        sims = []
        for name in sorted_tasks:
            proto = prototypes[name]
            sim = torch.sum(pooled_norm * proto.unsqueeze(0), dim=1)
            sims.append(sim)
        sims = torch.stack(sims, dim=1)
        routing_container["weights"] = torch.softmax(beta * sims, dim=1)
        
    anchor_layer = getattr(eval_backbone, anchor_layer_name)
    hooks.append(anchor_layer.register_forward_hook(srls_anchor_hook))
    
    # 3. Layer-wise Scaling Hook
    def make_bn_hook(l_name):
        def bn_hook(module, input, output):
            routing_weights = routing_container["weights"]
            B, C, H, W = output.shape
            
            if routing_weights is None or routing_weights.shape[0] != B:
                return output
                
            # Stack gammas across tasks: [3]
            gamma_k = torch.stack([gammas[task_name][l_name] for task_name in sorted_tasks], dim=0)
            
            # Dynamic interpolation: [B]
            gamma_interp = torch.matmul(routing_weights, gamma_k) # [B]
            
            # Reshape for broadcasting to [B, 1, 1, 1]
            gamma_interp = gamma_interp.view(B, 1, 1, 1)
            
            return output * gamma_interp
        return bn_hook
        
    for name, module in eval_backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if target_bn_layers is not None and not any(pat in name for pat in target_bn_layers):
                continue
            hooks.append(module.register_forward_hook(make_bn_hook(name)))
            
    # Evaluation
    accuracies = {}
    for name in sorted_tasks:
        test_loader = DataLoader(subsets[name]['test'], batch_size=256, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                features = eval_backbone(x)
                routing_weights = routing_container["weights"]
                
                if routing_weights is None:
                    routing_weights = torch.ones(x.shape[0], 3, device=device) / 3.0
                    
                logits_mnist = heads['mnist'](features)
                logits_fmnist = heads['fmnist'](features)
                logits_cifar = heads['cifar'](features)
                logits_all = torch.stack([logits_mnist, logits_fmnist, logits_cifar], dim=1)
                
                logits = torch.sum(routing_weights.unsqueeze(-1) * logits_all, dim=1)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        accuracies[name] = (correct / total) * 100.0
        
    for hook in hooks:
        hook.remove()
        
    accuracies['average'] = np.mean([accuracies[k] for k in sorted_tasks])
    return accuracies

def main():
    subsets = get_datasets()
    sorted_tasks = ['mnist', 'fmnist', 'cifar']
    expert_paths = [f"expert_{name}.pth" for name in sorted_tasks]
    
    # Load heads
    experts = {}
    for name in sorted_tasks:
        ckpt = torch.load(f"expert_{name}.pth", map_location=device)
        backbone = create_base_resnet().to(device)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(ckpt['head_state_dict'])
        experts[name] = ExpertModel(backbone, head).to(device)
        
    heads = {name: experts[name].head for name in sorted_tasks}
    merged_backbone = merge_backbones(expert_paths)
    
    gammas = compute_srls_gammas(expert_paths, merged_backbone, subsets, sorted_tasks)
    
    experiments = [
        {"name": "SRLS All BN Layers", "layers": None},
        {"name": "SRLS Deep Layers (layer3, layer4)", "layers": ["layer3", "layer4"]},
        {"name": "SRLS Final Layer (layer4 only)", "layers": ["layer4"]}
    ]
    
    for exp in experiments:
        print(f"\n==================== {exp['name']} ====================")
        for beta in [5.0, 15.0, 30.0, 50.0]:
            accs = run_srls_evaluation(
                merged_backbone, 
                heads, 
                subsets, 
                gammas, 
                sorted_tasks, 
                beta=beta, 
                anchor_layer_name="layer2", 
                target_bn_layers=exp['layers']
            )
            print(f"  Beta={beta:<5} | Avg: {accs['average']:.2f}% (MNIST: {accs['mnist']:.2f}%, F-MNIST: {accs['fmnist']:.2f}%, CIFAR: {accs['cifar']:.2f}%)")

if __name__ == "__main__":
    main()
