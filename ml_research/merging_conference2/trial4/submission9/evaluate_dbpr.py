import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from run_experiments import get_datasets, create_base_resnet, ExpertModel, merge_backbones

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def extract_prototypes(backbone, subsets, sorted_tasks, anchor_layer_name="layer2"):
    backbone.eval()
    prototypes = {}
    anchor_act = None
    
    def anchor_hook(module, input, output):
        nonlocal anchor_act
        anchor_act = output.detach()
        
    # Get the anchor layer
    anchor_layer = getattr(backbone, anchor_layer_name)
    hook_handle = anchor_layer.register_forward_hook(anchor_hook)
    
    for name in sorted_tasks:
        cal_loader = DataLoader(subsets[name]['cal'], batch_size=128, shuffle=False)
        with torch.no_grad():
            for x, _ in cal_loader:
                x = x.to(device)
                backbone(x)
                break # Extract from first batch
        # Apply global average pooling
        pooled = anchor_act.mean(dim=[2, 3]) # [B, C]
        proto = pooled.mean(dim=0) # [C]
        proto = proto / (proto.norm(p=2) + 1e-8) # L2 Normalization
        prototypes[name] = proto
        
    hook_handle.remove()
    return prototypes

def run_dbpr_evaluation(merged_backbone, heads, subsets, expert_bn_params, sorted_tasks, beta=30.0, anchor_layer_name="layer2", target_bn_layers=None):
    # Dictionary to share routing weights from anchor hook
    routing_container = {"weights": None}
    
    prototypes = extract_prototypes(merged_backbone, subsets, sorted_tasks, anchor_layer_name)
    
    eval_backbone = create_base_resnet().to(device)
    eval_backbone.load_state_dict(merged_backbone.state_dict())
    eval_backbone.eval()
    
    hooks = []
    
    # 1. Pre-forward hook to reset routing weights at the start of each forward pass
    def reset_pre_hook(module, input):
        routing_container["weights"] = None
    hooks.append(eval_backbone.register_forward_pre_hook(reset_pre_hook))
    
    # 2. Anchor Hook to compute routing weights
    def dbpr_anchor_hook(module, input, output):
        B = output.shape[0]
        pooled = output.mean(dim=[2, 3]) # [B, C]
        pooled_norm = pooled / (pooled.norm(p=2, dim=1, keepdim=True) + 1e-8)
        
        sims = []
        for name in sorted_tasks:
            proto = prototypes[name]
            sim = torch.sum(pooled_norm * proto.unsqueeze(0), dim=1)
            sims.append(sim)
        sims = torch.stack(sims, dim=1) # [B, 3]
        routing_container["weights"] = torch.softmax(beta * sims, dim=1)
        
    anchor_layer = getattr(eval_backbone, anchor_layer_name)
    hooks.append(anchor_layer.register_forward_hook(dbpr_anchor_hook))
    
    # 3. Calibration Hook
    def make_bn_hook(l_name):
        def bn_hook(module, input, output):
            routing_weights = routing_container["weights"]
            x = input[0]
            B, C, H, W = x.shape
            
            # Defensive checks: if routing weights are not computed yet, or if batch size mismatch
            if routing_weights is None or routing_weights.shape[0] != B:
                return output
                
            eps = module.eps
            
            mean_experts = torch.stack([expert_bn_params[task_name][l_name]['running_mean'] for task_name in sorted_tasks], dim=0)
            var_experts = torch.stack([expert_bn_params[task_name][l_name]['running_var'] for task_name in sorted_tasks], dim=0)
            weight_experts = torch.stack([expert_bn_params[task_name][l_name]['weight'] for task_name in sorted_tasks], dim=0)
            bias_experts = torch.stack([expert_bn_params[task_name][l_name]['bias'] for task_name in sorted_tasks], dim=0)
            
            mean_interp = torch.matmul(routing_weights, mean_experts).view(B, C, 1, 1)
            var_interp = torch.matmul(routing_weights, var_experts).view(B, C, 1, 1)
            weight_interp = torch.matmul(routing_weights, weight_experts).view(B, C, 1, 1)
            bias_interp = torch.matmul(routing_weights, bias_experts).view(B, C, 1, 1)
            
            std_interp = torch.sqrt(var_interp + eps)
            x_normalized = (x - mean_interp) / std_interp
            y = x_normalized * weight_interp + bias_interp
            return y
        return bn_hook
        
    for name, module in eval_backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if target_bn_layers is not None and not any(pat in name for pat in target_bn_layers):
                continue
            hooks.append(module.register_forward_hook(make_bn_hook(name)))
            
    # Run evaluation
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
                
                # If routing_weights is None (e.g. if we are using an anchor layer that hasn't run), fallback
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

if __name__ == "__main__":
    subsets = get_datasets()
    sorted_tasks = ['mnist', 'fmnist', 'cifar']
    expert_paths = [f"expert_{name}.pth" for name in sorted_tasks]
    
    expert_bn_params = {}
    experts = {}
    for name in sorted_tasks:
        ckpt = torch.load(f"expert_{name}.pth", map_location=device)
        backbone = create_base_resnet().to(device)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(ckpt['head_state_dict'])
        experts[name] = ExpertModel(backbone, head).to(device)
        experts[name].eval()
        
        expert_bn_params[name] = {}
        for b_name, module in backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                expert_bn_params[name][b_name] = {
                    'running_mean': module.running_mean.clone().to(device),
                    'running_var': module.running_var.clone().to(device),
                    'weight': module.weight.clone().to(device),
                    'bias': module.bias.clone().to(device)
                }
                
    heads = {name: experts[name].head for name in sorted_tasks}
    merged_backbone = merge_backbones(expert_paths)
    
    # 1. Base Uncalibrated WA Accuracy
    print("\nEvaluating Uncalibrated WA Base...")
    correct_base = {name: 0 for name in sorted_tasks}
    total_base = {name: 0 for name in sorted_tasks}
    for name in sorted_tasks:
        test_loader = DataLoader(subsets[name]['test'], batch_size=256, shuffle=False)
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                features = merged_backbone(x)
                if name == 'mnist' and i == 0:
                    print(f"[DEBUG MNIST Batch 0] Features mean: {features.mean().item():.6f}, var: {features.var().item():.6f}")
                logits = heads[name](features)
                correct_base[name] += (logits.argmax(dim=1) == y).sum().item()
                total_base[name] += y.size(0)
    for name in sorted_tasks:
        print(f"  Uncalibrated WA on {name.upper()}: {(correct_base[name]/total_base[name])*100:.2f}%")
        
    experiments = [
        {"name": "DBPR All BN Layers", "layers": None},
        {"name": "DBPR Deep Layers (layer3, layer4)", "layers": ["layer3", "layer4"]},
        {"name": "DBPR Final Layer (layer4 only)", "layers": ["layer4"]}
    ]
    
    for exp in experiments:
        print(f"\n==================== {exp['name']} ====================")
        for beta in [5.0, 15.0, 30.0, 50.0]:
            accs = run_dbpr_evaluation(
                merged_backbone, 
                heads, 
                subsets, 
                expert_bn_params, 
                sorted_tasks, 
                beta=beta, 
                anchor_layer_name="layer2", 
                target_bn_layers=exp['layers']
            )
            print(f"  Beta={beta:<5} | Avg: {accs['average']:.2f}% (MNIST: {accs['mnist']:.2f}%, F-MNIST: {accs['fmnist']:.2f}%, CIFAR: {accs['cifar']:.2f}%)")
