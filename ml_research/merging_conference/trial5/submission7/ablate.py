import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import json

from eval_ttmm import (
    load_experts,
    extract_prototypes,
    compute_fisher_information,
    get_dataset,
    get_transforms,
    load_base_model,
    get_blended_params,
    project_simplex
)
from torch.func import functional_call

def run_evaluation_ablation(
    stream_type,
    corruption,
    experts,
    prototypes,
    joint_fisher,
    use_opr=True,
    projection_space='fisher', # 'fisher', 'euclidean', 'none'
    precondition=True,
    alpha=0.5,
    lr=0.1,
    beta=0.1,
    num_steps=1
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test datasets with appropriate corruptions
    transform = get_transforms(corruption)
    test_datasets = {
        'mnist': get_dataset('mnist', train=False, transform=transform),
        'fashion': get_dataset('fashion', train=False, transform=transform),
        'kmnist': get_dataset('kmnist', train=False, transform=transform)
    }
    
    # Subset to 1,600 samples each (50 batches of size 32)
    torch.manual_seed(42)
    subset_indices = {task: torch.randperm(len(test_datasets[task]))[:1600] for task in test_datasets}
    loaders = {
        task: DataLoader(Subset(test_datasets[task], subset_indices[task]), batch_size=32, shuffle=False)
        for task in test_datasets
    }
    
    # Construct stream batches
    mnist_batches = list(loaders['mnist'])
    fashion_batches = list(loaders['fashion'])
    kmnist_batches = list(loaders['kmnist'])
    
    stream_batches = []
    if stream_type == 'alternating':
        for i in range(50):
            stream_batches.append(('mnist', mnist_batches[i]))
            stream_batches.append(('fashion', fashion_batches[i]))
            stream_batches.append(('kmnist', kmnist_batches[i]))
    elif stream_type == 'sequential':
        for b in mnist_batches:
            stream_batches.append(('mnist', b))
        for b in fashion_batches:
            stream_batches.append(('fashion', b))
        for b in kmnist_batches:
            stream_batches.append(('kmnist', b))
            
    # Base and Merged model instances
    base_weights = load_base_model().state_dict()
    
    # Move base weights and experts to device once
    base_weights_dev = {k: v.to(device) for k, v in base_weights.items()}
    experts_dev = {
        'mnist': {k: v.to(device) for k, v in experts['mnist'].items()},
        'fashion': {k: v.to(device) for k, v in experts['fashion'].items()},
        'kmnist': {k: v.to(device) for k, v in experts['kmnist'].items()}
    }
    merged_model = load_base_model().to(device)
    
    # parameters only (for gradient tracking)
    backbone_layers = [name for name, param in merged_model.named_parameters() if 'fc' not in name]
    # all float state dict keys (parameters + buffers)
    all_backbone_keys = [name for name, param in merged_model.state_dict().items() if 'fc' not in name and param.is_floating_point()]
    
    # Initialize coefficients lambdas for all backbone parameters and buffers
    lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True) for name in all_backbone_keys}
    
    # Prepare Fisher preconditioning weights G_w = (F_w + eps)^alpha
    fisher_weights = {}
    for name in backbone_layers:
        f_val = joint_fisher.get(name, 1e-5)
        g_val = (f_val + 1e-8) ** alpha
        fisher_weights[name] = g_val
        
    # EMA loss tracking for OPR reset
    ema_loss = 0.0
    beta_ema = 0.90
    opr_alpha = 2.5 if corruption != 'clean' else 4.0
    
    correct = 0
    total = 0
    
    # Run stream
    desc = f"Ablation ({stream_type}, {corruption}): OPR={use_opr}, Proj={projection_space}, Precond={precondition}, alpha={alpha}"
    pbar = tqdm(stream_batches, desc=desc[:60])
    for step, (task_name, (images, labels)) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # 1. Prototype-driven Dynamic Routing (PD-Routing) to reset / initialize
        temp_lambdas = {name: torch.tensor([1/3, 1/3, 1/3], device=device) for name in all_backbone_keys}
        blended_params = get_blended_params(base_weights_dev, experts_dev, temp_lambdas, backbone_layers, task_name)
        
        # Temporary replace fc for feature extraction
        original_fc = merged_model.fc
        merged_model.fc = nn.Identity()
        with torch.no_grad():
            anchor_features = functional_call(merged_model, blended_params, images) # [B, 512]
            anchor_features = anchor_features / anchor_features.norm(p=2, dim=1, keepdim=True) # Normalize
        merged_model.fc = original_fc
        
        # Compute cosine similarities with prototypes of each task
        S = []
        for t_idx, t in enumerate(['mnist', 'fashion', 'kmnist']):
            t_proto = prototypes[t].to(device) # [10, 512]
            sims = torch.mm(anchor_features, t_proto.t())
            max_sims, _ = sims.max(dim=1)
            S_k = max_sims.mean().item()
            S.append(S_k)
            
        # Sharp softmax routing prior
        S_tensor = torch.tensor(S, device=device)
        lambda_prior = F.softmax(S_tensor / 0.02, dim=0)
        
        # Re-initialize coefficients to the routing prior
        with torch.no_grad():
            for name in all_backbone_keys:
                lambdas[name].copy_(lambda_prior)
                    
        # 2. Test-Time Adaptation Steps
        for sub_step in range(num_steps):
            # Differentiable forward pass for outputs (predictions)
            blended_params = get_blended_params(base_weights_dev, experts_dev, lambdas, backbone_layers, task_name)
            outputs = functional_call(merged_model, blended_params, images)
            
            probs = F.softmax(outputs, dim=1)
            entropy_loss_samples = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            
            # Differentiable forward pass for feature extraction
            original_fc = merged_model.fc
            merged_model.fc = nn.Identity()
            features = functional_call(merged_model, blended_params, images)
            features_norm = features / features.norm(p=2, dim=1, keepdim=True)
            merged_model.fc = original_fc
            
            # Get active task class prototypes
            active_proto = prototypes[task_name].to(device) # [10, 512]
            M = torch.mm(features_norm, active_proto.t())
            
            # Mask based on prediction confidence
            max_probs, pred_labels = probs.max(dim=1)
            mask = max_probs > 0.85
            
            # Contrastive loss (InfoNCE over classes)
            if mask.sum() > 0:
                M_masked = M[mask] # [num_masked, 10]
                pred_labels_masked = pred_labels[mask]
                
                exp_sims = torch.exp(M_masked / 0.1)
                pos_sims = torch.exp(M_masked[torch.arange(M_masked.size(0)), pred_labels_masked] / 0.1)
                contrast_loss_samples = -torch.log(pos_sims / exp_sims.sum(dim=1))
                
                contrast_loss = torch.zeros(images.size(0), device=device)
                contrast_loss[mask] = contrast_loss_samples
            else:
                contrast_loss = torch.zeros(images.size(0), device=device)
                
            # Sample-wise total loss
            loss_samples = entropy_loss_samples + beta * contrast_loss
            
            # Unsupervised task boundary detection (OPR) via loss spike
            if use_opr:
                mean_loss = loss_samples.mean().item()
                if step == 0:
                    ema_loss = mean_loss
                else:
                    if mean_loss > opr_alpha * ema_loss:
                        with torch.no_grad():
                            for name in all_backbone_keys:
                                lambdas[name].copy_(torch.tensor([1/3, 1/3, 1/3], device=device))
                        ema_loss = mean_loss
                    else:
                        ema_loss = beta_ema * ema_loss + (1 - beta_ema) * mean_loss
                
            # Class-specific Gradient Projection
            active_classes = torch.unique(pred_labels)
            class_grads = {}
            for c in active_classes:
                c_mask = pred_labels == c
                if c_mask.sum() > 0:
                    c_loss = loss_samples[c_mask].mean()
                    c_grads = torch.autograd.grad(c_loss, [lambdas[name] for name in backbone_layers], retain_graph=True, allow_unused=True)
                    class_grads[c.item()] = {name: grad.clone() for name, grad in zip(backbone_layers, c_grads) if grad is not None}
                        
            # Pairwise gradient surgery
            if projection_space != 'none':
                for c_a in class_grads.keys():
                    for c_b in class_grads.keys():
                        if c_a != c_b:
                            dot_prod = 0.0
                            norm_b_sq = 0.0
                            for name in backbone_layers:
                                if name in class_grads[c_a] and name in class_grads[c_b]:
                                    if projection_space == 'fisher':
                                        metric_w = fisher_weights[name]
                                        dot_prod += metric_w * torch.sum(class_grads[c_a][name] * class_grads[c_b][name]).item()
                                        norm_b_sq += metric_w * torch.sum(class_grads[c_b][name] * class_grads[c_b][name]).item()
                                    else: # 'euclidean'
                                        dot_prod += torch.sum(class_grads[c_a][name] * class_grads[c_b][name]).item()
                                        norm_b_sq += torch.sum(class_grads[c_b][name] * class_grads[c_b][name]).item()
                                        
                            if dot_prod < 0:
                                for name in backbone_layers:
                                    if name in class_grads[c_a] and name in class_grads[c_b]:
                                        class_grads[c_a][name] -= (dot_prod / (norm_b_sq + 1e-8)) * class_grads[c_b][name]
                                        
            # Sum conflict-free class gradients and update coefficients
            with torch.no_grad():
                for name in backbone_layers:
                    final_grad = torch.zeros_like(lambdas[name])
                    for c in class_grads.keys():
                        if name in class_grads[c]:
                            final_grad += class_grads[c][name]
                    
                    if precondition:
                        scaled_lr = lr / fisher_weights[name]
                        lambdas[name] -= scaled_lr * final_grad
                    else:
                        lambdas[name] -= lr * final_grad
                    lambdas[name] = project_simplex(lambdas[name])
                        
        # 3. Final Forward Pass and Accuracy Computation
        with torch.no_grad():
            final_blended_params = get_blended_params(base_weights_dev, experts_dev, lambdas, backbone_layers, task_name)
            final_outputs = functional_call(merged_model, final_blended_params, images)
            _, predicted = torch.max(final_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            acc = 100 * correct / total
            pbar.set_postfix(Acc=f"{acc:.2f}%")
            
    final_acc = 100 * correct / total
    return final_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parallel TTMM Ablation Runner")
    parser.add_argument("--mode", type=str, required=True, choices=["component", "alpha"])
    parser.add_argument("--stream", type=str, required=True, choices=["alternating", "sequential"])
    parser.add_argument("--corruption", type=str, required=True, choices=["noise", "contrast"])
    parser.add_argument("--config_name", type=str, choices=["Full_IGGS_Merge", "No_OPR", "Euclidean_Proj", "No_Proj", "No_Preconditioning"])
    parser.add_argument("--alpha", type=float, default=0.5)
    
    args = parser.parse_args()
    
    print(f"Loading experts, prototypes, and Fisher information...")
    experts = load_experts()
    prototypes = extract_prototypes(experts, num_samples=100)
    joint_fisher = compute_fisher_information(experts, num_samples=250)
    
    os.makedirs("ablation_parts", exist_ok=True)
    
    if args.mode == "component":
        configs = {
            'Full_IGGS_Merge': {'use_opr': True, 'projection_space': 'fisher', 'precondition': True},
            'No_OPR': {'use_opr': False, 'projection_space': 'fisher', 'precondition': True},
            'Euclidean_Proj': {'use_opr': True, 'projection_space': 'euclidean', 'precondition': True},
            'No_Proj': {'use_opr': True, 'projection_space': 'none', 'precondition': True},
            'No_Preconditioning': {'use_opr': True, 'projection_space': 'fisher', 'precondition': False}
        }
        cfg = configs[args.config_name]
        print(f"Running Component Ablation: {args.config_name} | Stream: {args.stream} | Corruption: {args.corruption}")
        acc = run_evaluation_ablation(
            args.stream, args.corruption, experts, prototypes, joint_fisher,
            use_opr=cfg['use_opr'],
            projection_space=cfg['projection_space'],
            precondition=cfg['precondition'],
            alpha=0.5
        )
        print(f"Final Accuracy: {acc:.2f}%")
        
        # Save part
        res = {
            "mode": "component",
            "stream": args.stream,
            "corruption": args.corruption,
            "config_name": args.config_name,
            "accuracy": acc
        }
        filename = f"ablation_parts/component_{args.stream}_{args.corruption}_{args.config_name}.json"
        with open(filename, "w") as f:
            json.dump(res, f)
            
    elif args.mode == "alpha":
        print(f"Running alpha Sweep: alpha={args.alpha} | Stream: {args.stream} | Corruption: {args.corruption}")
        acc = run_evaluation_ablation(
            args.stream, args.corruption, experts, prototypes, joint_fisher,
            use_opr=True,
            projection_space='fisher',
            precondition=True,
            alpha=args.alpha
        )
        print(f"Final Accuracy: {acc:.2f}%")
        
        # Save part
        res = {
            "mode": "alpha",
            "stream": args.stream,
            "corruption": args.corruption,
            "alpha": args.alpha,
            "accuracy": acc
        }
        filename = f"ablation_parts/alpha_{args.stream}_{args.corruption}_{args.alpha}.json"
        with open(filename, "w") as f:
            json.dump(res, f)
