import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import merge_models_weight_space

def project_simplex_pytorch(v):
    """
    Projects a vector v onto the probability simplex: sum(x) = 1, x >= 0.
    v: tensor of shape [K] or [B, K]
    """
    if v.dim() == 1:
        v_sorted, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(v_sorted, dim=0)
        ind = torch.arange(1, len(v) + 1, device=v.device, dtype=v.dtype)
        cond = v_sorted - (cssv - 1.0) / ind > 0
        # Find the last index where condition is true
        idx = torch.nonzero(cond).squeeze()
        if idx.numel() == 0:
            idx = 0
        else:
            idx = idx.max().item()
        theta = (cssv[idx] - 1.0) / (idx + 1)
        return torch.clamp(v - theta, min=0.0)
    else:
        # Batch mode
        B, K = v.shape
        v_sorted, _ = torch.sort(v, dim=1, descending=True)
        cssv = torch.cumsum(v_sorted, dim=1)
        ind = torch.arange(1, K + 1, device=v.device, dtype=v.dtype).unsqueeze(0).expand(B, K)
        cond = v_sorted - (cssv - 1.0) / ind > 0
        
        projected = torch.zeros_like(v)
        for i in range(B):
            idx = torch.nonzero(cond[i]).squeeze()
            if idx.numel() == 0:
                idx = 0
            else:
                idx = idx.max().item()
            theta = (cssv[i, idx] - 1.0) / (idx + 1)
            projected[i] = torch.clamp(v[i] - theta, min=0.0)
        return projected

def evaluate_model(model, inputs, targets, device):
    """
    Evaluates model accuracy on a single batch.
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, preds = outputs.max(1)
        correct = preds.eq(targets).sum().item()
        acc = 100.0 * correct / targets.size(0)
    return acc, outputs

# 1. Static Merging Baseline
def run_static_baseline(experts, stream, device):
    print("Running Static (Uniform) Merging Baseline...")
    K = len(experts)
    lambda_static = torch.tensor([1.0 / K] * K, device=device)
    merged_model = merge_models_weight_space(experts, lambda_static)
    
    accuracies = []
    for t, (inputs, targets, domain) in enumerate(stream):
        acc, _ = evaluate_model(merged_model, inputs, targets, device)
        accuracies.append(acc)
    print(f"Static Merging Accuracy: {np.mean(accuracies):.2f}%")
    return accuracies

# 2. CPA-Merge Baseline (Closed-World Entropy Confidence routing)
def run_cpa_merge(experts, stream, alpha=0.99, device="cuda"):
    print("Running CPA-Merge Baseline...")
    K = len(experts)
    lambda_t = torch.tensor([1.0 / K] * K, device=device)
    
    accuracies = []
    lambda_history = []
    
    for t, (inputs, targets, domain) in enumerate(stream):
        # 1. Evaluate merged model on active batch
        merged_model = merge_models_weight_space(experts, lambda_t)
        acc, _ = evaluate_model(merged_model, inputs, targets, device)
        accuracies.append(acc)
        lambda_history.append(lambda_t.cpu().numpy().copy())
        
        # 2. Compute predictive confidence (negative entropy) of each individual expert
        confidences = []
        for expert in experts:
            expert.eval()
            expert.to(device)
            with torch.no_grad():
                outputs = expert(inputs.to(device))
                probs = F.softmax(outputs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()
                confidences.append(-entropy)
                
        # 3. Route to the expert with highest confidence (lowest entropy)
        best_expert = np.argmax(confidences)
        one_hot = torch.zeros(K, device=device)
        one_hot[best_expert] = 1.0
        
        # 4. Update lambda via EMA
        lambda_t = (1.0 - alpha) * lambda_t + alpha * one_hot
        lambda_t = lambda_t / lambda_t.sum() # Normalize to ensure sum=1
        
    print(f"CPA-Merge Accuracy: {np.mean(accuracies):.2f}%")
    return accuracies, lambda_history

# 3. PC-Merge Baseline (Closed-World Class Prototype Cohesion routing)
def run_pc_merge(experts, stream, known_prototypes, alpha=0.99, tau_conf=0.90, device="cuda"):
    """
    known_prototypes: list of K tensors of shape [10, 512] (class prototypes for each expert)
    """
    print("Running PC-Merge Baseline...")
    K = len(experts)
    lambda_t = torch.tensor([1.0 / K] * K, device=device)
    
    accuracies = []
    lambda_history = []
    
    for t, (inputs, targets, domain) in enumerate(stream):
        # 1. Evaluate merged model
        merged_model = merge_models_weight_space(experts, lambda_t)
        acc, outputs = evaluate_model(merged_model, inputs, targets, device)
        accuracies.append(acc)
        lambda_history.append(lambda_t.cpu().numpy().copy())
        
        # 2. Extract features and pseudo-labels from merged model
        merged_model.eval()
        with torch.no_grad():
            feats = merged_model.extract_features(inputs.to(device)) # Shape: [B, 512]
            feats = feats / (feats.norm(p=2, dim=1, keepdim=True) + 1e-8)
            probs = F.softmax(outputs, dim=1)
            conf, pseudo_labels = probs.max(dim=1)
            
            # Filter high confidence samples
            mask = conf > tau_conf
            
        if mask.sum() > 0:
            filtered_feats = feats[mask]
            filtered_pseudo = pseudo_labels[mask]
            
            # 3. Compute class-specific cohesion scores for each expert
            cohesions = []
            for k in range(len(known_prototypes)):
                # Retrieve expert k's class prototypes: shape [10, 512]
                proto_k = known_prototypes[k]
                
                # For each sample, compute cosine similarity between its feature and the expert's prototype of its pseudo-label
                proto_samples = proto_k[filtered_pseudo] # Shape: [M, 512]
                cos_sim = torch.sum(filtered_feats * proto_samples, dim=1) # Shape: [M]
                cohesions.append(cos_sim.mean().item())
                
            best_expert = np.argmax(cohesions)
            one_hot = torch.zeros(K, device=device)
            one_hot[best_expert] = 1.0
            
            # Update lambda via EMA
            lambda_t = (1.0 - alpha) * lambda_t + alpha * one_hot
            lambda_t = lambda_t / lambda_t.sum()
            
    print(f"PC-Merge Accuracy: {np.mean(accuracies):.2f}%")
    return accuracies, lambda_history
