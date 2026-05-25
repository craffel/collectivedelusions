import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
import numpy as np
from models import get_resnet18_32x32, merge_models_weight_space
from baselines import project_simplex_pytorch, evaluate_model

def get_merged_params_and_buffers(experts, coefficients):
    """
    Differentiably merges parameters and buffers of the experts.
    coefficients: tensor of shape [K] with requires_grad=True
    """
    K = len(experts)
    merged_dict = {}
    
    # Merge parameters
    expert0_params = dict(experts[0].named_parameters())
    for name, param in expert0_params.items():
        # Differentiable weighted sum
        merged_val = torch.zeros_like(param)
        for k in range(K):
            expert_k_param = dict(experts[k].named_parameters())[name]
            merged_val = merged_val + coefficients[k] * expert_k_param
        merged_dict[name] = merged_val
        
    # Merge buffers (like batchnorm running stats)
    expert0_buffers = dict(experts[0].named_buffers())
    for name, buffer in expert0_buffers.items():
        merged_val = torch.zeros_like(buffer, dtype=torch.float32)
        for k in range(K):
            expert_k_buffer = dict(experts[k].named_buffers())[name].to(torch.float32)
            # Use detached coefficients for buffers to prevent requires_grad=True on batchnorm stats
            merged_val = merged_val + coefficients[k].detach() * expert_k_buffer
        merged_dict[name] = merged_val.to(buffer.dtype).detach()
        
    return merged_dict

def run_proto_ttmm(experts, stream, known_prototypes, 
                   tau_N=0.50, tau_conf=0.90, alpha=0.99, 
                   gamma_b=0.10, gamma_p=0.10, temp=0.10, 
                   eta=0.01, opt_steps=5, device="cuda"):
    """
    Standard PROTO-TTMM Algorithm.
    """
    print("Running PROTO-TTMM Baseline...")
    K = len(experts)
    model = get_resnet18_32x32().to(device)
    
    # Initialize coefficients (deployed configured as Expert 0)
    lambda_t = torch.zeros(K, device=device)
    lambda_t[0] = 1.0
    
    # Initialize IFC bias vector
    v_bias = torch.zeros(512, device=device)
    
    # Novel prototypes slot
    P_novel = None
    
    accuracies = []
    lambda_history = []
    novel_detection_history = []
    
    for t, (inputs, targets, domain) in enumerate(stream):
        inputs = inputs.to(device)
        
        # 1. Differentiable merged dictionary for forward pass
        merged_dict = get_merged_params_and_buffers(experts, lambda_t)
        
        # 2. Extract features and predictions
        with torch.no_grad():
            outputs = functional_call(model, merged_dict, inputs)
            feats = functional_call(model, merged_dict, inputs, kwargs={"return_features": True})
            
        # Compute accuracy on active batch
        _, preds = outputs.max(1)
        correct = preds.eq(targets.to(device)).sum().item()
        acc = 100.0 * correct / targets.size(0)
        accuracies.append(acc)
        lambda_history.append(lambda_t.cpu().numpy().copy())
        
        # 3. Update Isotropic Feature Centering (IFC)
        with torch.no_grad():
            batch_mean_feat = feats.mean(dim=0)
            if t == 0:
                v_bias = batch_mean_feat.clone()
            else:
                v_bias = (1.0 - gamma_b) * v_bias + gamma_b * batch_mean_feat
                
            # Center and L2-normalize features
            feats_centered = feats - v_bias
            feats_normalized = feats_centered / (feats_centered.norm(p=2, dim=1, keepdim=True) + 1e-8)
            
            # Generate pseudo-labels
            probs = F.softmax(outputs, dim=1)
            conf, pseudo_labels = probs.max(dim=1)
            mask = conf > tau_conf
            
        # 4. Decoupled Unbiased Routing and Novelty Detection via Individual Expert Cohesion
        cohesions = []
        for k in range(len(known_prototypes)):
            expert_k = experts[k]
            expert_k.eval()
            with torch.no_grad():
                outputs_k = expert_k(inputs)
                feats_k = expert_k.extract_features(inputs)
                # Center features to match isotropic feature space
                feats_k_centered = feats_k - feats_k.mean(dim=0, keepdim=True)
                feats_k_norm = feats_k_centered / (feats_k_centered.norm(p=2, dim=1, keepdim=True) + 1e-8)
                
                probs_k = F.softmax(outputs_k, dim=1)
                _, pseudo_k = probs_k.max(dim=1)
                
                proto_k = known_prototypes[k]
                proto_samples = proto_k[pseudo_k] # Use expert's own pseudo labels to prevent feedback trap
                cos_sim = torch.sum(feats_k_norm * proto_samples, dim=1)
                cohesions.append(cos_sim.mean().item())
                
        max_cohesion = max(cohesions)
        best_expert = np.argmax(cohesions)
        is_novel = (max_cohesion < tau_N)
        
        # Record if we detected novelty (domain == "fmnist" is the true novel domain)
        novel_detection_history.append(is_novel)
        
        if not is_novel:
            # Case 1: Known Domain - route to best matching expert using EMA
            if mask.sum() > 0:
                one_hot = torch.zeros(K, device=device)
                one_hot[best_expert] = 1.0
                lambda_t = (1.0 - alpha) * lambda_t + alpha * one_hot
                lambda_t = lambda_t / lambda_t.sum()
        else:
            # Case 2: Novel Domain - initialize / update novel prototypes and update lambda via contrastive alignment
            if mask.sum() > 0:
                filtered_feats = feats_normalized[mask]
                filtered_pseudo = pseudo_labels[mask]
            else:
                filtered_feats = feats_normalized
                filtered_pseudo = pseudo_labels
                # Force mask to be all True for contrastive alignment to use all samples
                mask = torch.ones_like(mask, dtype=torch.bool)
                
            if P_novel is None:
                P_novel = torch.zeros(10, 512, device=device)
                # Initialize with current centered features
                for c in range(10):
                    c_mask = (filtered_pseudo == c)
                    if c_mask.sum() > 0:
                        class_mean = filtered_feats[c_mask].mean(dim=0)
                        P_novel[c] = class_mean / (class_mean.norm(p=2) + 1e-8)
            else:
                # Update with EMA
                for c in range(10):
                    c_mask = (filtered_pseudo == c)
                    if c_mask.sum() > 0:
                        class_mean = filtered_feats[c_mask].mean(dim=0)
                        updated_proto = (1.0 - gamma_p) * P_novel[c] + gamma_p * class_mean
                        P_novel[c] = updated_proto / (updated_proto.norm(p=2) + 1e-8)
                        
            # Precompute expert features once
            expert_feats = []
            for expert in experts:
                expert.eval()
                with torch.no_grad():
                    feats_k = expert.extract_features(inputs)
                    expert_feats.append(feats_k)
                    
            # Differentiable update of lambda using contrastive alignment loss
            # Define lambda as a differentiable parameter
            lambda_param = nn.Parameter(lambda_t.clone())
            optimizer = torch.optim.SGD([lambda_param], lr=eta)
            
            # Concatenate all prototypes: [30, 512]
            # Known expert prototypes: known_prototypes[0] is [10, 512], known_prototypes[1] is [10, 512]
            all_protos = torch.cat([known_prototypes[0], known_prototypes[1], P_novel], dim=0)
            
            for step in range(opt_steps):
                optimizer.zero_grad()
                
                # Linearly merge features in activation-space: sum_k lambda_k * feats_k
                feats_step = torch.zeros_like(expert_feats[0])
                for k in range(K):
                    feats_step = feats_step + lambda_param[k] * expert_feats[k]
                
                # Center and normalize features
                feats_step_centered = feats_step - v_bias
                feats_step_norm = feats_step_centered / (feats_step_centered.norm(p=2, dim=1, keepdim=True) + 1e-8)
                
                # Filter high-confidence samples
                feats_step_filtered = feats_step_norm[mask]
                pseudo_step_filtered = filtered_pseudo
                
                # Compute contrastive loss
                # Logits shape: [M, 30] representing similarity of M samples to 30 prototypes
                similarity_matrix = torch.matmul(feats_step_filtered, all_protos.T) / temp
                
                # Target prototype index is K * 10 + pseudo_label (which puts it in the novel prototypes slot)
                # Here K = 2 (known experts)
                targets_step = 2 * 10 + pseudo_step_filtered
                
                loss = F.cross_entropy(similarity_matrix, targets_step)
                loss.backward()
                optimizer.step()
                
                # Project back onto the probability simplex
                with torch.no_grad():
                    lambda_param.copy_(project_simplex_pytorch(lambda_param))
                    
            lambda_t = lambda_param.data.clone()
            
    print(f"PROTO-TTMM Accuracy: {np.mean(accuracies):.2f}%")
    return accuracies, lambda_history, novel_detection_history
