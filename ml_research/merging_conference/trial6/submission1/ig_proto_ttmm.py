import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
import numpy as np
from models import get_resnet18_32x32
from baselines import project_simplex_pytorch, evaluate_model
from proto_ttmm import get_merged_params_and_buffers

def run_ig_proto_ttmm(experts, stream, known_prototypes, expert_fishers,
                      tau_N=0.50, tau_conf=0.90, alpha=0.99, 
                      gamma_b=0.10, gamma_p=0.10, temp=0.10, 
                      eta=0.01, opt_steps=5, 
                      damping_alpha=0.5, eps_scale=1e-5, use_iggs=True, device="cuda"):
    """
    Proposed IG-PROTO-TTMM Algorithm.
    Integrates open-world TTMM with layer-wise Riemannian optimization, 
    Fisher preconditioning, and Information-Geometric Gradient Surgery.
    """
    print("Running Proposed IG-PROTO-TTMM...")
    K = len(experts)
    model = get_resnet18_32x32().to(device)
    
    # 1. Precompute layer-wise joint Fisher and metric G_w
    G = {}
    for name, param in experts[0].named_parameters():
        if param.requires_grad:
            # Average Fisher across experts
            joint_fisher_tensor = torch.zeros_like(param, device=device)
            for k in range(K):
                joint_fisher_tensor += expert_fishers[k][name].to(device)
            joint_fisher_tensor /= K
            
            # Compute scalar sensitivity of the layer (mean Fisher)
            F_w_scalar = joint_fisher_tensor.mean().item()
            
            # Compute Riemannian metric scalar G_w
            G[name] = (F_w_scalar + eps_scale) ** damping_alpha
            
    # 2. Initialize layer-wise merging coefficients
    # Each parameter tensor has its own vector of size K, initialized as Expert 0
    lambda_dict = {}
    for name, param in experts[0].named_parameters():
        init_coeff = torch.zeros(K, device=device)
        init_coeff[0] = 1.0
        lambda_dict[name] = init_coeff
        
    # Initialize IFC bias vector
    v_bias = torch.zeros(512, device=device)
    
    # Novel prototypes slot
    P_novel = None
    
    accuracies = []
    lambda_history = []
    novel_detection_history = []
    
    for t, (inputs, targets, domain) in enumerate(stream):
        inputs = inputs.to(device)
        
        # 3. Differentiable merged dictionary for forward pass
        # Since lambda_dict is a dict, we pass it to get_merged_params_and_buffers modified to handle dict-based coefficients
        # Let's write a helper to merge weights with layer-wise coefficients
        merged_dict = {}
        for name, param in experts[0].named_parameters():
            coeff = lambda_dict[name]
            merged_val = torch.zeros_like(param)
            for k in range(K):
                expert_k_param = dict(experts[k].named_parameters())[name]
                merged_val = merged_val + coeff[k] * expert_k_param
            merged_dict[name] = merged_val
            
        # Merge buffers using the average coefficient
        # Buffers (like BatchNorm running stats) can use the average coefficient across all layers for simplicity
        avg_coeff = torch.stack(list(lambda_dict.values())).mean(dim=0)
        for name, buffer in experts[0].named_buffers():
            merged_val = torch.zeros_like(buffer, dtype=torch.float32)
            for k in range(K):
                expert_k_buffer = dict(experts[k].named_buffers())[name].to(torch.float32)
                merged_val = merged_val + avg_coeff[k].detach() * expert_k_buffer
            merged_dict[name] = merged_val.to(buffer.dtype).detach()
            
        # 4. Extract features and predictions
        with torch.no_grad():
            outputs = functional_call(model, merged_dict, inputs)
            feats = functional_call(model, merged_dict, inputs, kwargs={"return_features": True})
            
        # Compute accuracy
        _, preds = outputs.max(1)
        correct = preds.eq(targets.to(device)).sum().item()
        acc = 100.0 * correct / targets.size(0)
        accuracies.append(acc)
        lambda_history.append(avg_coeff.cpu().numpy().copy())
        
        # 5. Update Isotropic Feature Centering (IFC)
        with torch.no_grad():
            batch_mean_feat = feats.mean(dim=0)
            if t == 0:
                v_bias = batch_mean_feat.clone()
            else:
                v_bias = (1.0 - gamma_b) * v_bias + gamma_b * batch_mean_feat
                
            feats_centered = feats - v_bias
            feats_normalized = feats_centered / (feats_centered.norm(p=2, dim=1, keepdim=True) + 1e-8)
            
            probs = F.softmax(outputs, dim=1)
            conf, pseudo_labels = probs.max(dim=1)
            mask = conf > tau_conf
            
        # 6. Decoupled Unbiased Routing and Novelty Detection via Individual Expert Cohesion
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
                for name in lambda_dict.keys():
                    lambda_dict[name] = (1.0 - alpha) * lambda_dict[name] + alpha * one_hot
                    lambda_dict[name] = lambda_dict[name] / lambda_dict[name].sum()
        else:
            # Case 2: Novel Domain - initialize / update novel prototypes
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
                for c in range(10):
                    c_mask = (filtered_pseudo == c)
                    if c_mask.sum() > 0:
                        class_mean = filtered_feats[c_mask].mean(dim=0)
                        P_novel[c] = class_mean / (class_mean.norm(p=2) + 1e-8)
            else:
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
                    
            # Differentiable update of lambda_dict via Information-Geometric Gradient Surgery
            all_protos = torch.cat([known_prototypes[0], known_prototypes[1], P_novel], dim=0)
            
            # Setup differentiable parameters for optimization
            lambda_params = {name: nn.Parameter(val.clone()) for name, val in lambda_dict.items()}
            
            for step in range(opt_steps):
                # We perform gradient surgery over classes present in the batch
                classes_present = torch.unique(filtered_pseudo).tolist()
                class_gradients = {c: {} for c in classes_present}
                
                # Compute average coefficient for linear activation merging
                step_avg_coeff = torch.stack(list(lambda_params.values())).mean(dim=0)
                
                # Compute class-specific gradients
                for c in classes_present:
                    # Filter inputs of class c
                    c_mask = (filtered_pseudo == c)
                    
                    # Linearly merge features in activation-space for class c
                    feats_step_c = torch.zeros_like(expert_feats[0][mask][c_mask])
                    for k in range(K):
                        feats_step_c = feats_step_c + step_avg_coeff[k] * expert_feats[k][mask][c_mask]
                        
                    # Center and normalize features
                    feats_step_centered = feats_step_c - v_bias
                    feats_step_norm = feats_step_centered / (feats_step_centered.norm(p=2, dim=1, keepdim=True) + 1e-8)
                    
                    similarity_matrix = torch.matmul(feats_step_norm, all_protos.T) / temp
                    targets_step = torch.tensor([2 * 10 + c] * feats_step_norm.size(0), device=device)
                    
                    loss_c = F.cross_entropy(similarity_matrix, targets_step)
                    
                    # Zero gradients and backward to get class-specific gradients
                    for p in lambda_params.values():
                        if p.grad is not None:
                            p.grad.zero_()
                            
                    loss_c.backward()
                    
                    # Store gradients
                    for name, p in lambda_params.items():
                        if p.grad is not None:
                            class_gradients[c][name] = p.grad.data.clone()
                        else:
                            class_gradients[c][name] = torch.zeros_like(lambda_params[name])
                            
                # Perform Information-Geometric Gradient Surgery (IGGS)
                # Compute pairwise Riemannian inner products
                # Inner product: <g_a, g_b>_F = sum_w G_w * (g_a_w . g_b_w)
                projected_gradients = {c: {name: grad.clone() for name, grad in class_gradients[c].items()} for c in classes_present}
                
                if use_iggs:
                    keys = list(lambda_params.keys())
                    G_tensor = torch.tensor([G[name] for name in keys], device=device)
                    
                    stacked_grads = {}
                    for c in classes_present:
                        stacked_grads[c] = torch.stack([class_gradients[c][name] for name in keys])
                        
                    for i in range(len(classes_present)):
                        for j in range(len(classes_present)):
                            if i == j:
                                continue
                            ca = classes_present[i]
                            cb = classes_present[j]
                            
                            # Tensorized Riemannian inner product
                            g_a = stacked_grads[ca]
                            g_b = stacked_grads[cb]
                            
                            inner_prod = torch.sum(G_tensor * torch.sum(g_a * g_b, dim=1))
                            norm_b = torch.sum(G_tensor * torch.sum(g_b * g_b, dim=1))
                            
                            # If conflict, project
                            if inner_prod.item() < 0:
                                for name in keys:
                                    g_a_w = projected_gradients[ca][name]
                                    g_b_w = class_gradients[cb][name]
                                    projected_gradients[ca][name] = g_a_w - (inner_prod / (norm_b + 1e-8)) * g_b_w
                                
                # Sum projected gradients
                final_gradients = {}
                for name in lambda_params.keys():
                    final_gradients[name] = torch.zeros_like(lambda_params[name])
                    for c in classes_present:
                        final_gradients[name] += projected_gradients[c][name]
                        
                # Perform Riemannian Gradient Descent update: lambda_w = lambda_w - eta * G_w^-1 * g_final_w
                with torch.no_grad():
                    for name, p in lambda_params.items():
                        g_final_w = final_gradients[name]
                        # Riemannian gradient update scaled by G_w^-1 (preconditioning)
                        update_val = (eta / G[name]) * g_final_w
                        p.copy_(project_simplex_pytorch(p - update_val))
                        
            # Store updated coefficients
            with torch.no_grad():
                for name in lambda_dict.keys():
                    lambda_dict[name] = lambda_params[name].data.clone()
                    
    print(f"Proposed IG-PROTO-TTMM Accuracy: {np.mean(accuracies):.2f}%")
    return accuracies, lambda_history, novel_detection_history
