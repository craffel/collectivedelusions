import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# SimpleCNN Architecture definition
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        features = F.relu(self.fc1(x))
        logits = self.fc2(self.dropout(features))
        return logits, features

# Helper to merge model parameters (including MoG BN Buffer Fusion)
def merge_models(model0, model1, lambdas):
    # lambdas can be a dictionary mapping parameter/layer names to merging coefficients
    # or a dictionary mapping layer prefix to float values
    merged_model = SimpleCNN()
    state_dict0 = model0.state_dict()
    state_dict1 = model1.state_dict()
    merged_state_dict = {}
    
    # Layer groups for lambda mapping
    layer_groups = ['conv1', 'bn1', 'conv2', 'bn2', 'fc1', 'fc2']
    
    for key in state_dict0.keys():
        # Identify layer group
        group = None
        for g in layer_groups:
            if g in key:
                group = g
                break
        
        lambda_val = lambdas[group] if group in lambdas else 0.5
        
        if 'running_mean' in key:
            # BN running mean: linear interpolation
            merged_state_dict[key] = (1.0 - lambda_val) * state_dict0[key] + lambda_val * state_dict1[key]
        elif 'running_var' in key:
            # BN running var: MoG moment-matching formula
            mean_key = key.replace('running_var', 'running_mean')
            mu0 = state_dict0[mean_key]
            mu1 = state_dict1[mean_key]
            mu_fused = (1.0 - lambda_val) * mu0 + lambda_val * mu1
            
            var0 = state_dict0[key]
            var1 = state_dict1[key]
            
            merged_state_dict[key] = (1.0 - lambda_val) * (var0 + (mu0 - mu_fused)**2) + lambda_val * (var1 + (mu1 - mu_fused)**2)
        elif 'num_batches_tracked' in key:
            merged_state_dict[key] = state_dict0[key]
        else:
            # Parameter weights and biases
            merged_state_dict[key] = (1.0 - lambda_val) * state_dict0[key] + lambda_val * state_dict1[key]
            
    merged_model.load_state_dict(merged_state_dict)
    return merged_model

# Hoyer's Sparsity helper
def hoyer_sparsity(x):
    # Flatten batch and compute Hoyer's sparsity
    # x is a tensor of shape [B, C, H, W]
    flat = x.view(x.size(0), -1) # [B, D]
    l1 = torch.norm(flat, p=1, dim=1)
    l2 = torch.norm(flat, p=2, dim=1)
    d = flat.size(1)
    
    # Avoid division by zero
    l2 = torch.where(l2 == 0, torch.ones_like(l2) * 1e-8, l2)
    sparsity = (np.sqrt(d) - l1 / l2) / (np.sqrt(d) - 1.0)
    return sparsity.mean().item()

# Denoised Hoyer's sparsity helper (from Paper 4 / BK-AHR)
def denoised_hoyer_sparsity(x):
    # Map normalized pixels in [-1.0, 1.0] back to [0.0, 1.0]
    x_pos = (x + 1.0) / 2.0
    # Threshold denoising: keep only values > 0.35
    x_denoised = torch.where(x_pos > 0.35, x_pos, torch.zeros_like(x_pos))
    return hoyer_sparsity(x_denoised)

# Precompute class prototypes on clean data
def precompute_prototypes(model, dataloader, device):
    model.eval()
    features_all = [[] for _ in range(10)]
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _, feats = model(images)
            # Normalize features on unit sphere
            feats_norm = feats / (torch.norm(feats, p=2, dim=1, keepdim=True) + 1e-8)
            for f, l in zip(feats_norm, labels):
                features_all[l.item()].append(f)
                
    prototypes = torch.zeros(10, 128).to(device)
    for c in range(10):
        if len(features_all[c]) > 0:
            prototypes[c] = torch.stack(features_all[c]).mean(dim=0)
            prototypes[c] = prototypes[c] / (torch.norm(prototypes[c], p=2) + 1e-8)
    return prototypes

# Compute predictive entropy of model predictions
def compute_entropy(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy.mean()

# Compute Routing Priors (AHR-SATS-DUN)
def compute_routing_priors(batch, model0, model1, proto0, proto1, device):
    # 1. Sparsity Estimation
    h_batch = denoised_hoyer_sparsity(batch)
    
    # 2. Metric selection
    if h_batch >= 0.50:
        routing_type = 0 # Normalized L2 Distance
    else:
        routing_type = 1 # Angular Distance
        
    model0.eval()
    model1.eval()
    
    with torch.no_grad():
        _, feats0 = model0(batch.to(device))
        _, feats1 = model1(batch.to(device))
        
        # Normalize features
        feats0_norm = feats0 / (torch.norm(feats0, p=2, dim=1, keepdim=True) + 1e-8)
        feats1_norm = feats1 / (torch.norm(feats1, p=2, dim=1, keepdim=True) + 1e-8)
        
        # Calculate distances
        d0_batch = []
        d1_batch = []
        
        B = batch.size(0)
        for i in range(B):
            if routing_type == 0:
                # Squared L2 distance bounded in [0, 4]
                dist0 = torch.norm(feats0_norm[i] - proto0, p=2, dim=1)**2
                dist1 = torch.norm(feats1_norm[i] - proto1, p=2, dim=1)**2
            else:
                # Cosine distance (1.0 - cos(theta))
                dist0 = 1.0 - torch.clamp(torch.mv(proto0, feats0_norm[i]), -1.0, 1.0)
                dist1 = 1.0 - torch.clamp(torch.mv(proto1, feats1_norm[i]), -1.0, 1.0)
                
            d0_batch.append(dist0.min().item())
            d1_batch.append(dist1.min().item())
            
        D0 = np.mean(d0_batch)
        D1 = np.mean(d1_batch)
        gap = np.abs(D0 - D1)
        
        # 3. Compute predictive entropy for temperature scaling
        # Run standard logits
        logits0, _ = model0(batch.to(device))
        logits1, _ = model1(batch.to(device))
        H_avg = 0.5 * (compute_entropy(logits0).item() + compute_entropy(logits1).item())
        
        # Decisive Under Noise (DUN) scaling
        gamma_dun = 2.0
        ϵ_base = 0.08 if routing_type == 0 else 0.04
        ϵ_stab = ϵ_base / (1.0 + gamma_dun * H_avg)
        
        tau = (gap / 3.0) + ϵ_stab
        
        # Compute SCTS routing prior
        w1 = np.exp(-D1 / tau) / (np.exp(-D0 / tau) + np.exp(-D1 / tau) + 1e-8)
        w0 = 1.0 - w1
        
    return w0, w1, h_batch, H_avg

def run_test_stream(model0, model1, proto0, proto1, stream_batches, device, method_name="static", **kwargs):
    # Initialize list of overall accuracies
    accuracies = []
    
    # Store layer names for SimpleCNN
    layer_groups = ['conv1', 'bn1', 'conv2', 'bn2', 'fc1', 'fc2']
    
    # Global coefficients parameter trackers for test-time adaptation
    w_global = 0.0
    deltas = {g: 0.0 for g in layer_groups}
    
    # Track segment accuracies
    segment_accuracies = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    # Calculate dynamic phase length (always 5 phases)
    phase_len = len(stream_batches) // 5
    
    for b_idx, (batch_images, batch_labels) in enumerate(stream_batches):
        phase_idx = b_idx // phase_len
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        
        # Compute routing priors w0, w1
        w0, w1, h_batch, H_avg = compute_routing_priors(batch_images, model0, model1, proto0, proto1, device)
        
        # Reset coefficients if starting a new phase (domain transition)
        if b_idx % phase_len == 0:
            w_global = np.log(w1 / (w0 + 1e-8) + 1e-8)
            deltas = {g: 0.0 for g in layer_groups}
            
        if method_name == "static":
            # Static merging of 0.5
            lambdas = {g: 0.5 for g in layer_groups}
            merged = merge_models(model0, model1, lambdas).to(device)
            merged.eval()
            with torch.no_grad():
                logits, _ = merged(batch_images)
                _, preds = logits.max(1)
                acc = preds.eq(batch_labels).sum().item() / batch_labels.size(0)
                
        elif method_name == "fixed_tta":
            # Test-time adaptation (Entropy minimization)
            # Define parameters to optimize
            w_global_t = torch.tensor(w_global, requires_grad=True, device=device)
            deltas_t = {g: torch.tensor(deltas[g], requires_grad=True, device=device) for g in layer_groups}
            
            optimizer = torch.optim.Adam([w_global_t] + list(deltas_t.values()), lr=kwargs.get('lr', 0.05))
            
            for step in range(kwargs.get('steps', 5)):
                optimizer.zero_grad()
                # Compute lambdas
                lambdas = {g: torch.sigmoid(w_global_t + deltas_t[g]) for g in layer_groups}
                # Create merged model
                # Note: to allow backpropagation through weights, we must compute output of merged model
                # Since SimpleCNN is small, we can reconstruct the merged model and run forward pass
                # In PyTorch, constructing the model is fast. We do it in-place.
                merged = merge_models(model0, model1, {g: lambdas[g].item() for g in layer_groups}).to(device)
                merged.eval()
                
                # Forward pass to compute loss (unsupervised prediction entropy)
                # To make merged weights differentiable, we do functional-like forwarding
                # But since coefficients are small, standard TTMM optimizes the logit parameter directly
                # via an approximation or we can backprop to w_global_t/deltas_t if we implement standard TTA.
                # In our SimpleCNN, we can compute the gradients of prediction entropy w.r.t lambdas directly.
                # Let's write a differentiable forward pass or backpropagate.
                # Since we want to optimize w_global_t and deltas_t, let's make a forward pass that is differentiable
                # with respect to lambdas!
                # Let's define a differentiable forward function:
                logits = differentiable_forward(batch_images, model0, model1, lambdas, device)
                loss = compute_entropy(logits)
                loss.backward()
                optimizer.step()
                
            # Extract final lambdas
            lambdas_final = {g: torch.sigmoid(w_global_t + deltas_t[g]).item() for g in layer_groups}
            w_global = w_global_t.item()
            deltas = {g: deltas_t[g].item() for g in layer_groups}
            
            # Evaluate final model
            merged = merge_models(model0, model1, lambdas_final).to(device)
            merged.eval()
            with torch.no_grad():
                logits, _ = merged(batch_images)
                _, preds = logits.max(1)
                acc = preds.eq(batch_labels).sum().item() / batch_labels.size(0)
                
        elif method_name == "ahr_sats_dun":
            # Method C from Paper 2 (AHR-SATS-DUN + EALR)
            w_global_t = torch.tensor(w_global, requires_grad=True, device=device)
            deltas_t = {g: torch.tensor(deltas[g], requires_grad=True, device=device) for g in layer_groups}
            
            # Entropy-Adaptive Learning Rate (EALR) calculation
            gamma_ealr = 5.0
            eta = kwargs.get('lr', 0.05)
            eta_t = eta / (1.0 + gamma_ealr * H_avg)
            
            optimizer = torch.optim.Adam([w_global_t] + list(deltas_t.values()), lr=eta_t)
            
            for step in range(kwargs.get('steps', 5)):
                optimizer.zero_grad()
                lambdas = {g: torch.sigmoid(w_global_t + deltas_t[g]) for g in layer_groups}
                logits = differentiable_forward(batch_images, model0, model1, lambdas, device)
                loss = compute_entropy(logits)
                loss.backward()
                optimizer.step()
                
            lambdas_final = {g: torch.sigmoid(w_global_t + deltas_t[g]).item() for g in layer_groups}
            w_global = w_global_t.item()
            deltas = {g: deltas_t[g].item() for g in layer_groups}
            
            merged = merge_models(model0, model1, lambdas_final).to(device)
            merged.eval()
            with torch.no_grad():
                logits, _ = merged(batch_images)
                _, preds = logits.max(1)
                acc = preds.eq(batch_labels).sum().item() / batch_labels.size(0)
                
        elif method_name == "sam_ttmm":
            # Method D: SAM-TTMM with preconditioned sharpness perturbation
            w_global_t = torch.tensor(w_global, requires_grad=True, device=device)
            deltas_t = {g: torch.tensor(deltas[g], requires_grad=True, device=device) for g in layer_groups}
            
            # Fixed perturbation scale
            rho = kwargs.get('rho', 0.05)
            eta = kwargs.get('lr', 0.005)
            epsilon_stab = 0.1
            
            # Step 1: First forward-backward pass to get gradients
            lambdas = {g: torch.sigmoid(w_global_t + deltas_t[g]) for g in layer_groups}
            logits = differentiable_forward(batch_images, model0, model1, lambdas, device)
            # Prior KL loss: D_KL([lambda_bar, 1 - lambda_bar] || [w0, w1])
            lambda_bar = torch.stack(list(lambdas.values())).mean()
            q = torch.stack([lambda_bar, 1.0 - lambda_bar])
            p = torch.tensor([w1, w0], device=device)
            loss_kl = F.kl_div(torch.log(q + 1e-8), p, reduction='sum')
            loss = compute_entropy(logits) + 0.01 * loss_kl
            
            loss.backward()
            
            # Compute sensitivities F_j = E[g_j^2] and normalize them (from parameter gradients)
            F_sens = {}
            for g in layer_groups:
                grad_val = deltas_t[g].grad
                if grad_val is not None:
                    F_sens[g] = torch.mean(grad_val**2).item()
                else:
                    F_sens[g] = 1e-8
            
            # Normalize sensitivities
            total_F = sum(F_sens.values()) + 1e-8
            F_sens_norm = {g: F_sens[g] / total_F for g in layer_groups}
            
            # preconditioned direction vectors
            d_w = w_global_t.grad.item() if w_global_t.grad is not None else 0.0
            d_deltas = {}
            for g in layer_groups:
                g_val = deltas_t[g].grad.item() if deltas_t[g].grad is not None else 0.0
                d_deltas[g] = g_val / (F_sens_norm[g] + epsilon_stab)
                
            # Compute Combined Norm (fixed stabilizer bug)
            D_norm = np.sqrt(d_w**2 + sum(val**2 for val in d_deltas.values())) + 1e-12
            
            # Perturbations
            epsilon_w = rho * d_w / D_norm
            epsilon_deltas = {g: rho * d_deltas[g] / D_norm for g in layer_groups}
            
            # Apply perturbation
            w_global_pert = w_global_t.item() + epsilon_w
            deltas_pert = {g: deltas_t[g].item() + epsilon_deltas[g] for g in layer_groups}
            
            # Clear gradients
            w_global_t.grad = None
            for g in layer_groups:
                deltas_t[g].grad = None
                
            # Step 2: Second forward-backward pass on perturbed parameters
            lambdas_pert = {g: torch.sigmoid(torch.tensor(w_global_pert, device=device) + torch.tensor(deltas_pert[g], device=device)) for g in layer_groups}
            logits_pert = differentiable_forward(batch_images, model0, model1, lambdas_pert, device)
            loss_pert = compute_entropy(logits_pert)
            
            # Backprop to original variables
            # To do this cleanly, we calculate gradients on perturbed variables and apply to original
            w_global_pert_t = torch.tensor(w_global_pert, requires_grad=True, device=device)
            deltas_pert_t = {g: torch.tensor(deltas_pert[g], requires_grad=True, device=device) for g in layer_groups}
            lambdas_p = {g: torch.sigmoid(w_global_pert_t + deltas_pert_t[g]) for g in layer_groups}
            logits_p = differentiable_forward(batch_images, model0, model1, lambdas_p, device)
            loss_p = compute_entropy(logits_p)
            loss_p.backward()
            
            # Update original variables
            w_global = w_global_t.item() - eta * (w_global_pert_t.grad.item() if w_global_pert_t.grad is not None else 0.0)
            for g in layer_groups:
                g_grad = deltas_pert_t[g].grad.item() if deltas_pert_t[g].grad is not None else 0.0
                deltas[g] = deltas_t[g].item() - eta * (1.0 / (F_sens_norm[g] + epsilon_stab)) * g_grad
                
            # Evaluate final model
            lambdas_final = {g: torch.sigmoid(torch.tensor(w_global + deltas[g])).item() for g in layer_groups}
            merged = merge_models(model0, model1, lambdas_final).to(device)
            merged.eval()
            with torch.no_grad():
                logits, _ = merged(batch_images)
                _, preds = logits.max(1)
                acc = preds.eq(batch_labels).sum().item() / batch_labels.size(0)
                
        elif method_name == "sam_ttmm_ealr":
            # Ablation Method: SAM-TTMM with EALR but constant perturbation scale (rho)
            w_global_t = torch.tensor(w_global, requires_grad=True, device=device)
            deltas_t = {g: torch.tensor(deltas[g], requires_grad=True, device=device) for g in layer_groups}
            
            rho = kwargs.get('rho', 0.05)
            gamma_ealr = 5.0
            eta = kwargs.get('lr', 0.005)
            eta_t = eta / (1.0 + gamma_ealr * H_avg)
            epsilon_stab = 0.1
            
            # Step 1: First forward-backward pass
            lambdas = {g: torch.sigmoid(w_global_t + deltas_t[g]) for g in layer_groups}
            logits = differentiable_forward(batch_images, model0, model1, lambdas, device)
            lambda_bar = torch.stack(list(lambdas.values())).mean()
            q = torch.stack([lambda_bar, 1.0 - lambda_bar])
            p = torch.tensor([w1, w0], device=device)
            loss_kl = F.kl_div(torch.log(q + 1e-8), p, reduction='sum')
            loss = compute_entropy(logits) + 0.01 * loss_kl
            loss.backward()
            
            # Compute sensitivities
            F_sens = {}
            for g in layer_groups:
                grad_val = deltas_t[g].grad
                if grad_val is not None:
                    F_sens[g] = torch.mean(grad_val**2).item()
                else:
                    F_sens[g] = 1e-8
            total_F = sum(F_sens.values()) + 1e-8
            F_sens_norm = {g: F_sens[g] / total_F for g in layer_groups}
            
            d_w = w_global_t.grad.item() if w_global_t.grad is not None else 0.0
            d_deltas = {}
            for g in layer_groups:
                g_val = deltas_t[g].grad.item() if deltas_t[g].grad is not None else 0.0
                d_deltas[g] = g_val / (F_sens_norm[g] + epsilon_stab)
                
            D_norm = np.sqrt(d_w**2 + sum(val**2 for val in d_deltas.values())) + 1e-12
            epsilon_w = rho * d_w / D_norm
            epsilon_deltas = {g: rho * d_deltas[g] / D_norm for g in layer_groups}
            
            w_global_pert = w_global_t.item() + epsilon_w
            deltas_pert = {g: deltas_t[g].item() + epsilon_deltas[g] for g in layer_groups}
            
            w_global_t.grad = None
            for g in layer_groups:
                deltas_t[g].grad = None
                
            w_global_pert_t = torch.tensor(w_global_pert, requires_grad=True, device=device)
            deltas_pert_t = {g: torch.tensor(deltas_pert[g], requires_grad=True, device=device) for g in layer_groups}
            lambdas_p = {g: torch.sigmoid(w_global_pert_t + deltas_pert_t[g]) for g in layer_groups}
            logits_p = differentiable_forward(batch_images, model0, model1, lambdas_p, device)
            loss_p = compute_entropy(logits_p)
            loss_p.backward()
            
            w_global = w_global_t.item() - eta_t * (w_global_pert_t.grad.item() if w_global_pert_t.grad is not None else 0.0)
            for g in layer_groups:
                g_grad = deltas_pert_t[g].grad.item() if deltas_pert_t[g].grad is not None else 0.0
                deltas[g] = deltas_t[g].item() - eta_t * (1.0 / (F_sens_norm[g] + epsilon_stab)) * g_grad
                
            lambdas_final = {g: torch.sigmoid(torch.tensor(w_global + deltas[g])).item() for g in layer_groups}
            merged = merge_models(model0, model1, lambdas_final).to(device)
            merged.eval()
            with torch.no_grad():
                logits, _ = merged(batch_images)
                _, preds = logits.max(1)
                acc = preds.eq(batch_labels).sum().item() / batch_labels.size(0)
                
        elif method_name == "sw_sam_ttmm_no_ealr":
            # Ablation Method: SAM-TTMM with Sparsity-Weighted Perturbation but constant learning rate (eta)
            w_global_t = torch.tensor(w_global, requires_grad=True, device=device)
            deltas_t = {g: torch.tensor(deltas[g], requires_grad=True, device=device) for g in layer_groups}
            
            rho = kwargs.get('rho', 0.05)
            if h_batch >= 0.50:
                rho_adaptive = rho * (1.0 - h_batch)
            else:
                rho_adaptive = rho
                
            eta = kwargs.get('lr', 0.005)
            epsilon_stab = 0.1
            
            # Step 1: First forward-backward pass
            lambdas = {g: torch.sigmoid(w_global_t + deltas_t[g]) for g in layer_groups}
            logits = differentiable_forward(batch_images, model0, model1, lambdas, device)
            lambda_bar = torch.stack(list(lambdas.values())).mean()
            q = torch.stack([lambda_bar, 1.0 - lambda_bar])
            p = torch.tensor([w1, w0], device=device)
            loss_kl = F.kl_div(torch.log(q + 1e-8), p, reduction='sum')
            loss = compute_entropy(logits) + 0.01 * loss_kl
            loss.backward()
            
            # Compute sensitivities
            F_sens = {}
            for g in layer_groups:
                grad_val = deltas_t[g].grad
                if grad_val is not None:
                    F_sens[g] = torch.mean(grad_val**2).item()
                else:
                    F_sens[g] = 1e-8
            total_F = sum(F_sens.values()) + 1e-8
            F_sens_norm = {g: F_sens[g] / total_F for g in layer_groups}
            
            d_w = w_global_t.grad.item() if w_global_t.grad is not None else 0.0
            d_deltas = {}
            for g in layer_groups:
                g_val = deltas_t[g].grad.item() if deltas_t[g].grad is not None else 0.0
                d_deltas[g] = g_val / (F_sens_norm[g] + epsilon_stab)
                
            D_norm = np.sqrt(d_w**2 + sum(val**2 for val in d_deltas.values())) + 1e-12
            epsilon_w = rho_adaptive * d_w / D_norm
            epsilon_deltas = {g: rho_adaptive * d_deltas[g] / D_norm for g in layer_groups}
            
            w_global_pert = w_global_t.item() + epsilon_w
            deltas_pert = {g: deltas_t[g].item() + epsilon_deltas[g] for g in layer_groups}
            
            w_global_t.grad = None
            for g in layer_groups:
                deltas_t[g].grad = None
                
            w_global_pert_t = torch.tensor(w_global_pert, requires_grad=True, device=device)
            deltas_pert_t = {g: torch.tensor(deltas_pert[g], requires_grad=True, device=device) for g in layer_groups}
            lambdas_p = {g: torch.sigmoid(w_global_pert_t + deltas_pert_t[g]) for g in layer_groups}
            logits_p = differentiable_forward(batch_images, model0, model1, lambdas_p, device)
            loss_p = compute_entropy(logits_p)
            loss_p.backward()
            
            w_global = w_global_t.item() - eta * (w_global_pert_t.grad.item() if w_global_pert_t.grad is not None else 0.0)
            for g in layer_groups:
                g_grad = deltas_pert_t[g].grad.item() if deltas_pert_t[g].grad is not None else 0.0
                deltas[g] = deltas_t[g].item() - eta * (1.0 / (F_sens_norm[g] + epsilon_stab)) * g_grad
                
            lambdas_final = {g: torch.sigmoid(torch.tensor(w_global + deltas[g])).item() for g in layer_groups}
            merged = merge_models(model0, model1, lambdas_final).to(device)
            merged.eval()
            with torch.no_grad():
                logits, _ = merged(batch_images)
                _, preds = logits.max(1)
                acc = preds.eq(batch_labels).sum().item() / batch_labels.size(0)
                
        elif method_name == "sw_sam_ttmm":
            # Our Proposed Method: Sparsity-Weighted SAM-TTMM
            w_global_t = torch.tensor(w_global, requires_grad=True, device=device)
            deltas_t = {g: torch.tensor(deltas[g], requires_grad=True, device=device) for g in layer_groups}
            
            # 1. Sparsity-Weighted Perturbation Neighborhood Size
            rho = kwargs.get('rho', 0.05)
            # If batch is sparse, scale down perturbation size to mitigate random background noise
            if h_batch >= 0.50:
                # MNIST/sparse background
                rho_adaptive = rho * (1.0 - h_batch)
            else:
                # FashionMNIST/dense domain
                rho_adaptive = rho
                
            # 2. Entropy-Adaptive Learning Rate (EALR)
            gamma_ealr = 5.0
            eta = kwargs.get('lr', 0.005)
            eta_t = eta / (1.0 + gamma_ealr * H_avg)
            
            epsilon_stab = 0.1
            
            # Step 1: First forward-backward pass
            lambdas = {g: torch.sigmoid(w_global_t + deltas_t[g]) for g in layer_groups}
            logits = differentiable_forward(batch_images, model0, model1, lambdas, device)
            lambda_bar = torch.stack(list(lambdas.values())).mean()
            q = torch.stack([lambda_bar, 1.0 - lambda_bar])
            p = torch.tensor([w1, w0], device=device)
            loss_kl = F.kl_div(torch.log(q + 1e-8), p, reduction='sum')
            loss = compute_entropy(logits) + 0.01 * loss_kl
            loss.backward()
            
            # Compute sensitivities
            F_sens = {}
            for g in layer_groups:
                grad_val = deltas_t[g].grad
                if grad_val is not None:
                    F_sens[g] = torch.mean(grad_val**2).item()
                else:
                    F_sens[g] = 1e-8
            total_F = sum(F_sens.values()) + 1e-8
            F_sens_norm = {g: F_sens[g] / total_F for g in layer_groups}
            
            # Preconditioned direction vectors
            d_w = w_global_t.grad.item() if w_global_t.grad is not None else 0.0
            d_deltas = {}
            for g in layer_groups:
                g_val = deltas_t[g].grad.item() if deltas_t[g].grad is not None else 0.0
                d_deltas[g] = g_val / (F_sens_norm[g] + epsilon_stab)
                
            # Compute Combined Norm (fixed stabilizer bug)
            D_norm = np.sqrt(d_w**2 + sum(val**2 for val in d_deltas.values())) + 1e-12
            
            # Apply adaptive perturbations
            epsilon_w = rho_adaptive * d_w / D_norm
            epsilon_deltas = {g: rho_adaptive * d_deltas[g] / D_norm for g in layer_groups}
            
            w_global_pert = w_global_t.item() + epsilon_w
            deltas_pert = {g: deltas_t[g].item() + epsilon_deltas[g] for g in layer_groups}
            
            w_global_t.grad = None
            for g in layer_groups:
                deltas_t[g].grad = None
                
            # Step 2: Second forwardpass
            w_global_pert_t = torch.tensor(w_global_pert, requires_grad=True, device=device)
            deltas_pert_t = {g: torch.tensor(deltas_pert[g], requires_grad=True, device=device) for g in layer_groups}
            lambdas_p = {g: torch.sigmoid(w_global_pert_t + deltas_pert_t[g]) for g in layer_groups}
            logits_p = differentiable_forward(batch_images, model0, model1, lambdas_p, device)
            loss_p = compute_entropy(logits_p)
            loss_p.backward()
            
            # Update original variables using EALR
            w_global = w_global_t.item() - eta_t * (w_global_pert_t.grad.item() if w_global_pert_t.grad is not None else 0.0)
            for g in layer_groups:
                g_grad = deltas_pert_t[g].grad.item() if deltas_pert_t[g].grad is not None else 0.0
                deltas[g] = deltas_t[g].item() - eta_t * (1.0 / (F_sens_norm[g] + epsilon_stab)) * g_grad
                
            # Evaluate final model
            lambdas_final = {g: torch.sigmoid(torch.tensor(w_global + deltas[g])).item() for g in layer_groups}
            merged = merge_models(model0, model1, lambdas_final).to(device)
            merged.eval()
            with torch.no_grad():
                logits, _ = merged(batch_images)
                _, preds = logits.max(1)
                acc = preds.eq(batch_labels).sum().item() / batch_labels.size(0)
                
        accuracies.append(acc)
        segment_accuracies[phase_idx].append(acc)
        
    avg_acc = np.mean(accuracies)
    segment_means = {phase: np.mean(accs) for phase, accs in segment_accuracies.items()}
    return avg_acc, segment_means

# Differentiable forward pass using weighted blending of logits/representations
# This provides gradient pathways from logits to merging coefficients
def differentiable_forward(images, model0, model1, lambdas, device):
    # To perform on-the-fly backpropagation without breaking graphs,
    # we can run the inputs through both models and blend their logits/features:
    # Under standard TTMM parameterization, model outputs (logits) are blended:
    # z_merged = (1 - lambda) * z_0 + lambda * z_1
    # However, since lambdas can be layer-wise, we can interpolate the logits of the layers.
    # In a linear CNN, linear blending of layer outputs is highly accurate and differentiable.
    # Let's run forward passes through both experts, extract layer-wise pre-activations,
    # and blend them at each layer!
    # For a precise and fast implementation:
    model0.eval()
    model1.eval()
    
    # Layer 1
    x0 = model0.conv1(images)
    x1 = model1.conv1(images)
    l_conv1 = lambdas['conv1']
    x = (1.0 - l_conv1) * x0 + l_conv1 * x1
    
    # BN1 & Relu & Pool
    # Note: we use blended BN statistics as in Moment Matching MoG BN Fusion
    # For simplicity of differentiable graph, we can run the blended BN:
    # Since BN layers contain weights and running stats, we can blend their weights, biases,
    # and running stats directly to create a merged BN state.
    # Let's perform standard PyTorch functional BN or merge parameters to construct the merged BN:
    # But wait! A simpler, mathematically equivalent approach is:
    # Let's run BN1 on x using merged weights/biases/running stats!
    merged_bn1_weight = (1.0 - lambdas['bn1']) * model0.bn1.weight + lambdas['bn1'] * model1.bn1.weight
    merged_bn1_bias = (1.0 - lambdas['bn1']) * model0.bn1.bias + lambdas['bn1'] * model1.bn1.bias
    
    l_bn1_det = lambdas['bn1'].detach()
    merged_bn1_mean = (1.0 - l_bn1_det) * model0.bn1.running_mean + l_bn1_det * model1.bn1.running_mean
    # MoG var fusion
    merged_bn1_var = (1.0 - l_bn1_det) * (model0.bn1.running_var + (model0.bn1.running_mean - merged_bn1_mean)**2) + \
                      l_bn1_det * (model1.bn1.running_var + (model1.bn1.running_mean - merged_bn1_mean)**2)
                      
    x = F.batch_norm(x, merged_bn1_mean, merged_bn1_var, merged_bn1_weight, merged_bn1_bias, training=False)
    x = model0.pool(F.relu(x))
    
    # Layer 2 (Conv2)
    x0 = model0.conv2(x)
    x1 = model1.conv2(x)
    l_conv2 = lambdas['conv2']
    x = (1.0 - l_conv2) * x0 + l_conv2 * x1
    
    # BN2 & Relu & Pool
    merged_bn2_weight = (1.0 - lambdas['bn2']) * model0.bn2.weight + lambdas['bn2'] * model1.bn2.weight
    merged_bn2_bias = (1.0 - lambdas['bn2']) * model0.bn2.bias + lambdas['bn2'] * model1.bn2.bias
    
    l_bn2_det = lambdas['bn2'].detach()
    merged_bn2_mean = (1.0 - l_bn2_det) * model0.bn2.running_mean + l_bn2_det * model1.bn2.running_mean
    merged_bn2_var = (1.0 - l_bn2_det) * (model0.bn2.running_var + (model0.bn2.running_mean - merged_bn2_mean)**2) + \
                      l_bn2_det * (model1.bn2.running_var + (model1.bn2.running_mean - merged_bn2_mean)**2)
                      
    x = F.batch_norm(x, merged_bn2_mean, merged_bn2_var, merged_bn2_weight, merged_bn2_bias, training=False)
    x = model0.pool(F.relu(x))
    
    # Flatten
    x = x.view(x.size(0), -1)
    
    # FC1
    x0 = model0.fc1(x)
    x1 = model1.fc1(x)
    l_fc1 = lambdas['fc1']
    x = (1.0 - l_fc1) * x0 + l_fc1 * x1
    # Dropout & Relu
    # We disable dropout during test-time evaluation/adaptation to get deterministic and stable entropy
    x = F.relu(x)
    
    # FC2 (Output logits)
    x0 = model0.fc2(x)
    x1 = model1.fc2(x)
    l_fc2 = lambdas['fc2']
    logits = (1.0 - l_fc2) * x0 + l_fc2 * x1
    
    return logits

def construct_stream(mnist_test, fmnist_test, kmnist_test, seed, batches_per_phase=50):
    # Set random seeds right before constructing stream to ensure absolute reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # We use batch size 64. Total 250 batches.
    stream_batches = []
    
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)
    fmnist_loader = DataLoader(fmnist_test, batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True)
    
    mnist_iter = iter(mnist_loader)
    fmnist_iter = iter(fmnist_loader)
    kmnist_iter = iter(kmnist_loader)
    
    # Phase 0: Clean MNIST
    for _ in range(batches_per_phase):
        stream_batches.append(next(mnist_iter))
        
    # Phase 1: Noisy MNIST (sigma=0.6)
    for _ in range(batches_per_phase):
        images, labels = next(mnist_iter)
        noise = torch.randn_like(images) * 0.6
        images_noisy = torch.clamp(images + noise, -1.0, 1.0)
        stream_batches.append((images_noisy, labels))
        
    # Phase 2: Clean FashionMNIST
    for _ in range(batches_per_phase):
        stream_batches.append(next(fmnist_iter))
        
    # Phase 3: Noisy FashionMNIST (sigma=0.6)
    for _ in range(batches_per_phase):
        images, labels = next(fmnist_iter)
        noise = torch.randn_like(images) * 0.6
        images_noisy = torch.clamp(images + noise, -1.0, 1.0)
        stream_batches.append((images_noisy, labels))
        
    # Phase 4: Novel KMNIST (unseen OOD)
    for _ in range(batches_per_phase):
        stream_batches.append(next(kmnist_iter))
        
    return stream_batches

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Pre-trained Experts
    print("Loading expert models...")
    model0 = SimpleCNN()
    model1 = SimpleCNN()
    model0.load_state_dict(torch.load('checkpoints/expert0.pth', map_location=device))
    model1.load_state_dict(torch.load('checkpoints/expert1.pth', map_location=device))
    model0.to(device)
    model1.to(device)
    
    # 2. Precompute Class Prototypes on Clean Calibration Subsets
    # We use 1000 clean test samples as calibration data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_cal = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_cal = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_cal_loader = DataLoader(Subset(mnist_cal, list(range(1000))), batch_size=128, shuffle=False)
    fmnist_cal_loader = DataLoader(Subset(fmnist_cal, list(range(1000))), batch_size=128, shuffle=False)
    
    print("Precomputing expert class prototypes...")
    proto0 = precompute_prototypes(model0, mnist_cal_loader, device)
    proto1 = precompute_prototypes(model1, fmnist_cal_loader, device)
    
    # 3. Load all datasets
    print("Loading test datasets...")
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Define seeds to evaluate
    seeds = [42, 43, 44]
    batches_per_phase = 50
    
    methods = [
        ("static", {}),
        ("fixed_tta", {"lr": 0.05, "steps": 5}),
        ("ahr_sats_dun", {"lr": 0.05, "steps": 5}),
        ("sam_ttmm", {"lr": 0.5, "rho": 0.05}),
        ("sam_ttmm_ealr", {"lr": 2.0, "rho": 0.05}),
        ("sw_sam_ttmm_no_ealr", {"lr": 0.5, "rho": 0.05}),
        ("sw_sam_ttmm", {"lr": 2.0, "rho": 0.05})
    ]
    
    # Keep track of all seed results
    all_results = {name: {'overall': [], 0: [], 1: [], 2: [], 3: [], 4: []} for name, _ in methods}
    
    for s_idx, seed in enumerate(seeds):
        print(f"\n==========================================")
        print(f"RUNNING FOR SEED: {seed} ({s_idx + 1}/{len(seeds)})")
        print(f"==========================================")
        stream_batches = construct_stream(mnist_test, fmnist_test, kmnist_test, seed, batches_per_phase)
        
        for name, kwargs in methods:
            print(f"Evaluating Method: {name.upper()}...")
            avg_acc, segment_accs = run_test_stream(model0, model1, proto0, proto1, stream_batches, device, method_name=name, **kwargs)
            
            all_results[name]['overall'].append(avg_acc)
            for phase in range(5):
                all_results[name][phase].append(segment_accs[phase])
                
    # Calculate means and standard deviations
    summary_results = {}
    for name, _ in methods:
        summary_results[name] = {}
        # Overall
        summary_results[name]['overall_mean'] = np.mean(all_results[name]['overall'])
        summary_results[name]['overall_std'] = np.std(all_results[name]['overall'])
        # Segments
        for phase in range(5):
            summary_results[name][f'{phase}_mean'] = np.mean(all_results[name][phase])
            summary_results[name][f'{phase}_std'] = np.std(all_results[name][phase])
            
    # Print comparison table
    print("\n" + "="*120)
    print(f"{'Method':<22} | {'MNIST':<14} | {'N-MNIST':<14} | {'Fashion':<14} | {'N-Fash':<14} | {'KMNIST':<14} | {'Overall':<14}")
    print("="*120)
    for name, _ in methods:
        m = summary_results[name]
        print(f"{name.upper():<22} | "
              f"{m['0_mean']*100:5.2f}±{m['0_std']*100:4.2f}% | "
              f"{m['1_mean']*100:5.2f}±{m['1_std']*100:4.2f}% | "
              f"{m['2_mean']*100:5.2f}±{m['2_std']*100:4.2f}% | "
              f"{m['3_mean']*100:5.2f}±{m['3_std']*100:4.2f}% | "
              f"{m['4_mean']*100:5.2f}±{m['4_std']*100:4.2f}% | "
              f"{m['overall_mean']*100:5.2f}±{m['overall_std']*100:4.2f}%")
    print("="*120)

if __name__ == "__main__":
    main()
