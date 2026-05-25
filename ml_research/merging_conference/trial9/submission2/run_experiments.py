import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- Model Definitions ---

class SimpleCNN(nn.Module):
    def __init__(self, use_cosface=False, s=30.0, m=0.35):
        super().__init__()
        self.use_cosface = use_cosface
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.25)
        
        if use_cosface:
            self.classifier = CosFaceLinear(128, 10, s=s, m=m)
        else:
            self.classifier = nn.Linear(128, 10)

    def get_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

    def forward(self, x, label=None):
        features = self.get_features(x)
        features_drop = self.dropout(features)
        if self.use_cosface:
            logits = self.classifier(features_drop, label)
        else:
            logits = self.classifier(features_drop)
        return logits

class CosFaceLinear(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label=None):
        # Normalize weights and features to the unit sphere
        w_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        # Cosine similarity
        cosine = F.linear(x_norm, w_norm)
        
        if label is not None and self.training:
            # Apply additive angular margin
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1), 1.0)
            output = self.s * (cosine - one_hot * self.m)
        else:
            output = self.s * cosine
        return output

# --- Data Preparation ---

def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download MNIST, FashionMNIST, KMNIST
    train_mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_fashion = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_fashion = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    test_kmnist = datasets.KMNIST('./data', train=False, download=True, transform=transform)
    
    return train_mnist, test_mnist, train_fashion, test_fashion, test_kmnist

# --- Expert Training ---

def train_expert(model, train_dataset, device, save_path, epochs=2, batch_size=64, subset_size=10000):
    model = model.to(device)
    model.train()
    
    # Take a random subset of training data as per paper
    indices = np.random.choice(len(train_dataset), subset_size, replace=False)
    subset = Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training model for {epochs} epochs on a subset of {subset_size} samples...")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if model.use_cosface:
                outputs = model(images, labels)
            else:
                outputs = model(images)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

# --- Evaluation and Baseline Pipelines ---

def compute_fisher_and_prototypes(model, dataset, device, num_samples=256, is_cosface=False):
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 1. Compute Class-wise Prototypes
    class_features = {c: [] for c in range(10)}
    collected = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        label = labels.item()
        if len(class_features[label]) < num_samples:
            with torch.no_grad():
                feat = model.get_features(images)
                if is_cosface:
                    # Spherical normalization
                    feat = F.normalize(feat, p=2, dim=1)
                class_features[label].append(feat.cpu().squeeze(0).numpy())
                collected += 1
        if all(len(class_features[c]) >= num_samples for c in range(10)):
            break
            
    # Average class features to create prototypes
    prototypes = {}
    for c in range(10):
        feats = np.array(class_features[c])
        avg_feat = np.mean(feats, axis=0)
        if is_cosface:
            # Project back onto unit sphere
            avg_feat = avg_feat / (np.linalg.norm(avg_feat) + 1e-8)
        prototypes[c] = torch.tensor(avg_feat, dtype=torch.float32, device=device)
        
    # 2. Compute Parameter sensitivities (Fisher Information)
    # Define loss function for Fisher
    criterion = nn.CrossEntropyLoss()
    param_grads = {name: [] for name, param in model.named_parameters()}
    
    # Use 256 samples to compute sensitivities
    calib_indices = np.random.choice(len(dataset), 256, replace=False)
    calib_subset = Subset(dataset, calib_indices)
    calib_loader = DataLoader(calib_subset, batch_size=1, shuffle=False)
    
    for images, labels in calib_loader:
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        if model.use_cosface:
            outputs = model(images, labels)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_grads[name].append(param.grad.detach().cpu().numpy() ** 2)
                
    # Average squared gradients per parameter tensor
    fisher_sens = {}
    for name, grads in param_grads.items():
        if len(grads) > 0:
            avg_grad_sq = np.mean(np.array(grads), axis=0)
            fisher_sens[name] = float(np.mean(avg_grad_sq))
        else:
            fisher_sens[name] = 1e-5
            
    return prototypes, fisher_sens

# --- Hoyer Sparsity Measure ---

def hoyer_sparsity(f):
    # f: PyTorch tensor of shape (batch_size, dim) or (dim,)
    dim = f.shape[-1]
    l1 = torch.sum(torch.abs(f), dim=-1)
    l2 = torch.sqrt(torch.sum(f**2, dim=-1) + 1e-8)
    sparsity = (math.sqrt(dim) - (l1 / l2)) / (math.sqrt(dim) - 1)
    return sparsity

# --- Test-Time Adaptation Loop ---

def run_test_time_merging(method_name,
                          expert0, expert1,
                          stream_batches,
                          device,
                          prototypes0=None, prototypes1=None,
                          fisher_sens0=None, fisher_sens1=None,
                          is_cosface=False,
                          tau_sparse=0.6,
                          use_data_free_prototypes=False,
                          gamma_dun=2.0,
                          gamma_ealr=5.0):
    
    print(f"\n--- Running Evaluation: {method_name} ---")
    
    # Extract states
    state0 = {k: v.clone() for k, v in expert0.state_dict().items()}
    state1 = {k: v.clone() for k, v in expert1.state_dict().items()}
    
    # Map BN layers
    bn_stats0 = {}
    for name, module in expert0.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_stats0[name] = {
                'running_mean': module.running_mean.clone(),
                'running_var': module.running_var.clone()
            }
            
    bn_stats1 = {}
    for name, module in expert1.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_stats1[name] = {
                'running_mean': module.running_mean.clone(),
                'running_var': module.running_var.clone()
            }
            
    # Parameter naming and grouping for adaptation
    param_names = [name for name, _ in expert0.named_parameters()]
    num_layers = len(param_names)
    param_name_to_index = {name: i for i, name in enumerate(param_names)}
    
    # Precompute global normalized joint Fisher sensitivities
    if fisher_sens0 is not None and fisher_sens1 is not None:
        joint_sens = []
        for name in param_names:
            sens = 0.5 * (fisher_sens0.get(name, 1e-5) + fisher_sens1.get(name, 1e-5))
            joint_sens.append(sens)
        joint_sens = np.array(joint_sens)
        # Globally normalize
        F_tilde = joint_sens / (np.max(joint_sens) + 1e-8)
    else:
        F_tilde = np.ones(num_layers)
        
    # Create evaluation model (merged model structure)
    eval_model = SimpleCNN(use_cosface=is_cosface).to(device)
    
    # High-confidence online prototypes tracker (for data-free version)
    online_prototypes0 = {c: None for c in range(10)}
    online_prototypes1 = {c: None for c in range(10)}
    
    # Performance tracking
    batch_accuracies = []
    merging_coefficients = []
    selected_routing_types = [] # 0 for L2, 1 for Angular
    hoyer_sparsities = []
    
    # Hyperparameters from paper
    beta = 1.5
    gamma = 0.02
    eta = 0.05
    N_step = 5
    
    for batch_idx, (X_t, Y_t, domain) in enumerate(stream_batches):
        X_t, Y_t = X_t.to(device), Y_t.to(device)
        
        # 1. Extract feature activations to compute Hoyer Sparsity and choose routing type
        with torch.no_grad():
            f0_batch = expert0.get_features(X_t)
            f1_batch = expert1.get_features(X_t)
            
            h_f0 = hoyer_sparsity(f0_batch) # shape: (batch_size,)
            h_f1 = hoyer_sparsity(f1_batch)
            h_batch = 0.5 * (torch.mean(h_f0) + torch.mean(h_f1)).item()
            hoyer_sparsities.append(h_batch)
            
            # Compute predictive entropy for Decisive Under Noise (DUN) scaling
            expert0.eval()
            expert1.eval()
            out0_ent = expert0(X_t)
            out1_ent = expert1(X_t)
            p0_ent = F.softmax(out0_ent, dim=1)
            p1_ent = F.softmax(out1_ent, dim=1)
            h0_ent = -torch.mean(torch.sum(p0_ent * torch.log(p0_ent + 1e-8), dim=1)).item()
            h1_ent = -torch.mean(torch.sum(p1_ent * torch.log(p1_ent + 1e-8), dim=1)).item()
            h_avg = 0.5 * (h0_ent + h1_ent)
            
        # 2. Determine routing metric (L2 Euclidean vs Angular Cosine)
        routing_type = 1 # Default: Angular Cosine
        if "AHR-SAN" in method_name:
            if h_batch >= tau_sparse:
                routing_type = 0 # Euclidean L2
            else:
                routing_type = 1 # Angular Cosine
            selected_routing_types.append(routing_type)
        else:
            if "L2" in method_name or "Method B" in method_name or "Method A" in method_name:
                routing_type = 0 # Hardcoded L2
            else:
                routing_type = 1 # Hardcoded Angular
                
        # 3. Compute routing prior
        w0, w1 = 0.5, 0.5 # fallback uniform prior
        
        # Decide which prototypes to use
        curr_prot0 = online_prototypes0 if use_data_free_prototypes else prototypes0
        curr_prot1 = online_prototypes1 if use_data_free_prototypes else prototypes1
        
        # Check if prototypes are available/initialized
        p0_classes = [c for c in range(10) if curr_prot0 is not None and curr_prot0[c] is not None]
        p1_classes = [c for c in range(10) if curr_prot1 is not None and curr_prot1[c] is not None]
        
        prototypes_available = (len(p0_classes) == 10 and len(p1_classes) == 10)
        any_prototypes_available = (len(p0_classes) > 0 or len(p1_classes) > 0)
        
        if any_prototypes_available:
            with torch.no_grad():
                # Perform batch extraction
                f0_extracted = expert0.get_features(X_t)
                f1_extracted = expert1.get_features(X_t)
                
                # Compute batch average prototype distances for Expert 0
                if len(p0_classes) > 0:
                    dist0_list = []
                    for i in range(X_t.size(0)):
                        f0_i = f0_extracted[i]
                        if routing_type == 0:
                            if is_cosface:
                                f0_i_norm = F.normalize(f0_i, p=2, dim=0)
                                d0_min = min(torch.sum((f0_i_norm - curr_prot0[c])**2).item() for c in p0_classes)
                            else:
                                d0_min = min(torch.sum((f0_i - curr_prot0[c])**2).item() for c in p0_classes)
                        else:
                            f0_i_norm = F.normalize(f0_i, p=2, dim=0)
                            d0_min = min((1.0 - torch.dot(f0_i_norm, curr_prot0[c])).item() for c in p0_classes)
                        dist0_list.append(d0_min)
                    D0_bar = np.mean(dist0_list)
                else:
                    D_base0 = 1.5 if routing_type == 0 else 0.85
                    D0_bar = D_base0 * (h0_ent + 0.1)
                    
                # Compute batch average prototype distances for Expert 1
                if len(p1_classes) > 0:
                    dist1_list = []
                    for i in range(X_t.size(0)):
                        f1_i = f1_extracted[i]
                        if routing_type == 0:
                            if is_cosface:
                                f1_i_norm = F.normalize(f1_i, p=2, dim=0)
                                d1_min = min(torch.sum((f1_i_norm - curr_prot1[c])**2).item() for c in p1_classes)
                            else:
                                d1_min = min(torch.sum((f1_i - curr_prot1[c])**2).item() for c in p1_classes)
                        else:
                            f1_i_norm = F.normalize(f1_i, p=2, dim=0)
                            d1_min = min((1.0 - torch.dot(f1_i_norm, curr_prot1[c])).item() for c in p1_classes)
                        dist1_list.append(d1_min)
                    D1_bar = np.mean(dist1_list)
                else:
                    D_base1 = 1.5 if routing_type == 0 else 0.85
                    D1_bar = D_base1 * (h1_ent + 0.1)
                
                D_min = min(D0_bar, D1_bar)
                D_second = max(D0_bar, D1_bar)
                gap = abs(D_second - D_min)
                
                # SCTS Temperature scaling
                s_factor = 3.0
                if "AHR-SAN" in method_name:
                    # Proposed Decisive Under Noise (DUN) scaling: scale eps_stab down when uncertainty is high
                    if routing_type == 0:
                        # For Euclidean L2
                        eps_stab = 0.08 / (1.0 + gamma_dun * h_avg)
                    else:
                        # For Angular Cosine
                        eps_stab = 0.04 / (1.0 + gamma_dun * h_avg)
                    tau = (gap / s_factor) + eps_stab
                else:
                    if routing_type == 0:
                        if is_cosface:
                            eps_stab = 0.08
                        else:
                            eps_stab = 150.0 if not use_data_free_prototypes else 5.0
                        tau = (gap / s_factor) + eps_stab
                    else:
                        eps_stab = 0.04
                        tau = (gap / s_factor) + eps_stab
                    
                # Softmax routing prior
                w0 = math.exp(-D0_bar / tau)
                w1 = math.exp(-D1_bar / tau)
                sum_w = w0 + w1
                w0 /= sum_w
                w1 /= sum_w
        else:
            # Fully data-free initialization phase: route based on entropy
            with torch.no_grad():
                expert0.eval()
                expert1.eval()
                out0 = expert0(X_t)
                out1 = expert1(X_t)
                
                p0 = F.softmax(out0, dim=1)
                p1 = F.softmax(out1, dim=1)
                
                h0 = -torch.mean(torch.sum(p0 * torch.log(p0 + 1e-8), dim=1)).item()
                h1 = -torch.mean(torch.sum(p1 * torch.log(p1 + 1e-8), dim=1)).item()
                
                # Soft posterior routing
                gamma_entropy = 5.0
                w0 = math.exp(-gamma_entropy * h0)
                w1 = math.exp(-gamma_entropy * h1)
                sum_w = w0 + w1
                w0 /= sum_w
                w1 /= sum_w
                
        # 4. Merging weight initialization
        # Prior-Guided Initialization (PG-Init)
        # We parameterize expert 1 as lambda, expert 0 as 1 - lambda
        # So p = w1 is the routing prior for Expert 1
        p_prior = w1
        # Bound prior away from 0 and 1 for numerical stability
        p_prior = max(min(p_prior, 0.999), 0.001)
        
        w_global = torch.tensor(math.log(p_prior / (1.0 - p_prior)), requires_grad=True, device=device)
        delta = torch.zeros(num_layers, requires_grad=True, device=device)
        
        # 5. Test-Time Adaptation Step
        if "Method A" in method_name:
            # Fixed TTA (No offset adaptation, simply use prior weights, reset every batch)
            with torch.no_grad():
                # Perform static merge using w0, w1
                merged_state = {}
                for name in param_names:
                    merged_state[name] = (1.0 - p_prior) * state0[name] + p_prior * state1[name]
                eval_model.load_state_dict(merged_state, strict=False)
                
                # Blend BN
                for name, module in eval_model.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        mean0 = bn_stats0[name]['running_mean']
                        var0 = bn_stats0[name]['running_var']
                        mean1 = bn_stats1[name]['running_mean']
                        var1 = bn_stats1[name]['running_var']
                        mean_f = w0 * mean0 + w1 * mean1
                        var_f = w0 * (var0 + (mean0 - mean_f)**2) + w1 * (var1 + (mean1 - mean_f)**2)
                        module.running_mean.copy_(mean_f)
                        module.running_var.copy_(var_f)
                        
            # Run one forward pass to measure accuracy
            eval_model.eval()
            with torch.no_grad():
                preds = eval_model(X_t)
                acc = (preds.max(1)[1] == Y_t).float().mean().item() * 100.0
                batch_accuracies.append(acc)
                merging_coefficients.append(p_prior)
        else:
            # CL W-Fisher / Proposed optimization loop
            for step in range(N_step):
                # a. Create merged state dictionary
                merged_state = {}
                for name in param_names:
                    idx = param_name_to_index[name]
                    lambda_j = torch.sigmoid(w_global + delta[idx])
                    merged_state[name] = (1.0 - lambda_j) * state0[name] + lambda_j * state1[name]
                    
                # b. Copy to eval_model
                eval_model.train() # Enable gradients
                for name, param in eval_model.named_parameters():
                    param.data.copy_(merged_state[name])
                    
                # c. Blend BN statistics (moment-matching MoG)
                for name, module in eval_model.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        mean0 = bn_stats0[name]['running_mean']
                        var0 = bn_stats0[name]['running_var']
                        mean1 = bn_stats1[name]['running_mean']
                        var1 = bn_stats1[name]['running_var']
                        mean_f = w0 * mean0 + w1 * mean1
                        var_f = w0 * (var0 + (mean0 - mean_f)**2) + w1 * (var1 + (mean1 - mean_f)**2)
                        module.running_mean.copy_(mean_f)
                        module.running_var.copy_(var_f)
                        
                # d. Forward pass & Joint Loss
                outputs = eval_model(X_t)
                probs = F.softmax(outputs, dim=1)
                L_entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
                
                L_kl = 0.0
                for name in param_names:
                    idx = param_name_to_index[name]
                    lambda_j = torch.sigmoid(w_global + delta[idx])
                    # KL divergence to prior
                    kl_j = p_prior * torch.log(p_prior / (lambda_j + 1e-8)) + (1.0 - p_prior) * torch.log((1.0 - p_prior) / (1.0 - lambda_j + 1e-8))
                    L_kl += kl_j
                L_kl /= num_layers
                
                L_coherence = torch.sum(delta ** 2)
                
                loss = L_entropy + beta * L_kl + gamma * L_coherence
                
                # e. Backward pass
                loss.backward()
                
                # f. Parameter update (preconditioned gradient step)
                with torch.no_grad():
                    eta_t = eta / (1.0 + gamma_ealr * h_avg) if "AHR-SAN" in method_name else eta
                    w_global -= eta_t * w_global.grad
                    for j in range(num_layers):
                        sens = F_tilde[j]
                        delta[j] -= eta_t * (1.0 / (sens + 1e-2)) * delta.grad[j]
                        
                    w_global.grad.zero_()
                    delta.grad.zero_()
                    
            # Compute final evaluation accuracy on the batch
            eval_model.eval()
            with torch.no_grad():
                # Rebuild merged model with final optimized coefficients
                final_state = {}
                final_lambdas = []
                for name in param_names:
                    idx = param_name_to_index[name]
                    lambda_j = torch.sigmoid(w_global + delta[idx]).item()
                    final_lambdas.append(lambda_j)
                    final_state[name] = (1.0 - lambda_j) * state0[name] + lambda_j * state1[name]
                eval_model.load_state_dict(final_state, strict=False)
                
                preds = eval_model(X_t)
                acc = (preds.max(1)[1] == Y_t).float().mean().item() * 100.0
                batch_accuracies.append(acc)
                merging_coefficients.append(np.mean(final_lambdas))
                
                # --- On-the-fly Dynamic Prototype Adaptation (FDF-DPA) ---
                if use_data_free_prototypes:
                    # Update running prototypes using high-confidence predictions from the active expert
                    # Decide which expert is active based on the final routing prior
                    active_idx = 1 if w1 > w0 else 0
                    active_expert = expert1 if active_idx == 1 else expert0
                    active_prototypes = online_prototypes1 if active_idx == 1 else online_prototypes0
                    
                    # Entropy margin check: only update prototypes of active expert if it is significantly more confident
                    # h0_ent and h1_ent are the predictive entropies of expert 0 and expert 1
                    entropy_margin = 0.08
                    is_confident = False
                    if active_idx == 1 and (h0_ent - h1_ent) >= entropy_margin:
                        # Only update if the active expert itself is confident/low-entropy (prevents noise contamination)
                        if h1_ent < 0.6:
                            is_confident = True
                    elif active_idx == 0 and (h1_ent - h0_ent) >= entropy_margin:
                        if h0_ent < 0.6:
                            is_confident = True
                        
                    if is_confident:
                        active_expert.eval()
                        outputs_active = active_expert(X_t)
                        probs_active = F.softmax(outputs_active, dim=1)
                        
                        conf_vals, pred_classes = probs_active.max(1)
                        features_active = active_expert.get_features(X_t)
                        
                        alpha_proto = 0.95
                        for i in range(X_t.size(0)):
                            if conf_vals[i] >= 0.95:
                                c_pred = pred_classes[i].item()
                                feat_i = features_active[i].detach()
                                if is_cosface: # ALWAYS normalize on the sphere for CosFace
                                    feat_i = F.normalize(feat_i, p=2, dim=0)

                                if active_prototypes[c_pred] is None:
                                    active_prototypes[c_pred] = feat_i
                                else:
                                    updated_feat = alpha_proto * active_prototypes[c_pred] + (1.0 - alpha_proto) * feat_i
                                    if is_cosface: # ALWAYS normalize on the sphere for CosFace
                                        updated_feat = updated_feat / (torch.norm(updated_feat) + 1e-8)
                                    active_prototypes[c_pred] = updated_feat                                
        # Print progress periodically
        if "Fully Data-Free" in method_name:
            p0_non_none = sum(1 for c in range(10) if online_prototypes0[c] is not None)
            p1_non_none = sum(1 for c in range(10) if online_prototypes1[c] is not None)
            print(f"Method F - Batch {batch_idx+1:2d} ({domain:13s}): routing={routing_type} | w0={w0:.4f}, w1={w1:.4f} | Acc: {acc:.2f}% | Lambda_mean: {merging_coefficients[-1]:.4f} | Prot0: {p0_non_none:2d}/10, Prot1: {p1_non_none:2d}/10 | Avail: {prototypes_available}")
        elif (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/50 - Domain: {domain:15s} - Acc: {acc:.2f}% - Lambda_mean: {merging_coefficients[-1]:.4f} - Hoyer Sparsity: {h_batch:.4f}")
            
    # Calculate Segment-wise performance
    # Phase divisions: Clean MNIST (0-9), Noisy MNIST (10-19), Clean Fashion (20-29), Noisy Fashion (30-39), Novel KMNIST (40-49)
    segments = {
        "Clean MNIST": batch_accuracies[0:10],
        "Noisy MNIST": batch_accuracies[10:20],
        "Clean Fashion": batch_accuracies[20:30],
        "Noisy Fashion": batch_accuracies[30:40],
        "Novel KMNIST": batch_accuracies[40:50]
    }
    
    overall_mean = np.mean(batch_accuracies)
    print(f"\nResults for {method_name}:")
    for seg, accs in segments.items():
        print(f"  {seg:15s}: {np.mean(accs):.2f}%")
    print(f"  {'Overall Average':15s}: {overall_mean:.2f}%")
    
    return batch_accuracies, merging_coefficients, hoyer_sparsities

# --- Main Workflow ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Datasets
    print("Loading datasets...")
    train_mnist, test_mnist, train_fashion, test_fashion, test_kmnist = get_datasets()
    
    # 2. Build or Load Experts
    os.makedirs("./models", exist_ok=True)
    
    # Standard Experts
    mnist_std_path = "./models/mnist_std.pt"
    fashion_std_path = "./models/fashion_std.pt"
    
    expert_mnist_std = SimpleCNN(use_cosface=False)
    if os.path.exists(mnist_std_path):
        print("Loading pre-trained standard MNIST expert...")
        expert_mnist_std.load_state_dict(torch.load(mnist_std_path, map_location=device))
    else:
        print("Training standard MNIST expert...")
        expert_mnist_std = train_expert(expert_mnist_std, train_mnist, device, mnist_std_path, epochs=2)
        
    expert_fashion_std = SimpleCNN(use_cosface=False)
    if os.path.exists(fashion_std_path):
        print("Loading pre-trained standard FashionMNIST expert...")
        expert_fashion_std.load_state_dict(torch.load(fashion_std_path, map_location=device))
    else:
        print("Training standard FashionMNIST expert...")
        expert_fashion_std = train_expert(expert_fashion_std, train_fashion, device, fashion_std_path, epochs=2)
        
    # CosFace Experts
    mnist_cos_path = "./models/mnist_cosface.pt"
    fashion_cos_path = "./models/fashion_cosface.pt"
    
    expert_mnist_cos = SimpleCNN(use_cosface=True)
    if os.path.exists(mnist_cos_path):
        print("Loading pre-trained CosFace MNIST expert...")
        expert_mnist_cos.load_state_dict(torch.load(mnist_cos_path, map_location=device))
    else:
        print("Training CosFace MNIST expert...")
        expert_mnist_cos = train_expert(expert_mnist_cos, train_mnist, device, mnist_cos_path, epochs=2)
        
    expert_fashion_cos = SimpleCNN(use_cosface=True)
    if os.path.exists(fashion_cos_path):
        print("Loading pre-trained CosFace FashionMNIST expert...")
        expert_fashion_cos.load_state_dict(torch.load(fashion_cos_path, map_location=device))
    else:
        print("Training CosFace FashionMNIST expert...")
        expert_fashion_cos = train_expert(expert_fashion_cos, train_fashion, device, fashion_cos_path, epochs=2)
        
    # 3. Calibration / Precomputing Fisher and Prototypes
    print("\nComputing prototypes and sensitivities for Standard Experts...")
    prototypes_std0, fisher_std0 = compute_fisher_and_prototypes(expert_mnist_std, test_mnist, device, is_cosface=False)
    prototypes_std1, fisher_std1 = compute_fisher_and_prototypes(expert_fashion_std, test_fashion, device, is_cosface=False)
    
    print("Computing prototypes and sensitivities for CosFace Experts...")
    prototypes_cos0, fisher_cos0 = compute_fisher_and_prototypes(expert_mnist_cos, test_mnist, device, is_cosface=True)
    prototypes_cos1, fisher_cos1 = compute_fisher_and_prototypes(expert_fashion_cos, test_fashion, device, is_cosface=True)
    
    # 4. Generate the non-stationary test stream
    print("\nGenerating non-stationary test-time stream...")
    # Standardize stream batch generation
    # 50 batches of size 64: Clean MNIST (0-9), Noisy MNIST (10-19), Clean Fashion (20-29), Noisy Fashion (30-39), Novel KMNIST (40-49)
    # We draw from test sets using consistent seeds
    mnist_loader = DataLoader(test_mnist, batch_size=64, shuffle=False)
    fashion_loader = DataLoader(test_fashion, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(test_kmnist, batch_size=64, shuffle=False)
    
    mnist_iter = iter(mnist_loader)
    fashion_iter = iter(fashion_loader)
    kmnist_iter = iter(kmnist_loader)
    
    stream_batches = []
    
    # Batches 0-9: Clean MNIST
    for _ in range(10):
        images, labels = next(mnist_iter)
        stream_batches.append((images, labels, "Clean MNIST"))
        
    # Batches 10-19: Noisy MNIST (Gaussian noise with sigma = 0.6)
    for _ in range(10):
        images, labels = next(mnist_iter)
        noise = torch.randn_like(images) * 0.6
        noisy_images = torch.clamp(images + noise, -1.0, 1.0)
        stream_batches.append((noisy_images, labels, "Noisy MNIST"))
        
    # Batches 20-29: Clean Fashion
    for _ in range(10):
        images, labels = next(fashion_iter)
        stream_batches.append((images, labels, "Clean Fashion"))
        
    # Batches 30-39: Noisy Fashion (Gaussian noise with sigma = 0.6)
    for _ in range(10):
        images, labels = next(fashion_iter)
        noise = torch.randn_like(images) * 0.6
        noisy_images = torch.clamp(images + noise, -1.0, 1.0)
        stream_batches.append((noisy_images, labels, "Noisy Fashion"))
        
    # Batches 40-49: Novel KMNIST
    for _ in range(10):
        images, labels = next(kmnist_iter)
        stream_batches.append((images, labels, "Novel KMNIST"))
        
    print(f"Generated stream with {len(stream_batches)} batches of size 64.")
    
    # 5. Execute Evaluations
    results = {}
    
    # Method A: Fixed TTA + Reset (L2 SCTS baseline with no parameter offset updates)
    acc_A, lambda_A, hoy_A = run_test_time_merging(
        "Method A: Fixed TTA + Reset (L2)",
        expert_mnist_std, expert_fashion_std,
        stream_batches, device,
        prototypes0=prototypes_std0, prototypes1=prototypes_std1,
        fisher_sens0=fisher_std0, fisher_sens1=fisher_std1,
        is_cosface=False
    )
    results["Method A"] = (acc_A, lambda_A)
    
    # Method B: CL W-Fisher + SCTS (L2) (Standard experts)
    acc_B, lambda_B, hoy_B = run_test_time_merging(
        "Method B: CL W-Fisher + SCTS (L2)",
        expert_mnist_std, expert_fashion_std,
        stream_batches, device,
        prototypes0=prototypes_std0, prototypes1=prototypes_std1,
        fisher_sens0=fisher_std0, fisher_sens1=fisher_std1,
        is_cosface=False
    )
    results["Method B"] = (acc_B, lambda_B)
    
    # Method C: CL W-Fisher + A-SCTS (Standard experts with Angular routing)
    acc_C, lambda_C, hoy_C = run_test_time_merging(
        "Method C: CL W-Fisher + A-SCTS",
        expert_mnist_std, expert_fashion_std,
        stream_batches, device,
        prototypes0=prototypes_std0, prototypes1=prototypes_std1,
        fisher_sens0=fisher_std0, fisher_sens1=fisher_std1,
        is_cosface=False
    )
    results["Method C"] = (acc_C, lambda_C)
    
    # Method D: CP-AM (Ours) (CosFace experts with Angular routing)
    acc_D, lambda_D, hoy_D = run_test_time_merging(
        "Method D: CP-AM (Ours)",
        expert_mnist_cos, expert_fashion_cos,
        stream_batches, device,
        prototypes0=prototypes_cos0, prototypes1=prototypes_cos1,
        fisher_sens0=fisher_cos0, fisher_sens1=fisher_cos1,
        is_cosface=True
    )
    results["Method D"] = (acc_D, lambda_D)
    
    # Method E: Proposed AHR-SAN (CosFace experts, Adaptive routing with Sparsity-Aware Normalization, Offline prototypes)
    acc_E, lambda_E, hoy_E = run_test_time_merging(
        "Method E: AHR-SAN (Proposed, Offline Prototypes)",
        expert_mnist_cos, expert_fashion_cos,
        stream_batches, device,
        prototypes0=prototypes_cos0, prototypes1=prototypes_cos1,
        fisher_sens0=fisher_cos0, fisher_sens1=fisher_cos1,
        is_cosface=True,
        tau_sparse=0.6,
        use_data_free_prototypes=False
    )
    results["Method E"] = (acc_E, lambda_E)
    
    # Method F: Proposed AHR-SAN (Data-Free) (Proposed, on-the-fly prototypes estimation)
    acc_F, lambda_F, hoy_F = run_test_time_merging(
        "Method F: AHR-SAN (Proposed, Fully Data-Free)",
        expert_mnist_cos, expert_fashion_cos,
        stream_batches, device,
        prototypes0=None, prototypes1=None,
        fisher_sens0=fisher_cos0, fisher_sens1=fisher_cos1,
        is_cosface=True,
        tau_sparse=0.6,
        use_data_free_prototypes=True
    )
    results["Method F"] = (acc_F, lambda_F)
    
    # Method G: Proposed AHR-SAN (Proposed, Unsupervised Calibrated Data-Free)
    acc_G, lambda_G, hoy_G = run_test_time_merging(
        "Method G: AHR-SAN (Proposed, Unsupervised Calibrated Data-Free)",
        expert_mnist_cos, expert_fashion_cos,
        stream_batches, device,
        prototypes0=None, prototypes1=None,
        fisher_sens0=fisher_cos0, fisher_sens1=fisher_cos1,
        is_cosface=True,
        tau_sparse=0.6,
        use_data_free_prototypes=True,
        gamma_dun=0.87,
        gamma_ealr=2.67
    )
    results["Method G"] = (acc_G, lambda_G)
    
    # 6. Plotting Results
    print("\nSaving plots and final report...")
    plt.figure(figsize=(14, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(acc_B, label="Method B: CL W-Fisher (L2 SCTS)", color="blue", alpha=0.7)
    plt.plot(acc_D, label="Method D: CP-AM (Angular SCTS)", color="green", alpha=0.7)
    plt.plot(acc_E, label="Method E: AHR-SAN (Proposed, Offline)", color="red", linewidth=2)
    plt.plot(acc_F, label="Method F: AHR-SAN (Proposed, Data-Free)", color="purple", linestyle="--", linewidth=2)
    plt.plot(acc_G, label="Method G: AHR-SAN (Proposed, Unsupervised Calibrated)", color="magenta", linestyle=":", linewidth=2)
    plt.axvline(x=10, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(x=20, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(x=30, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(x=40, color="gray", linestyle="--", alpha=0.5)
    plt.title("Test-Time Model Merging Accuracy Tracking")
    plt.xlabel("Batch Index")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sparsity / Routing Coefficients plot
    plt.subplot(1, 2, 2)
    plt.plot(lambda_B, label="Lambda (L2 SCTS)", color="blue", alpha=0.6)
    plt.plot(lambda_D, label="Lambda (CP-AM)", color="green", alpha=0.6)
    plt.plot(lambda_E, label="Lambda (AHR-SAN, Offline)", color="red", linewidth=2)
    plt.plot(lambda_G, label="Lambda (AHR-SAN, Calibrated)", color="magenta", linestyle=":", alpha=0.8)
    plt.plot(hoy_E, label="Hoyer Sparsity S(f)", color="orange", linestyle="-.", alpha=0.8)
    plt.axvline(x=10, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(x=20, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(x=30, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(x=40, color="gray", linestyle="--", alpha=0.5)
    plt.title("Merging Coefficient & Hoyer Sparsity Tracking")
    plt.xlabel("Batch Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("./plots", exist_ok=True)
    plt.savefig("./plots/stream_results.png", dpi=300)
    print("Saved stream_results.png in ./plots/")
    
    # Save statistics report
    with open("./plots/metrics.txt", "w") as f:
        f.write("Test-Time Model Merging Performance Report\n")
        f.write("==========================================\n\n")
        for key, name in [("Method A", "Method A: Fixed TTA + Reset (L2)"),
                          ("Method B", "Method B: CL W-Fisher + SCTS (L2)"),
                          ("Method C", "Method C: CL W-Fisher + A-SCTS"),
                          ("Method D", "Method D: CP-AM (Ours)"),
                          ("Method E", "Method E: AHR-SAN (Proposed, Offline)"),
                          ("Method F", "Method F: AHR-SAN (Proposed, Data-Free)"),
                          ("Method G", "Method G: AHR-SAN (Proposed, Unsupervised Calibrated, Data-Free)")]:
            acc, _ = results[key]
            f.write(f"{name}:\n")
            f.write(f"  Clean MNIST (0-9):  {np.mean(acc[0:10]):.2f}%\n")
            f.write(f"  Noisy MNIST (10-19): {np.mean(acc[10:20]):.2f}%\n")
            f.write(f"  Clean Fashion (20-29): {np.mean(acc[20:30]):.2f}%\n")
            f.write(f"  Noisy Fashion (30-39): {np.mean(acc[30:40]):.2f}%\n")
            f.write(f"  Novel KMNIST (40-49): {np.mean(acc[40:50]):.2f}%\n")
            f.write(f"  Overall Average:     {np.mean(acc):.2f}%\n\n")
            
    print("Metrics written to ./plots/metrics.txt")

if __name__ == "__main__":
    main()
