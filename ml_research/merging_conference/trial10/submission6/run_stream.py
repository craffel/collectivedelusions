import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.func import functional_call
import os
import numpy as np
import copy

# SimpleCNN class identical to the pre-training script
class SimpleCNN(nn.Module):
    def __init__(self, use_cosface=False, s=30.0, m=0.35):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.25)
        
        self.use_cosface = use_cosface
        if use_cosface:
            self.classifier_weight = nn.Parameter(torch.randn(10, 128))
            self.s = s
            self.m = m
        else:
            self.classifier = nn.Linear(128, 10)
            
    def extract_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.bn3(self.fc1(x))))
        return x
        
    def forward(self, x, labels=None):
        features = self.extract_features(x)
        if self.use_cosface:
            f_norm = F.normalize(features, p=2, dim=1)
            w_norm = F.normalize(self.classifier_weight, p=2, dim=1)
            cosine = F.linear(f_norm, w_norm)
            if self.training and labels is not None:
                one_hot = torch.zeros_like(cosine)
                one_hot.scatter_(1, labels.view(-1, 1), 1.0)
                output = self.s * (cosine - one_hot * self.m)
            else:
                output = self.s * cosine
        else:
            output = self.classifier(features)
        return output

# Helper to load experts
def load_experts():
    experts = {}
    
    # Load Standard MNIST and Fashion experts
    mnist_std_ckpt = torch.load("checkpoints/expert_mnist_std.pt", map_location='cpu')
    mnist_std = SimpleCNN(use_cosface=False)
    mnist_std.load_state_dict(mnist_std_ckpt['state_dict'])
    experts['mnist_std'] = {
        'model': mnist_std,
        'prototypes': mnist_std_ckpt['prototypes']
    }
    
    fashion_std_ckpt = torch.load("checkpoints/expert_fashion_std.pt", map_location='cpu')
    fashion_std = SimpleCNN(use_cosface=False)
    fashion_std.load_state_dict(fashion_std_ckpt['state_dict'])
    experts['fashion_std'] = {
        'model': fashion_std,
        'prototypes': fashion_std_ckpt['prototypes']
    }
    
    # Load CosFace MNIST and Fashion experts
    mnist_cos_ckpt = torch.load("checkpoints/expert_mnist_cos.pt", map_location='cpu')
    mnist_cos = SimpleCNN(use_cosface=True)
    mnist_cos.load_state_dict(mnist_cos_ckpt['state_dict'])
    experts['mnist_cos'] = {
        'model': mnist_cos,
        'prototypes': mnist_cos_ckpt['prototypes']
    }
    
    fashion_cos_ckpt = torch.load("checkpoints/expert_fashion_cos.pt", map_location='cpu')
    fashion_cos = SimpleCNN(use_cosface=True)
    fashion_cos.load_state_dict(fashion_cos_ckpt['state_dict'])
    experts['fashion_cos'] = {
        'model': fashion_cos,
        'prototypes': fashion_cos_ckpt['prototypes']
    }
    
    return experts

# Denoised Hoyer sparsity estimator
def compute_denoised_hoyer_sparsity(x):
    # Map from [-1, 1] to [0, 1]
    x_pos = (x + 1.0) / 2.0
    # Threshold denoising
    x_denoised = torch.where(x_pos > 0.35, x_pos, torch.zeros_like(x_pos))
    # Flatten
    x_flat = x_denoised.view(x_denoised.size(0), -1)
    
    # Hoyer's Sparsity: (sqrt(d) - ||f||1 / ||f||2) / (sqrt(d) - 1)
    d = x_flat.size(1)
    l1 = torch.norm(x_flat, p=1, dim=1)
    l2 = torch.norm(x_flat, p=2, dim=1) + 1e-8
    
    sparsity = (np.sqrt(d) - l1 / l2) / (np.sqrt(d) - 1.0)
    return sparsity.mean().item()

# Compute batch-wise prototype distance D_m
def compute_prototype_distance(features, prototypes, metric='angular'):
    # features: (B, 128), prototypes: (10, 128)
    B = features.size(0)
    if metric == 'angular':
        f_norm = F.normalize(features, p=2, dim=1)
        p_norm = F.normalize(prototypes, p=2, dim=1)
        # dist = 2 - 2 * cos(theta)
        cos_sim = torch.mm(f_norm, p_norm.t()) # (B, 10)
        dist = 2.0 - 2.0 * cos_sim
    else: # Euclidean
        # dist = ||f - P||_2
        # Expand shapes: features is (B, 1, 128), prototypes is (1, 10, 128)
        diff = features.unsqueeze(1) - prototypes.unsqueeze(0) # (B, 10, 128)
        dist = torch.norm(diff, p=2, dim=2) # (B, 10)
        
    # Batch-wise prototype distance is 1/B * sum_{i} min_{c} dist(f_i, P_{m,c})
    min_dist, _ = dist.min(dim=1)
    return min_dist.mean().item()

# Compute SCTS routing prior w0, w1
def compute_scts_prior(d0, d1, eps_stab=0.04):
    gap = abs(d0 - d1)
    tau = gap / 3.0 + eps_stab
    w0 = np.exp(-d0 / tau) / (np.exp(-d0 / tau) + np.exp(-d1 / tau) + 1e-8)
    w1 = 1.0 - w0
    return w0, w1

# Load data and prepare 50-batch non-stationary stream
def prepare_test_stream():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_val = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fashion_val = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_val = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    # 50 batches of size 64
    B = 64
    stream_batches = []
    
    # Batches 0-9: Clean MNIST
    for i in range(10):
        subset = Subset(mnist_val, list(range(i * B, (i + 1) * B)))
        loader = DataLoader(subset, batch_size=B, shuffle=False)
        x, y = next(iter(loader))
        stream_batches.append((x, y, 'C-MN'))
        
    # Batches 10-19: Noisy MNIST (sigma=0.6)
    for i in range(10):
        subset = Subset(mnist_val, list(range((10 + i) * B, (11 + i) * B)))
        loader = DataLoader(subset, batch_size=B, shuffle=False)
        x, y = next(iter(loader))
        noisy_x = torch.clamp(x + torch.randn_like(x) * 0.6, -1.0, 1.0)
        stream_batches.append((noisy_x, y, 'N-MN'))
        
    # Batches 20-29: Clean FashionMNIST
    for i in range(10):
        subset = Subset(fashion_val, list(range(i * B, (i + 1) * B)))
        loader = DataLoader(subset, batch_size=B, shuffle=False)
        x, y = next(iter(loader))
        stream_batches.append((x, y, 'C-FN'))
        
    # Batches 30-39: Noisy FashionMNIST (sigma=0.6)
    for i in range(10):
        subset = Subset(fashion_val, list(range((10 + i) * B, (11 + i) * B)))
        loader = DataLoader(subset, batch_size=B, shuffle=False)
        x, y = next(iter(loader))
        noisy_x = torch.clamp(x + torch.randn_like(x) * 0.6, -1.0, 1.0)
        stream_batches.append((noisy_x, y, 'N-FN'))
        
    # Batches 40-49: Novel KMNIST
    for i in range(10):
        subset = Subset(kmnist_val, list(range(i * B, (i + 1) * B)))
        loader = DataLoader(subset, batch_size=B, shuffle=False)
        x, y = next(iter(loader))
        stream_batches.append((x, y, 'Nov-K'))
        
    return stream_batches

# Precompute Offline Fisher for Expert 0 and Expert 1 on clean validation samples
def precompute_offline_fisher(expert_mnist, expert_fashion):
    print("Precomputing offline Fisher matrices...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_val = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fashion_val = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    loader_mnist = DataLoader(Subset(mnist_val, list(range(8000, 8256))), batch_size=64, shuffle=False)
    loader_fashion = DataLoader(Subset(fashion_val, list(range(8000, 8256))), batch_size=64, shuffle=False)
    
    def compute_fisher(model, loader):
        model.eval()
        fisher = {}
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param)
            
        count = 0
        for x, _ in loader:
            model.zero_grad()
            features = model.extract_features(x)
            outputs = model(x)
            probs = F.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            entropy.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * x.size(0)
            count += x.size(0)
            
        for name in fisher:
            fisher[name] /= count
            # Bound below by a small constant
            fisher[name] = torch.clamp(fisher[name], min=1e-5)
        return fisher
        
    fisher_mnist = compute_fisher(expert_mnist, loader_mnist)
    fisher_fashion = compute_fisher(expert_fashion, loader_fashion)
    
    # Combined Fisher matrix (weighted mean of both experts)
    combined_fisher = {}
    for name in fisher_mnist:
        combined_fisher[name] = 0.5 * fisher_mnist[name] + 0.5 * fisher_fashion[name]
        
    return combined_fisher

# Evaluate a full TTA stream using a given method config
def evaluate_method(method_name, experts, stream_batches, precomputed_fisher=None, s_momentum=0.8, depth_coherence=False, eta=0.05, beta=1.5, gamma_c=0.02, eps_damping=0.02, depth_factors=None):
    torch.manual_seed(42)
    print(f"\nEvaluating: {method_name}")
    
    accuracies = {'C-MN': [], 'N-MN': [], 'C-FN': [], 'N-FN': [], 'Nov-K': []}
    overall_correct = 0
    overall_total = 0
    
    # Initialize EMA of on-the-fly sensitivities for Method F (SMT)
    smoothed_sensitivity_sparse = None
    smoothed_sensitivity_dense = None
    
    # Initialize EMA of raw gradients for SNR-Adaptive Dynamic Damping (SADD)
    smoothed_gradient_sparse = None
    smoothed_gradient_dense = None
    
    # Define layer depth coherence scaling factors for Method F (LDAC)
    # Shallow layers are heavily regularized to stay near the consensus logit w_global,
    # while deep task-specific heads are granted higher flexibility to adapt.
    # If depth_factors is None, we will dynamically compute them inside the batch loop
    # using our Continuous Depth Weighting (CDW) formulation to eliminate manual tuning.
    use_dynamic_cdw = (depth_factors is None)
    if not use_dynamic_cdw:
        # If manual depth factors are provided, use them
        depth_factors = depth_factors
    else:
        depth_factors = {}
    
    for batch_idx, (x, y, domain) in enumerate(stream_batches):
        # 1. Determine which experts and routing domain to use
        if method_name in ['Method A (Fixed TTA + Reset)', 'Method B (CL W-Fisher + SCTS L2)', 'Method C (CL W-Fisher + A-SCTS)']:
            expert0_model = experts['mnist_std']['model']
            expert1_model = experts['mnist_std']['model'] # wait, standard uses standard model architecture
            expert0 = experts['mnist_std']
            expert1 = experts['fashion_std']
            cosface_mode = False
        elif method_name in ['Method D (CP-AM)']:
            expert0 = experts['mnist_cos']
            expert1 = experts['fashion_cos']
            cosface_mode = True
        else: # Method E (BK-AHR) and Method F (SMT-LDAC, Ours)
            # Sparsity-aware gating
            sparsity = compute_denoised_hoyer_sparsity(x)
            if sparsity >= 0.50:
                # Sparse domain: Standard experts
                expert0 = experts['mnist_std']
                expert1 = experts['fashion_std']
                cosface_mode = False
            else:
                # Dense domain: CosFace experts
                expert0 = experts['mnist_cos']
                expert1 = experts['fashion_cos']
                cosface_mode = True
                
        # 2. Extract features of current batch to compute routing prior
        expert0['model'].eval()
        expert1['model'].eval()
        with torch.no_grad():
            f0 = expert0['model'].extract_features(x)
            f1 = expert1['model'].extract_features(x)
            
        # Select routing metric
        if method_name in ['Method A (Fixed TTA + Reset)', 'Method B (CL W-Fisher + SCTS L2)']:
            metric = 'euclidean'
        elif method_name in ['Method C (CL W-Fisher + A-SCTS)', 'Method D (CP-AM)']:
            metric = 'angular'
        else: # BK-AHR and Ours
            metric = 'euclidean' if sparsity >= 0.50 else 'angular'
            
        d0 = compute_prototype_distance(f0, expert0['prototypes'], metric)
        d1 = compute_prototype_distance(f1, expert1['prototypes'], metric)
        
        # SCTS prior routing coefficents
        w0, w1 = compute_scts_prior(d0, d1)
        
        # 3. Model parameters and offsets setup
        # Reconstruct base model to run stateless functional adaptation
        base_model = SimpleCNN(use_cosface=cosface_mode)
        
        # Trainable parameters names
        param_names = [n for n, p in base_model.named_parameters() if p.requires_grad]
        
        # Initialize damping factors for SNR-Adaptive Dynamic Damping (SADD)
        eps_damping_dict = {name: eps_damping for name in param_names}
        
        # Dynamically compute Continuous Depth Weighting (CDW) if requested
        if depth_coherence and use_dynamic_cdw:
            N = len(param_names)
            depth_factors = {
                name: 2.5 * (1.0 - idx / (N - 1)) + 0.4 * (idx / (N - 1)) if N > 1 else 1.0
                for idx, name in enumerate(param_names)
            }
        
        # Experts parameters dictionaries
        theta1 = dict(expert0['model'].named_parameters())
        theta2 = dict(expert1['model'].named_parameters())
        
        # Batch Normalization running buffers
        buffers1 = dict(expert0['model'].named_buffers())
        buffers2 = dict(expert1['model'].named_buffers())
        
        # Initialization
        # w_global is the global consensus logit
        w_global = torch.tensor(np.log(w0 / (w1 + 1e-8)), requires_grad=True)
        # delta_j are layer-specific offsets
        delta = {name: torch.zeros_like(theta1[name], requires_grad=True) for name in param_names}
        
        # Test-Time Batch Normalization (TTBN)
        # Put base model BN layers in training mode to use batch statistics under Method E and F (and A)
        use_ttbn = method_name in ['Method A (Fixed TTA + Reset)', 'Method E (BK-AHR with TTBN, SOTA)', 'Method F (SMT-LDAC, Ours)']
        
        # Extract initial on-the-fly Kronecker sensitivity Fj using gradients of the entropy loss
        if method_name in ['Method E (BK-AHR with TTBN, SOTA)', 'Method F (SMT-LDAC, Ours)']:
            # Construct initial merged parameters with routing priors (delta = 0)
            merged_params_init = {}
            for name in param_names:
                merged_params_init[name] = w0 * theta1[name] + w1 * theta2[name]
                
            # Run a functional forward pass to compute entropy
            base_model.train(use_ttbn)
            outputs_init = functional_call(base_model, merged_params_init, x)
            probs_init = F.softmax(outputs_init, dim=1)
            entropy_init = -torch.sum(probs_init * torch.log(probs_init + 1e-8), dim=1).mean()
            
            # Compute gradients of entropy with respect to merged parameters
            grads = torch.autograd.grad(entropy_init, list(merged_params_init.values()))
            grads_dict = {name: g for name, g in zip(param_names, grads)}
            
            # Raw sensitivity Fj = mean squared gradient
            F_raw = {}
            for name in param_names:
                g = grads_dict[name]
                F_raw[name] = g.pow(2).mean() # trace-like sensitivity estimate
                
            # Method F: Sensitivity Momentum Tracking (EMA)
            if method_name == 'Method F (SMT-LDAC, Ours)':
                if sparsity >= 0.50:
                    if smoothed_sensitivity_sparse is None:
                        smoothed_sensitivity_sparse = {name: F_raw[name].clone().detach() for name in param_names}
                        smoothed_gradient_sparse = {name: grads_dict[name].clone().detach() for name in param_names}
                    else:
                        for name in param_names:
                            smoothed_sensitivity_sparse[name] = s_momentum * smoothed_sensitivity_sparse[name] + (1.0 - s_momentum) * F_raw[name].detach()
                            smoothed_gradient_sparse[name] = s_momentum * smoothed_gradient_sparse[name] + (1.0 - s_momentum) * grads_dict[name].detach()
                    F_opt = smoothed_sensitivity_sparse
                    g_opt = smoothed_gradient_sparse
                else:
                    if smoothed_sensitivity_dense is None:
                        smoothed_sensitivity_dense = {name: F_raw[name].clone().detach() for name in param_names}
                        smoothed_gradient_dense = {name: grads_dict[name].clone().detach() for name in param_names}
                    else:
                        for name in param_names:
                            smoothed_sensitivity_dense[name] = s_momentum * smoothed_sensitivity_dense[name] + (1.0 - s_momentum) * F_raw[name].detach()
                            smoothed_gradient_dense[name] = s_momentum * smoothed_gradient_dense[name] + (1.0 - s_momentum) * grads_dict[name].detach()
                    F_opt = smoothed_sensitivity_dense
                    g_opt = smoothed_gradient_dense
            else: # Method E: BK-AHR standard single-batch sensitivity
                F_opt = F_raw
                g_opt = None
                
            # Normalize sensitivities globally
            total_F = sum(F_opt.values()) + 1e-8
            F_tilde = {name: F_opt[name] / total_F for name in param_names}
            
            # Compute SNR-Adaptive Dynamic Damping (SADD) for Method F
            if method_name == 'Method F (SMT-LDAC, Ours)':
                for name in param_names:
                    signal = g_opt[name].pow(2).mean()
                    noise = F_opt[name]
                    snr = signal / (noise + 1e-6)
                    snr = torch.clamp(snr, 0.0, 1.0)
                    eps_damping_dict[name] = eps_damping * (1.0 + 5.0 * (1.0 - snr.item()))
            
        # 4. TTA adaptation loop: 5 steps of gradient descent
        for step in range(5):
            # Sigmoid weights for each layer
            lambda_layer = {}
            for name in param_names:
                lambda_layer[name] = torch.sigmoid(w_global + delta[name])
                
            # Merge parameters
            merged_params = {}
            for name in param_names:
                merged_params[name] = lambda_layer[name] * theta1[name] + (1.0 - lambda_layer[name]) * theta2[name]
                
            # Merge BN buffers via Soft Bayesian MoG formulation
            merged_buffers = {}
            lambda_det = torch.sigmoid(w_global).detach()
            
            for buf_name in buffers1:
                if 'running_mean' in buf_name:
                    mu1 = buffers1[buf_name]
                    mu2 = buffers2[buf_name]
                    mu_f = lambda_det * mu1 + (1.0 - lambda_det) * mu2
                    merged_buffers[buf_name] = mu_f
                    
                    # Compute fused variance for the matching variance buffer
                    var_name = buf_name.replace('running_mean', 'running_var')
                    var1 = buffers1[var_name]
                    var2 = buffers2[var_name]
                    var_f = lambda_det * (var1 + (mu1 - mu_f).pow(2)) + (1.0 - lambda_det) * (var2 + (mu2 - mu_f).pow(2))
                    merged_buffers[var_name] = var_f
                elif 'num_batches_tracked' in buf_name:
                    merged_buffers[buf_name] = buffers1[buf_name]
                    
            # Run forward pass on base model functionally
            base_model.train(use_ttbn)
            outputs = functional_call(base_model, (merged_params, merged_buffers), x)
            probs = F.softmax(outputs, dim=1)
            loss_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            
            # Prior penalty D_KL(w || lambda)
            loss_prior = 0.0
            for name in param_names:
                # D_KL between routing prior w0 and lambda_layer
                kl_div = w0 * torch.log((w0 + 1e-8) / (lambda_layer[name] + 1e-8)) + w1 * torch.log((w1 + 1e-8) / (1.0 - lambda_layer[name] + 1e-8))
                loss_prior += kl_div.mean()
            loss_prior = beta * loss_prior / len(param_names)
            
            # Coherence penalty
            loss_coherence = 0.0
            if method_name in ['Method B (CL W-Fisher + SCTS L2)', 'Method C (CL W-Fisher + A-SCTS)', 'Method D (CP-AM)']:
                # Standard offline Fisher-weighted coherence
                for name in param_names:
                    fisher_name = name
                    if name == 'classifier_weight':
                        fisher_name = 'classifier.weight'
                    f_val = precomputed_fisher[fisher_name].mean() if (precomputed_fisher is not None and fisher_name in precomputed_fisher) else 1.0
                    loss_coherence += f_val * delta[name].pow(2).mean()
                loss_coherence = gamma_c * loss_coherence
            elif method_name in ['Method E (BK-AHR with TTBN, SOTA)', 'Method F (SMT-LDAC, Ours)']:
                # On-the-fly preconditioned coherence
                for name in param_names:
                    factor = depth_factors[name] if (depth_coherence and name in depth_factors) else 1.0
                    loss_coherence += factor * F_tilde[name] * delta[name].pow(2).sum()
                loss_coherence = gamma_c * loss_coherence
                
            # Total TTA adaptation loss
            loss_total = loss_entropy + loss_prior + loss_coherence
            
            # Extract gradients
            vars_to_opt = [w_global] + list(delta.values())
            grads_opt = torch.autograd.grad(loss_total, vars_to_opt, allow_unused=True)
            
            # Update parameters
            with torch.no_grad():
                # w_global update
                if grads_opt[0] is not None:
                    w_global.copy_(w_global - eta * grads_opt[0])
                    
                # delta offsets update with preconditioning
                for i, name in enumerate(param_names):
                    g_delta = grads_opt[i + 1]
                    if g_delta is not None:
                        # Determine preconditioning sensitivity value
                        if method_name in ['Method B (CL W-Fisher + SCTS L2)', 'Method C (CL W-Fisher + A-SCTS)', 'Method D (CP-AM)']:
                            fisher_name = name
                            if name == 'classifier_weight':
                                fisher_name = 'classifier.weight'
                            sens = precomputed_fisher[fisher_name] if (precomputed_fisher is not None and fisher_name in precomputed_fisher) else torch.ones_like(g_delta)
                        elif method_name in ['Method E (BK-AHR with TTBN, SOTA)', 'Method F (SMT-LDAC, Ours)']:
                            # Global relative sensitivity scalar preconditioning
                            sens = F_tilde[name]
                        else: # Method A: unpreconditioned standard gradient descent
                            sens = 1.0
                            
                        # Learning rate preconditioning step: delta = delta - eta * 1 / (sens + eps) * grad
                        delta[name].copy_(delta[name] - eta * (1.0 / (sens + eps_damping_dict[name])) * g_delta)
                        
        # 5. Inference / final prediction using optimized parameters on current batch
        with torch.no_grad():
            lambda_layer_final = {}
            for name in param_names:
                lambda_layer_final[name] = torch.sigmoid(w_global + delta[name])
                
            merged_params_final = {}
            for name in param_names:
                merged_params_final[name] = lambda_layer_final[name] * theta1[name] + (1.0 - lambda_layer_final[name]) * theta2[name]
                
            merged_buffers_final = {}
            lambda_det_final = torch.sigmoid(w_global).detach()
            for buf_name in buffers1:
                if 'running_mean' in buf_name:
                    mu1 = buffers1[buf_name]
                    mu2 = buffers2[buf_name]
                    mu_f = lambda_det_final * mu1 + (1.0 - lambda_det_final) * mu2
                    merged_buffers_final[buf_name] = mu_f
                    
                    var_name = buf_name.replace('running_mean', 'running_var')
                    var1 = buffers1[var_name]
                    var2 = buffers2[var_name]
                    var_f = lambda_det_final * (var1 + (mu1 - mu_f).pow(2)) + (1.0 - lambda_det_final) * (var2 + (mu2 - mu_f).pow(2))
                    merged_buffers_final[var_name] = var_f
                elif 'num_batches_tracked' in buf_name:
                    merged_buffers_final[buf_name] = buffers1[buf_name]
                    
            base_model.train(use_ttbn) # TTBN mode for inference
            final_outputs = functional_call(base_model, (merged_params_final, merged_buffers_final), x)
            _, predicted = final_outputs.max(dim=1)
            
            # Compute classification accuracy on batch
            correct_cnt = predicted.eq(y).sum().item()
            acc = 100.0 * correct_cnt / y.size(0)
            
            accuracies[domain].append(acc)
            overall_correct += correct_cnt
            overall_total += y.size(0)
            
    # Compute average accuracies across segments
    results = {}
    for dom in accuracies:
        results[dom] = np.mean(accuracies[dom])
    results['Overall'] = 100.0 * overall_correct / overall_total
    
    # Print results
    print(f"Results for {method_name}:")
    print(f"  Clean MNIST (C-MN): {results['C-MN']:.2f}%")
    print(f"  Noisy MNIST (N-MN): {results['N-MN']:.2f}%")
    print(f"  Clean Fashion (C-FN): {results['C-FN']:.2f}%")
    print(f"  Noisy Fashion (N-FN): {results['N-FN']:.2f}%")
    print(f"  Novel KMNIST (Nov-K): {results['Nov-K']:.2f}%")
    print(f"  Overall Accuracy: {results['Overall']:.2f}%")
    
    return results

def main():
    torch.manual_seed(42)
    experts = load_experts()
    stream_batches = prepare_test_stream()
    
    # Precompute offline Fisher matrices using MNIST and Fashion standard models on validation samples
    precomputed_fisher = precompute_offline_fisher(experts['mnist_std']['model'], experts['fashion_std']['model'])
    
    all_results = {}
    
    # Evaluate Method A
    all_results['Method A'] = evaluate_method('Method A (Fixed TTA + Reset)', experts, stream_batches)
    
    # Evaluate Method B
    all_results['Method B'] = evaluate_method('Method B (CL W-Fisher + SCTS L2)', experts, stream_batches, precomputed_fisher)
    
    # Evaluate Method C
    all_results['Method C'] = evaluate_method('Method C (CL W-Fisher + A-SCTS)', experts, stream_batches, precomputed_fisher)
    
    # Evaluate Method D
    all_results['Method D'] = evaluate_method('Method D (CP-AM)', experts, stream_batches, precomputed_fisher)
    
    # Evaluate Method E (BK-AHR SOTA)
    all_results['Method E'] = evaluate_method('Method E (BK-AHR with TTBN, SOTA)', experts, stream_batches)
    
    # Evaluate Method F (SMT-LDAC, Ours)
    # Uses Sensitivity Momentum (alpha=0.85) and Layer-Depth-Aware Coherence (beta=2.0, gamma_c=0.02, eta=0.07)
    all_results['Method F'] = evaluate_method('Method F (SMT-LDAC, Ours)', experts, stream_batches, s_momentum=0.85, depth_coherence=True, eta=0.07, beta=2.0, gamma_c=0.02)
    
    # Let's save the results to a numpy or text file
    print("\n--- Summary Table ---")
    headers = ["Method", "C-MN", "N-MN", "C-FN", "N-FN", "Nov-K", "Overall"]
    print(f"{headers[0]:<12} {headers[1]:<8} {headers[2]:<8} {headers[3]:<8} {headers[4]:<8} {headers[5]:<8} {headers[6]:<8}")
    for m in all_results:
        r = all_results[m]
        print(f"{m:<12} {r['C-MN']:>7.2f}% {r['N-MN']:>7.2f}% {r['C-FN']:>7.2f}% {r['N-FN']:>7.2f}% {r['Nov-K']:>7.2f}% {r['Overall']:>7.2f}%")
        
    np.save("checkpoints/results.npy", all_results)

if __name__ == "__main__":
    main()
