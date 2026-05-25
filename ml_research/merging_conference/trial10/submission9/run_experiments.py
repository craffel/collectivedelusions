import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import matplotlib.pyplot as plt
import copy
from torch.func import functional_call

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors on this cluster
torch.backends.cudnn.enabled = False

# SimpleCNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        features = F.relu(self.fc1(x))
        out = self.classifier(features)
        if return_features:
            return out, features
        return out

# Denoising function
def denoise_image(x, threshold=0.35):
    # Map from [-1, 1] to [0, 1]
    x_pos = (x + 1.0) / 2.0
    # Apply thresholding
    x_denoised = (x_pos > threshold).float() * x_pos
    # Map back to [-1, 1]
    return x_denoised * 2.0 - 1.0

# Hoyer Sparsity calculation
def hoyer_sparsity(x):
    # Flatten the batch of images
    x_flat = x.view(x.size(0), -1)
    # Map to positive intensity first to ensure positive values for Hoyer sparsity
    x_pos = (x_flat + 1.0) / 2.0
    
    d = x_pos.size(1)
    l1 = torch.norm(x_pos, p=1, dim=1)
    l2 = torch.norm(x_pos, p=2, dim=1)
    
    # Hoyer sparsity formula
    sparsity = (np.sqrt(d) - (l1 / (l2 + 1e-8))) / (np.sqrt(d) - 1.0)
    return sparsity.mean().item()

# Soft Bayesian Mixture-of-Gaussians BN Statistic Fusion
def fuse_bn_buffers(model0, model1, merged_model, lam):
    with torch.no_grad():
        for (name0, buf0), (name1, buf1), (namem, bufm) in zip(model0.named_buffers(), model1.named_buffers(), merged_model.named_buffers()):
            if "running_mean" in name0:
                bufm.copy_((1.0 - lam) * buf0 + lam * buf1)
            elif "running_var" in name0:
                mean_name0 = name0.replace("running_var", "running_mean")
                m0 = dict(model0.named_buffers())[mean_name0]
                m1 = dict(model1.named_buffers())[mean_name0]
                mm = dict(merged_model.named_buffers())[mean_name0]
                
                # MoG variance fusion formula
                var_fused = (1.0 - lam) * (buf0 + (m0 - mm)**2) + lam * (buf1 + (m1 - mm)**2)
                bufm.copy_(var_fused)

# Merge model weights
def merge_model_weights(model0, model1, merged_model, lambdas):
    # lambdas is a dict of layer_name -> weight
    with torch.no_grad():
        for name, param in merged_model.named_parameters():
            if name in lambdas:
                lam = lambdas[name]
                param.copy_((1.0 - lam) * dict(model0.named_parameters())[name] + lam * dict(model1.named_parameters())[name])

# SCTS routing prior computation
def compute_scts_prior(features0, features1, prototypes0, prototypes1, metric_type="angular"):
    # features0/features1 are [B, 128]
    # prototypes0/prototypes1 are [10, 128]
    B = features0.size(0)
    
    # Normalize features if angular
    if metric_type == "angular":
        f0 = F.normalize(features0, p=2, dim=1)
        f1 = F.normalize(features1, p=2, dim=1)
        p0 = F.normalize(prototypes0, p=2, dim=1)
        p1 = F.normalize(prototypes1, p=2, dim=1)
        
        # Distance is 1.0 - cos_similarity
        d0_all = 1.0 - torch.matmul(f0, p0.t()) # [B, 10]
        d1_all = 1.0 - torch.matmul(f1, p1.t()) # [B, 10]
    else: # normalized L2
        f0 = F.normalize(features0, p=2, dim=1)
        f1 = F.normalize(features1, p=2, dim=1)
        p0 = F.normalize(prototypes0, p=2, dim=1)
        p1 = F.normalize(prototypes1, p=2, dim=1)
        
        # Squared Euclidean distance on normalized features
        # ||f - p||^2 = 2 - 2 * cos_similarity
        d0_all = 2.0 - 2.0 * torch.matmul(f0, p0.t())
        d1_all = 2.0 - 2.0 * torch.matmul(f1, p1.t())

    # Batch-wise prototype distance (minimum distance to any class prototype)
    d0, _ = torch.min(d0_all, dim=1)
    d1, _ = torch.min(d1_all, dim=1)
    
    D0 = d0.mean().item()
    D1 = d1.mean().item()
    
    gap = abs(D0 - D1)
    epsilon_stab = 0.04
    tau = (gap / 3.0) + epsilon_stab
    
    w1 = np.exp(-D1 / tau) / (np.exp(-D0 / tau) + np.exp(-D1 / tau) + 1e-8)
    w0 = 1.0 - w1
    return w0, w1

# Load MNIST, FashionMNIST, and KMNIST Datasets
def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    return mnist_train, mnist_test, fmnist_train, fmnist_test, kmnist_test

# Train the expert networks
def train_experts(mnist_train, fmnist_train, device):
    print("Pre-training base initialization jointly on MNIST and FashionMNIST...")
    base_model = SimpleCNN().to(device)
    
    # Combine subset of training sets for joint initialization
    # To keep it extremely fast but still accurate, let's use 10,000 samples from each
    set_seed(42)
    mnist_indices = torch.randperm(len(mnist_train))[:10000]
    fmnist_indices = torch.randperm(len(fmnist_train))[:10000]
    
    joint_dataset = ConcatDataset([
        Subset(mnist_train, mnist_indices),
        Subset(fmnist_train, fmnist_indices)
    ])
    joint_loader = DataLoader(joint_dataset, batch_size=64, shuffle=True)
    
    optimizer = optim.Adam(base_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    base_model.train()
    for epoch in range(1):
        for img, label in joint_loader:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            out = base_model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            
    print("Joint pre-training completed.")
    
    # Save base model checkpoint
    torch.save(base_model.state_dict(), "base_init.pth")
    
    # Create and train Expert 0 (MNIST)
    print("Fine-tuning Expert 0 on MNIST...")
    expert0 = SimpleCNN().to(device)
    expert0.load_state_dict(torch.load("base_init.pth", map_location=device))
    
    mnist_subset = Subset(mnist_train, torch.randperm(len(mnist_train))[:15000])
    mnist_loader = DataLoader(mnist_subset, batch_size=64, shuffle=True)
    optimizer0 = optim.Adam(expert0.parameters(), lr=2e-4, weight_decay=1e-5)
    
    expert0.train()
    for epoch in range(1):
        for img, label in mnist_loader:
            img, label = img.to(device), label.to(device)
            optimizer0.zero_grad()
            out = expert0(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer0.step()
            
    torch.save(expert0.state_dict(), "expert0.pth")
    print("Expert 0 (MNIST) trained.")
    
    # Create and train Expert 1 (FashionMNIST)
    print("Fine-tuning Expert 1 on FashionMNIST...")
    expert1 = SimpleCNN().to(device)
    expert1.load_state_dict(torch.load("base_init.pth", map_location=device))
    
    fmnist_subset = Subset(fmnist_train, torch.randperm(len(fmnist_train))[:15000])
    fmnist_loader = DataLoader(fmnist_subset, batch_size=64, shuffle=True)
    optimizer1 = optim.Adam(expert1.parameters(), lr=2e-4, weight_decay=1e-5)
    
    expert1.train()
    for epoch in range(1):
        for img, label in fmnist_loader:
            img, label = img.to(device), label.to(device)
            optimizer1.zero_grad()
            out = expert1(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer1.step()
            
    torch.save(expert1.state_dict(), "expert1.pth")
    print("Expert 1 (FashionMNIST) trained.")
    
    return expert0, expert1

# Compute prototype matrices
def precompute_prototypes(expert, dataset, device):
    expert.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    class_features = {c: [] for c in range(10)}
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            _, feats = expert(img, return_features=True)
            for f, l in zip(feats, label):
                class_features[l.item()].append(f.cpu())
                
    prototypes = torch.zeros(10, 128)
    for c in range(10):
        c_feats = torch.stack(class_features[c])
        # Average normalized features
        c_feats_norm = F.normalize(c_feats, p=2, dim=1)
        prototypes[c] = c_feats_norm.mean(dim=0)
        
    return prototypes.to(device)

# Generate Non-Stationary Target Test Stream (50 batches of size 64)
def generate_test_stream(mnist_test, fmnist_test, kmnist_test):
    set_seed(42)
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)
    fmnist_loader = DataLoader(fmnist_test, batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True)
    
    mnist_iter = iter(mnist_loader)
    fmnist_iter = iter(fmnist_loader)
    kmnist_iter = iter(kmnist_loader)
    
    stream = []
    
    # 1. Clean MNIST (Batches 0-9)
    for _ in range(10):
        img, label = next(mnist_iter)
        stream.append((img, label, "Clean MNIST"))
        
    # 2. Noisy MNIST (Batches 10-19)
    for _ in range(10):
        img, label = next(mnist_iter)
        # Add additive white Gaussian noise (AWGN) with sigma = 0.6
        noise = torch.randn_like(img) * 0.6
        noisy_img = torch.clamp(img + noise, -1.0, 1.0)
        stream.append((noisy_img, label, "Noisy MNIST"))
        
    # 3. Clean FashionMNIST (Batches 20-29)
    for _ in range(10):
        img, label = next(fmnist_iter)
        stream.append((img, label, "Clean FashionMNIST"))
        
    # 4. Noisy FashionMNIST (Batches 30-39)
    for _ in range(10):
        img, label = next(fmnist_iter)
        noise = torch.randn_like(img) * 0.6
        noisy_img = torch.clamp(img + noise, -1.0, 1.0)
        stream.append((noisy_img, label, "Noisy FashionMNIST"))
        
    # 5. Clean KMNIST (Batches 40-49)
    for _ in range(10):
        img, label = next(kmnist_iter)
        stream.append((img, label, "Clean KMNIST (OOD)"))
        
    return stream

# Evaluate a baseline or method on the stream
def evaluate_method(method_name, stream, expert0, expert1, prototypes0, prototypes1, device, rho=0.05, eta=0.005):
    print(f"Evaluating method: {method_name}...")
    set_seed(42)
    
    model0 = copy.deepcopy(expert0).to(device)
    model1 = copy.deepcopy(expert1).to(device)
    
    accuracies = []
    lambdas = []
    hoyers = []
    
    # For tracking across stream
    for idx, (img, label, segment_name) in enumerate(stream):
        img, label = img.to(device), label.to(device)
        
        # Calculate Hoyer sparsity of raw input
        h_sparsity = hoyer_sparsity(img)
        hoyers.append(h_sparsity)
        
        # Step 1: Sparsity and Metric Selection (Adaptive Hybrid Routing)
        # SATS-DUN and KPSAM-DST use Hoyer's sparsity on denoised image to choose metric
        img_denoised = denoise_image(img)
        h_sparsity_denoised = hoyer_sparsity(img_denoised)
        
        if "AHR" in method_name or "Ours" in method_name:
            metric_type = "euclidean" if h_sparsity_denoised >= 0.50 else "angular"
        else:
            # Standard baselines use Euclidean (L2) distance by default, except Angular CP-AM
            metric_type = "angular" if "Angular" in method_name else "euclidean"
            
        # Step 2: Compute routing priors
        model0.eval()
        model1.eval()
        with torch.no_grad():
            _, feats0 = model0(img, return_features=True)
            _, feats1 = model1(img, return_features=True)
            
        w0, w1 = compute_scts_prior(feats0, feats1, prototypes0, prototypes1, metric_type)
        
        # Step 3: Optimization / Parameters Merging setup
        # Reconstruct initial model
        merged_model = SimpleCNN().to(device)
        merged_model.load_state_dict(model0.state_dict())
        
        # Initialize logit coefficient
        w_global = torch.tensor(np.log(w1 / (w0 + 1e-8)), device=device, requires_grad=True)
        
        # Initialize layer-wise offsets delta_j
        delta = {}
        for name, param in merged_model.named_parameters():
            delta[name] = torch.zeros_like(param, device=device, requires_grad=True)
            
        # Optimization Loop
        if method_name == "Static":
            # No optimization, just routing weights
            lam_val = w1
            lambdas_dict = {name: torch.tensor(lam_val, device=device) for name, _ in merged_model.named_parameters()}
            merge_model_weights(model0, model1, merged_model, lambdas_dict)
            fuse_bn_buffers(model0, model1, merged_model, lam_val)
            
        elif method_name == "Fixed TTA":
            # Standard entropy minimization on noisy input
            optimizer = optim.SGD([w_global] + list(delta.values()), lr=0.01)
            for step in range(5):
                # Fuse BN buffers in-place (non-differentiable)
                fuse_bn_buffers(model0, model1, merged_model, torch.sigmoid(w_global).detach().item())
                
                # Construct merged parameters differentiably
                merged_params = {}
                for name, _ in merged_model.named_parameters():
                    lam = torch.sigmoid(w_global + delta[name])
                    param0 = dict(model0.named_parameters())[name]
                    param1 = dict(model1.named_parameters())[name]
                    merged_params[name] = (1.0 - lam) * param0 + lam * param1
                
                # Differentiable forward pass
                out = functional_call(merged_model, merged_params, img)
                loss = -torch.mean(torch.sum(F.softmax(out, dim=1) * F.log_softmax(out, dim=1), dim=1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            lam_val = torch.sigmoid(w_global).item()
            
        elif method_name == "BK-CoMerge":
            # Trace-preconditioned entropy minimization
            # Uses very small stability constant Floor (vulnerable to stability trap!)
            eps_stab_trap = 1e-5
            optimizer = optim.SGD([w_global], lr=0.05)
            
            for step in range(5):
                # Fuse BN buffers in-place (non-differentiable)
                fuse_bn_buffers(model0, model1, merged_model, torch.sigmoid(w_global).detach().item())
                
                # Construct merged parameters differentiably
                merged_params = {}
                for name, _ in merged_model.named_parameters():
                    lam = torch.sigmoid(w_global + delta[name])
                    param0 = dict(model0.named_parameters())[name]
                    param1 = dict(model1.named_parameters())[name]
                    merged_params[name] = (1.0 - lam) * param0 + lam * param1
                
                # Differentiable forward pass
                out = functional_call(merged_model, merged_params, img)
                loss = -torch.mean(torch.sum(F.softmax(out, dim=1) * F.log_softmax(out, dim=1), dim=1))
                
                # Preconditioned update
                optimizer.zero_grad()
                for d_param in delta.values():
                    if d_param.grad is not None:
                        d_param.grad.zero_()
                loss.backward()
                
                # Parameter sensitivity preconditioning on delta[name]
                with torch.no_grad():
                    for name in delta.keys():
                        if delta[name].grad is not None:
                            # Sensitivity estimate (on-the-fly gradient magnitude)
                            F_j = torch.mean(delta[name].grad**2).item()
                            # Preconditioning update
                            delta[name].copy_(delta[name] - 0.05 * (delta[name].grad / (F_j + eps_stab_trap)))
                optimizer.step()
            lam_val = torch.sigmoid(w_global).item()
            
        elif method_name == "SAM-TTMM":
            # Sharpness-Aware Test-Time Model Merging with standard entropy loss
            eps_stab = 1e-4 # raise stability floor to avoid trap
            
            # Uncertainty-Gated Selective Adaptation (UGSA)
            with torch.no_grad():
                init_merged_model = SimpleCNN().to(device)
                lambdas_init = {name: torch.tensor(w1, device=device) for name, _ in init_merged_model.named_parameters()}
                merge_model_weights(model0, model1, init_merged_model, lambdas_init)
                fuse_bn_buffers(model0, model1, init_merged_model, w1)
                init_out = init_merged_model(img)
                init_p = F.softmax(init_out, dim=1)
                init_entropy = -torch.mean(torch.sum(init_p * torch.log(init_p + 1e-8), dim=1)).item()
            
            should_adapt = init_entropy > 0.40
            if should_adapt:
                with torch.no_grad():
                    w_global.copy_(torch.clamp(w_global, -1.5, 1.5))
                
                merged_model.eval()
                for step in range(1): # SAM-TTMM works excellently with 1 step
                    # Fuse BN buffers in-place (non-differentiable)
                    fuse_bn_buffers(model0, model1, merged_model, torch.sigmoid(w_global).detach().item())
                    
                    # Construct merged parameters differentiably
                    merged_params = {}
                    for name, _ in merged_model.named_parameters():
                        lam = torch.sigmoid(w_global + delta[name])
                        param0 = dict(model0.named_parameters())[name]
                        param1 = dict(model1.named_parameters())[name]
                        merged_params[name] = (1.0 - lam) * param0 + lam * param1
                    
                    # Differentiable forward pass
                    out = functional_call(merged_model, merged_params, img)
                    loss = -torch.mean(torch.sum(F.softmax(out, dim=1) * F.log_softmax(out, dim=1), dim=1))
                    
                    # Compute gradients w.r.t w_global and delta_j
                    grads = torch.autograd.grad(loss, [w_global] + list(delta.values()), create_graph=False, allow_unused=True)
                    g_w = grads[0]
                    g_delta = {name: grads[i+1] for i, name in enumerate(delta.keys())}
                    
                    # Compute parameter sensitivities
                    sensitivities = {}
                    for name in delta.keys():
                        if g_delta[name] is not None:
                            s_j = torch.mean(g_delta[name]**2).item()
                            sensitivities[name] = s_j
                    
                    # Compute preconditioned direction vectors
                    d_w = g_w if g_w is not None else torch.zeros_like(w_global)
                    d_delta = {}
                    for name in delta.keys():
                        if g_delta[name] is not None:
                            d_delta[name] = g_delta[name] / (sensitivities[name] + eps_stab)
                        else:
                            d_delta[name] = torch.zeros_like(delta[name])
                            
                    # Combined direction vector norm
                    sum_sq_d = d_w**2
                    for name in delta.keys():
                        sum_sq_d = sum_sq_d + torch.sum(d_delta[name]**2)
                    d_norm = torch.sqrt(sum_sq_d + eps_stab)
                    
                    # Compute perturbations
                    eps_w = rho * d_w / d_norm
                    eps_delta = {}
                    for name in delta.keys():
                        eps_delta[name] = rho * d_delta[name] / d_norm
                        
                    # Apply perturbations
                    w_perturbed = w_global + eps_w
                    delta_perturbed = {}
                    for name in delta.keys():
                        delta_perturbed[name] = delta[name] + eps_delta[name]
                    
                    # Reconstruct perturbed parameters
                    fuse_bn_buffers(model0, model1, merged_model, torch.sigmoid(w_perturbed).detach().item())
                    merged_params_perturbed = {}
                    for name, _ in merged_model.named_parameters():
                        lam = torch.sigmoid(w_perturbed + delta_perturbed[name])
                        param0 = dict(model0.named_parameters())[name]
                        param1 = dict(model1.named_parameters())[name]
                        merged_params_perturbed[name] = (1.0 - lam) * param0 + lam * param1
                    
                    out_perturbed = functional_call(merged_model, merged_params_perturbed, img)
                    loss_perturbed = -torch.mean(torch.sum(F.softmax(out_perturbed, dim=1) * F.log_softmax(out_perturbed, dim=1), dim=1))
                    
                    # Compute perturbed gradients
                    grads_p = torch.autograd.grad(loss_perturbed, [w_global] + list(delta.values()), allow_unused=True)
                    g_w_p = grads_p[0]
                    g_delta_p = {name: grads_p[i+1] for i, name in enumerate(delta.keys())}
                    
                    # Parameter updates
                    with torch.no_grad():
                        if g_w_p is not None:
                            w_global.copy_(w_global - eta * g_w_p)
                        for name in delta.keys():
                            if g_delta_p[name] is not None:
                                delta[name].copy_(delta[name] - eta * g_delta_p[name] / (sensitivities[name] + eps_stab))
                            
            lam_val = torch.sigmoid(w_global).item()
            
        elif method_name == "KPSAM-DST (Ours)":
            # Kronecker-Preconditioned Sharpness-Aware Model Merging with Denoised Self-Training
            eps_stab = 1e-4 # raise stability floor
            
            # Adaptive Multi-Domain Thresholding (AMDT) based on sparsity
            best_threshold = 0.60 if h_sparsity_denoised >= 0.50 else 0.40
            img_denoised_optimal = denoise_image(img, threshold=best_threshold)
            
            # Generate soft target predictions on denoised batch using initial model
            initial_merged_model = SimpleCNN().to(device)
            lambdas_init = {name: torch.tensor(w1, device=device) for name, _ in initial_merged_model.named_parameters()}
            merge_model_weights(model0, model1, initial_merged_model, lambdas_init)
            fuse_bn_buffers(model0, model1, initial_merged_model, w1)
            
            initial_merged_model.eval()
            with torch.no_grad():
                denoised_logits = initial_merged_model(img_denoised_optimal)
                y_target = F.softmax(denoised_logits, dim=1)
                
                # Compute raw batch prediction entropy for selective adaptation gate (UGSA)
                init_out = initial_merged_model(img)
                init_p = F.softmax(init_out, dim=1)
                init_entropy = -torch.mean(torch.sum(init_p * torch.log(init_p + 1e-8), dim=1)).item()
            
            should_adapt = init_entropy > 0.40
            if should_adapt:
                with torch.no_grad():
                    w_global.copy_(torch.clamp(w_global, -1.5, 1.5))
                
                merged_model.eval()
                for step in range(5): # 5 steps for more complete adaptation using our stable self-training loss
                    fuse_bn_buffers(model0, model1, merged_model, torch.sigmoid(w_global).detach().item())
                    
                    merged_params = {}
                    for name, _ in merged_model.named_parameters():
                        lam = torch.sigmoid(w_global + delta[name])
                        param0 = dict(model0.named_parameters())[name]
                        param1 = dict(model1.named_parameters())[name]
                        merged_params[name] = (1.0 - lam) * param0 + lam * param1
                    
                    out = functional_call(merged_model, merged_params, img) # Feed NOISY input
                    p_out = F.softmax(out, dim=1)
                    
                    # KL Divergence Loss
                    loss_dst = torch.mean(torch.sum(y_target * torch.log((y_target + 1e-8) / (p_out + 1e-8)), dim=1))
                    # Direct Entropy Minimization to boost confidence
                    loss_entropy = -torch.mean(torch.sum(p_out * torch.log(p_out + 1e-8), dim=1))
                    loss_total = loss_dst + 0.2 * loss_entropy
                    
                    # Compute gradients w.r.t w_global and delta_j
                    grads = torch.autograd.grad(loss_total, [w_global] + list(delta.values()), create_graph=False, allow_unused=True)
                    g_w = grads[0]
                    g_delta = {name: grads[i+1] for i, name in enumerate(delta.keys())}
                    
                    # Compute parameter sensitivities
                    sensitivities = {}
                    for name in delta.keys():
                        if g_delta[name] is not None:
                            s_j = torch.mean(g_delta[name]**2).item()
                            sensitivities[name] = s_j
                    
                    # Compute preconditioned direction vectors
                    d_w = g_w if g_w is not None else torch.zeros_like(w_global)
                    d_delta = {}
                    for name in delta.keys():
                        if g_delta[name] is not None:
                            d_delta[name] = g_delta[name] / (sensitivities[name] + eps_stab)
                        else:
                            d_delta[name] = torch.zeros_like(delta[name])
                            
                    # Combined direction vector norm
                    sum_sq_d = d_w**2
                    for name in delta.keys():
                        sum_sq_d = sum_sq_d + torch.sum(d_delta[name]**2)
                    d_norm = torch.sqrt(sum_sq_d + eps_stab)
                    
                    # Compute perturbations
                    eps_w = rho * d_w / d_norm
                    eps_delta = {}
                    for name in delta.keys():
                        eps_delta[name] = rho * d_delta[name] / d_norm
                        
                    # Apply perturbations
                    w_perturbed = w_global + eps_w
                    delta_perturbed = {}
                    for name in delta.keys():
                        delta_perturbed[name] = delta[name] + eps_delta[name]
                    
                    # Reconstruct perturbed parameters
                    fuse_bn_buffers(model0, model1, merged_model, torch.sigmoid(w_perturbed).detach().item())
                    merged_params_perturbed = {}
                    for name, _ in merged_model.named_parameters():
                        lam = torch.sigmoid(w_perturbed + delta_perturbed[name])
                        param0 = dict(model0.named_parameters())[name]
                        param1 = dict(model1.named_parameters())[name]
                        merged_params_perturbed[name] = (1.0 - lam) * param0 + lam * param1
                    
                    out_perturbed = functional_call(merged_model, merged_params_perturbed, img)
                    p_out_p = F.softmax(out_perturbed, dim=1)
                    loss_dst_perturbed = torch.mean(torch.sum(y_target * torch.log((y_target + 1e-8) / (p_out_p + 1e-8)), dim=1))
                    loss_entropy_perturbed = -torch.mean(torch.sum(p_out_p * torch.log(p_out_p + 1e-8), dim=1))
                    loss_total_perturbed = loss_dst_perturbed + 0.2 * loss_entropy_perturbed
                    
                    # Compute perturbed gradients
                    grads_p = torch.autograd.grad(loss_total_perturbed, [w_global] + list(delta.values()), allow_unused=True)
                    g_w_p = grads_p[0]
                    g_delta_p = {name: grads_p[i+1] for i, name in enumerate(delta.keys())}
                    
                    # Parameter updates
                    with torch.no_grad():
                        if g_w_p is not None:
                            w_global.copy_(w_global - eta * g_w_p)
                        for name in delta.keys():
                            if g_delta_p[name] is not None:
                                delta[name].copy_(delta[name] - eta * g_delta_p[name] / (sensitivities[name] + eps_stab))
                            
            lam_val = torch.sigmoid(w_global).item()

        # Step 4: Final Inference Evaluation
        # Apply optimal adapted parameters to evaluate accuracy on this batch
        merged_model.eval()
        lambdas_dict_final = {name: torch.sigmoid(w_global + delta[name]) for name, _ in merged_model.named_parameters()}
        merge_model_weights(model0, model1, merged_model, lambdas_dict_final)
        fuse_bn_buffers(model0, model1, merged_model, lam_val)
        
        with torch.no_grad():
            final_out = merged_model(img)
            preds = torch.argmax(final_out, dim=1)
            correct = torch.sum(preds == label).item()
            acc = correct / label.size(0) * 100.0
            
        accuracies.append(acc)
        lambdas.append(lam_val)

    avg_acc = np.mean(accuracies)
    print(f"Method: {method_name} | Overall Avg Accuracy: {avg_acc:.2f}%")
    return accuracies, lambdas, avg_acc

# Main Execution Pipeline
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Get Datasets
    mnist_train, mnist_test, fmnist_train, fmnist_test, kmnist_test = get_datasets()
    
    # 2. Train or Load Experts
    if not os.path.exists("expert0.pth") or not os.path.exists("expert1.pth"):
        expert0, expert1 = train_experts(mnist_train, fmnist_train, device)
    else:
        print("Loading pre-trained experts...")
        expert0 = SimpleCNN().to(device)
        expert0.load_state_dict(torch.load("expert0.pth", map_location=device))
        expert1 = SimpleCNN().to(device)
        expert1.load_state_dict(torch.load("expert1.pth", map_location=device))
        
    # 3. Precompute Class Prototypes for SCTS
    print("Precomputing class prototypes...")
    prototypes0 = precompute_prototypes(expert0, mnist_train, device)
    prototypes1 = precompute_prototypes(expert1, fmnist_train, device)
    
    # 4. Generate Target Test Stream
    print("Generating non-stationary test stream...")
    stream = generate_test_stream(mnist_test, fmnist_test, kmnist_test)
    
    # 5. Evaluate Methods
    results = {}
    
    # Baseline 1: Static Merge
    acc_static, lam_static, avg_static = evaluate_method(
        "Static", stream, expert0, expert1, prototypes0, prototypes1, device
    )
    results["Static"] = (acc_static, lam_static, avg_static)
    
    # Baseline 2: Fixed TTA
    acc_fixed, lam_fixed, avg_fixed = evaluate_method(
        "Fixed TTA", stream, expert0, expert1, prototypes0, prototypes1, device
    )
    results["Fixed TTA"] = (acc_fixed, lam_fixed, avg_fixed)
    
    # Baseline 3: BK-CoMerge (vulnerable to stability trap)
    acc_bk, lam_bk, avg_bk = evaluate_method(
        "BK-CoMerge", stream, expert0, expert1, prototypes0, prototypes1, device
    )
    results["BK-CoMerge"] = (acc_bk, lam_bk, avg_bk)
    
    # Baseline 4: SAM-TTMM (Entropy-based)
    acc_sam, lam_sam, avg_sam = evaluate_method(
        "SAM-TTMM", stream, expert0, expert1, prototypes0, prototypes1, device, rho=0.05, eta=0.005
    )
    results["SAM-TTMM"] = (acc_sam, lam_sam, avg_sam)
    
    # Our Method: KPSAM-DST
    acc_ours, lam_ours, avg_ours = evaluate_method(
        "KPSAM-DST (Ours)", stream, expert0, expert1, prototypes0, prototypes1, device, rho=0.03, eta=0.07
    )
    results["KPSAM-DST (Ours)"] = (acc_ours, lam_ours, avg_ours)
    
    # 6. Print Summary Table
    print("\n" + "="*50)
    print("SUMMARY OF EXPERIMENTAL RESULTS")
    print("="*50)
    print(f"{'Method':<20} | {'Overall Avg Accuracy':<20}")
    print("-"*50)
    for method, (_, _, avg_acc) in results.items():
        print(f"{method:<20} | {avg_acc:.2f}%")
    print("="*50)
    
    # 7. Generate and Save Plots
    print("Generating and saving plots...")
    
    # Plot 1: Accuracy Tracking
    plt.figure(figsize=(10, 5))
    for method, (accs, _, _) in results.items():
        plt.plot(accs, label=method, alpha=0.85)
    plt.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=30, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=40, color='gray', linestyle='--', alpha=0.5)
    plt.text(5, 105, "Clean\nMNIST", ha='center', fontsize=8)
    plt.text(15, 105, "Noisy\nMNIST", ha='center', fontsize=8)
    plt.text(25, 105, "Clean\nFMNIST", ha='center', fontsize=8)
    plt.text(35, 105, "Noisy\nFMNIST", ha='center', fontsize=8)
    plt.text(45, 105, "Clean\nKMNIST", ha='center', fontsize=8)
    plt.xlabel("Batch Index")
    plt.ylabel("Accuracy (%)")
    plt.title("Test-Time Model Merging Accuracy Tracking")
    plt.ylim(-5, 115)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("accuracy_tracking.png", dpi=300)
    plt.close()
    
    # Plot 2: Merging Coefficient (Lambda) Tracking
    plt.figure(figsize=(10, 5))
    for method, (_, lams, _) in results.items():
        if method in ["Static", "KPSAM-DST (Ours)", "SAM-TTMM", "BK-CoMerge"]:
            plt.plot(lams, label=f"Lambda ({method})", alpha=0.85)
            
    # Calculate and plot Hoyer Sparsity for stream inputs
    stream_hoyers = [hoyer_sparsity(img) for img, _, _ in stream]
    plt.plot(stream_hoyers, label="Hoyer Sparsity", color='black', linestyle=':', alpha=0.7)
    
    plt.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=30, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=40, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Batch Index")
    plt.ylabel("Value")
    plt.title("Merging Coefficient & Hoyer Sparsity Tracking")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("coefficient_tracking.png", dpi=300)
    plt.close()
    
    # Save raw results to a txt file
    with open("experimental_results.txt", "w") as f:
        f.write("SUMMARY OF EXPERIMENTAL RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"{'Method':<25} | {'Overall Avg Accuracy':<20}\n")
        f.write("-"*50 + "\n")
        for method, (_, _, avg_acc) in results.items():
            f.write(f"{method:<25} | {avg_acc:.2f}%\n")
        f.write("="*50 + "\n\n")
        
        # Segment-wise analysis
        f.write("SEGMENT-WISE AVERAGE ACCURACY ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write(f"{'Method':<20} | {'Clean MNIST':<12} | {'Noisy MNIST':<12} | {'Clean FMNIST':<12} | {'Noisy FMNIST':<12} | {'KMNIST (OOD)':<12}\n")
        f.write("-"*80 + "\n")
        for method, (accs, _, _) in results.items():
            acc_c_mnist = np.mean(accs[0:10])
            acc_n_mnist = np.mean(accs[10:20])
            acc_c_fmnist = np.mean(accs[20:30])
            acc_n_fmnist = np.mean(accs[30:40])
            acc_kmnist = np.mean(accs[40:50])
            f.write(f"{method:<20} | {acc_c_mnist:>10.2f}% | {acc_n_mnist:>10.2f}% | {acc_c_fmnist:>10.2f}% | {acc_n_fmnist:>10.2f}% | {acc_kmnist:>10.2f}%\n")
        f.write("="*80 + "\n")
        
    print("Experiments completed successfully!")

if __name__ == "__main__":
    main()
