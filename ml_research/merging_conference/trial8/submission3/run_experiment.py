import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- MODEL ARCHITECTURE ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        features = self.fc1(x)
        x = torch.relu(features)
        x = self.dropout2(x)
        logits = self.fc2(x)
        if return_features:
            return logits, features
        return logits

# --- CUSTOM BATCH NORM FOR ADAPTIVE KL-DRIVEN DIAGONAL BLENDING (AdaKL-BN) ---
class AdaKLBatchNorm2D(nn.Module):
    def __init__(self, bn_module):
        super().__init__()
        self.num_features = bn_module.num_features
        self.eps = bn_module.eps
        self.momentum = bn_module.momentum
        self.affine = bn_module.affine
        if self.affine:
            self.weight = nn.Parameter(bn_module.weight.clone())
            self.bias = nn.Parameter(bn_module.bias.clone())
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.register_buffer('running_mean', bn_module.running_mean.clone())
        self.register_buffer('running_var', bn_module.running_var.clone())
        
        # Expert buffers
        self.register_buffer('expert0_mean', bn_module.running_mean.clone())
        self.register_buffer('expert0_var', bn_module.running_var.clone())
        self.register_buffer('expert1_mean', bn_module.running_mean.clone())
        self.register_buffer('expert1_var', bn_module.running_var.clone())
        
        self.use_adakl = False
        self.w = None  # [w0, w1] posterior weights
        
    def set_expert_stats(self, mean0, var0, mean1, var1):
        self.expert0_mean.copy_(mean0)
        self.expert0_var.copy_(var0)
        self.expert1_mean.copy_(mean1)
        self.expert1_var.copy_(var1)
        
    def forward(self, x):
        if self.use_adakl and self.w is not None:
            w0, w1 = self.w[0], self.w[1]
            # Compute soft fused statistics
            mu_fused = w0 * self.expert0_mean + w1 * self.expert1_mean
            var_fused = w0 * (self.expert0_var + (self.expert0_mean - mu_fused)**2) + \
                        w1 * (self.expert1_var + (self.expert1_mean - mu_fused)**2)
            
            # Compute current batch statistics (on-the-fly)
            batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            
            # Reshape for broadcasting
            mu_fused_bc = mu_fused.view(1, -1, 1, 1)
            var_fused_bc = var_fused.view(1, -1, 1, 1)
            
            # Compute KL divergence to measure domain shift / out-of-distribution severity
            eps_stab = 1e-5
            bm = batch_mean.view(-1)
            bv = batch_var.view(-1)
            fm = mu_fused.view(-1)
            fv = var_fused.view(-1)
            kl = 0.5 * (torch.log((fv + eps_stab) / (bv + eps_stab)) + (bv + (bm - fm)**2) / (fv + eps_stab) - 1.0)
            kl_val = kl.mean().clamp(min=0.0, max=10.0)
            
            # Adaptive alpha-adaptation blending (A3-AdaKL-BN)
            # Modulate alpha dynamically based on the KL divergence
            alpha = torch.clamp(0.005 * kl_val, min=0.01, max=0.20).item()
            
            mean_adapted = alpha * batch_mean + (1.0 - alpha) * mu_fused_bc
            var_adapted = alpha * batch_var + (1.0 - alpha) * var_fused_bc
            
            x_scaled = (x - mean_adapted) / torch.sqrt(var_adapted + self.eps)
            
            if self.affine:
                w_bc = self.weight.view(1, -1, 1, 1)
                b_bc = self.bias.view(1, -1, 1, 1)
                return x_scaled * w_bc + b_bc
            else:
                return x_scaled
        else:
            # Standard BN evaluation
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                w_bc = self.weight.view(1, -1, 1, 1)
                b_bc = self.bias.view(1, -1, 1, 1)
                return x_norm * w_bc + b_bc
            else:
                return x_norm

# Replace standard BN with AdaKL-BN in a model
def convert_to_adakl_bn(model):
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, name, AdaKLBatchNorm2D(child))
        else:
            convert_to_adakl_bn(child)

# --- EXPERT TRAINING FUNCTION ---
def train_expert(model, loader, epochs=2):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/total:.4f} | Acc: {correct/total*100:.2f}%")
    return model

# --- PROTOTYPE PRECOMPUTATION ---
def precompute_prototypes(expert_model, loader, num_samples=256):
    expert_model.eval()
    features_list = []
    labels_list = []
    samples_collected = 0
    with torch.no_grad():
        for x, y in loader:
            _, feats = expert_model(x, return_features=True)
            features_list.append(feats)
            labels_list.append(y)
            samples_collected += x.size(0)
            if samples_collected >= num_samples:
                break
    features = torch.cat(features_list, dim=0)[:num_samples]
    labels = torch.cat(labels_list, dim=0)[:num_samples]
    
    prototypes = {}
    for c in range(10):
        mask = (labels == c)
        if mask.sum() > 0:
            prototypes[c] = features[mask].mean(dim=0)
        else:
            prototypes[c] = torch.zeros(128)
    return prototypes

# --- PROTOTYPE ROUTING & SCTS ---
def compute_expert_distances(x, base_model, expert0_state, expert1_state, prototypes0, prototypes1):
    base_model.eval()
    B = x.size(0)
    
    # Expert 0
    base_model.load_state_dict(expert0_state, strict=False)
    with torch.no_grad():
        _, feats0 = base_model(x, return_features=True)
    # L2-normalize features and prototypes for scale invariance
    feats0_norm = feats0 / (torch.norm(feats0, p=2, dim=1, keepdim=True) + 1e-8)
    prototypes0_norm = {c: p / (torch.norm(p, p=2) + 1e-8) for c, p in prototypes0.items()}
    
    dists0 = []
    for i in range(B):
        f = feats0_norm[i]
        min_d = min([torch.sum((f - prototypes0_norm[c])**2).item() for c in range(10)])
        dists0.append(min_d)
    D0 = sum(dists0) / B
    
    # Expert 1
    base_model.load_state_dict(expert1_state, strict=False)
    with torch.no_grad():
        _, feats1 = base_model(x, return_features=True)
    # L2-normalize features and prototypes for scale invariance
    feats1_norm = feats1 / (torch.norm(feats1, p=2, dim=1, keepdim=True) + 1e-8)
    prototypes1_norm = {c: p / (torch.norm(p, p=2) + 1e-8) for c, p in prototypes1.items()}
    
    dists1 = []
    for i in range(B):
        f = feats1_norm[i]
        min_d = min([torch.sum((f - prototypes1_norm[c])**2).item() for c in range(10)])
        dists1.append(min_d)
    D1 = sum(dists1) / B
    
    return D0, D1

# --- WEIGHT MERGING HELPER ---
def merge_weights(base_model, expert0_state, expert1_state, lambda0, lambda1):
    merged_state = {}
    for k in expert0_state.keys():
        if 'running' in k or 'num_batches_tracked' in k:
            continue  # BN buffers handled separately
        merged_state[k] = lambda0 * expert0_state[k] + lambda1 * expert1_state[k]
    base_model.load_state_dict(merged_state, strict=False)

def merge_layer_weights_adaptive(base_model, expert0_state, expert1_state, lambdas):
    merged_state = {}
    layer_names = ['conv1', 'conv2', 'fc1', 'fc2']
    for l_idx, l_name in enumerate(layer_names):
        lam = lambdas[l_idx]
        for k in expert0_state.keys():
            if k.startswith(l_name + '.'):
                if 'running' in k or 'num_batches_tracked' in k:
                    continue
                merged_state[k] = lam * expert0_state[k] + (1.0 - lam) * expert1_state[k]
        bn_name = 'bn1' if l_name == 'conv1' else ('bn2' if l_name == 'conv2' else None)
        if bn_name is not None:
            for k in expert0_state.keys():
                if k.startswith(bn_name + '.'):
                    if 'running' in k or 'num_batches_tracked' in k:
                        continue
                    merged_state[k] = lam * expert0_state[k] + (1.0 - lam) * expert1_state[k]
    base_model.load_state_dict(merged_state, strict=False)

# --- HOOKS FOR SENSITIVITY TRACKING (KT-FISHER) ---
activations = {}
gradients = {}

def get_forward_hook(name):
    def hook(module, input, output):
        activations[name] = input[0].detach().clone()
    return hook

def get_backward_hook(name):
    def hook(module, grad_input, grad_output):
        if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
            gradients[name] = grad_output[0].detach().clone()
    return hook

def compute_layer_sensitivities(model):
    sensitivities = {}
    
    # Conv1
    a = activations['conv1']
    g = gradients['conv1']
    c_in = model.conv1.in_channels
    c_out = model.conv1.out_channels
    k_h, k_w = model.conv1.kernel_size
    l2_a = torch.sum(a**2, dim=(1, 2, 3)).mean().item()
    l2_g = torch.sum(g**2, dim=(1, 2, 3)).mean().item()
    h_in, w_in = a.shape[2], a.shape[3]
    h_out, w_out = g.shape[2], g.shape[3]
    A_bar = l2_a / (h_in * w_in)
    G_bar = l2_g / (h_out * w_out)
    sensitivities['conv1'] = (G_bar * A_bar) / (c_out * c_in * k_h * k_w)
    
    # Conv2
    a = activations['conv2']
    g = gradients['conv2']
    c_in = model.conv2.in_channels
    c_out = model.conv2.out_channels
    k_h, k_w = model.conv2.kernel_size
    l2_a = torch.sum(a**2, dim=(1, 2, 3)).mean().item()
    l2_g = torch.sum(g**2, dim=(1, 2, 3)).mean().item()
    h_in, w_in = a.shape[2], a.shape[3]
    h_out, w_out = g.shape[2], g.shape[3]
    A_bar = l2_a / (h_in * w_in)
    G_bar = l2_g / (h_out * w_out)
    sensitivities['conv2'] = (G_bar * A_bar) / (c_out * c_in * k_h * k_w)
    
    # FC1
    a = activations['fc1']
    g = gradients['fc1']
    d_in = model.fc1.in_features
    d_out = model.fc1.out_features
    l2_a = torch.sum(a**2, dim=1).mean().item()
    l2_g = torch.sum(g**2, dim=1).mean().item()
    sensitivities['fc1'] = (l2_a * l2_g) / (d_out * d_in)
    
    # FC2
    a = activations['fc2']
    g = gradients['fc2']
    d_in = model.fc2.in_features
    d_out = model.fc2.out_features
    l2_a = torch.sum(a**2, dim=1).mean().item()
    l2_g = torch.sum(g**2, dim=1).mean().item()
    sensitivities['fc2'] = (l2_a * l2_g) / (d_out * d_in)
    
    return sensitivities

# --- SOFT BN STATS HELPER ---
def apply_soft_bn_stats(model, w0, w1):
    for m in model.modules():
        if isinstance(m, AdaKLBatchNorm2D):
            mu_fused = w0 * m.expert0_mean + w1 * m.expert1_mean
            var_fused = w0 * (m.expert0_var + (m.expert0_mean - mu_fused)**2) + \
                        w1 * (m.expert1_var + (m.expert1_mean - mu_fused)**2)
            m.running_mean.copy_(mu_fused)
            m.running_var.copy_(var_fused)

# --- PREDICTION ENTROPY ---
def compute_entropy(logits):
    probs = torch.softmax(logits, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()

# --- MAIN EXPERIMENTAL RUN ---
def main():
    print("--- STARTING COVARIANCE-AWARE TTMM EXPERIMENTS ---")
    
    # Setup directories
    os.makedirs("./data", exist_ok=True)
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading datasets...")
    # Train datasets (subset of 10,000 for fast training)
    mnist_train_full = datasets.MNIST("./data", train=True, download=True, transform=transform)
    mnist_train = DataLoader(Subset(mnist_train_full, range(10000)), batch_size=64, shuffle=True)
    
    fmnist_train_full = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    fmnist_train = DataLoader(Subset(fmnist_train_full, range(10000)), batch_size=64, shuffle=True)
    
    # Test loaders (batch size 64)
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST("./data", train=False, download=True, transform=transform)
    
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    fmnist_loader = DataLoader(fmnist_test, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=False)
    
    # Train/load Expert 0 (MNIST)
    expert0 = SimpleCNN()
    if os.path.exists("mnist_expert.pth"):
        print("Loading Expert 0 (MNIST) from checkpoint...")
        expert0.load_state_dict(torch.load("mnist_expert.pth", map_location="cpu"))
    else:
        print("Training Expert 0 (MNIST)...")
        expert0 = train_expert(expert0, mnist_train, epochs=2)
        torch.save(expert0.state_dict(), "mnist_expert.pth")
    
    # Train/load Expert 1 (FashionMNIST)
    expert1 = SimpleCNN()
    if os.path.exists("fashion_expert.pth"):
        print("Loading Expert 1 (FashionMNIST) from checkpoint...")
        expert1.load_state_dict(torch.load("fashion_expert.pth", map_location="cpu"))
    else:
        print("Training Expert 1 (FashionMNIST)...")
        expert1 = train_expert(expert1, fmnist_train, epochs=2)
        torch.save(expert1.state_dict(), "fashion_expert.pth")
    
    # Load state dicts
    expert0_state = torch.load("mnist_expert.pth")
    expert1_state = torch.load("fashion_expert.pth")
    
    # Setup Base Model with AdaKL-BN
    base_model = SimpleCNN()
    convert_to_adakl_bn(base_model)
    
    # Load BN expert stats from models
    for name, m in base_model.named_modules():
        if isinstance(m, AdaKLBatchNorm2D):
            # Extract standard BN statistics from expert checkpoints
            m0_mean = expert0_state[name + '.running_mean']
            m0_var = expert0_state[name + '.running_var']
            m1_mean = expert1_state[name + '.running_mean']
            m1_var = expert1_state[name + '.running_var']
            m.set_expert_stats(m0_mean, m0_var, m1_mean, m1_var)
            
    print("Precomputing class-wise prototypes...")
    prototypes0 = precompute_prototypes(expert0, mnist_loader, num_samples=256)
    prototypes1 = precompute_prototypes(expert1, fmnist_loader, num_samples=256)
    
    # --- CONSTRUCT NON-STATIONARY TEST STREAM ---
    print("Constructing non-stationary test stream...")
    # Extract batches
    stream_batches = []
    
    # Phase 0: Clean MNIST (10 batches)
    mnist_iter = iter(mnist_loader)
    for _ in range(10):
        x, y = next(mnist_iter)
        stream_batches.append((x, y, "Clean MNIST"))
        
    # Phase 1: Noisy MNIST (10 batches)
    for _ in range(10):
        x, y = next(mnist_iter)
        # Unnormalize (mean=0.1307, std=0.3081)
        x_raw = x * 0.3081 + 0.1307
        # Add noise
        noise = torch.randn_like(x_raw) * 0.15
        # Clamp to raw pixel bounds [0.0, 1.0]
        x_raw_noisy = torch.clamp(x_raw + noise, 0.0, 1.0)
        # Re-normalize
        x_noisy = (x_raw_noisy - 0.1307) / 0.3081
        stream_batches.append((x_noisy, y, "Noisy MNIST"))

    # Phase 2: Clean FashionMNIST (10 batches)
    fmnist_iter = iter(fmnist_loader)
    for _ in range(10):
        x, y = next(fmnist_iter)
        stream_batches.append((x, y, "Clean Fashion"))

    # Phase 3: Noisy FashionMNIST (10 batches)
    for _ in range(10):
        x, y = next(fmnist_iter)
        # Unnormalize (mean=0.1307, std=0.3081)
        x_raw = x * 0.3081 + 0.1307
        # Add noise
        noise = torch.randn_like(x_raw) * 0.15
        # Clamp to raw pixel bounds [0.0, 1.0]
        x_raw_noisy = torch.clamp(x_raw + noise, 0.0, 1.0)
        # Re-normalize
        x_noisy = (x_raw_noisy - 0.1307) / 0.3081
        stream_batches.append((x_noisy, y, "Noisy Fashion"))        
    # Phase 4: Novel KMNIST (10 batches)
    kmnist_iter = iter(kmnist_loader)
    for _ in range(10):
        x, y = next(kmnist_iter)
        stream_batches.append((x, y, "Novel KMNIST"))
        
    print(f"Stream constructed successfully with {len(stream_batches)} batches.")
    
    # --- DEFINE EVALUATION METHODS ---
    methods = ["Static Merging", "Fixed TTA", "Soft BN Fusion", "KT-Fisher", "AdaKL-BN (Ours)"]
    results = {m: [] for m in methods}
    
    # Evaluation loop
    for method in methods:
        print(f"\nEvaluating Method: {method}")
        correct_by_phase = {p: 0 for p in ["Clean MNIST", "Noisy MNIST", "Clean Fashion", "Noisy Fashion", "Novel KMNIST"]}
        total_by_phase = {p: 0 for p in ["Clean MNIST", "Noisy MNIST", "Clean Fashion", "Noisy Fashion", "Novel KMNIST"]}
        
        # Process each batch
        for b_idx, (x, y, phase) in enumerate(stream_batches):
            # Reset AdaKL-BN configuration for each batch before any forward pass (distances, entropy, etc.)
            for m in base_model.modules():
                if isinstance(m, AdaKLBatchNorm2D):
                    m.use_adakl = False
                    m.w = None
                    
            # 1. Compute prototype similarities & SCTS (with L2-normalization, s=5.0, eps_stab=0.01)
            D0, D1 = compute_expert_distances(x, base_model, expert0_state, expert1_state, prototypes0, prototypes1)
            delta = abs(D0 - D1)

            # SCTS Temperature
            s = 5.0
            eps_stab = 0.01
            tau_self = (delta / s) + eps_stab

            # Routing prior
            w0 = 1.0 / (1.0 + np.exp((D0 - D1) / tau_self))
            w1 = 1.0 - w0
            w_prior = torch.tensor([w0, w1])            
            # 2. Novelty detection using average expert entropy
            base_model.load_state_dict(expert0_state, strict=False)
            with torch.no_grad():
                H0 = compute_entropy(base_model(x)).item()
            base_model.load_state_dict(expert1_state, strict=False)
            with torch.no_grad():
                H1 = compute_entropy(base_model(x)).item()
                
            H_bar = 0.5 * (H0 + H1)
            is_novel = H_bar > 1.2
            
            # If novel, routing falls back to uniform prior
            if is_novel:
                w0, w1 = 0.5, 0.5
                w_prior = torch.tensor([0.5, 0.5])
                
            # 3. Method-specific configuration
            if method == "Static Merging":
                # Static uniform coefficients, no adaptation
                merge_weights(base_model, expert0_state, expert1_state, 0.5, 0.5)
                # Apply standard uniform BN statistics
                apply_soft_bn_stats(base_model, 0.5, 0.5)
                
            elif method == "Soft BN Fusion":
                # Use routing prior directly for weight and soft BN statistics
                merge_weights(base_model, expert0_state, expert1_state, w0, w1)
                apply_soft_bn_stats(base_model, w0, w1)
                
            else:
                # TTA methods: Fixed TTA, KT-Fisher, AdaKL-BN
                layer_names = ['conv1', 'conv2', 'fc1', 'fc2']
                # Initialize merging parameter logit using PG-Init
                # We clamp w0 to avoid log(0)
                p_init = np.clip(w0, 1e-4, 1.0 - 1e-4)
                w_param = torch.tensor([np.log(p_init / (1.0 - p_init))] * 4, requires_grad=True)
                
                # Setup custom BN mode
                for m in base_model.modules():
                    if isinstance(m, AdaKLBatchNorm2D):
                        if method == "AdaKL-BN (Ours)":
                            m.use_adakl = True
                            m.w = w_prior
                        else:
                            m.use_adakl = False
                            m.w = None
                            # Apply soft BN stats as baseline for standard TTA/KT-Fisher
                            apply_soft_bn_stats(base_model, w0, w1)
                
                # Precompute layer sensitivities on the first step
                sensitivities = {}
                if method in ["KT-Fisher", "AdaKL-BN (Ours)"]:
                    # Load initial merged weights into the model
                    merge_weights(base_model, expert0_state, expert1_state, w0, w1)
                    
                    # Register forward/backward hooks
                    hooks = []
                    hooks.append(base_model.conv1.register_forward_hook(get_forward_hook('conv1')))
                    hooks.append(base_model.conv1.register_full_backward_hook(get_backward_hook('conv1')))
                    hooks.append(base_model.conv2.register_forward_hook(get_forward_hook('conv2')))
                    hooks.append(base_model.conv2.register_full_backward_hook(get_backward_hook('conv2')))
                    hooks.append(base_model.fc1.register_forward_hook(get_forward_hook('fc1')))
                    hooks.append(base_model.fc1.register_full_backward_hook(get_backward_hook('fc1')))
                    hooks.append(base_model.fc2.register_forward_hook(get_forward_hook('fc2')))
                    hooks.append(base_model.fc2.register_full_backward_hook(get_backward_hook('fc2')))
                    
                    # Perform dummy forward/backward of entropy loss
                    logits = base_model(x)
                    loss_ent = compute_entropy(logits)
                    base_model.zero_grad()
                    loss_ent.backward()
                    
                    # Compute sensitivities
                    sensitivities = compute_layer_sensitivities(base_model)
                    
                    # Normalize sensitivities to prevent division by extremely small absolute values
                    sens_vals = list(sensitivities.values())
                    if len(sens_vals) > 0 and sum(sens_vals) > 0:
                        mean_sens = sum(sens_vals) / len(sens_vals)
                        for k in sensitivities:
                            sensitivities[k] /= mean_sens
                    
                    # Remove hooks
                    for h in hooks:
                        h.remove()
                        
                # Perform 5 TTA adaptation steps
                eta = 0.1 if method == "Fixed TTA" else 0.05
                beta_reg = 1.5
                for step in range(5):
                    # Load current merged weights
                    lambdas = torch.sigmoid(w_param)
                    
                    # Merge weights for each layer
                    merge_layer_weights_adaptive(base_model, expert0_state, expert1_state, lambdas)
                    
                    # Forward pass
                    logits = base_model(x)
                    loss_ent = compute_entropy(logits)
                    
                    # KL Regularization Loss
                    loss_kl = 0.0
                    for l_idx in range(4):
                        lam = lambdas[l_idx]
                        loss_kl += w_prior[0] * torch.log(w_prior[0] / (lam + 1e-10) + 1e-10) + \
                                  w_prior[1] * torch.log(w_prior[1] / (1.0 - lam + 1e-10) + 1e-10)
                                  
                    loss_total = loss_ent + beta_reg * loss_kl
                    
                    # Gradient step
                    if w_param.grad is not None:
                        w_param.grad.zero_()
                    loss_total.backward()

                    with torch.no_grad():
                        grad = w_param.grad.clone()
                        if method in ["KT-Fisher", "AdaKL-BN (Ours)"]:                            # Precondition gradients
                            scale_eps = 1e-5
                            for l_idx, l_name in enumerate(layer_names):
                                sens = sensitivities.get(l_name, 1.0)
                                grad[l_idx] /= (sens + scale_eps)
                            # Clip preconditioned gradients to prevent explosion due to vanishing backprop magnitudes
                            grad = torch.clamp(grad, -1.0, 1.0)
                                
                        # Update coefficients
                        w_param -= eta * grad
                        
                # Load final merged weights into the base model
                with torch.no_grad():
                    lambdas = torch.sigmoid(w_param)
                    merge_layer_weights_adaptive(base_model, expert0_state, expert1_state, lambdas)
                    
            # 4. Predict on current batch using adapted model
            base_model.eval()
            with torch.no_grad():
                logits = base_model(x)
                preds = logits.argmax(dim=1)
                correct = (preds == y).sum().item()
                
            if b_idx % 10 < 2:
                print(f"  [DEBUG {method}] Batch {b_idx} ({phase}): D0={D0:.1f}, D1={D1:.1f}, w0={w0:.3f}, w1={w1:.3f}, H_bar={H_bar:.3f}, is_novel={is_novel}, batch_acc={correct / x.size(0) * 100:.1f}%")
                
            correct_by_phase[phase] += correct
            total_by_phase[phase] += x.size(0)
            
        # Log accuracies for this method
        method_accs = []
        print(f"Results for {method}:")
        for phase in ["Clean MNIST", "Noisy MNIST", "Clean Fashion", "Noisy Fashion", "Novel KMNIST"]:
            acc = correct_by_phase[phase] / total_by_phase[phase] * 100
            print(f"  {phase}: {acc:.2f}%")
            method_accs.append(acc)
        results[method] = method_accs
        
    # --- PLOT COMPARISON ---
    phases = ["Clean MNIST", "Noisy MNIST", "Clean Fashion", "Noisy Fashion", "Novel KMNIST"]
    x_indices = np.arange(len(phases))
    width = 0.15
    
    plt.figure(figsize=(12, 7))
    for idx, method in enumerate(methods):
        plt.bar(x_indices + (idx - 2) * width, results[method], width, label=method)
        
    plt.xlabel("Test Stream Phase")
    plt.ylabel("Accuracy (%)")
    plt.title("Test-Time Model Merging Performance Under Open-World Shifts")
    plt.xticks(x_indices, phases)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("results.png")
    print("\nChart saved as results.png.")
    
    # --- WRITE RESULTS TO LATEX TABLE ---
    print("\n--- Summary Table (LaTeX format) ---")
    header = " & ".join(["Method"] + phases) + " \\\\"
    print(header)
    for method in methods:
        row = f"{method} & " + " & ".join([f"{val:.2f}\\%" for val in results[method]]) + " \\\\"
        print(row)
        
    print("\n--- EXPERIMENTATION PHASE COMPLETE ---")

if __name__ == "__main__":
    main()
