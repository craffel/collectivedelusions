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

# --- MODEL ARCHITECTURE: RESNET-9 ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + x)

class ResNet9(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.layer1_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer1_res = ResidualBlock(128)
        
        self.layer2_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.layer3_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3_res = ResidualBlock(512)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x, return_features=False):
        out = self.prep(x)
        out = self.layer1_conv(out)
        out = self.layer1_res(out)
        out = self.layer2_conv(out)
        out = self.layer3_conv(out)
        out = self.layer3_res(out)
        out = self.pool(out)
        features = torch.flatten(out, 1)
        logits = self.fc(features)
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
            # Optimal hyperparameters from sweep: multiplier = 0.015, max_alpha = 0.25
            alpha = torch.clamp(0.015 * kl_val, min=0.01, max=0.25).item()
            
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

# Replace standard BN with AdaKL-BN in a model recursively
def convert_to_adakl_bn(model):
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, name, AdaKLBatchNorm2D(child))
        else:
            convert_to_adakl_bn(child)

# --- EXPERT TRAINING FUNCTION ---
def train_expert(model, loader, device, epochs=2):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f}")
    return model

# --- PRECOMPUTE PROTOTYPES ---
def precompute_prototypes(expert_model, loader, device, num_samples=256):
    expert_model.eval()
    features_list = []
    labels_list = []
    samples_collected = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, feats = expert_model(x, return_features=True)
            features_list.append(feats.cpu())
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
            feat_dim = features.size(1)
            prototypes[c] = torch.zeros(feat_dim)
    return prototypes

# --- PROTOTYPE ROUTING & SCTS ---
def compute_expert_distances(x, base_model, expert0_state, expert1_state, prototypes0, prototypes1, device):
    base_model.eval()
    B = x.size(0)
    
    # Expert 0
    base_model.load_state_dict(expert0_state, strict=False)
    with torch.no_grad():
        _, feats0 = base_model(x.to(device), return_features=True)
    feats0 = feats0.cpu()
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
        _, feats1 = base_model(x.to(device), return_features=True)
    feats1 = feats1.cpu()
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

# --- GENERIC LAYER GROUPING HELPER FOR ADAPTIVE MERGING ---
def get_layer_groups(model):
    groups = []
    param_to_group_idx = {}
    current_group_idx = -1
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            groups.append(name)
            current_group_idx = len(groups) - 1
            for p_name, _ in module.named_parameters():
                full_p_name = f"{name}.{p_name}" if name else p_name
                param_to_group_idx[full_p_name] = current_group_idx
        elif isinstance(module, nn.BatchNorm2d):
            if current_group_idx != -1:
                for p_name, _ in module.named_parameters():
                    full_p_name = f"{name}.{p_name}" if name else p_name
                    param_to_group_idx[full_p_name] = current_group_idx
    return groups, param_to_group_idx

# --- WEIGHT MERGING HELPERS ---
def merge_weights(base_model, expert0_state, expert1_state, lambda0, lambda1):
    merged_state = {}
    for k in expert0_state.keys():
        if 'running' in k or 'num_batches_tracked' in k:
            continue  # BN buffers handled separately
        merged_state[k] = lambda0 * expert0_state[k] + lambda1 * expert1_state[k]
    base_model.load_state_dict(merged_state, strict=False)

def merge_layer_weights_adaptive(base_model, expert0_state, expert1_state, lambdas, param_to_group_idx):
    merged_state = {}
    for k in expert0_state.keys():
        if 'running' in k or 'num_batches_tracked' in k:
            continue
        if k in param_to_group_idx:
            g_idx = param_to_group_idx[k]
            lam = lambdas[g_idx]
            merged_state[k] = lam * expert0_state[k] + (1.0 - lam) * expert1_state[k]
        else:
            merged_state[k] = 0.5 * expert0_state[k] + 0.5 * expert1_state[k]
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

def compute_sensitivities_generic(model, groups):
    sensitivities = {}
    for name in groups:
        module = dict(model.named_modules())[name]
        a = activations.get(name)
        g = gradients.get(name)
        if a is None or g is None:
            sensitivities[name] = 1.0
            continue
        
        if isinstance(module, nn.Conv2d):
            c_in = module.in_channels
            c_out = module.out_channels
            k_h, k_w = module.kernel_size
            l2_a = torch.sum(a**2, dim=(1, 2, 3)).mean().item()
            l2_g = torch.sum(g**2, dim=(1, 2, 3)).mean().item()
            h_in, w_in = a.shape[2], a.shape[3]
            h_out, w_out = g.shape[2], g.shape[3]
            A_bar = l2_a / (h_in * w_in)
            G_bar = l2_g / (h_out * w_out)
            sensitivities[name] = (G_bar * A_bar) / (c_out * c_in * k_h * k_w)
        elif isinstance(module, nn.Linear):
            d_in = module.in_features
            d_out = module.out_features
            l2_a = torch.sum(a**2, dim=1).mean().item()
            l2_g = torch.sum(g**2, dim=1).mean().item()
            sensitivities[name] = (l2_a * l2_g) / (d_out * d_in)
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
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    return entropy.mean()

def main():
    print("--- STARTING RESNET-9 SCALABILITY EXPERIMENTS ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup directories
    os.makedirs("./data", exist_ok=True)
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading datasets...")
    mnist_train_full = datasets.MNIST("./data", train=True, download=True, transform=transform)
    mnist_train = DataLoader(Subset(mnist_train_full, range(10000)), batch_size=64, shuffle=True)
    
    fmnist_train_full = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    fmnist_train = DataLoader(Subset(fmnist_train_full, range(10000)), batch_size=64, shuffle=True)
    
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST("./data", train=False, download=True, transform=transform)
    
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    fmnist_loader = DataLoader(fmnist_test, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=False)
    
    # Train/load ResNet-9 Expert 0 (MNIST)
    expert0 = ResNet9().to(device)
    if os.path.exists("mnist_resnet_expert.pth"):
        print("Loading ResNet-9 Expert 0 (MNIST) from checkpoint...")
        expert0.load_state_dict(torch.load("mnist_resnet_expert.pth", map_location=device))
    else:
        print("Training ResNet-9 Expert 0 (MNIST)...")
        expert0 = train_expert(expert0, mnist_train, device, epochs=2)
        torch.save(expert0.state_dict(), "mnist_resnet_expert.pth")
    
    # Train/load ResNet-9 Expert 1 (FashionMNIST)
    expert1 = ResNet9().to(device)
    if os.path.exists("fashion_resnet_expert.pth"):
        print("Loading ResNet-9 Expert 1 (FashionMNIST) from checkpoint...")
        expert1.load_state_dict(torch.load("fashion_resnet_expert.pth", map_location=device))
    else:
        print("Training ResNet-9 Expert 1 (FashionMNIST)...")
        expert1 = train_expert(expert1, fmnist_train, device, epochs=2)
        torch.save(expert1.state_dict(), "fashion_resnet_expert.pth")
    
    # Load state dicts
    expert0_state = torch.load("mnist_resnet_expert.pth", map_location="cpu")
    expert1_state = torch.load("fashion_resnet_expert.pth", map_location="cpu")
    
    # Setup Base Model with AdaKL-BN
    base_model = ResNet9().to(device)
    convert_to_adakl_bn(base_model)
    
    # Load BN expert stats from models
    for name, m in base_model.named_modules():
        if isinstance(m, AdaKLBatchNorm2D):
            m0_mean = expert0_state[name + '.running_mean']
            m0_var = expert0_state[name + '.running_var']
            m1_mean = expert1_state[name + '.running_mean']
            m1_var = expert1_state[name + '.running_var']
            m.set_expert_stats(m0_mean, m0_var, m1_mean, m1_var)
            
    print("Precomputing class-wise prototypes...")
    prototypes0 = precompute_prototypes(expert0, mnist_loader, device, num_samples=256)
    prototypes1 = precompute_prototypes(expert1, fmnist_loader, device, num_samples=256)
    
    # Get Layer Groups for the architecture
    groups, param_to_group_idx = get_layer_groups(base_model)
    print(f"Identified {len(groups)} trainable layer groups for ResNet-9 merging.")
    
    # --- CONSTRUCT NON-STATIONARY TEST STREAM ---
    print("Constructing non-stationary test stream...")
    stream_batches = []
    
    # Phase 0: Clean MNIST (10 batches)
    mnist_iter = iter(mnist_loader)
    for _ in range(10):
        x, y = next(mnist_iter)
        stream_batches.append((x, y, "Clean MNIST"))
        
    # Phase 1: Noisy MNIST (10 batches)
    for _ in range(10):
        x, y = next(mnist_iter)
        x_raw = x * 0.3081 + 0.1307
        noise = torch.randn_like(x_raw) * 0.15
        x_raw_noisy = torch.clamp(x_raw + noise, 0.0, 1.0)
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
        x_raw = x * 0.3081 + 0.1307
        noise = torch.randn_like(x_raw) * 0.15
        x_raw_noisy = torch.clamp(x_raw + noise, 0.0, 1.0)
        x_noisy = (x_raw_noisy - 0.1307) / 0.3081
        stream_batches.append((x_noisy, y, "Noisy Fashion"))        

    # Phase 4: Novel KMNIST (10 batches)
    kmnist_iter = iter(kmnist_loader)
    for _ in range(10):
        x, y = next(kmnist_iter)
        stream_batches.append((x, y, "Novel KMNIST"))
        
    print(f"Stream constructed with {len(stream_batches)} batches.")
    
    # --- DEFINE EVALUATION METHODS ---
    methods = ["Static Merging", "Fixed TTA", "Soft BN Fusion", "KT-Fisher", "AdaKL-BN (Ours)"]
    results = {m: [] for m in methods}
    
    for method in methods:
        print(f"\nEvaluating Method: {method}")
        correct_by_phase = {p: 0 for p in ["Clean MNIST", "Noisy MNIST", "Clean Fashion", "Noisy Fashion", "Novel KMNIST"]}
        total_by_phase = {p: 0 for p in ["Clean MNIST", "Noisy MNIST", "Clean Fashion", "Noisy Fashion", "Novel KMNIST"]}
        
        for b_idx, (x, y, phase) in enumerate(stream_batches):
            x, y = x.to(device), y.to(device)
            # Reset AdaKL-BN configuration
            for m in base_model.modules():
                if isinstance(m, AdaKLBatchNorm2D):
                    m.use_adakl = False
                    m.w = None
                    
            D0, D1 = compute_expert_distances(x.cpu(), base_model, expert0_state, expert1_state, prototypes0, prototypes1, device)
            delta = abs(D0 - D1)

            s = 5.0
            eps_stab = 0.01
            tau_self = (delta / s) + eps_stab

            w0 = 1.0 / (1.0 + np.exp((D0 - D1) / tau_self))
            w1 = 1.0 - w0
            w_prior = torch.tensor([w0, w1], device=device)
            
            # Novelty detection
            base_model.load_state_dict(expert0_state, strict=False)
            with torch.no_grad():
                H0 = compute_entropy(base_model(x)).item()
            base_model.load_state_dict(expert1_state, strict=False)
            with torch.no_grad():
                H1 = compute_entropy(base_model(x)).item()
                
            H_bar = 0.5 * (H0 + H1)
            is_novel = H_bar > 1.2
            
            if is_novel:
                w0, w1 = 0.5, 0.5
                w_prior = torch.tensor([0.5, 0.5], device=device)
                
            if method == "Static Merging":
                merge_weights(base_model, expert0_state, expert1_state, 0.5, 0.5)
                apply_soft_bn_stats(base_model, 0.5, 0.5)
                
            elif method == "Soft BN Fusion":
                merge_weights(base_model, expert0_state, expert1_state, w0, w1)
                apply_soft_bn_stats(base_model, w0, w1)
                
            else:
                p_init = np.clip(w0, 1e-4, 1.0 - 1e-4)
                # Initialize merging logs for all groups
                w_param = torch.tensor([np.log(p_init / (1.0 - p_init))] * len(groups), requires_grad=True, device=device)
                
                for m in base_model.modules():
                    if isinstance(m, AdaKLBatchNorm2D):
                        if method == "AdaKL-BN (Ours)":
                            m.use_adakl = True
                            m.w = w_prior.cpu()
                        else:
                            m.use_adakl = False
                            m.w = None
                            apply_soft_bn_stats(base_model, w0, w1)
                
                # sensitivities computation for KT-Fisher or AdaKL-BN
                sensitivities = {}
                if method in ["KT-Fisher", "AdaKL-BN (Ours)"]:
                    merge_weights(base_model, expert0_state, expert1_state, w0, w1)
                    
                    hooks = []
                    for g_name in groups:
                        mod = dict(base_model.named_modules())[g_name]
                        hooks.append(mod.register_forward_hook(get_forward_hook(g_name)))
                        hooks.append(mod.register_full_backward_hook(get_backward_hook(g_name)))
                    
                    # Dummy pass to get activations & gradients
                    logits = base_model(x)
                    loss_ent = compute_entropy(logits)
                    base_model.zero_grad()
                    loss_ent.backward()
                    
                    sensitivities = compute_sensitivities_generic(base_model, groups)
                    
                    # Normalize sensitivities
                    sens_vals = list(sensitivities.values())
                    if len(sens_vals) > 0 and sum(sens_vals) > 0:
                        mean_sens = sum(sens_vals) / len(sens_vals)
                        for k in sensitivities:
                            sensitivities[k] /= mean_sens
                    
                    for h in hooks:
                        h.remove()
                        
                eta = 0.1 if method == "Fixed TTA" else 0.05
                beta_reg = 1.5
                for step in range(5):
                    lambdas = torch.sigmoid(w_param)
                    merge_layer_weights_adaptive(base_model, expert0_state, expert1_state, lambdas.cpu(), param_to_group_idx)
                    
                    logits = base_model(x)
                    loss_ent = compute_entropy(logits)
                    
                    loss_kl = 0.0
                    for l_idx in range(len(groups)):
                        lam = lambdas[l_idx]
                        loss_kl += w_prior[0] * torch.log(w_prior[0] / (lam + 1e-10) + 1e-10) + \
                                  w_prior[1] * torch.log(w_prior[1] / (1.0 - lam + 1e-10) + 1e-10)
                                  
                    loss_total = loss_ent + beta_reg * loss_kl
                    
                    if w_param.grad is not None:
                        w_param.grad.zero_()
                    loss_total.backward()

                    with torch.no_grad():
                        grad = w_param.grad.clone()
                        if method in ["KT-Fisher", "AdaKL-BN (Ours)"]:
                            scale_eps = 1e-5
                            for l_idx, g_name in enumerate(groups):
                                sens = sensitivities.get(g_name, 1.0)
                                grad[l_idx] /= (sens + scale_eps)
                            grad = torch.clamp(grad, -1.0, 1.0)
                                
                        w_param -= eta * grad
                        
                with torch.no_grad():
                    lambdas = torch.sigmoid(w_param)
                    merge_layer_weights_adaptive(base_model, expert0_state, expert1_state, lambdas.cpu(), param_to_group_idx)
                    
            base_model.eval()
            with torch.no_grad():
                logits = base_model(x)
                preds = logits.argmax(dim=1)
                correct = (preds == y).sum().item()
                
            if b_idx % 10 < 2:
                print(f"  [DEBUG {method}] Batch {b_idx} ({phase}): batch_acc={correct / x.size(0) * 100:.1f}%")
                
            correct_by_phase[phase] += correct
            total_by_phase[phase] += x.size(0)
            
        method_accs = []
        print(f"Results for {method} with ResNet-9:")
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
    plt.title("ResNet-9 Test-Time Model Merging Performance Under Open-World Shifts")
    plt.xticks(x_indices, phases)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("results_resnet.png")
    print("\nResNet Chart saved as results_resnet.png.")
    
    # --- WRITE RESULTS TO LATEX TABLE ---
    print("\n--- Summary Table for ResNet-9 (LaTeX format) ---")
    header = " & ".join(["Method"] + phases) + " \\\\"
    print(header)
    for method in methods:
        row = f"{method} & " + " & ".join([f"{val:.2f}\\%" for val in results[method]]) + " \\\\"
        print(row)
        
    print("\n--- RESNET-9 EXPERIMENTATION COMPLETE ---")

if __name__ == "__main__":
    main()
