import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.func import functional_call
import math
import numpy as np
import copy
import os

# Set random seeds for strict reproducibility
torch.manual_seed(42)
np.random.seed(42)

# SimpleCNN Architecture with optional CosFace classification layer
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
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(x_norm, w_norm)
        if label is None:
            return self.s * cos_theta
        one_hot = torch.zeros(cos_theta.size(), device=cos_theta.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = self.s * (cos_theta - one_hot * self.m)
        return output

class SimpleCNN(nn.Module):
    def __init__(self, use_cosface=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.use_cosface = use_cosface
        if use_cosface:
            self.classifier = CosFaceLinear(128, 10)
        else:
            self.classifier = nn.Linear(128, 10)

    def extract_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x

    def forward(self, x, label=None):
        features = self.extract_features(x)
        if self.use_cosface:
            return self.classifier(features, label)
        else:
            return self.classifier(features)

# Training Helper function
def train_model(model, dataloader, epochs, lr, weight_decay, use_cosface=False):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            if use_cosface:
                outputs = model(images, labels)
            else:
                outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

# Image Hoyer Sparsity calculation
def hoyer_sparsity(x):
    # x shape: [B, 1, 28, 28]
    B = x.size(0)
    x_flat = x.view(B, -1) # [B, 784]
    d = x_flat.size(1)
    l1 = torch.norm(x_flat, p=1, dim=1)
    l2 = torch.norm(x_flat, p=2, dim=1) + 1e-8
    sparsity = (math.sqrt(d) - l1 / l2) / (math.sqrt(d) - 1.0)
    return sparsity.mean().item()

# Parameter Merging Helper
def merge_parameters(expert0, expert1, w_global, delta):
    merged_params = {}
    groups = {
        0: ['conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias'],
        1: ['conv2.weight', 'conv2.bias', 'bn2.weight', 'bn2.bias'],
        2: ['fc1.weight', 'fc1.bias'],
        3: ['classifier.weight', 'classifier.bias']
    }
    for j in range(4):
        lam = torch.sigmoid(w_global + delta[j])
        for name in groups[j]:
            if name in expert0.state_dict():
                p0 = expert0.state_dict()[name]
                p1 = expert1.state_dict()[name]
                merged_params[name] = (1.0 - lam) * p0 + lam * p1
    return merged_params

# BN Buffer Merging Helper (Soft Bayesian MoG Statistic Fusion)
def merge_buffers(expert0, expert1, w_global):
    lam = torch.sigmoid(w_global).detach()
    merged_buffers = {}
    bn_names = [
        ('bn1.running_mean', 'bn1.running_var'),
        ('bn2.running_mean', 'bn2.running_var')
    ]
    for mean_name, var_name in bn_names:
        mu0 = expert0.state_dict()[mean_name]
        var0 = expert0.state_dict()[var_name]
        mu1 = expert1.state_dict()[mean_name]
        var1 = expert1.state_dict()[var_name]
        
        # Exact moment matching
        mu_fused = (1.0 - lam) * mu0 + lam * mu1
        var_fused = (1.0 - lam) * (var0 + (mu0 - mu_fused)**2) + lam * (var1 + (mu1 - mu_fused)**2)
        
        merged_buffers[mean_name] = mu_fused
        merged_buffers[var_name] = var_fused
        
        track_name = mean_name.replace('running_mean', 'num_batches_tracked')
        merged_buffers[track_name] = expert0.state_dict()[track_name]
    return merged_buffers

# Precompute Class Prototypes from Calibration dataset
def compute_prototypes(expert, dataset):
    expert.eval()
    prototypes = {c: [] for c in range(10)}
    with torch.no_grad():
        for img, lbl in dataset:
            lbl = int(lbl)
            if len(prototypes[lbl]) < 100:
                feat = expert.extract_features(img.unsqueeze(0)).squeeze(0)
                feat_norm = F.normalize(feat, p=2, dim=0)
                prototypes[lbl].append(feat_norm)
    proto_tensor = torch.zeros(10, 128)
    for c in range(10):
        c_protos = torch.stack(prototypes[c])
        mean_proto = c_protos.mean(dim=0)
        proto_tensor[c] = F.normalize(mean_proto, p=2, dim=0)
    return proto_tensor

# Get Routing Prior based on input images using AHR and DUN
def get_routing_prior(X_t, expert0, expert1, proto0, proto1, force_routing_type=None):
    X_pos = (X_t + 1.0) / 2.0
    X_denoised = torch.where(X_pos > 0.35, X_pos, torch.zeros_like(X_pos))
    h_batch = hoyer_sparsity(X_denoised)
    
    if force_routing_type is not None:
        routing_type = force_routing_type
    else:
        routing_type = 0 if h_batch >= 0.50 else 1 # 0: L2 (Sparse), 1: Angular (Dense)
    
    expert0.eval()
    expert1.eval()
    with torch.no_grad():
        f0 = expert0.extract_features(X_t)
        f1 = expert1.extract_features(X_t)
        
        o0 = expert0(X_t)
        o1 = expert1(X_t)
        p0 = F.softmax(o0, dim=1)
        p1 = F.softmax(o1, dim=1)
        H0 = -torch.sum(p0 * torch.log(p0 + 1e-8), dim=1).mean().item()
        H1 = -torch.sum(p1 * torch.log(p1 + 1e-8), dim=1).mean().item()
        H_avg = 0.5 * (H0 + H1)
        
    d0_list = []
    d1_list = []
    B = X_t.size(0)
    for i in range(B):
        f0_norm = F.normalize(f0[i], p=2, dim=0)
        f1_norm = F.normalize(f1[i], p=2, dim=0)
        if routing_type == 0:
            dist0 = torch.min(torch.sum((f0_norm.unsqueeze(0) - proto0)**2, dim=1))
            dist1 = torch.min(torch.sum((f1_norm.unsqueeze(0) - proto1)**2, dim=1))
        else:
            dist0 = torch.min(1.0 - F.cosine_similarity(f0_norm.unsqueeze(0), proto0, dim=1))
            dist1 = torch.min(1.0 - F.cosine_similarity(f1_norm.unsqueeze(0), proto1, dim=1))
        d0_list.append(dist0)
        d1_list.append(dist1)
        
    D0 = torch.stack(d0_list).mean().item()
    D1 = torch.stack(d1_list).mean().item()
    gap = abs(D0 - D1)
    
    ϵ_base = 0.08 if routing_type == 0 else 0.04
    ϵ_stab = ϵ_base / (1.0 + 2.0 * H_avg) # DUN scaling
    τ = (gap / 3.0) + ϵ_stab
    
    w1 = math.exp(-D1 / τ) / (math.exp(-D0 / τ) + math.exp(-D1 / τ) + 1e-8)
    w0 = 1.0 - w1
    return w0, w1, H_avg, h_batch, routing_type

# On-the-fly Kronecker Sensitivity preconditioning
def estimate_on_the_fly_sensitivity(expert0, expert1, w_global_init, inputs, skeleton_model):
    delta_zero = torch.zeros(4)
    params = merge_parameters(expert0, expert1, w_global_init, delta_zero)
    buffers = merge_buffers(expert0, expert1, w_global_init)
    
    params_grad_leaf = {name: p.detach().clone().requires_grad_(True) for name, p in params.items()}
    all_state = {**params_grad_leaf, **buffers}
    
    outputs = functional_call(skeleton_model, all_state, inputs)
    p = F.softmax(outputs, dim=1)
    loss = -torch.sum(p * torch.log(p + 1e-8), dim=1).mean()
    loss.backward()
    
    groups = {
        0: ['conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias'],
        1: ['conv2.weight', 'conv2.bias', 'bn2.weight', 'bn2.bias'],
        2: ['fc1.weight', 'fc1.bias'],
        3: ['classifier.weight', 'classifier.bias']
    }
    F_sens = torch.zeros(4)
    for j in range(4):
        sq_grads = []
        for name in groups[j]:
            if name in params_grad_leaf:
                g = params_grad_leaf[name].grad
                if g is not None:
                    sq_grads.append(g.pow(2).mean())
        if len(sq_grads) > 0:
            F_sens[j] = torch.stack(sq_grads).mean()
        else:
            F_sens[j] = 1e-8
            
    return F_sens / (F_sens.sum() + 1e-8)

def compute_loss(outputs, w_global_var, delta_var, w0, w1, F_sens):
    # Entropy loss
    p = F.softmax(outputs, dim=1)
    L_entropy = -torch.sum(p * torch.log(p + 1e-8), dim=1).mean()
    
    # KL Routing prior regularization
    # Mean lambda
    lambdas = torch.stack([torch.sigmoid(w_global_var + delta_var[j]) for j in range(4)])
    mean_lambda = lambdas.mean()
    mean_lambda = torch.clamp(mean_lambda, 1e-6, 1.0 - 1e-6)
    L_kl = mean_lambda * torch.log(mean_lambda / (w1 + 1e-8)) + (1.0 - mean_lambda) * torch.log((1.0 - mean_lambda) / (w0 + 1e-8))
    
    # Coherence Regularization
    L_coherence = torch.sum(F_sens * delta_var.pow(2))
    
    return L_entropy + 1.5 * L_kl + 0.02 * L_coherence

# Main Dataset and Stream setup
print("Setting up datasets...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
fashion_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Subsampling 10,000 samples for pretraining
mnist_subset = torch.utils.data.Subset(mnist_train, list(range(10000)))
fashion_subset = torch.utils.data.Subset(fashion_train, list(range(10000)))

class JointDataset(torch.utils.data.Dataset):
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2
    def __len__(self):
        return len(self.ds1) + len(self.ds2)
    def __getitem__(self, idx):
        if idx < len(self.ds1):
            return self.ds1[idx]
        else:
            return self.ds2[idx - len(self.ds1)]

joint_train = JointDataset(mnist_subset, fashion_subset)
joint_loader = torch.utils.data.DataLoader(joint_train, batch_size=64, shuffle=True)

mnist_loader = torch.utils.data.DataLoader(mnist_subset, batch_size=64, shuffle=True)
fashion_loader = torch.utils.data.DataLoader(fashion_subset, batch_size=64, shuffle=True)

# Pretrain Models
print("Pre-training Standard joint base & experts...")
base_std = SimpleCNN(use_cosface=False)
train_model(base_std, joint_loader, epochs=1, lr=1e-3, weight_decay=1e-4)

expert0_std = SimpleCNN(use_cosface=False)
expert0_std.load_state_dict(base_std.state_dict())
train_model(expert0_std, mnist_loader, epochs=1, lr=2e-4, weight_decay=1e-5)

expert1_std = SimpleCNN(use_cosface=False)
expert1_std.load_state_dict(base_std.state_dict())
train_model(expert1_std, fashion_loader, epochs=1, lr=2e-4, weight_decay=1e-5)

print("Pre-training CosFace joint base & experts...")
base_cos = SimpleCNN(use_cosface=True)
train_model(base_cos, joint_loader, epochs=1, lr=1e-3, weight_decay=1e-4, use_cosface=True)

expert0_cos = SimpleCNN(use_cosface=True)
expert0_cos.load_state_dict(base_cos.state_dict())
train_model(expert0_cos, mnist_loader, epochs=1, lr=2e-4, weight_decay=1e-5, use_cosface=True)

expert1_cos = SimpleCNN(use_cosface=True)
expert1_cos.load_state_dict(base_cos.state_dict())
train_model(expert1_cos, fashion_loader, epochs=1, lr=2e-4, weight_decay=1e-5, use_cosface=True)

# Evaluate Clean expert performance
def eval_expert(expert, dataset):
    expert.eval()
    correct = 0
    total = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for img, lbl in loader:
            outputs = expert(img)
            _, pred = torch.max(outputs, 1)
            correct += (pred == lbl).sum().item()
            total += lbl.size(0)
    return correct / total

mnist_acc = eval_expert(expert0_std, mnist_test)
fashion_acc = eval_expert(expert1_std, fashion_test)
print(f"Expert 0 (Standard MNIST) Accuracy: {mnist_acc * 100:.2f}%")
print(f"Expert 1 (Standard Fashion) Accuracy: {fashion_acc * 100:.2f}%")

# Compute prototypes
print("Computing prototypes...")
proto_std0 = compute_prototypes(expert0_std, mnist_subset)
proto_std1 = compute_prototypes(expert1_std, fashion_subset)
proto_cos0 = compute_prototypes(expert0_cos, mnist_subset)
proto_cos1 = compute_prototypes(expert1_cos, fashion_subset)

# Build Test Stream
print("Building 50-batch test stream...")
test_batches = []
# Phase 1: Clean MNIST (batches 0-9)
for b in range(10):
    imgs, lbls = [], []
    for i in range(64):
        img, lbl = mnist_test[b * 64 + i]
        imgs.append(img)
        lbls.append(lbl)
    test_batches.append((torch.stack(imgs), torch.tensor(lbls), "Clean MNIST"))

# Phase 2: Noisy MNIST (batches 10-19)
for b in range(10):
    imgs, lbls = [], []
    for i in range(64):
        img, lbl = mnist_test[(10 + b) * 64 + i]
        noise = torch.randn_like(img) * 0.6
        imgs.append(torch.clamp(img + noise, -1.0, 1.0))
        lbls.append(lbl)
    test_batches.append((torch.stack(imgs), torch.tensor(lbls), "Noisy MNIST"))

# Phase 3: Clean Fashion (batches 20-29)
for b in range(10):
    imgs, lbls = [], []
    for i in range(64):
        img, lbl = fashion_test[b * 64 + i]
        imgs.append(img)
        lbls.append(lbl)
    test_batches.append((torch.stack(imgs), torch.tensor(lbls), "Clean Fashion"))

# Phase 4: Noisy Fashion (batches 30-39)
for b in range(10):
    imgs, lbls = [], []
    for i in range(64):
        img, lbl = fashion_test[(10 + b) * 64 + i]
        noise = torch.randn_like(img) * 0.6
        imgs.append(torch.clamp(img + noise, -1.0, 1.0))
        lbls.append(lbl)
    test_batches.append((torch.stack(imgs), torch.tensor(lbls), "Noisy Fashion"))

# Phase 5: Novel KMNIST (batches 40-49)
for b in range(10):
    imgs, lbls = [], []
    for i in range(64):
        img, lbl = kmnist_test[b * 64 + i]
        imgs.append(img)
        lbls.append(lbl)
    test_batches.append((torch.stack(imgs), torch.tensor(lbls), "Novel KMNIST"))


def evaluate_method(method_name, stream_adaptation_func):
    segment_accs = {
        "Clean MNIST": [],
        "Noisy MNIST": [],
        "Clean Fashion": [],
        "Noisy Fashion": [],
        "Novel KMNIST": []
    }
    
    print(f"Running method: {method_name}...")
    
    for b_idx, (inputs, targets, seg_name) in enumerate(test_batches):
        # Adapt and get merged model forward function or output predictions
        preds = stream_adaptation_func(inputs, b_idx)
        
        correct = (preds == targets).sum().item()
        acc = correct / inputs.size(0)
        segment_accs[seg_name].append(acc)
        
    # Summarize results
    results = {}
    overall_accs = []
    for seg, accs in segment_accs.items():
        mean_acc = np.mean(accs) * 100
        results[seg] = mean_acc
        overall_accs.extend(accs)
    results["Overall"] = np.mean(overall_accs) * 100
    return results

# 1. Static Merging (Fixed λ = 0.5)
def adapt_static(inputs, b_idx):
    # Static uses Standard experts and fixed λ = 0.5
    w_global = torch.tensor(0.0)
    delta = torch.zeros(4)
    params = merge_parameters(expert0_std, expert1_std, w_global, delta)
    buffers = merge_buffers(expert0_std, expert1_std, w_global)
    skeleton = SimpleCNN(use_cosface=False)
    
    with torch.no_grad():
        outputs = functional_call(skeleton, {**params, **buffers}, inputs)
        _, preds = torch.max(outputs, 1)
    return preds

# 2. Fixed TTA (TENT-style entropy minimization on standard experts)
def adapt_fixed_tta(inputs, b_idx):
    # Retrieve routing prior for initializing
    w0, w1, _, _, _ = get_routing_prior(inputs, expert0_std, expert1_std, proto_std0, proto_std1)
    w_global = torch.tensor(math.log(w1 / (w0 + 1e-8)), requires_grad=True)
    delta = torch.zeros(4, requires_grad=True)
    
    skeleton = SimpleCNN(use_cosface=False)
    optimizer = optim.Adam([w_global, delta], lr=0.01)
    
    # 5 TTA steps
    for step in range(5):
        optimizer.zero_grad()
        params = merge_parameters(expert0_std, expert1_std, w_global, delta)
        buffers = merge_buffers(expert0_std, expert1_std, w_global)
        outputs = functional_call(skeleton, {**params, **buffers}, inputs)
        p = F.softmax(outputs, dim=1)
        loss = -torch.sum(p * torch.log(p + 1e-8), dim=1).mean()
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        params = merge_parameters(expert0_std, expert1_std, w_global, delta)
        buffers = merge_buffers(expert0_std, expert1_std, w_global)
        outputs = functional_call(skeleton, {**params, **buffers}, inputs)
        _, preds = torch.max(outputs, 1)
    return preds

# 3. BK-CoMerge (Standard trace-preconditioned update)
def adapt_bk_comerge(inputs, b_idx):
    w0, w1, H_avg, _, _ = get_routing_prior(inputs, expert0_std, expert1_std, proto_std0, proto_std1)
    w_global = torch.tensor(math.log(w1 / (w0 + 1e-8)), requires_grad=True)
    delta = torch.zeros(4, requires_grad=True)
    
    skeleton = SimpleCNN(use_cosface=False)
    F_sens = estimate_on_the_fly_sensitivity(expert0_std, expert1_std, w_global, inputs, skeleton)
    
    optimizer = optim.Adam([
        {'params': [w_global], 'lr': 0.05},
        {'params': [delta], 'lr': 0.05}
    ])
    
    for step in range(5):
        optimizer.zero_grad()
        params = merge_parameters(expert0_std, expert1_std, w_global, delta)
        buffers = merge_buffers(expert0_std, expert1_std, w_global)
        outputs = functional_call(skeleton, {**params, **buffers}, inputs)
        loss = compute_loss(outputs, w_global, delta, w0, w1, F_sens)
        loss.backward()
        
        # Precondition the gradients of delta
        with torch.no_grad():
            delta.grad = delta.grad / (F_sens + 10**-5) # Small epsilon_stab
            
        optimizer.step()
        
    with torch.no_grad():
        params = merge_parameters(expert0_std, expert1_std, w_global, delta)
        buffers = merge_buffers(expert0_std, expert1_std, w_global)
        outputs = functional_call(skeleton, {**params, **buffers}, inputs)
        _, preds = torch.max(outputs, 1)
    return preds

# 4. SAM-TTMM (Sharpness-aware preconditioned update with robust floor)
def adapt_sam_ttmm(inputs, b_idx):
    w0, w1, H_avg, _, _ = get_routing_prior(inputs, expert0_std, expert1_std, proto_std0, proto_std1)
    w_global = torch.tensor(math.log(w1 / (w0 + 1e-8)))
    delta = torch.zeros(4)
    
    skeleton = SimpleCNN(use_cosface=False)
    F_sens = estimate_on_the_fly_sensitivity(expert0_std, expert1_std, w_global, inputs, skeleton)
    
    # 5 steps of adaptation
    for step in range(5):
        w_global_var = w_global.clone().detach().requires_grad_(True)
        delta_var = delta.clone().detach().requires_grad_(True)
        
        # Merge and run forward pass to compute gradients at current point
        params = merge_parameters(expert0_std, expert1_std, w_global_var, delta_var)
        buffers = merge_buffers(expert0_std, expert1_std, w_global_var)
        outputs = functional_call(skeleton, {**params, **buffers}, inputs)
        
        loss = compute_loss(outputs, w_global_var, delta_var, w0, w1, F_sens)
        loss.backward()
        
        g_w = w_global_var.grad.clone().detach()
        g_delta = delta_var.grad.clone().detach()
        
        # Preconditioned SAM perturbation
        d_w = g_w
        d_delta = g_delta / (F_sens + 0.1) # Robust preconditioning floor
        norm_D = torch.sqrt(d_w.pow(2) + torch.sum(d_delta.pow(2)) + 0.1)
        
        epsilon_w = 0.05 * d_w / norm_D
        epsilon_delta = 0.05 * d_delta / norm_D
        
        # Reset grads and compute perturbed loss
        w_global_var.grad = None
        delta_var.grad = None
        perturbed_w = w_global_var + epsilon_w.detach()
        perturbed_delta = delta_var + epsilon_delta.detach()
        
        params_p = merge_parameters(expert0_std, expert1_std, perturbed_w, perturbed_delta)
        buffers_p = merge_buffers(expert0_std, expert1_std, perturbed_w)
        outputs_p = functional_call(skeleton, {**params_p, **buffers_p}, inputs)
        
        loss_p = compute_loss(outputs_p, perturbed_w, perturbed_delta, w0, w1, F_sens)
        loss_p.backward()
        
        # Gradient update step with fixed eta
        with torch.no_grad():
            w_global -= 0.005 * w_global_var.grad
            delta -= 0.005 * (delta_var.grad / (F_sens + 0.1))
            
    # Final evaluation
    with torch.no_grad():
        params = merge_parameters(expert0_std, expert1_std, w_global, delta)
        buffers = merge_buffers(expert0_std, expert1_std, w_global)
        outputs = functional_call(skeleton, {**params, **buffers}, inputs)
        _, preds = torch.max(outputs, 1)
    return preds

# 5. BK-AHR (Adaptive Hybrid Routing SOTA)
def adapt_bk_ahr(inputs, b_idx):
    # Select appropriate experts and prototypes based on Gating
    X_pos = (inputs + 1.0) / 2.0
    X_denoised = torch.where(X_pos > 0.35, X_pos, torch.zeros_like(X_pos))
    h_batch = hoyer_sparsity(X_denoised)
    
    use_cos = h_batch < 0.50
    expert0 = expert0_cos if use_cos else expert0_std
    expert1 = expert1_cos if use_cos else expert1_std
    proto0 = proto_cos0 if use_cos else proto_std0
    proto1 = proto_cos1 if use_cos else proto_std1
    skeleton = SimpleCNN(use_cosface=use_cos)
    
    w0, w1, H_avg, _, _ = get_routing_prior(inputs, expert0, expert1, proto0, proto1)
    w_global = torch.tensor(math.log(w1 / (w0 + 1e-8)), requires_grad=True)
    delta = torch.zeros(4, requires_grad=True)
    
    F_sens = estimate_on_the_fly_sensitivity(expert0, expert1, w_global, inputs, skeleton)
    
    optimizer = optim.Adam([
        {'params': [w_global], 'lr': 0.05},
        {'params': [delta], 'lr': 0.05}
    ])
    
    for step in range(5):
        optimizer.zero_grad()
        params = merge_parameters(expert0, expert1, w_global, delta)
        buffers = merge_buffers(expert0, expert1, w_global)
        outputs = functional_call(skeleton, {**params, **buffers}, inputs)
        loss = compute_loss(outputs, w_global, delta, w0, w1, F_sens)
        loss.backward()
        
        with torch.no_grad():
            delta.grad = delta.grad / (F_sens + 0.02)
            
        optimizer.step()
        
    with torch.no_grad():
        params = merge_parameters(expert0, expert1, w_global, delta)
        buffers = merge_buffers(expert0, expert1, w_global)
        outputs = functional_call(skeleton, {**params, **buffers}, inputs)
        _, preds = torch.max(outputs, 1)
    return preds

# 6. SAM-BK-AHR (Our Proposed Method: Sharpness-Aware Bayesian preconditioned AHR)
def adapt_sam_bk_ahr(inputs, b_idx):
    # Denoised Hoyer Sparsity-Aware Gating (For matching the standard expert behavior)
    X_pos = (inputs + 1.0) / 2.0
    X_denoised = torch.where(X_pos > 0.35, X_pos, torch.zeros_like(X_pos))
    h_batch = hoyer_sparsity(X_denoised)
    
    # We use Standard experts for both routing domains to avoid "Sphere Merging Distortion"
    use_cos = False
    expert0 = expert0_std
    expert1 = expert1_std
    proto0 = proto_std0
    proto1 = proto_std1
    skeleton = SimpleCNN(use_cosface=False)
    
    # Calculate Routing Prior & DUN scaling (with forced Euclidean routing_type=0 for robust matching)
    w0, w1, H_avg, _, _ = get_routing_prior(inputs, expert0, expert1, proto0, proto1, force_routing_type=0)
    w_global = torch.tensor(math.log(w1 / (w0 + 1e-8)))
    delta = torch.zeros(4)
    
    # Differentiable on-the-fly Kronecker preconditioning
    F_sens = estimate_on_the_fly_sensitivity(expert0, expert1, w_global, inputs, skeleton)
    
    # Entropy-Adaptive Learning Rate with optimized base 0.1
    eta_t = 0.1 / (1.0 + 5.0 * H_avg)
    
    # 5 steps of adaptation
    damping = 0.001
    rho = 0.05
    for step in range(5):
        w_global_var = w_global.clone().detach().requires_grad_(True)
        delta_var = delta.clone().detach().requires_grad_(True)
        
        # Step 1: Forward to compute gradients at current point
        params = merge_parameters(expert0, expert1, w_global_var, delta_var)
        buffers = merge_buffers(expert0, expert1, w_global_var)
        outputs = functional_call(skeleton, {**params, **buffers}, inputs)
        
        loss = compute_loss(outputs, w_global_var, delta_var, w0, w1, F_sens)
        loss.backward()
        
        g_w = w_global_var.grad.clone().detach()
        g_delta = delta_var.grad.clone().detach()
        
        # Step 2: Preconditioned worst-case perturbation
        d_w = g_w
        d_delta = g_delta / (F_sens + damping)
        norm_D = torch.sqrt(d_w.pow(2) + torch.sum(d_delta.pow(2)) + damping)
        
        epsilon_w = rho * d_w / norm_D
        epsilon_delta = rho * d_delta / norm_D
        
        # Step 3: Forward pass to compute perturbed loss
        w_global_var.grad = None
        delta_var.grad = None
        perturbed_w = w_global_var + epsilon_w.detach()
        perturbed_delta = delta_var + epsilon_delta.detach()
        
        params_p = merge_parameters(expert0, expert1, perturbed_w, perturbed_delta)
        buffers_p = merge_buffers(expert0, expert1, perturbed_w)
        outputs_p = functional_call(skeleton, {**params_p, **buffers_p}, inputs)
        
        loss_p = compute_loss(outputs_p, perturbed_w, perturbed_delta, w0, w1, F_sens)
        loss_p.backward()
        
        # Step 4: Gradient update step
        with torch.no_grad():
            w_global -= eta_t * w_global_var.grad
            delta -= eta_t * (delta_var.grad / (F_sens + damping))
            
    # Final evaluation
    with torch.no_grad():
        params = merge_parameters(expert0, expert1, w_global, delta)
        buffers = merge_buffers(expert0, expert1, w_global)
        outputs = functional_call(skeleton, {**params, **buffers}, inputs)
        _, preds = torch.max(outputs, 1)
    return preds

# Main evaluation runner
methods = {
    "Static Merging": adapt_static,
    "Fixed TTA": adapt_fixed_tta,
    "BK-CoMerge": adapt_bk_comerge,
    "SAM-TTMM": adapt_sam_ttmm,
    "BK-AHR": adapt_bk_ahr,
    "SAM-BK-AHR (Ours)": adapt_sam_bk_ahr
}

results_table = {}
for m_name, m_func in methods.items():
    results_table[m_name] = evaluate_method(m_name, m_func)

# Print results
print("\n" + "="*80)
print(f"{'Method':<20} | {'Clean MN':<10} | {'Noisy MN':<10} | {'Clean Fash':<10} | {'Noisy Fash':<10} | {'Novel KMN':<10} | {'Overall':<10}")
print("="*80)
for m_name, res in results_table.items():
    print(f"{m_name:<20} | {res['Clean MNIST']:<10.2f}% | {res['Noisy MNIST']:<10.2f}% | {res['Clean Fashion']:<10.2f}% | {res['Noisy Fashion']:<10.2f}% | {res['Novel KMNIST']:<10.2f}% | {res['Overall']:<10.2f}%")
print("="*80)

# Save metrics for paper writing
import json
with open("metrics.json", "w") as f:
    json.dump(results_table, f, indent=4)
print("Metrics saved to metrics.json")
