import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import os
from torch.func import functional_call

# Define the network architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Noise helper
def add_gaussian_noise(x, sigma=0.6):
    return x + torch.randn_like(x) * sigma

# Load datasets and create the non-stationary stream
def get_stream_batches(batch_size=64):
    torch.manual_seed(42)
    np.random.seed(42)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fashion_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_set = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_loader = torch.utils.data.DataLoader(mnist_set, batch_size=batch_size, shuffle=False)
    fashion_loader = torch.utils.data.DataLoader(fashion_set, batch_size=batch_size, shuffle=False)
    kmnist_loader = torch.utils.data.DataLoader(kmnist_set, batch_size=batch_size, shuffle=False)
    
    batches = []
    
    # 1. Clean MNIST (0-9)
    mnist_iter = iter(mnist_loader)
    for _ in range(10):
        x, y = next(mnist_iter)
        batches.append((x, y, "Clean MNIST"))
        
    # 2. Noisy MNIST (10-19)
    for _ in range(10):
        x, y = next(mnist_iter)
        batches.append((add_gaussian_noise(x, 0.6), y, "Noisy MNIST"))
        
    # 3. Clean FashionMNIST (20-29)
    fashion_iter = iter(fashion_loader)
    for _ in range(10):
        x, y = next(fashion_iter)
        batches.append((x, y, "Clean Fashion"))
        
    # 4. Noisy FashionMNIST (30-39)
    for _ in range(10):
        x, y = next(fashion_iter)
        batches.append((add_gaussian_noise(x, 0.6), y, "Noisy Fashion"))
        
    # 5. Novel KMNIST (40-49)
    kmnist_iter = iter(kmnist_loader)
    for _ in range(10):
        x, y = next(kmnist_iter)
        batches.append((x, y, "Novel KMNIST"))
        
    return batches

# Precompute offline joint Fisher Information for CLW-Fisher
def compute_offline_fisher(expert_0, expert_1, num_samples=100):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    fashion_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    mnist_loader = torch.utils.data.DataLoader(mnist_set, batch_size=1, shuffle=True)
    fashion_loader = torch.utils.data.DataLoader(fashion_set, batch_size=1, shuffle=True)
    
    fisher = {}
    model = SimpleCNN()
    
    params_dict = make_merged_parameters(expert_0, expert_1, 0.5)
    fuse_bn_buffers(expert_0, expert_1, model, 0.5)
    model.eval()
    
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)
            
    criterion = nn.CrossEntropyLoss()
    
    # Collect gradients over MNIST
    mnist_iter = iter(mnist_loader)
    for _ in range(num_samples // 2):
        try:
            x, y = next(mnist_iter)
        except StopIteration:
            break
        model.zero_grad()
        p_dict = {k: v.clone().detach().requires_grad_(True) for k, v in params_dict.items()}
        out = functional_call(model, p_dict, x)
        loss = criterion(out, y)
        loss.backward()
        for name in fisher.keys():
            if name in p_dict and p_dict[name].grad is not None:
                fisher[name] += p_dict[name].grad.data ** 2
                
    # Collect gradients over FashionMNIST
    fashion_iter = iter(fashion_loader)
    for _ in range(num_samples // 2):
        try:
            x, y = next(fashion_iter)
        except StopIteration:
            break
        model.zero_grad()
        p_dict = {k: v.clone().detach().requires_grad_(True) for k, v in params_dict.items()}
        out = functional_call(model, p_dict, x)
        loss = criterion(out, y)
        loss.backward()
        for name in fisher.keys():
            if name in p_dict and p_dict[name].grad is not None:
                fisher[name] += p_dict[name].grad.data ** 2
                
    # Normalize
    for name in fisher.keys():
        fisher[name] = (fisher[name] / num_samples).clamp(min=1e-5)
        
    return fisher

# Helper to merge weights differentiably
def make_merged_parameters(expert_0, expert_1, coefficients):
    e0_params = dict(expert_0.named_parameters())
    e1_params = dict(expert_1.named_parameters())
    
    merged_params = {}
    for key in e0_params.keys():
        p0 = e0_params[key]
        p1 = e1_params[key]
        
        if p0.dtype.is_floating_point:
            if isinstance(coefficients, dict):
                prefix = key.split('.')[0]
                lam = coefficients.get(prefix, 0.5)
            else:
                lam = coefficients
            
            merged_params[key] = (1.0 - lam) * p0 + lam * p1
        else:
            merged_params[key] = p0
            
    return merged_params

# Helper to fuse BN buffers in-place
def fuse_bn_buffers(expert_0, expert_1, merged_model, w_expert_1):
    w1 = w_expert_1
    w0 = 1.0 - w1
    
    e0_buffers = dict(expert_0.named_buffers())
    e1_buffers = dict(expert_1.named_buffers())
    merged_buffers = dict(merged_model.named_buffers())
    
    for key in e0_buffers.keys():
        if 'running_mean' in key:
            mean0 = e0_buffers[key]
            mean1 = e1_buffers[key]
            mean_fused = w0 * mean0 + w1 * mean1
            merged_buffers[key].copy_(mean_fused)
            
            var_key = key.replace('running_mean', 'running_var')
            var0 = e0_buffers[var_key]
            var1 = e1_buffers[var_key]
            
            var_fused = w0 * (var0 + (mean0 - mean_fused) ** 2) + w1 * (var1 + (mean1 - mean_fused) ** 2)
            merged_buffers[var_key].copy_(var_fused)

# Shannon entropy of a model on a batch
def compute_batch_entropy(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
        return entropy.mean().item()

# Forward/backward hooks to calculate Kronecker trace sensitivity on-the-fly
class KroneckerTraceTracker:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.gradients = {}
        self.f_hooks = []
        self.b_hooks = []
        self.setup_hooks()
        
    def setup_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.f_hooks.append(module.register_forward_hook(self._make_f_hook(name)))
                self.b_hooks.append(module.register_full_backward_hook(self._make_b_hook(name)))
                
    def _make_f_hook(self, name):
        return lambda module, inp, out: self._f_hook_fn(name, inp[0])
        
    def _make_b_hook(self, name):
        return lambda module, grad_inp, grad_out: self._b_hook_fn(name, grad_out[0])
        
    def _f_hook_fn(self, name, inp):
        self.activations[name] = inp.detach()
        
    def _b_hook_fn(self, name, grad_out):
        self.gradients[name] = grad_out[0].detach()
        
    def compute_sensitivities(self):
        sensitivities = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if name in self.activations and name in self.gradients:
                    act = self.activations[name]
                    grad = self.gradients[name]
                    
                    act_norm = (act ** 2).sum(dim=list(range(1, act.ndim))).mean().item()
                    grad_norm = (grad ** 2).sum(dim=list(range(1, grad.ndim))).mean().item()
                    
                    if isinstance(module, nn.Conv2d):
                        din = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                        dout = module.out_channels
                    else:
                        din = module.in_features
                        dout = module.out_features
                        
                    sens = (act_norm * grad_norm) / (din * dout)
                    sensitivities[name] = sens
                else:
                    sensitivities[name] = 1.0
        
        total = sum(sensitivities.values()) + 1e-12
        normalized_sens = {k: v / total for k, v in sensitivities.items()}
        return normalized_sens
        
    def remove(self):
        for h in self.f_hooks:
            h.remove()
        for h in self.b_hooks:
            h.remove()

# KL Divergence between Bernoulli(p) and Bernoulli(q)
def kl_bernoulli(p, q, eps=1e-6):
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    return p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))

# Evaluation of static model
def run_static_merging(expert_0, expert_1, batches):
    print("Evaluating Static Merging...")
    merged_model = SimpleCNN()
    fuse_bn_buffers(expert_0, expert_1, merged_model, 0.5)
    params = make_merged_parameters(expert_0, expert_1, 0.5)
    
    accuracies = []
    for batch_idx, (x, y, name) in enumerate(batches):
        with torch.no_grad():
            out = functional_call(merged_model, params, x)
            pred = out.argmax(dim=1)
            acc = (pred == y).float().mean().item() * 100
            accuracies.append(acc)
    return accuracies

# Evaluation of unconstrained TTA (Fixed TTA)
def run_fixed_tta(expert_0, expert_1, batches, lr=0.01, steps=3):
    print("Evaluating Fixed TTA (TENT)...")
    model = SimpleCNN()
    fuse_bn_buffers(expert_0, expert_1, model, 0.5)
    
    init_params = make_merged_parameters(expert_0, expert_1, 0.5)
    params = {k: nn.Parameter(v.clone()) for k, v in init_params.items()}
    
    accuracies = []
    for batch_idx, (x, y, name) in enumerate(batches):
        params = {k: nn.Parameter(v.clone()) for k, v in init_params.items()}
        optimizer = torch.optim.SGD(params.values(), lr=lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            logits = functional_call(model, params, x)
            probs = F.softmax(logits, dim=1)
            loss = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            out = functional_call(model, params, x)
            pred = out.argmax(dim=1)
            acc = (pred == y).float().mean().item() * 100
            accuracies.append(acc)
            
    return accuracies

# Evaluation of CLW-Fisher
def run_clw_fisher(expert_0, expert_1, batches, offline_fisher, lr=0.05, steps=3, gamma_c=0.01):
    print("Evaluating CLW-Fisher...")
    accuracies = []
    layer_names = ['conv1', 'conv2', 'fc1', 'fc2']
    
    layer_sens = {}
    for name in layer_names:
        weights_fisher = offline_fisher[name + '.weight']
        bias_fisher = offline_fisher[name + '.bias']
        layer_sens[name] = (weights_fisher.mean() + bias_fisher.mean()).item()
    
    total_sens = sum(layer_sens.values()) + 1e-12
    for name in layer_sens.keys():
        layer_sens[name] /= total_sens
        
    model = SimpleCNN()
    
    for batch_idx, (x, y, name) in enumerate(batches):
        w_global = 0.0
        delta = {name: 0.0 for name in layer_names}
        
        w_g = torch.tensor(w_global, requires_grad=True)
        deltas = {k: torch.tensor(v, requires_grad=True) for k, v in delta.items()}
        
        for _ in range(steps):
            lambdas = {}
            for l_name in layer_names:
                lambdas[l_name] = torch.sigmoid(w_g + deltas[l_name])
                
            params = make_merged_parameters(expert_0, expert_1, lambdas)
            mean_lam = torch.stack(list(lambdas.values())).mean()
            fuse_bn_buffers(expert_0, expert_1, model, mean_lam.item())
            
            logits = functional_call(model, params, x)
            probs = F.softmax(logits, dim=1)
            L_entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()
            
            L_coherence = 0.0
            for l_name in layer_names:
                L_coherence += gamma_c * layer_sens[l_name] * (deltas[l_name] ** 2)
                
            loss = L_entropy + L_coherence
            
            grads = torch.autograd.grad(loss, [w_g] + list(deltas.values()))
            w_g_grad = grads[0]
            deltas_grads = {k: g for k, g in zip(layer_names, grads[1:])}
            
            with torch.no_grad():
                w_g -= lr * w_g_grad
                for l_name in layer_names:
                    grad_scale = 1.0 / (layer_sens[l_name] + 1e-5)
                    deltas[l_name] -= lr * grad_scale * deltas_grads[l_name]
                    
        w_global = w_g.item()
        delta = {k: v.item() for k, v in deltas.items()}
        
        with torch.no_grad():
            lambdas = {k: torch.sigmoid(torch.tensor(v)).item() for k, v in delta.items()}
            params = make_merged_parameters(expert_0, expert_1, lambdas)
            mean_lam = np.mean(list(lambdas.values()))
            fuse_bn_buffers(expert_0, expert_1, model, mean_lam)
            out = functional_call(model, params, x)
            pred = out.argmax(dim=1)
            acc = (pred == y).float().mean().item() * 100
            accuracies.append(acc)
            
        if batch_idx in [0, 9, 10, 19, 20, 29, 30, 39, 40, 49]:
            print(f"Batch {batch_idx:02d} ({name}) | w_global: {w_global:.3f} | mean_lam: {mean_lam:.3f} | Acc: {acc:.2f}%")
            
    return accuracies

# Evaluation of KT-Fisher
def run_kt_fisher(expert_0, expert_1, batches, lr=0.01, steps=3):
    print("Evaluating KT-Fisher...")
    accuracies = []
    layer_names = ['conv1', 'conv2', 'fc1', 'fc2']
    
    model = SimpleCNN()
    
    for batch_idx, (x, y, name) in enumerate(batches):
        h0 = compute_batch_entropy(expert_0, x)
        h1 = compute_batch_entropy(expert_1, x)
        w_expert = 0.0 if h0 < h1 else 1.0
        
        w_global = -4.0 if w_expert == 0.0 else 4.0
        delta = {name: 0.0 for name in layer_names}
        
        lambdas = {k: torch.sigmoid(torch.tensor(v)).item() for k, v in delta.items()}
        params_base = make_merged_parameters(expert_0, expert_1, lambdas)
        mean_lam = np.mean(list(lambdas.values()))
        fuse_bn_buffers(expert_0, expert_1, model, mean_lam)
        
        p_dict = {k: nn.Parameter(v.clone()) for k, v in params_base.items()}
        tracker = KroneckerTraceTracker(model)
        logits = functional_call(model, p_dict, x)
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()
        
        for p in p_dict.values():
            if p.grad is not None:
                p.grad.zero_()
        entropy.backward()
        
        sensitivities = tracker.compute_sensitivities()
        tracker.remove()
        
        w_g = torch.tensor(w_global, requires_grad=True)
        deltas = {k: torch.tensor(v, requires_grad=True) for k, v in delta.items()}
        
        for _ in range(steps):
            curr_lambdas = {k: torch.sigmoid(w_g + deltas[k]) for k in layer_names}
            params = make_merged_parameters(expert_0, expert_1, curr_lambdas)
            mean_lam_curr = torch.stack(list(curr_lambdas.values())).mean()
            fuse_bn_buffers(expert_0, expert_1, model, mean_lam_curr.item())
            
            logits = functional_call(model, params, x)
            probs = F.softmax(logits, dim=1)
            L_entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()
            
            p_mean = torch.stack(list(curr_lambdas.values())).mean()
            L_kl = kl_bernoulli(p_mean, torch.tensor(w_expert))
            
            loss = L_entropy + 0.1 * L_kl
            
            grads = torch.autograd.grad(loss, [w_g] + list(deltas.values()))
            w_g_grad = grads[0]
            deltas_grads = {k: g for k, g in zip(layer_names, grads[1:])}
            
            with torch.no_grad():
                w_g -= lr * w_g_grad
                for l_name in layer_names:
                    grad_scale = 1.0 / (sensitivities[l_name] + 1e-5)
                    deltas[l_name] -= lr * grad_scale * deltas_grads[l_name]
                    
        w_global = w_g.item()
        delta = {k: v.item() for k, v in deltas.items()}
        
        with torch.no_grad():
            lambdas = {k: torch.sigmoid(torch.tensor(v)).item() for k, v in delta.items()}
            params = make_merged_parameters(expert_0, expert_1, lambdas)
            mean_lam = np.mean(list(lambdas.values()))
            fuse_bn_buffers(expert_0, expert_1, model, mean_lam)
            out = functional_call(model, params, x)
            pred = out.argmax(dim=1)
            acc = (pred == y).float().mean().item() * 100
            accuracies.append(acc)
            
        if batch_idx in [0, 9, 10, 19, 20, 29, 30, 39, 40, 49]:
            print(f"Batch {batch_idx:02d} ({name}) | h0: {h0:.3f} | h1: {h1:.3f} | Route W1: {w_expert:.3f} | w_global: {w_global:.3f} | mean_lam: {mean_lam:.3f} | Acc: {acc:.2f}%")
            
    return accuracies

# Evaluation of DF-Bayes-TTMM
def run_df_bayes_ttmm(expert_0, expert_1, batches, lr=0.01, steps=3):
    print("Evaluating DF-Bayes-TTMM...")
    accuracies = []
    layer_names = ['conv1', 'conv2', 'fc1', 'fc2']
    
    model = SimpleCNN()
    
    for batch_idx, (x, y, name) in enumerate(batches):
        h0 = compute_batch_entropy(expert_0, x)
        h1 = compute_batch_entropy(expert_1, x)
        
        tau_self = abs(h0 - h1) / 1.0 + 1e-5
        w0 = np.exp(-h0 / tau_self)
        w1 = np.exp(-h1 / tau_self)
        routing_weight = w1 / (w0 + w1 + 1e-12)
        
        w_global = np.log(routing_weight / (1.0 - routing_weight + 1e-12))
        w_global = np.clip(w_global, -4.0, 4.0)
        delta = {name: 0.0 for name in layer_names}
        
        lambdas = {k: torch.sigmoid(torch.tensor(v)).item() for k, v in delta.items()}
        params_base = make_merged_parameters(expert_0, expert_1, lambdas)
        mean_lam = np.mean(list(lambdas.values()))
        fuse_bn_buffers(expert_0, expert_1, model, mean_lam)
        
        p_dict = {k: nn.Parameter(v.clone()) for k, v in params_base.items()}
        tracker = KroneckerTraceTracker(model)
        logits = functional_call(model, p_dict, x)
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()
        
        for p in p_dict.values():
            if p.grad is not None:
                p.grad.zero_()
        entropy.backward()
        
        sensitivities = tracker.compute_sensitivities()
        tracker.remove()
        
        w_g = torch.tensor(w_global, requires_grad=True)
        deltas = {k: torch.tensor(v, requires_grad=True) for k, v in delta.items()}
        
        for _ in range(steps):
            curr_lambdas = {k: torch.sigmoid(w_g + deltas[k]) for k in layer_names}
            params = make_merged_parameters(expert_0, expert_1, curr_lambdas)
            mean_lam_curr = torch.stack(list(curr_lambdas.values())).mean()
            fuse_bn_buffers(expert_0, expert_1, model, mean_lam_curr.item())
            
            logits = functional_call(model, params, x)
            probs = F.softmax(logits, dim=1)
            L_entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()
            
            p_mean = torch.stack(list(curr_lambdas.values())).mean()
            L_kl = kl_bernoulli(p_mean, torch.tensor(routing_weight))
            
            loss = L_entropy + 0.1 * L_kl
            
            grads = torch.autograd.grad(loss, [w_g] + list(deltas.values()))
            w_g_grad = grads[0]
            deltas_grads = {k: g for k, g in zip(layer_names, grads[1:])}
            
            with torch.no_grad():
                w_g -= lr * w_g_grad
                for l_name in layer_names:
                    grad_scale = 1.0 / (sensitivities[l_name] + 1e-5)
                    deltas[l_name] -= lr * grad_scale * deltas_grads[l_name]
                    
        w_global = w_g.item()
        delta = {k: v.item() for k, v in deltas.items()}
        
        with torch.no_grad():
            lambdas = {k: torch.sigmoid(torch.tensor(v)).item() for k, v in delta.items()}
            params = make_merged_parameters(expert_0, expert_1, lambdas)
            mean_lam = np.mean(list(lambdas.values()))
            fuse_bn_buffers(expert_0, expert_1, model, mean_lam)
            out = functional_call(model, params, x)
            pred = out.argmax(dim=1)
            acc = (pred == y).float().mean().item() * 100
            accuracies.append(acc)
            
        if batch_idx in [0, 9, 10, 19, 20, 29, 30, 39, 40, 49]:
            print(f"Batch {batch_idx:02d} ({name}) | h0: {h0:.3f} | h1: {h1:.3f} | Route W1: {routing_weight:.3f} | w_global: {w_global:.3f} | mean_lam: {mean_lam:.3f} | Acc: {acc:.2f}%")
            
    return accuracies

# Unified function for BK-CoMerge, TS-BK-CoMerge, EGA-BK-CoMerge, TS-EGA-BK-CoMerge, and ATT-BK-CoMerge
def run_bk_comerge(expert_0, expert_1, batches, lr=0.02, steps=3, gamma_c=0.01, beta=0.1, s_scale=1.0, use_ts=False, ema_factor=0.9, use_ega=False, tau_gate=0.65, alpha_gate=2.0, use_att_bn=False, rho=0.1):
    if use_att_bn:
        method_name = f"ATT-BK-CoMerge (rho={rho})"
    elif use_ts:
        method_name = f"TS-BK-CoMerge (EGA={use_ega})"
    else:
        method_name = f"BK-CoMerge (EGA={use_ega})"
    print(f"Evaluating {method_name}...")
        
    accuracies = []
    layer_names = ['conv1', 'conv2', 'fc1', 'fc2']
    
    ema_delta = 0.0
    model = SimpleCNN()
    
    for batch_idx, (x, y, name) in enumerate(batches):
        h0 = compute_batch_entropy(expert_0, x)
        h1 = compute_batch_entropy(expert_1, x)
        
        avg_h = 0.5 * (h0 + h1)
        tau_N = 1.30
        
        if avg_h > tau_N:
            routing_weight = 0.5
        else:
            diff_h = abs(h0 - h1)
            if use_ts:
                ema_delta = ema_factor * ema_delta + (1.0 - ema_factor) * diff_h
                tau_self = ema_delta / s_scale + 1e-5
            else:
                tau_self = diff_h / s_scale + 1e-5
                
            w0 = np.exp(-h0 / tau_self)
            w1 = np.exp(-h1 / tau_self)
            routing_weight = w1 / (w0 + w1 + 1e-12)
            
        w_global = np.log(routing_weight / (1.0 - routing_weight + 1e-12))
        w_global = np.clip(w_global, -4.0, 4.0)
        delta = {name: 0.0 for name in layer_names}
        
        lambdas = {k: torch.sigmoid(torch.tensor(v)).item() for k, v in delta.items()}
        params_base = make_merged_parameters(expert_0, expert_1, lambdas)
        mean_lam = np.mean(list(lambdas.values()))
        fuse_bn_buffers(expert_0, expert_1, model, mean_lam)
        
        # Adaptive Test-Time BN calibration (ATT-BN)
        if use_att_bn:
            model.train()
            # Set BN momentum
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = rho
            with torch.no_grad():
                _ = functional_call(model, params_base, x)
            model.eval()
            
        p_dict = {k: nn.Parameter(v.clone()) for k, v in params_base.items()}
        tracker = KroneckerTraceTracker(model)
        logits = functional_call(model, p_dict, x)
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()
        
        for p in p_dict.values():
            if p.grad is not None:
                p.grad.zero_()
        entropy.backward()
        
        sensitivities = tracker.compute_sensitivities()
        tracker.remove()
        
        current_pred_entropy = entropy.item()
        
        gating_scale = 1.0
        if use_ega:
            if current_pred_entropy > tau_gate:
                gating_scale = np.exp(-alpha_gate * (current_pred_entropy - tau_gate))
                
        w_g = torch.tensor(w_global, requires_grad=True)
        deltas = {k: torch.tensor(v, requires_grad=True) for k, v in delta.items()}
        
        effective_lr = lr * gating_scale
        
        for _ in range(steps):
            curr_lambdas = {k: torch.sigmoid(w_g + deltas[k]) for k in layer_names}
            params = make_merged_parameters(expert_0, expert_1, curr_lambdas)
            mean_lam_curr = torch.stack(list(curr_lambdas.values())).mean()
            fuse_bn_buffers(expert_0, expert_1, model, mean_lam_curr.item())
            
            # If ATT-BN is enabled, re-calibrate BN statistics during adaptation steps too
            if use_att_bn:
                model.train()
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.momentum = rho
                with torch.no_grad():
                    _ = functional_call(model, params, x)
                model.eval()
                
            logits = functional_call(model, params, x)
            probs = F.softmax(logits, dim=1)
            L_entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()
            
            p_mean = torch.stack(list(curr_lambdas.values())).mean()
            L_kl = kl_bernoulli(p_mean, torch.tensor(routing_weight))
            
            L_coherence = 0.0
            for k in layer_names:
                L_coherence += gamma_c * sensitivities[k] * (deltas[k] ** 2)
                
            loss = L_entropy + beta * L_kl + L_coherence
            
            grads = torch.autograd.grad(loss, [w_g] + list(deltas.values()))
            w_g_grad = grads[0]
            deltas_grads = {k: g for k, g in zip(layer_names, grads[1:])}
            
            with torch.no_grad():
                w_g -= effective_lr * w_g_grad
                for l_name in layer_names:
                    grad_scale = 1.0 / (sensitivities[l_name] + 1e-5)
                    deltas[l_name] -= effective_lr * grad_scale * deltas_grads[l_name]
                    
        w_global = w_g.item()
        delta = {k: v.item() for k, v in deltas.items()}
        
        with torch.no_grad():
            lambdas = {k: torch.sigmoid(torch.tensor(v)).item() for k, v in delta.items()}
            params = make_merged_parameters(expert_0, expert_1, lambdas)
            mean_lam = np.mean(list(lambdas.values()))
            fuse_bn_buffers(expert_0, expert_1, model, mean_lam)
            
            if use_att_bn:
                model.train()
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.momentum = rho
                _ = functional_call(model, params, x)
                model.eval()
                
            out = functional_call(model, params, x)
            pred = out.argmax(dim=1)
            acc = (pred == y).float().mean().item() * 100
            accuracies.append(acc)
            
        if batch_idx in [0, 9, 10, 19, 20, 29, 30, 39, 40, 49]:
            print(f"Batch {batch_idx:02d} ({name}) | h0: {h0:.3f} | h1: {h1:.3f} | Route W1: {routing_weight:.3f} | w_global: {w_global:.3f} | mean_lam: {mean_lam:.3f} | Acc: {acc:.2f}%")
            
    return accuracies

# Print segment-wise and overall accuracies
def print_results_table(results_dict):
    print("\n" + "="*80)
    print(f"{'Method':<35} | {'Cl. MNIST':<9} | {'No. MNIST':<9} | {'Cl. Fash':<9} | {'No. Fash':<9} | {'Nov. KM':<9} | {'Overall':<9}")
    print("-"*101)
    for method, accs in results_dict.items():
        m_mnist = np.mean(accs[0:10])
        n_mnist = np.mean(accs[10:20])
        m_fash = np.mean(accs[20:30])
        n_fash = np.mean(accs[30:40])
        n_kmnist = np.mean(accs[40:50])
        overall = np.mean(accs)
        print(f"{method:<35} | {m_mnist:9.2f} | {n_mnist:9.2f} | {m_fash:9.2f} | {n_fash:9.2f} | {n_kmnist:9.2f} | {overall:9.2f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    if not (os.path.exists("./checkpoints/expert_0.pt") and os.path.exists("./checkpoints/expert_1.pt")):
        print("Expert checkpoints not found! Please run train_experts.py first.")
        exit(1)
        
    print("Loading experts...")
    expert_0 = SimpleCNN()
    expert_0.load_state_dict(torch.load("./checkpoints/expert_0.pt", map_location="cpu"))
    expert_0.eval()
    
    expert_1 = SimpleCNN()
    expert_1.load_state_dict(torch.load("./checkpoints/expert_1.pt", map_location="cpu"))
    expert_1.eval()
    
    print("Generating stream batches...")
    batches = get_stream_batches()
    
    print("Precomputing offline joint Fisher...")
    offline_fisher = compute_offline_fisher(expert_0, expert_1, num_samples=100)
    
    # Run all methods
    results = {}
    
    # 1. Static Merging
    results["Static Merging"] = run_static_merging(expert_0, expert_1, batches)
    
    # 2. Fixed TTA (TENT)
    results["Fixed TTA"] = run_fixed_tta(expert_0, expert_1, batches, lr=0.01, steps=3)
    
    # 3. CLW-Fisher
    results["CLW-Fisher"] = run_clw_fisher(expert_0, expert_1, batches, offline_fisher, lr=0.05, steps=3, gamma_c=0.01)
    
    # 4. KT-Fisher
    results["KT-Fisher"] = run_kt_fisher(expert_0, expert_1, batches, lr=0.01, steps=3)
    
    # 5. DF-Bayes-TTMM
    results["DF-Bayes-TTMM"] = run_df_bayes_ttmm(expert_0, expert_1, batches, lr=0.01, steps=3)
    
    # 6. BK-CoMerge
    results["BK-CoMerge (Ours-T9S9)"] = run_bk_comerge(expert_0, expert_1, batches, lr=0.02, steps=3, gamma_c=0.01, beta=0.1, s_scale=1.0, use_ts=False, use_ega=False)
    
    # 7. TS-BK-CoMerge
    results["TS-BK-CoMerge (Ours-T9S9)"] = run_bk_comerge(expert_0, expert_1, batches, lr=0.02, steps=3, gamma_c=0.01, beta=0.1, s_scale=1.0, use_ts=True, ema_factor=0.9, use_ega=False)
    
    # 8. EGA-BK-CoMerge (Proposed besttg=0.6, bestag=0.1)
    results["EGA-BK-CoMerge (Proposed)"] = run_bk_comerge(expert_0, expert_1, batches, lr=0.02, steps=3, gamma_c=0.01, beta=0.1, s_scale=1.0, use_ts=False, use_ega=True, tau_gate=0.6, alpha_gate=0.1)
    
    # 9. ATT-BK-CoMerge (Proposed flagship with Adaptive BN calibration!)
    results["ATT-BK-CoMerge (Ours-Proposed)"] = run_bk_comerge(expert_0, expert_1, batches, lr=0.02, steps=3, gamma_c=0.01, beta=0.1, s_scale=1.0, use_ts=False, use_ega=False, use_att_bn=True, rho=0.2)
    
    # 10. TS-EGA-BK-CoMerge (Ours-Proposed)
    results["TS-EGA-BK-CoMerge (Proposed)"] = run_bk_comerge(expert_0, expert_1, batches, lr=0.02, steps=3, gamma_c=0.01, beta=0.1, s_scale=1.0, use_ts=True, ema_factor=0.9, use_ega=True, tau_gate=0.8, alpha_gate=0.15)
    
    # Print results
    print_results_table(results)
