import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.func import functional_call
import numpy as np
import os
import json
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define SimpleCNN Architecture (Table 1)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def extract_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        feats = self.extract_features(x)
        feats = self.dropout2(feats)
        logits = self.fc2(feats)
        return logits

# Soft BN Buffer Fusion Helper
def merge_bn_buffers(merged_model, expert0, expert1, weights):
    w0, w1 = weights[0].item(), weights[1].item()
    with torch.no_grad():
        for name, buf in merged_model.named_buffers():
            if 'running_mean' in name:
                m0 = dict(expert0.named_buffers())[name]
                m1 = dict(expert1.named_buffers())[name]
                buf.copy_(w0 * m0 + w1 * m1)
            elif 'running_var' in name:
                v0 = dict(expert0.named_buffers())[name]
                v1 = dict(expert1.named_buffers())[name]
                mean_name = name.replace('running_var', 'running_mean')
                m0 = dict(expert0.named_buffers())[mean_name]
                m1 = dict(expert1.named_buffers())[mean_name]
                mu_fused = w0 * m0 + w1 * m1
                buf.copy_(w0 * (v0 + (m0 - mu_fused)**2) + w1 * (v1 + (m1 - mu_fused)**2))
            elif 'num_batches_tracked' in name:
                buf.copy_(dict(expert0.named_buffers())[name])

# SCTS & Prior-Guided Routing Helper
def compute_batch_distance(expert, x, prototypes):
    expert.eval()
    with torch.no_grad():
        feats = expert.extract_features(x)  # [B, 128]
        dists = []
        for i in range(feats.shape[0]):
            sample_feat = feats[i]
            sample_dists = [torch.sum((sample_feat - prototypes[c])**2) for c in range(10)]
            dists.append(min(sample_dists))
        return torch.stack(dists).mean().item()

# Compute Empirical Fisher Offline
def compute_offline_fisher(model, dataset, num_samples=100):
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad ** 2
        count += 1
        if count >= num_samples:
            break
    for name in fisher:
        fisher[name] /= count
    return fisher

def get_block_fisher(fisher_dict):
    blocks = ['conv1', 'bn1', 'conv2', 'bn2', 'fc1', 'fc2']
    block_fisher = {}
    for b in blocks:
        vals = []
        for name in fisher_dict:
            if name.startswith(b):
                vals.append(fisher_dict[name].mean().item())
        block_fisher[b] = sum(vals) / len(vals) if vals else 1.0
    return block_fisher

def main():
    # 1. Dataset Downloads
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Downloading datasets...")
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

    expert0 = SimpleCNN().to(device)
    expert1 = SimpleCNN().to(device)

    # 2. Train Experts (2 epochs, 10,000 samples)
    if os.path.exists('mnist_expert.pt') and os.path.exists('fashion_expert.pt'):
        print("Loading pre-trained experts...")
        expert0.load_state_dict(torch.load('mnist_expert.pt', map_location=device))
        expert1.load_state_dict(torch.load('fashion_expert.pt', map_location=device))
    else:
        print("Training experts...")
        # Subset of 10,000 samples
        idx0 = torch.randperm(len(mnist_train))[:10000]
        idx1 = torch.randperm(len(fmnist_train))[:10000]
        sub0 = Subset(mnist_train, idx0)
        sub1 = Subset(fmnist_train, idx1)

        loader0 = DataLoader(sub0, batch_size=64, shuffle=True)
        loader1 = DataLoader(sub1, batch_size=64, shuffle=True)

        for i, (loader, model, name) in enumerate([(loader0, expert0, 'mnist_expert.pt'), (loader1, expert1, 'fashion_expert.pt')]):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            model.train()
            print(f"Training {name}...")
            for epoch in range(2):
                correct = 0
                total = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    out = model(x)
                    loss = F.cross_entropy(out, y)
                    loss.backward()
                    optimizer.step()
                    correct += (out.argmax(dim=-1) == y).sum().item()
                    total += y.size(0)
                print(f"Epoch {epoch+1} Accuracy: {correct/total*100:.2f}%")
            torch.save(model.state_dict(), name)

    expert0.eval()
    expert1.eval()

    # 3. Compute class prototypes (500 samples per expert)
    print("Computing offline prototypes...")
    def compute_prototypes(expert, dataset):
        class_samples = {c: [] for c in range(10)}
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        count = 0
        for x, y in loader:
            y_val = y.item()
            if len(class_samples[y_val]) < 50:
                class_samples[y_val].append(x)
                count += 1
            if count >= 500:
                break
        prototypes = {}
        expert.eval()
        with torch.no_grad():
            for c in range(10):
                if class_samples[c]:
                    inputs = torch.cat(class_samples[c], dim=0).to(device)
                    feats = expert.extract_features(inputs)
                    prototypes[c] = feats.mean(dim=0)
                else:
                    prototypes[c] = torch.zeros(128, device=device)
        return prototypes

    prototypes0 = compute_prototypes(expert0, mnist_train)
    prototypes1 = compute_prototypes(expert1, fmnist_train)

    # Compute offline Fisher Information for CL W-Fisher
    print("Computing offline Fisher information...")
    sub_mnist_cal = Subset(mnist_train, torch.randperm(len(mnist_train))[:100])
    sub_fmnist_cal = Subset(fmnist_train, torch.randperm(len(fmnist_train))[:100])
    fisher0 = compute_offline_fisher(expert0, sub_mnist_cal)
    fisher1 = compute_offline_fisher(expert1, sub_fmnist_cal)
    block_fisher0 = get_block_fisher(fisher0)
    block_fisher1 = get_block_fisher(fisher1)
    joint_fisher = {b: block_fisher0[b] + block_fisher1[b] for b in block_fisher0}

    # 4. Define Test Stream
    # 50 batches of size 64:
    # 0-9: Clean MNIST, 10-19: Noisy MNIST, 20-29: Clean Fashion, 30-39: Noisy Fashion, 40-49: Novel KMNIST
    print("Preparing test stream...")
    loader_mnist = DataLoader(mnist_test, batch_size=64, shuffle=True)
    loader_fmnist = DataLoader(fmnist_test, batch_size=64, shuffle=True)
    loader_kmnist = DataLoader(kmnist_test, batch_size=64, shuffle=True)

    iter_mnist = iter(loader_mnist)
    iter_fmnist = iter(loader_fmnist)
    iter_kmnist = iter(loader_kmnist)

    stream_batches = []
    stream_labels = []
    for t in range(50):
        if t < 10:
            x, y = next(iter_mnist)
        elif t < 20:
            x, y = next(iter_mnist)
            x = x + torch.randn_like(x) * 0.6
        elif t < 30:
            x, y = next(iter_fmnist)
        elif t < 40:
            x, y = next(iter_fmnist)
            x = x + torch.randn_like(x) * 0.6
        else:
            x, y = next(iter_kmnist)
        stream_batches.append(x)
        stream_labels.append(y)

    # 5. Methods Evaluation
    methods = [
        "MNIST Expert Only",
        "Fashion-MNIST Expert Only",
        "Uniform Merging",
        "AdaMerging",
        "DF-Bayes-TTMM",
        "CL W-Fisher",
        "KT-Fisher",
        "KP-SCTS-Bayes-TTMM (Ours)",
        "Ours (No SCTS)",
        "Ours (No Soft BN)",
        "Ours (No Kronecker)",
        "Ours (s = 500)",
        "Ours (s = 2000)",
        "Ours (beta = 0.1)",
        "Ours (beta = 1.0)"
    ]

    results_acc = {m: [] for m in methods}
    results_lambda = {m: [] for m in methods}

    blocks = ['conv1', 'bn1', 'conv2', 'bn2', 'fc1', 'fc2']

    for method in methods:
        print(f"\nEvaluating method: {method}")
        # Initialize a fresh model structure for evaluation
        merged_model = SimpleCNN().to(device)

        for t, (x, y) in enumerate(zip(stream_batches, stream_labels)):
            x, y = x.to(device), y.to(device)

            # Define variables and adaptation behaviors
            if method == "MNIST Expert Only":
                # lambdas = [1, 1, 1, 1, 1, 1]
                lambdas = [1.0] * 6
                merge_bn_buffers(merged_model, expert0, expert1, torch.tensor([1.0, 0.0], device=device))
                # Merge weights
                with torch.no_grad():
                    for name, param in merged_model.named_parameters():
                        p0 = dict(expert0.named_parameters())[name]
                        param.copy_(p0)
                merged_model.eval()
                with torch.no_grad():
                    logits = merged_model(x)
                acc = (logits.argmax(dim=-1) == y).sum().item() / y.size(0)

            elif method == "Fashion-MNIST Expert Only":
                lambdas = [0.0] * 6
                merge_bn_buffers(merged_model, expert0, expert1, torch.tensor([0.0, 1.0], device=device))
                with torch.no_grad():
                    for name, param in merged_model.named_parameters():
                        p1 = dict(expert1.named_parameters())[name]
                        param.copy_(p1)
                merged_model.eval()
                with torch.no_grad():
                    logits = merged_model(x)
                acc = (logits.argmax(dim=-1) == y).sum().item() / y.size(0)

            elif method == "Uniform Merging":
                lambdas = [0.5] * 6
                merge_bn_buffers(merged_model, expert0, expert1, torch.tensor([0.5, 0.5], device=device))
                with torch.no_grad():
                    for name, param in merged_model.named_parameters():
                        p0 = dict(expert0.named_parameters())[name]
                        p1 = dict(expert1.named_parameters())[name]
                        param.copy_(0.5 * p0 + 0.5 * p1)
                merged_model.eval()
                with torch.no_grad():
                    logits = merged_model(x)
                acc = (logits.argmax(dim=-1) == y).sum().item() / y.size(0)

            elif method == "AdaMerging":
                # Standard unconstrained entropy minimization
                logits_p = nn.Parameter(torch.zeros(6, device=device))  # sigmoid(0) = 0.5
                optimizer = torch.optim.Adam([logits_p], lr=0.1)

                for step in range(3):
                    optimizer.zero_grad()
                    lambdas = torch.sigmoid(logits_p)
                    # Merge BN buffers
                    mean_lambda = lambdas.mean()
                    merge_bn_buffers(merged_model, expert0, expert1, torch.stack([mean_lambda, 1 - mean_lambda]))

                    # Merge weights dynamically
                    merged_params = {}
                    for name, param in merged_model.named_parameters():
                        block_idx = None
                        for i, block in enumerate(blocks):
                            if name.startswith(block):
                                block_idx = i
                                break
                        if block_idx is not None:
                            l = lambdas[block_idx]
                            p0 = dict(expert0.named_parameters())[name]
                            p1 = dict(expert1.named_parameters())[name]
                            merged_params[name] = l * p0 + (1 - l) * p1
                        else:
                            merged_params[name] = param

                    outputs = functional_call(merged_model, merged_params, x)
                    loss = - (F.softmax(outputs, dim=-1) * F.log_softmax(outputs, dim=-1)).sum(dim=-1).mean()
                    loss.backward()
                    optimizer.step()

                # Inference
                lambdas = torch.sigmoid(logits_p).detach().tolist()
                mean_lambda = sum(lambdas) / len(lambdas)
                merge_bn_buffers(merged_model, expert0, expert1, torch.tensor([mean_lambda, 1 - mean_lambda], device=device))
                with torch.no_grad():
                    merged_params = {}
                    for name, param in merged_model.named_parameters():
                        block_idx = None
                        for i, block in enumerate(blocks):
                            if name.startswith(block):
                                block_idx = i
                                break
                        if block_idx is not None:
                            l = lambdas[block_idx]
                            p0 = dict(expert0.named_parameters())[name]
                            p1 = dict(expert1.named_parameters())[name]
                            merged_params[name] = l * p0 + (1 - l) * p1
                        else:
                            merged_params[name] = param
                    logits = functional_call(merged_model, merged_params, x)
                acc = (logits.argmax(dim=-1) == y).sum().item() / y.size(0)

            elif method == "DF-Bayes-TTMM":
                # Evaluate expert prediction entropies
                expert0.eval()
                expert1.eval()
                with torch.no_grad():
                    out0 = expert0(x)
                    out1 = expert1(x)
                    h0 = - (F.softmax(out0, dim=-1) * F.log_softmax(out0, dim=-1)).sum(dim=-1).mean().item()
                    h1 = - (F.softmax(out1, dim=-1) * F.log_softmax(out1, dim=-1)).sum(dim=-1).mean().item()

                mean_h = (h0 + h1) / 2.0
                tau_n = 1.2
                if mean_h > tau_n:
                    w = torch.tensor([0.5, 0.5], device=device)
                else:
                    gamma = 2.0
                    w0 = np.exp(-gamma * h0)
                    w1 = np.exp(-gamma * h1)
                    s_w = w0 + w1
                    w = torch.tensor([w0 / s_w, w1 / s_w], device=device)

                # Initialize merging logits
                logits_layer = nn.Parameter(torch.log(w + 1e-6).unsqueeze(0).repeat(6, 1))  # [6, 2]
                logits_global = nn.Parameter(torch.log(w + 1e-6))  # [2]
                optimizer = torch.optim.Adam([logits_layer, logits_global], lr=0.1)

                # Estimate offline/online Fisher proxy (for gradient scaling)
                fisher_scale = {b: joint_fisher[b] for b in joint_fisher}

                for step in range(3):
                    optimizer.zero_grad()
                    w_layer = F.softmax(logits_layer, dim=-1)  # [6, 2]
                    w_global = F.softmax(logits_global, dim=-1)  # [2]

                    # Soft BN Fusion
                    merge_bn_buffers(merged_model, expert0, expert1, w_global)

                    # Merge weights
                    merged_params = {}
                    for name, param in merged_model.named_parameters():
                        block_idx = None
                        for i, block in enumerate(blocks):
                            if name.startswith(block):
                                block_idx = i
                                break
                        if block_idx is not None:
                            wl = w_layer[block_idx]
                            p0 = dict(expert0.named_parameters())[name]
                            p1 = dict(expert1.named_parameters())[name]
                            merged_params[name] = wl[0] * p0 + wl[1] * p1
                        else:
                            merged_params[name] = param

                    outputs = functional_call(merged_model, merged_params, x)
                    loss_entropy = - (F.softmax(outputs, dim=-1) * F.log_softmax(outputs, dim=-1)).sum(dim=-1).mean()
                    # MAP Loss with beta = 0.5
                    loss = loss_entropy + 0.25 * torch.sum((w_layer - w)**2) + 0.25 * torch.sum((w_global - w)**2)
                    loss.backward()

                    # Scale gradients inversely by Fisher sensitivities
                    with torch.no_grad():
                        for b_idx, b in enumerate(blocks):
                            f_val = fisher_scale[b]
                            if logits_layer.grad is not None:
                                logits_layer.grad[b_idx] /= (f_val + 1e-4)

                    optimizer.step()

                # Inference
                w_layer = F.softmax(logits_layer, dim=-1).detach()
                w_global = F.softmax(logits_global, dim=-1).detach()
                lambdas = w_layer[:, 0].tolist()
                merge_bn_buffers(merged_model, expert0, expert1, w_global)
                with torch.no_grad():
                    merged_params = {}
                    for name, param in merged_model.named_parameters():
                        block_idx = None
                        for i, block in enumerate(blocks):
                            if name.startswith(block):
                                block_idx = i
                                break
                        if block_idx is not None:
                            wl = w_layer[block_idx]
                            p0 = dict(expert0.named_parameters())[name]
                            p1 = dict(expert1.named_parameters())[name]
                            merged_params[name] = wl[0] * p0 + wl[1] * p1
                        else:
                            merged_params[name] = param
                    logits = functional_call(merged_model, merged_params, x)
                acc = (logits.argmax(dim=-1) == y).sum().item() / y.size(0)

            elif method == "CL W-Fisher":
                # Compute prototype distance gap
                d0 = compute_batch_distance(expert0, x, prototypes0)
                d1 = compute_batch_distance(expert1, x, prototypes1)
                delta = abs(d1 - d0)
                # SCTS: dynamic temperature
                tau = delta / 1000.0 + 1e-4
                # Routing prior
                s0 = -d0
                s1 = -d1
                max_s = max(s0, s1)
                w0 = np.exp((s0 - max_s) / tau)
                w1 = np.exp((s1 - max_s) / tau)
                sum_w = w0 + w1
                p_routing = max(1e-4, min(1 - 1e-4, w0 / sum_w))

                # PG-Init
                w_global = nn.Parameter(torch.tensor(np.log(p_routing / (1 - p_routing)), device=device))
                delta_layers = nn.Parameter(torch.zeros(6, device=device))
                optimizer = torch.optim.Adam([w_global, delta_layers], lr=0.1)

                # Offline precomputed Fisher sensitivities
                fisher_scale = {b: joint_fisher[b] for b in joint_fisher}

                for step in range(3):
                    optimizer.zero_grad()
                    lambdas = torch.sigmoid(w_global + delta_layers)
                    mean_lambda = lambdas.mean()
                    merge_bn_buffers(merged_model, expert0, expert1, torch.stack([mean_lambda, 1 - mean_lambda]))

                    # Merge weights
                    merged_params = {}
                    for name, param in merged_model.named_parameters():
                        block_idx = None
                        for i, block in enumerate(blocks):
                            if name.startswith(block):
                                block_idx = i
                                break
                        if block_idx is not None:
                            l = lambdas[block_idx]
                            p0 = dict(expert0.named_parameters())[name]
                            p1 = dict(expert1.named_parameters())[name]
                            merged_params[name] = l * p0 + (1 - l) * p1
                        else:
                            merged_params[name] = param

                    outputs = functional_call(merged_model, merged_params, x)
                    loss_entropy = - (F.softmax(outputs, dim=-1) * F.log_softmax(outputs, dim=-1)).sum(dim=-1).mean()
                    # Regularization with beta = 0.5, gamma = 0.1
                    loss = loss_entropy + 0.5 * ((mean_lambda - p_routing)**2) + 0.1 * torch.sum(delta_layers**2)
                    loss.backward()

                    # Preconditioned co-acting update
                    with torch.no_grad():
                        for b_idx, b in enumerate(blocks):
                            f_val = fisher_scale[b]
                            if delta_layers.grad is not None:
                                delta_layers.grad[b_idx] /= (f_val + 1e-4)

                    optimizer.step()

                # Inference
                lambdas = torch.sigmoid(w_global + delta_layers).detach()
                mean_lambda = lambdas.mean().item()
                lambdas = lambdas.tolist()
                merge_bn_buffers(merged_model, expert0, expert1, torch.tensor([mean_lambda, 1 - mean_lambda], device=device))
                with torch.no_grad():
                    merged_params = {}
                    for name, param in merged_model.named_parameters():
                        block_idx = None
                        for i, block in enumerate(blocks):
                            if name.startswith(block):
                                block_idx = i
                                break
                        if block_idx is not None:
                            l = lambdas[block_idx]
                            p0 = dict(expert0.named_parameters())[name]
                            p1 = dict(expert1.named_parameters())[name]
                            merged_params[name] = l * p0 + (1 - l) * p1
                        else:
                            merged_params[name] = param
                    logits = functional_call(merged_model, merged_params, x)
                acc = (logits.argmax(dim=-1) == y).sum().item() / y.size(0)

            elif method == "KT-Fisher":
                # Static Unified Space Cohesion Routing
                d0 = compute_batch_distance(expert0, x, prototypes0)
                d1 = compute_batch_distance(expert1, x, prototypes1)
                k_star = 0 if d0 < d1 else 1
                tau_n = 5.0  # Novelty threshold

                if min(d0, d1) > tau_n:
                    # Novel detected: entropy target routing
                    expert0.eval()
                    expert1.eval()
                    with torch.no_grad():
                        out0 = expert0(x)
                        out1 = expert1(x)
                        h0 = - (F.softmax(out0, dim=-1) * F.log_softmax(out0, dim=-1)).sum(dim=-1).mean().item()
                        h1 = - (F.softmax(out1, dim=-1) * F.log_softmax(out1, dim=-1)).sum(dim=-1).mean().item()
                    y_target = 0 if h0 < h1 else 1
                else:
                    y_target = k_star

                # Track activation and gradient traces
                activation_norms = {}
                gradient_norms = {}

                def f_hook(name):
                    def hook(module, inp, out):
                        act = inp[0].detach()
                        if isinstance(module, nn.Conv2d):
                            b_sz, c, h, w = act.shape
                            activation_norms[name] = (act**2).sum() / (b_sz * h * w)
                        else:
                            b_sz = act.shape[0]
                            activation_norms[name] = (act**2).sum() / b_sz
                    return hook

                def b_hook(name):
                    def hook(module, g_inp, g_out):
                        grad = g_out[0].detach()
                        if isinstance(module, nn.Conv2d):
                            b_sz, c, h, w = grad.shape
                            gradient_norms[name] = (grad**2).sum() / (b_sz * h * w)
                        else:
                            b_sz = grad.shape[0]
                            gradient_norms[name] = (grad**2).sum() / b_sz
                    return hook

                hooks = []
                for name, module in merged_model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        h_f = module.register_forward_hook(f_hook(name))
                        h_b = module.register_full_backward_hook(b_hook(name))
                        hooks.extend([h_f, h_b])

                # Run backward to estimate Kronecker sensitivities
                logits_p = nn.Parameter(torch.zeros(6, device=device))
                lambdas = torch.sigmoid(logits_p)
                mean_lambda = lambdas.mean()
                merge_bn_buffers(merged_model, expert0, expert1, torch.stack([mean_lambda, 1 - mean_lambda]))

                merged_params = {}
                for name, param in merged_model.named_parameters():
                    block_idx = None
                    for i, block in enumerate(blocks):
                        if name.startswith(block):
                            block_idx = i
                            break
                    if block_idx is not None:
                        l = lambdas[block_idx]
                        p0 = dict(expert0.named_parameters())[name]
                        p1 = dict(expert1.named_parameters())[name]
                        merged_params[name] = l * p0 + (1 - l) * p1
                    else:
                        merged_params[name] = param

                # Forward pass for backward activation tracking
                outputs = functional_call(merged_model, merged_params, x)
                loss_entropy = - (F.softmax(outputs, dim=-1) * F.log_softmax(outputs, dim=-1)).sum(dim=-1).mean()
                loss_entropy.backward()

                # Clean up hooks
                for hook in hooks:
                    hook.remove()

                # Calculate sensitivities
                sensitivities = {}
                for b in blocks:
                    # find matching hooked module
                    act_val, grad_val, num_p = 1.0, 1.0, 1.0
                    for name, module in merged_model.named_modules():
                        if name.startswith(b) and isinstance(module, (nn.Conv2d, nn.Linear)):
                            act_val = activation_norms.get(name, torch.tensor(1.0)).item()
                            grad_val = gradient_norms.get(name, torch.tensor(1.0)).item()
                            num_p = module.weight.numel()
                            break
                    sensitivities[b] = (act_val * grad_val) / num_p

                # Adapt parameters with preconditioned LR
                for step in range(3):
                    # update lambdas via simple step
                    with torch.no_grad():
                        for b_idx, b in enumerate(blocks):
                            lr_b = 0.1 * (sensitivities[b] + 1e-4)**(-0.5)
                            # Gradient towards y_target
                            g_dir = logits_p[b_idx].item() - (1.0 if y_target == 0 else 0.0)
                            logits_p[b_idx] -= lr_b * g_dir

                # Inference
                lambdas = torch.sigmoid(logits_p).detach()
                mean_lambda = lambdas.mean().item()
                lambdas = lambdas.tolist()
                merge_bn_buffers(merged_model, expert0, expert1, torch.tensor([mean_lambda, 1 - mean_lambda], device=device))
                with torch.no_grad():
                    merged_params = {}
                    for name, param in merged_model.named_parameters():
                        block_idx = None
                        for i, block in enumerate(blocks):
                            if name.startswith(block):
                                block_idx = i
                                break
                        if block_idx is not None:
                            l = lambdas[block_idx]
                            p0 = dict(expert0.named_parameters())[name]
                            p1 = dict(expert1.named_parameters())[name]
                            merged_params[name] = l * p0 + (1 - l) * p1
                        else:
                            merged_params[name] = param
                    logits = functional_call(merged_model, merged_params, x)
                acc = (logits.argmax(dim=-1) == y).sum().item() / y.size(0)

            elif method == "KP-SCTS-Bayes-TTMM (Ours)" or method.startswith("Ours"):
                # Retrieve customization params
                no_scts = "No SCTS" in method
                no_soft_bn = "No Soft BN" in method
                no_kronecker = "No Kronecker" in method
                
                s_val = 1000.0
                if "s = 500" in method:
                    s_val = 500.0
                elif "s = 2000" in method:
                    s_val = 2000.0
                    
                beta_val = 0.5
                if "beta = 0.1" in method:
                    beta_val = 0.1
                elif "beta = 1.0" in method:
                    beta_val = 1.0

                # 1. Novelty/Prior Guided Routing using SCTS + Bayes
                expert0.eval()
                expert1.eval()
                with torch.no_grad():
                    out0 = expert0(x)
                    out1 = expert1(x)
                    h0 = - (F.softmax(out0, dim=-1) * F.log_softmax(out0, dim=-1)).sum(dim=-1).mean().item()
                    h1 = - (F.softmax(out1, dim=-1) * F.log_softmax(out1, dim=-1)).sum(dim=-1).mean().item()

                mean_h = (h0 + h1) / 2.0
                tau_n = 1.2

                if mean_h > tau_n:
                    w = torch.tensor([0.5, 0.5], device=device)
                else:
                    if no_scts:
                        # Fixed temperature routing prior
                        tau = 1.2
                        d0 = compute_batch_distance(expert0, x, prototypes0)
                        d1 = compute_batch_distance(expert1, x, prototypes1)
                        s0 = -d0
                        s1 = -d1
                        max_s = max(s0, s1)
                        w0 = np.exp((s0 - max_s) / tau)
                        w1 = np.exp((s1 - max_s) / tau)
                        sum_w = w0 + w1
                        w = torch.tensor([w0 / sum_w, w1 / sum_w], device=device)
                    else:
                        # SCTS Prototype distance gap routing
                        d0 = compute_batch_distance(expert0, x, prototypes0)
                        d1 = compute_batch_distance(expert1, x, prototypes1)
                        delta = abs(d1 - d0)
                        tau = delta / s_val + 1e-4
                        s0 = -d0
                        s1 = -d1
                        max_s = max(s0, s1)
                        w0 = np.exp((s0 - max_s) / tau)
                        w1 = np.exp((s1 - max_s) / tau)
                        sum_w = w0 + w1
                        w = torch.tensor([w0 / sum_w, w1 / sum_w], device=device)

                # 2. Track activation and gradient traces (Kronecker Trace)
                activation_norms = {}
                gradient_norms = {}

                def f_hook(name):
                    def hook(module, inp, out):
                        act = inp[0].detach()
                        if isinstance(module, nn.Conv2d):
                            b_sz, c, h, w_sz = act.shape
                            activation_norms[name] = (act**2).sum() / (b_sz * h * w_sz)
                        else:
                            b_sz = act.shape[0]
                            activation_norms[name] = (act**2).sum() / b_sz
                    return hook

                def b_hook(name):
                    def hook(module, g_inp, g_out):
                        grad = g_out[0].detach()
                        if isinstance(module, nn.Conv2d):
                            b_sz, c, h, w_sz = grad.shape
                            gradient_norms[name] = (grad**2).sum() / (b_sz * h * w_sz)
                        else:
                            b_sz = grad.shape[0]
                            gradient_norms[name] = (grad**2).sum() / b_sz
                    return hook

                hooks = []
                for name, module in merged_model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        h_f = module.register_forward_hook(f_hook(name))
                        h_b = module.register_full_backward_hook(b_hook(name))
                        hooks.extend([h_f, h_b])

                # Get Kronecker sensitivities from standard forward/backward pass on current batch
                logits_layer = nn.Parameter(torch.log(w + 1e-6).unsqueeze(0).repeat(6, 1))  # [6, 2]
                logits_global = nn.Parameter(torch.log(w + 1e-6))  # [2]

                w_layer = F.softmax(logits_layer, dim=-1)
                w_global = F.softmax(logits_global, dim=-1)

                if no_soft_bn:
                    merge_bn_buffers(merged_model, expert0, expert1, torch.tensor([0.5, 0.5], device=device))
                else:
                    merge_bn_buffers(merged_model, expert0, expert1, w_global)

                merged_params = {}
                for name, param in merged_model.named_parameters():
                    block_idx = None
                    for i, block in enumerate(blocks):
                        if name.startswith(block):
                            block_idx = i
                            break
                    if block_idx is not None:
                        wl = w_layer[block_idx]
                        p0 = dict(expert0.named_parameters())[name]
                        p1 = dict(expert1.named_parameters())[name]
                        merged_params[name] = wl[0] * p0 + wl[1] * p1
                    else:
                        merged_params[name] = param

                outputs = functional_call(merged_model, merged_params, x)
                loss_entropy = - (F.softmax(outputs, dim=-1) * F.log_softmax(outputs, dim=-1)).sum(dim=-1).mean()
                loss_entropy.backward()

                for hook in hooks:
                    hook.remove()

                # Calculate sensitivities
                sensitivities = {}
                for b in blocks:
                    act_val, grad_val, num_p = 1.0, 1.0, 1.0
                    for name, module in merged_model.named_modules():
                        if name.startswith(b) and isinstance(module, (nn.Conv2d, nn.Linear)):
                            act_val = activation_norms.get(name, torch.tensor(1.0)).item()
                            grad_val = gradient_norms.get(name, torch.tensor(1.0)).item()
                            num_p = module.weight.numel()
                            break
                    sensitivities[b] = (act_val * grad_val) / num_p

                # 3. Perform MAP Optimization preconditioned by Kronecker Sensitivities
                optimizer = torch.optim.Adam([logits_layer, logits_global], lr=0.1)

                for step in range(3):
                    optimizer.zero_grad()
                    w_layer = F.softmax(logits_layer, dim=-1)
                    w_global = F.softmax(logits_global, dim=-1)

                    if no_soft_bn:
                        merge_bn_buffers(merged_model, expert0, expert1, torch.tensor([0.5, 0.5], device=device))
                    else:
                        merge_bn_buffers(merged_model, expert0, expert1, w_global)

                    merged_params = {}
                    for name, param in merged_model.named_parameters():
                        block_idx = None
                        for i, block in enumerate(blocks):
                            if name.startswith(block):
                                block_idx = i
                                break
                        if block_idx is not None:
                            wl = w_layer[block_idx]
                            p0 = dict(expert0.named_parameters())[name]
                            p1 = dict(expert1.named_parameters())[name]
                            merged_params[name] = wl[0] * p0 + wl[1] * p1
                        else:
                            merged_params[name] = param

                    outputs = functional_call(merged_model, merged_params, x)
                    loss_entropy = - (F.softmax(outputs, dim=-1) * F.log_softmax(outputs, dim=-1)).sum(dim=-1).mean()
                    # MAP Loss with beta parameter
                    loss = loss_entropy + 0.5 * beta_val * torch.sum((w_layer - w)**2) + 0.5 * beta_val * torch.sum((w_global - w)**2)
                    loss.backward()

                    # Scale gradients inversely by dynamic Kronecker sensitivities
                    if not no_kronecker:
                        with torch.no_grad():
                            for b_idx, b in enumerate(blocks):
                                f_val = sensitivities[b]
                                if logits_layer.grad is not None:
                                    logits_layer.grad[b_idx] /= (f_val + 1e-4)

                    optimizer.step()

                # Inference
                w_layer = F.softmax(logits_layer, dim=-1).detach()
                w_global = F.softmax(logits_global, dim=-1).detach()
                lambdas = w_layer[:, 0].tolist()

                if no_soft_bn:
                    merge_bn_buffers(merged_model, expert0, expert1, torch.tensor([0.5, 0.5], device=device))
                else:
                    merge_bn_buffers(merged_model, expert0, expert1, w_global)

                with torch.no_grad():
                    merged_params = {}
                    for name, param in merged_model.named_parameters():
                        block_idx = None
                        for i, block in enumerate(blocks):
                            if name.startswith(block):
                                block_idx = i
                                break
                        if block_idx is not None:
                            wl = w_layer[block_idx]
                            p0 = dict(expert0.named_parameters())[name]
                            p1 = dict(expert1.named_parameters())[name]
                            merged_params[name] = wl[0] * p0 + wl[1] * p1
                        else:
                            merged_params[name] = param
                    logits = functional_call(merged_model, merged_params, x)
                acc = (logits.argmax(dim=-1) == y).sum().item() / y.size(0)

            results_acc[method].append(acc)
            results_lambda[method].append(lambdas[0] if isinstance(lambdas, list) else lambdas)

            if t % 10 == 0:
                print(f"Batch {t}/50 | Accuracy: {acc*100:.2f}% | Lambda_0: {results_lambda[method][-1]:.4f}")

    # 6. Aggregate results per segment
    segments = {
        "Clean MNIST (0-9)": (0, 10),
        "Noisy MNIST (10-19)": (10, 20),
        "Clean Fashion (20-29)": (20, 30),
        "Noisy Fashion (30-39)": (30, 40),
        "Novel KMNIST (40-49)": (40, 50)
    }

    report = {}
    print("\n" + "="*50 + "\nSUMMARY METRICS PER SEGMENT\n" + "="*50)
    for seg_name, (start, end) in segments.items():
        print(f"\n--- {seg_name} ---")
        report[seg_name] = {}
        for method in methods:
            mean_acc = np.mean(results_acc[method][start:end]) * 100
            mean_lam = np.mean(results_lambda[method][start:end])
            print(f"{method:30s} | Accuracy: {mean_acc:6.2f}% | Mean Lambda_0: {mean_lam:.4f}")
            report[seg_name][method] = {
                "accuracy": float(mean_acc),
                "mean_lambda": float(mean_lam)
            }

    # Save results to JSON
    with open('results_metrics.json', 'w') as f:
        json.dump(report, f, indent=4)

    # 7. Generate beautiful plots
    plot_methods = [
        "MNIST Expert Only",
        "Fashion-MNIST Expert Only",
        "Uniform Merging",
        "AdaMerging",
        "DF-Bayes-TTMM",
        "CL W-Fisher",
        "KT-Fisher",
        "KP-SCTS-Bayes-TTMM (Ours)"
    ]
    plt.figure(figsize=(12, 6))
    for method in plot_methods:
        # Smooth with moving average of 3
        accs = np.array(results_acc[method])
        plt.plot(accs, label=method, alpha=0.8)
    plt.title("Test-Time Model Merging Accuracy Across the Stream")
    plt.xlabel("Batch index")
    plt.ylabel("Accuracy")
    plt.axvline(10, color='gray', linestyle='--')
    plt.axvline(20, color='gray', linestyle='--')
    plt.axvline(30, color='gray', linestyle='--')
    plt.axvline(40, color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('accuracy_trajectory.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    for method in ["AdaMerging", "DF-Bayes-TTMM", "CL W-Fisher", "KT-Fisher", "KP-SCTS-Bayes-TTMM (Ours)"]:
        plt.plot(results_lambda[method], label=method, alpha=0.8)
    plt.title("Trajectory of Merging Coefficient (Lambda_0) Across Stream")
    plt.xlabel("Batch index")
    plt.ylabel("Lambda_0 (MNIST weight)")
    plt.axvline(10, color='gray', linestyle='--')
    plt.axvline(20, color='gray', linestyle='--')
    plt.axvline(30, color='gray', linestyle='--')
    plt.axvline(40, color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lambda_trajectory.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
