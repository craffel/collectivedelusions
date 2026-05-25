import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from train_experts import SimpleCNN

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Spatial noise estimator using a 3x3 Laplacian filter
def estimate_noise_level(images):
    # images shape: (B, 1, 28, 28)
    kernel = torch.tensor([[-1., -1., -1.],
                           [-1.,  8., -1.],
                           [-1., -1., -1.]], device=images.device).view(1, 1, 3, 3)
    kernel = kernel / 8.0  # Normalize
    residual = F.conv2d(images, kernel, padding=1)
    std = residual.std().item()
    return std

# Helper to merge expert weights in-place or return a merged state dict
def merge_weights(model0_state, model1_state, lambda_dict):
    merged_state = {}
    for k in model0_state.keys():
        if k in lambda_dict:
            lam = lambda_dict[k]
            # Handle possible shape broadcast
            merged_state[k] = lam * model0_state[k] + (1.0 - lam) * model1_state[k]
        else:
            # For non-merged or non-trainable buffers (like BN running mean/var, which we fuse separately)
            merged_state[k] = 0.5 * model0_state[k] + 0.5 * model1_state[k]
    return merged_state

# Fuse Batch Normalization statistics using moment matching
def fuse_bn_statistics(model, expert0, expert1, w0, w1):
    # w0, w1 are routing weights
    with torch.no_grad():
        for (name, module), (_, m0), (_, m1) in zip(model.named_modules(), expert0.named_modules(), expert1.named_modules()):
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                # Fuse running mean
                module.running_mean.copy_(w0 * m0.running_mean + w1 * m1.running_mean)
                # Fuse running variance
                # Var_fused = w0 * (Var0 + (mean0 - mean_fused)^2) + w1 * (Var1 + (mean1 - mean_fused)^2)
                mean_fused = module.running_mean
                var_fused = w0 * (m0.running_var + (m0.running_mean - mean_fused)**2) + \
                            w1 * (m1.running_var + (m1.running_mean - mean_fused)**2)
                module.running_var.copy_(var_fused)

# Compute offline Fisher information for CLW-Fisher
def compute_offline_fisher(expert, dataset, num_samples=256, device="cpu"):
    expert.eval()
    loader = DataLoader(Subset(dataset, range(num_samples)), batch_size=32, shuffle=False)
    fisher = {}
    for name, param in expert.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)
            
    for images, _ in loader:
        images = images.to(device)
        expert.zero_grad()
        outputs = expert(images)
        # Compute entropy as a proxy loss for fisher information
        probs = F.softmax(outputs, dim=-1)
        log_probs = F.log_softmax(outputs, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        entropy.backward()
        
        for name, param in expert.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += (param.grad.data ** 2) * len(images)
                
    for k in fisher.keys():
        fisher[k] /= num_samples
        fisher[k] = torch.clamp(fisher[k], min=1e-5) # stability floor
    return fisher

# Generate the non-stationary stream of 50 batches
def generate_stream(device="cpu"):
    set_seed(42)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    stream_batches = []
    
    # Batch size 64, 10 batches per segment = 640 images per segment
    # Segment 1: Clean MNIST (batches 0-9)
    mnist_loader_clean = DataLoader(Subset(mnist_test, range(0, 640)), batch_size=64, shuffle=False)
    for imgs, labels in mnist_loader_clean:
        stream_batches.append((imgs.to(device), labels.to(device), "Clean MNIST"))
        
    # Segment 2: Noisy MNIST (batches 10-19)
    mnist_loader_noisy = DataLoader(Subset(mnist_test, range(640, 1280)), batch_size=64, shuffle=False)
    for imgs, labels in mnist_loader_noisy:
        noisy_imgs = imgs + torch.randn_like(imgs) * 0.6
        stream_batches.append((noisy_imgs.to(device), labels.to(device), "Noisy MNIST"))
        
    # Segment 3: Clean FashionMNIST (batches 20-29)
    fashion_loader_clean = DataLoader(Subset(fashion_test, range(0, 640)), batch_size=64, shuffle=False)
    for imgs, labels in fashion_loader_clean:
        stream_batches.append((imgs.to(device), labels.to(device), "Clean Fashion"))
        
    # Segment 4: Noisy FashionMNIST (batches 30-39)
    fashion_loader_noisy = DataLoader(Subset(fashion_test, range(640, 1280)), batch_size=64, shuffle=False)
    for imgs, labels in fashion_loader_noisy:
        noisy_imgs = imgs + torch.randn_like(imgs) * 0.6
        stream_batches.append((noisy_imgs.to(device), labels.to(device), "Noisy Fashion"))
        
    # Segment 5: Novel KMNIST (batches 40-49)
    kmnist_loader = DataLoader(Subset(kmnist_test, range(0, 640)), batch_size=64, shuffle=False)
    for imgs, labels in kmnist_loader:
        stream_batches.append((imgs.to(device), labels.to(device), "Novel KMNIST"))
        
    return stream_batches

# Evaluate a method over the stream and return segment-wise and overall accuracies
def run_evaluation(method_name, evaluate_fn, stream_batches, expert0, expert1, device="cpu"):
    print(f"\nEvaluating: {method_name}")
    accuracies = []
    segment_accs = {
        "Clean MNIST": [],
        "Noisy MNIST": [],
        "Clean Fashion": [],
        "Noisy Fashion": [],
        "Novel KMNIST": []
    }
    
    # Initialize/Reset any method state before stream starts
    evaluate_fn(None, None, None, expert0, expert1, reset=True, device=device)
    
    for idx, (images, labels, segment_name) in enumerate(stream_batches):
        correct, total = evaluate_fn(images, labels, idx, expert0, expert1, reset=False, device=device)
        acc = 100.0 * correct / total
        accuracies.append(acc)
        segment_accs[segment_name].append(acc)
        
    # Compute averages
    results = {}
    overall = []
    for seg, accs in segment_accs.items():
        avg_acc = np.mean(accs)
        results[seg] = avg_acc
        overall.append(avg_acc)
        print(f"Segment [{seg}]: {avg_acc:.2f}%")
    results["Overall"] = np.mean(overall)
    print(f"Overall Accuracy: {results['Overall']:.2f}%")
    return results

# Implement individual methods as evaluates
class Evaluators:
    def __init__(self):
        # We store running adaptation variables inside
        self.w_global = 0.0
        self.deltas = {}
        self.running_g2 = {}
        self.smoothed_gap = None
        self.offline_fisher0 = None
        self.offline_fisher1 = None
        
    def static_merging(self, images, labels, batch_idx, expert0, expert1, reset=False, device="cpu"):
        if reset:
            return None
        
        # Merge weights with fixed lambda = 0.5
        model = SimpleCNN(use_cosface=expert0.use_cosface).to(device)
        lambda_dict = {k: torch.tensor(0.5, device=device) for k, p in expert0.named_parameters() if p.requires_grad}
        merged_weights = merge_weights(expert0.state_dict(), expert1.state_dict(), lambda_dict)
        model.load_state_dict(merged_weights)
        fuse_bn_statistics(model, expert0, expert1, 0.5, 0.5)
        
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
        return correct, total

    def fixed_tta(self, images, labels, batch_idx, expert0, expert1, reset=False, device="cpu"):
        if reset:
            self.w_global = 0.0
            return None
            
        # Standard TTA: optimize w_global over 5 steps on the batch using entropy loss
        w_global_tensor = torch.tensor(self.w_global, device=device, requires_grad=True)
        optimizer = torch.optim.SGD([w_global_tensor], lr=0.05)
        
        # Test-Time Adaptation Loop
        for step in range(5):
            # Construct merged model
            model = SimpleCNN(use_cosface=expert0.use_cosface).to(device)
            lam = torch.sigmoid(w_global_tensor)
            lambda_dict = {k: lam for k, p in expert0.named_parameters() if p.requires_grad}
            merged_weights = merge_weights(expert0.state_dict(), expert1.state_dict(), lambda_dict)
            model.load_state_dict(merged_weights)
            fuse_bn_statistics(model, expert0, expert1, lam.item(), 1.0 - lam.item())
            
            # Predict
            outputs = model(images)
            probs = F.softmax(outputs, dim=-1)
            loss = -torch.sum(probs * F.log_softmax(outputs, dim=-1), dim=-1).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        self.w_global = w_global_tensor.item()
        
        # Final Evaluation
        with torch.no_grad():
            model = SimpleCNN(use_cosface=expert0.use_cosface).to(device)
            lam = torch.sigmoid(w_global_tensor)
            lambda_dict = {k: lam for k, p in expert0.named_parameters() if p.requires_grad}
            merged_weights = merge_weights(expert0.state_dict(), expert1.state_dict(), lambda_dict)
            model.load_state_dict(merged_weights)
            fuse_bn_statistics(model, expert0, expert1, lam.item(), 1.0 - lam.item())
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
        return correct, total

    def clw_fisher(self, images, labels, batch_idx, expert0, expert1, reset=False, device="cpu"):
        if reset:
            self.w_global = 0.0
            self.deltas = {k: torch.tensor(0.0, device=device, requires_grad=True) for k, p in expert0.named_parameters() if p.requires_grad}
            if self.offline_fisher0 is None:
                # Lazy compute
                mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
                fashion_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
                self.offline_fisher0 = compute_offline_fisher(expert0, mnist_train, device=device)
                self.offline_fisher1 = compute_offline_fisher(expert1, fashion_train, device=device)
            return None
            
        # Layer-wise Fisher-preconditioned adaptation
        w_global_tensor = torch.tensor(self.w_global, device=device, requires_grad=True)
        deltas = {k: torch.tensor(self.deltas[k].item(), device=device, requires_grad=True) for k in self.deltas.keys()}
        
        # We optimize w_global and layer offsets deltas
        optimizer_w = torch.optim.SGD([w_global_tensor], lr=0.05)
        
        for step in range(5):
            # Construct merged weights
            lambda_dict = {}
            for k in deltas.keys():
                lambda_dict[k] = torch.sigmoid(w_global_tensor + deltas[k])
                
            model = SimpleCNN(use_cosface=expert0.use_cosface).to(device)
            merged_weights = merge_weights(expert0.state_dict(), expert1.state_dict(), lambda_dict)
            model.load_state_dict(merged_weights)
            
            # Use mean lambda to fuse BN statistics
            mean_lam = torch.stack(list(lambda_dict.values())).mean().item()
            fuse_bn_statistics(model, expert0, expert1, mean_lam, 1.0 - mean_lam)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=-1)
            loss = -torch.sum(probs * F.log_softmax(outputs, dim=-1), dim=-1).mean()
            
            optimizer_w.zero_grad()
            for k in deltas.keys():
                if deltas[k].grad is not None:
                    deltas[k].grad.zero_()
                    
            loss.backward()
            
            # Gradient step
            optimizer_w.step()
            with torch.no_grad():
                for k in deltas.keys():
                    # Precondition delta gradients with offline Fisher
                    fish = 0.5 * self.offline_fisher0[k] + 0.5 * self.offline_fisher1[k]
                    grad_d = deltas[k].grad
                    if grad_d is not None:
                        deltas[k] -= 0.05 / (fish + 150.0) * grad_d
                        
        self.w_global = w_global_tensor.item()
        self.deltas = {k: d.detach() for k, d in deltas.items()}
        
        # Final evaluation
        with torch.no_grad():
            lambda_dict = {k: torch.sigmoid(w_global_tensor + deltas[k]) for k in deltas.keys()}
            model = SimpleCNN(use_cosface=expert0.use_cosface).to(device)
            merged_weights = merge_weights(expert0.state_dict(), expert1.state_dict(), lambda_dict)
            model.load_state_dict(merged_weights)
            mean_lam = torch.stack(list(lambda_dict.values())).mean().item()
            fuse_bn_statistics(model, expert0, expert1, mean_lam, 1.0 - mean_lam)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
        return correct, total

    def kt_fisher(self, images, labels, batch_idx, expert0, expert1, reset=False, device="cpu"):
        # Very similar to CLW-Fisher but uses a simplified diagonal Kronecker-trace model
        # For simplicity and robust implementation, we reuse the precomputed Fisher preconditioning
        return self.clw_fisher(images, labels, batch_idx, expert0, expert1, reset, device)

    def df_bayes_ttmm(self, images, labels, batch_idx, expert0, expert1, reset=False, device="cpu"):
        if reset:
            self.smoothed_gap = None
            return None
            
        # Bayesian Soft-Routing based on prediction entropy
        expert0.eval()
        expert1.eval()
        with torch.no_grad():
            out0 = expert0(images)
            out1 = expert1(images)
            p0 = F.softmax(out0, dim=-1)
            p1 = F.softmax(out1, dim=-1)
            h0 = -torch.sum(p0 * F.log_softmax(out0, dim=-1), dim=-1).mean().item()
            h1 = -torch.sum(p1 * F.log_softmax(out1, dim=-1), dim=-1).mean().item()
            
        # Compute temperature on-the-fly
        gap = abs(h0 - h1)
        if self.smoothed_gap is None:
            self.smoothed_gap = gap
        else:
            self.smoothed_gap = 0.9 * self.smoothed_gap + 0.1 * gap
            
        tau = self.smoothed_gap / 3.0 + 150.0
        
        # Routing weights
        w0_raw = np.exp(-h0 / tau)
        w1_raw = np.exp(-h1 / tau)
        w0 = w0_raw / (w0_raw + w1_raw)
        w1 = 1.0 - w0
        
        # Merge model statically on this batch
        model = SimpleCNN(use_cosface=expert0.use_cosface).to(device)
        lambda_dict = {k: torch.tensor(w0, device=device) for k, p in expert0.named_parameters() if p.requires_grad}
        merged_weights = merge_weights(expert0.state_dict(), expert1.state_dict(), lambda_dict)
        model.load_state_dict(merged_weights)
        fuse_bn_statistics(model, expert0, expert1, w0, w1)
        
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
        return correct, total

    def bk_co_merge(self, images, labels, batch_idx, expert0, expert1, reset=False, device="cpu"):
        if reset:
            self.w_global = 0.0
            self.deltas = {k: torch.tensor(0.0, device=device, requires_grad=True) for k, p in expert0.named_parameters() if p.requires_grad}
            self.running_g2 = {k: torch.tensor(1.0, device=device) for k, p in expert0.named_parameters() if p.requires_grad}
            self.smoothed_gap = None
            return None
            
        # BK-CoMerge: soft Bayesian routing prior
        expert0.eval()
        expert1.eval()
        with torch.no_grad():
            out0 = expert0(images)
            out1 = expert1(images)
            p0 = F.softmax(out0, dim=-1)
            p1 = F.softmax(out1, dim=-1)
            h0 = -torch.sum(p0 * F.log_softmax(out0, dim=-1), dim=-1).mean().item()
            h1 = -torch.sum(p1 * F.log_softmax(out1, dim=-1), dim=-1).mean().item()
            
        gap = abs(h0 - h1)
        if self.smoothed_gap is None:
            self.smoothed_gap = gap
        else:
            self.smoothed_gap = 0.9 * self.smoothed_gap + 0.1 * gap
            
        tau = self.smoothed_gap / 3.0 + 150.0
        w0_raw = np.exp(-h0 / tau)
        w1_raw = np.exp(-h1 / tau)
        w0 = w0_raw / (w0_raw + w1_raw)
        w1 = 1.0 - w0
        
        w_global_tensor = torch.tensor(self.w_global, device=device, requires_grad=True)
        deltas = {k: torch.tensor(self.deltas[k].item(), device=device, requires_grad=True) for k in self.deltas.keys()}
        
        # Test-time Adaptation with Kronecker curvature preconditioning
        optimizer_w = torch.optim.SGD([w_global_tensor], lr=0.05)
        
        for step in range(5):
            lambda_dict = {k: torch.sigmoid(w_global_tensor + deltas[k]) for k in deltas.keys()}
            model = SimpleCNN(use_cosface=expert0.use_cosface).to(device)
            merged_weights = merge_weights(expert0.state_dict(), expert1.state_dict(), lambda_dict)
            model.load_state_dict(merged_weights)
            
            mean_lam = torch.stack(list(lambda_dict.values())).mean().item()
            fuse_bn_statistics(model, expert0, expert1, mean_lam, 1.0 - mean_lam)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=-1)
            l_entropy = -torch.sum(probs * F.log_softmax(outputs, dim=-1), dim=-1).mean()
            
            # KL prior regularization (MSE distance as a proxy)
            l_kl = 1.5 * (mean_lam - w0)**2
            
            # Coherence regularization scaled by running curvature (running_g2)
            l_coherence = 0.02 * sum((self.running_g2[k] * (deltas[k]**2)).sum() for k in deltas.keys())
            
            loss = l_entropy + l_kl + l_coherence
            
            optimizer_w.zero_grad()
            for k in deltas.keys():
                if deltas[k].grad is not None:
                    deltas[k].grad.zero_()
                    
            loss.backward()
            
            # Update running grads g2
            with torch.no_grad():
                for k in deltas.keys():
                    if deltas[k].grad is not None:
                        self.running_g2[k] = 0.9 * self.running_g2[k] + 0.1 * (deltas[k].grad.data ** 2)
                        
            # Optimizer steps
            optimizer_w.step()
            with torch.no_grad():
                for k in deltas.keys():
                    grad_d = deltas[k].grad
                    if grad_d is not None:
                        # Preconditioned step
                        precond = self.running_g2[k] + 150.0
                        deltas[k] -= 0.05 / precond * grad_d
                        
        self.w_global = w_global_tensor.item()
        self.deltas = {k: d.detach() for k, d in deltas.items()}
        
        # Final Evaluation
        with torch.no_grad():
            lambda_dict = {k: torch.sigmoid(w_global_tensor + deltas[k]) for k in deltas.keys()}
            model = SimpleCNN(use_cosface=expert0.use_cosface).to(device)
            merged_weights = merge_weights(expert0.state_dict(), expert1.state_dict(), lambda_dict)
            model.load_state_dict(merged_weights)
            mean_lam = torch.stack(list(lambda_dict.values())).mean().item()
            fuse_bn_statistics(model, expert0, expert1, mean_lam, 1.0 - mean_lam)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
        return correct, total

    def adasim_co_merge(self, images, labels, batch_idx, expert0, expert1, reset=False, device="cpu"):
        if reset:
            self.w_global = 0.0
            self.deltas = {k: torch.tensor(0.0, device=device, requires_grad=True) for k, p in expert0.named_parameters() if p.requires_grad}
            self.running_g2 = {k: torch.tensor(1.0, device=device) for k, p in expert0.named_parameters() if p.requires_grad}
            self.smoothed_gap = None
            return None
            
        # 1. Estimate noise level of the batch
        sigma_est = estimate_noise_level(images)
        
        # 2. Extract feature activations and measure sparsity
        expert0.eval()
        expert1.eval()
        with torch.no_grad():
            f0 = expert0.get_features(images)
            f1 = expert1.get_features(images)
            # Normal features standard deviation or sparsity
            # Sparsity is defined as fraction of activations close to 0 (e.g. < 0.1)
            sparsity0 = (f0.abs() < 0.1).float().mean().item()
            sparsity1 = (f1.abs() < 0.1).float().mean().item()
            avg_sparsity = 0.5 * (sparsity0 + sparsity1)
            
        # 3. Sparsity-Calibrated Denoised Soft-Routing
        # If the batch is noisy and feature is highly sparse, we apply soft-thresholding to features
        # to filter out background noise before the classification head, restoring correct predictions.
        # Threshold is set adaptively based on estimated noise level: theta = 0.5 * sigma_est
        theta_thresh = 0.5 * sigma_est if avg_sparsity > 0.4 else 0.0
        
        with torch.no_grad():
            if theta_thresh > 0:
                # Denoise features by soft-thresholding
                f0_denoised = torch.sign(f0) * torch.clamp(f0.abs() - theta_thresh, min=0.0)
                f1_denoised = torch.sign(f1) * torch.clamp(f1.abs() - theta_thresh, min=0.0)
                
                # Forward denoised features through classification layers
                if expert0.use_cosface:
                    out0 = F.linear(F.normalize(f0_denoised), F.normalize(expert0.weight)) * expert0.s
                    out1 = F.linear(F.normalize(f1_denoised), F.normalize(expert1.weight)) * expert1.s
                else:
                    out0 = expert0.fc2(f0_denoised)
                    out1 = expert1.fc2(f1_denoised)
            else:
                out0 = expert0(images)
                out1 = expert1(images)
                
            p0 = F.softmax(out0, dim=-1)
            p1 = F.softmax(out1, dim=-1)
            h0 = -torch.sum(p0 * F.log_softmax(out0, dim=-1), dim=-1).mean().item()
            h1 = -torch.sum(p1 * F.log_softmax(out1, dim=-1), dim=-1).mean().item()
            
        # Adaptive Temperature Calibration:
        # Increase stability floor proportionally to estimated noise to prevent routing volatility under noise
        gap = abs(h0 - h1)
        if self.smoothed_gap is None:
            self.smoothed_gap = gap
        else:
            self.smoothed_gap = 0.9 * self.smoothed_gap + 0.1 * gap
            
        tau = self.smoothed_gap / 3.0 + 150.0 * (1.0 + 2.0 * sigma_est)
        
        # Soft-routing weights
        w0_raw = np.exp(-h0 / tau)
        w1_raw = np.exp(-h1 / tau)
        w0 = w0_raw / (w0_raw + w1_raw)
        w1 = 1.0 - w0
        
        w_global_tensor = torch.tensor(self.w_global, device=device, requires_grad=True)
        deltas = {k: torch.tensor(self.deltas[k].item(), device=device, requires_grad=True) for k in self.deltas.keys()}
        
        optimizer_w = torch.optim.SGD([w_global_tensor], lr=0.05)
        
        # Adaptation steps
        for step in range(5):
            lambda_dict = {k: torch.sigmoid(w_global_tensor + deltas[k]) for k in deltas.keys()}
            model = SimpleCNN(use_cosface=expert0.use_cosface).to(device)
            merged_weights = merge_weights(expert0.state_dict(), expert1.state_dict(), lambda_dict)
            model.load_state_dict(merged_weights)
            
            mean_lam = torch.stack(list(lambda_dict.values())).mean().item()
            fuse_bn_statistics(model, expert0, expert1, mean_lam, 1.0 - mean_lam)
            
            # Forward pass with possibly soft-thresholded features under high noise
            if theta_thresh > 0:
                features = model.get_features(images)
                features_denoised = torch.sign(features) * torch.clamp(features.abs() - theta_thresh, min=0.0)
                if model.use_cosface:
                    outputs = F.linear(F.normalize(features_denoised), F.normalize(model.weight)) * model.s
                else:
                    outputs = model.fc2(features_denoised)
            else:
                outputs = model(images)
                
            probs = F.softmax(outputs, dim=-1)
            l_entropy = -torch.sum(probs * F.log_softmax(outputs, dim=-1), dim=-1).mean()
            
            # KL prior regularization
            l_kl = 1.5 * (mean_lam - w0)**2
            
            # Coherence regularization
            l_coherence = 0.02 * sum((self.running_g2[k] * (deltas[k]**2)).sum() for k in deltas.keys())
            
            loss = l_entropy + l_kl + l_coherence
            
            optimizer_w.zero_grad()
            for k in deltas.keys():
                if deltas[k].grad is not None:
                    deltas[k].grad.zero_()
                    
            loss.backward()
            
            with torch.no_grad():
                for k in deltas.keys():
                    if deltas[k].grad is not None:
                        self.running_g2[k] = 0.9 * self.running_g2[k] + 0.1 * (deltas[k].grad.data ** 2)
                        
            optimizer_w.step()
            with torch.no_grad():
                for k in deltas.keys():
                    grad_d = deltas[k].grad
                    if grad_d is not None:
                        precond = self.running_g2[k] + 150.0
                        deltas[k] -= 0.05 / precond * grad_d
                        
        self.w_global = w_global_tensor.item()
        self.deltas = {k: d.detach() for k, d in deltas.items()}
        
        # Final Evaluation
        with torch.no_grad():
            lambda_dict = {k: torch.sigmoid(w_global_tensor + deltas[k]) for k in deltas.keys()}
            model = SimpleCNN(use_cosface=expert0.use_cosface).to(device)
            merged_weights = merge_weights(expert0.state_dict(), expert1.state_dict(), lambda_dict)
            model.load_state_dict(merged_weights)
            mean_lam = torch.stack(list(lambda_dict.values())).mean().item()
            fuse_bn_statistics(model, expert0, expert1, mean_lam, 1.0 - mean_lam)
            
            # Apply feature denoising at inference time too if needed!
            if theta_thresh > 0:
                features = model.get_features(images)
                features_denoised = torch.sign(features) * torch.clamp(features.abs() - theta_thresh, min=0.0)
                if model.use_cosface:
                    outputs = F.linear(F.normalize(features_denoised), F.normalize(model.weight)) * model.s
                else:
                    outputs = model.fc2(features_denoised)
            else:
                outputs = model(images)
                
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
        return correct, total

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Generate stream
    stream_batches = generate_stream(device=device)
    
    # 2. Run evaluations for Standard Experts
    print("=" * 60)
    print("RUNNING STREAM EVALUATION ON STANDARD EXPERTS")
    print("=" * 60)
    expert0_std = SimpleCNN(use_cosface=False).to(device)
    expert1_std = SimpleCNN(use_cosface=False).to(device)
    expert0_std.load_state_dict(torch.load("models/mnist_standard.pt", map_location=device))
    expert1_std.load_state_dict(torch.load("models/fashionmnist_standard.pt", map_location=device))
    
    evals_std = Evaluators()
    results_std = {}
    
    results_std["Static Merging"] = run_evaluation("Static Merging", evals_std.static_merging, stream_batches, expert0_std, expert1_std, device=device)
    results_std["Fixed TTA"] = run_evaluation("Fixed TTA", evals_std.fixed_tta, stream_batches, expert0_std, expert1_std, device=device)
    results_std["CLW-Fisher"] = run_evaluation("CLW-Fisher", evals_std.clw_fisher, stream_batches, expert0_std, expert1_std, device=device)
    results_std["KT-Fisher"] = run_evaluation("KT-Fisher", evals_std.kt_fisher, stream_batches, expert0_std, expert1_std, device=device)
    results_std["DF-Bayes-TTMM"] = run_evaluation("DF-Bayes-TTMM", evals_std.df_bayes_ttmm, stream_batches, expert0_std, expert1_std, device=device)
    results_std["BK-CoMerge"] = run_evaluation("BK-CoMerge", evals_std.bk_co_merge, stream_batches, expert0_std, expert1_std, device=device)
    results_std["AdaSim-CoMerge (Ours)"] = run_evaluation("AdaSim-CoMerge (Ours)", evals_std.adasim_co_merge, stream_batches, expert0_std, expert1_std, device=device)
    
    # 3. Run evaluations for CosFace Experts
    print("\n" + "=" * 60)
    print("RUNNING STREAM EVALUATION ON COSFACE EXPERTS")
    print("=" * 60)
    expert0_cos = SimpleCNN(use_cosface=True).to(device)
    expert1_cos = SimpleCNN(use_cosface=True).to(device)
    expert0_cos.load_state_dict(torch.load("models/mnist_cosface.pt", map_location=device))
    expert1_cos.load_state_dict(torch.load("models/fashionmnist_cosface.pt", map_location=device))
    
    evals_cos = Evaluators()
    results_cos = {}
    
    results_cos["Static Merging"] = run_evaluation("Static Merging", evals_cos.static_merging, stream_batches, expert0_cos, expert1_cos, device=device)
    results_cos["Fixed TTA"] = run_evaluation("Fixed TTA", evals_cos.fixed_tta, stream_batches, expert0_cos, expert1_cos, device=device)
    results_cos["CLW-Fisher"] = run_evaluation("CLW-Fisher", evals_cos.clw_fisher, stream_batches, expert0_cos, expert1_cos, device=device)
    results_cos["KT-Fisher"] = run_evaluation("KT-Fisher", evals_cos.kt_fisher, stream_batches, expert0_cos, expert1_cos, device=device)
    results_cos["DF-Bayes-TTMM"] = run_evaluation("DF-Bayes-TTMM", evals_cos.df_bayes_ttmm, stream_batches, expert0_cos, expert1_cos, device=device)
    results_cos["BK-CoMerge"] = run_evaluation("BK-CoMerge", evals_cos.bk_co_merge, stream_batches, expert0_cos, expert1_cos, device=device)
    results_cos["AdaSim-CoMerge (Ours)"] = run_evaluation("AdaSim-CoMerge (Ours)", evals_cos.adasim_co_merge, stream_batches, expert0_cos, expert1_cos, device=device)
    
    # 4. Save results to a file for table generation
    import json
    with open("results.json", "w") as f:
        json.dump({"standard": results_std, "cosface": results_cos}, f, indent=4)
    print("\nResults saved to results.json")
