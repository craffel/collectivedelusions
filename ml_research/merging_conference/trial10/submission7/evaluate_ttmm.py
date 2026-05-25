import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.func import functional_call
import copy
import numpy as np

# Define the SimpleCNN architecture exactly
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        feats = self.dropout(feats)
        out = self.fc2(feats)
        return out

def hoyer_sparsity(v):
    # v is a flattened tensor of size (D,)
    d = v.numel()
    l1 = torch.sum(torch.abs(v))
    l2 = torch.sum(v ** 2).sqrt()
    if l2 == 0:
        return 0.0
    return ((d**0.5 - l1/l2) / (d**0.5 - 1)).item()

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy.mean()

def get_distances(features, prototypes, metric="Euclidean"):
    if metric == "Euclidean":
        diff = features.unsqueeze(1) - prototypes.unsqueeze(0) # (B, 10, D)
        dist_sq = torch.sum(diff ** 2, dim=-1) # (B, 10)
        min_dists, _ = dist_sq.min(dim=-1) # (B,)
        return min_dists
    elif metric == "Angular":
        feats_norm = features / (features.norm(p=2, dim=-1, keepdim=True) + 1e-9)
        cos_sim = torch.mm(feats_norm, prototypes.t()) # (B, 10)
        angular_dist = 1.0 - cos_sim # (B, 10)
        min_dists, _ = angular_dist.min(dim=-1) # (B,)
        return min_dists

def stable_softmax(D0, D1, tau):
    # Standard stable softmax to avoid underflow/overflow to NaN
    val0 = -D0 / tau
    val1 = -D1 / tau
    max_val = max(val0, val1)
    e0 = np.exp(val0 - max_val)
    e1 = np.exp(val1 - max_val)
    w1 = e1 / (e0 + e1)
    return w1, 1.0 - w1

def set_bn_mode(model, train=True):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if train:
                m.train()
                m.track_running_stats = False # disable running stats update during TTA
            else:
                m.eval()

def fuse_bn_buffers(model_mnist, model_fashion, w1, w0):
    # w1 is weight for fashion, w0 is weight for mnist
    fused_buffers = {}
    state_mnist = model_mnist.state_dict()
    state_fashion = model_fashion.state_dict()
    
    for name in state_mnist.keys():
        if "running_mean" in name:
            mean0 = state_mnist[name]
            mean1 = state_fashion[name]
            fused_mean = w0 * mean0 + w1 * mean1
            fused_buffers[name] = fused_mean
            
            # Find corresponding running_var name
            var_name = name.replace("running_mean", "running_var")
            var0 = state_mnist[var_name]
            var1 = state_fashion[var_name]
            # MoG Variance Fusion
            fused_var = w0 * (var0 + (mean0 - fused_mean)**2) + w1 * (var1 + (mean1 - fused_mean)**2)
            fused_buffers[var_name] = fused_var
        elif "num_batches_tracked" in name:
            fused_buffers[name] = state_mnist[name]
            
    return fused_buffers

def run_tta(merged_model, params_mnist, params_fashion, initial_w_global, initial_deltas, 
            x_batch, w1, w0, preconditioning_sensitivities=None, use_coherence=True, lr=0.01, use_sam=False, rho=0.05):
    # Set up learnable parameters
    w_global = torch.tensor(initial_w_global, requires_grad=True, device=x_batch.device)
    deltas = {}
    for name, delta_init in initial_deltas.items():
        deltas[name] = torch.tensor(delta_init, requires_grad=True, device=x_batch.device)
        
    optimizer = torch.optim.Adam([w_global] + list(deltas.values()), lr=lr)
    
    # Run 5 steps of TTA
    for step in range(5):
        optimizer.zero_grad()
        
        # Helper to compute loss for given parameters
        def compute_loss(w_g, d_dict):
            merged_params = {}
            for name in params_mnist.keys():
                p0 = params_mnist[name]
                p1 = params_fashion[name]
                delta_j = d_dict[name] if name in d_dict else 0.0
                lambda_j = torch.sigmoid(w_g + delta_j)
                merged_params[name] = lambda_j * p1 + (1.0 - lambda_j) * p0
                
            # Run forward pass on the merged parameters
            out = functional_call(merged_model, merged_params, x_batch)
            
            # Loss terms
            L_entropy = compute_entropy(out)
            
            # KL Prior constraint
            lambda_global = torch.sigmoid(w_g)
            L_kl = w1 * torch.log(w1 / (lambda_global + 1e-9) + 1e-9) + w0 * torch.log(w0 / (1.0 - lambda_global + 1e-9) + 1e-9)
            
            # Coherence Regularizer
            L_coherence = 0.0
            if use_coherence and preconditioning_sensitivities is not None:
                for name, delta_j in d_dict.items():
                    sens = preconditioning_sensitivities.get(name, 0.0)
                    L_coherence += sens * torch.sum(delta_j ** 2)
                    
            return L_entropy + 1.5 * L_kl + 0.02 * L_coherence

        if not use_sam:
            loss = compute_loss(w_global, deltas)
            loss.backward()
            optimizer.step()
        else:
            # Preconditioned SGD SAM
            loss = compute_loss(w_global, deltas)
            loss.backward()
            
            with torch.no_grad():
                dw = w_global.grad.clone() if w_global.grad is not None else torch.zeros_like(w_global)
                d_deltas = {}
                sq_sum = dw.item() ** 2
                
                for name, delta_param in deltas.items():
                    if delta_param.grad is not None:
                        sens = preconditioning_sensitivities.get(name, 0.0) if preconditioning_sensitivities is not None else 0.0
                        d_deltas[name] = delta_param.grad / (sens + 1e-4)
                        sq_sum += torch.sum(d_deltas[name] ** 2).item()
                    else:
                        d_deltas[name] = torch.zeros_like(delta_param)
                
                norm_D = np.sqrt(sq_sum + 1e-4)
                eps_w = rho * dw / norm_D
                eps_deltas = {name: rho * d / norm_D for name, d in d_deltas.items()}
                
                # Compute perturbed coordinates
                w_g_perturbed = (w_global + eps_w).requires_grad_(True)
                deltas_perturbed = {name: (param + eps_deltas.get(name, 0.0)).requires_grad_(True) for name, param in deltas.items()}
                
            # Compute loss at perturbed coordinates
            loss_p = compute_loss(w_g_perturbed, deltas_perturbed)
            loss_p.backward()
            
            # Update original parameters using perturbed gradients with preconditioned learning rates
            with torch.no_grad():
                if w_g_perturbed.grad is not None:
                    w_global -= lr * w_g_perturbed.grad
                for name, delta_param in deltas.items():
                    p_grad = deltas_perturbed[name].grad
                    if p_grad is not None:
                        sens = preconditioning_sensitivities.get(name, 0.0) if preconditioning_sensitivities is not None else 0.0
                        delta_param -= lr * p_grad / (sens + 1e-4)
                        
            # Clear gradients manually
            if w_global.grad is not None:
                w_global.grad.zero_()
            for delta_param in deltas.values():
                if delta_param.grad is not None:
                    delta_param.grad.zero_()
        
    # Reconstruct final merged parameters
    final_params = {}
    with torch.no_grad():
        for name in params_mnist.keys():
            p0 = params_mnist[name]
            p1 = params_fashion[name]
            delta_j = deltas[name].detach() if name in deltas else 0.0
            lambda_j = torch.sigmoid(w_global.detach() + delta_j)
            final_params[name] = lambda_j * p1 + (1.0 - lambda_j) * p0
            
    return final_params

def estimate_sensitivities(merged_model, params_mnist, params_fashion, w1, w0, x_batch):
    # Construct initial merged weights
    init_params = {}
    for name in params_mnist.keys():
        init_params[name] = (w1 * params_fashion[name] + w0 * params_mnist[name]).clone().detach().requires_grad_(True)
        
    out = functional_call(merged_model, init_params, x_batch)
    loss = compute_entropy(out)
    
    # Compute gradients with respect to parameters
    grad_tensors = torch.autograd.grad(loss, list(init_params.values()), retain_graph=False, allow_unused=True)
    
    sensitivities = {}
    total_sens = 0.0
    for (name, param), grad in zip(init_params.items(), grad_tensors):
        if grad is not None:
            # Mean squared gradient
            sens = torch.mean(grad ** 2).item()
            sensitivities[name] = sens
            total_sens += sens
        else:
            sensitivities[name] = 0.0
            
    # Normalize sensitivities
    if total_sens > 0:
        for name in sensitivities.keys():
            sensitivities[name] /= total_sens
            
    return sensitivities


def evaluate_method(method, test_batches, model_mnist, model_fashion, protos, device):
    print(f"\nEvaluating: {method} ...")
    
    # Extract base parameters
    params_mnist = {k: v.to(device) for k, v in model_mnist.named_parameters()}
    params_fashion = {k: v.to(device) for k, v in model_fashion.named_parameters()}
    
    # Load prototypes
    proto_mnist_unnorm = protos["mnist_unnorm"].to(device)
    proto_mnist_norm = protos["mnist_norm"].to(device)
    proto_fashion_unnorm = protos["fashion_unnorm"].to(device)
    proto_fashion_norm = protos["fashion_norm"].to(device)
    
    phase_correct = [0] * 5
    phase_total = [0] * 5
    phase_hoyers = [[] for _ in range(5)]
    phase_w1s = [[] for _ in range(5)]
    
    # Set up a template model for functional evaluation
    merged_model = SimpleCNN().to(device)
    
    for b_idx, (x_batch, y_batch) in enumerate(test_batches):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        phase_idx = b_idx // 10
        
        # 1. Sparsity estimation on denoised pixels
        x_pos = (x_batch + 1.0) / 2.0
        x_denoised = torch.where(x_pos > 0.35, x_pos, torch.zeros_like(x_pos))
        S_hoyer = hoyer_sparsity(x_denoised)
        
        # 2. Extract features and entropies for routing
        model_mnist.eval()
        model_fashion.eval()
        # For methods with TTBN, set experts' BN layers to training mode during routing extraction
        use_ttbn = (method in [
            "Method E (Original BK-AHR, Scale-Mismatched)",
            "Method F (BK-AHR with Normalized L2)",
            "Method G (CSAIR, Ours)",
            "Method H (CSASAM, Ours)"
        ])
        if use_ttbn:
            set_bn_mode(model_mnist, train=True)
            set_bn_mode(model_fashion, train=True)
        with torch.no_grad():
            feats_mnist = model_mnist.forward_features(x_batch)
            feats_fashion = model_fashion.forward_features(x_batch)
            logits_mnist = model_mnist(x_batch)
            logits_fashion = model_fashion(x_batch)
            
            H_mnist = compute_entropy(logits_mnist).item()
            H_fashion = compute_entropy(logits_fashion).item()
            H_avg = 0.5 * (H_mnist + H_fashion)
            
        # 3. Compute routing weights based on method
        if method == "Method A (Fixed TTA + Reset)":
            w1, w0 = 0.5, 0.5
            routing_type = 0
            
        elif method == "Method B (CL W-Fisher + SCTS L2)":
            routing_type = 0
            d0 = get_distances(feats_mnist, proto_mnist_unnorm, "Euclidean")
            d1 = get_distances(feats_fashion, proto_fashion_unnorm, "Euclidean")
            D0, D1 = d0.mean().item(), d1.mean().item()
            gap = abs(D0 - D1)
            # gamma_dun = 5.0
            eps_stab = 0.08 / (1.0 + 5.0 * H_avg)
            tau = (gap / 3.0) + eps_stab
            w1, w0 = stable_softmax(D0, D1, tau)
            
        elif method == "Method C (CL W-Fisher + A-SCTS)":
            routing_type = 1
            d0 = get_distances(feats_mnist, proto_mnist_norm, "Angular")
            d1 = get_distances(feats_fashion, proto_fashion_norm, "Angular")
            D0, D1 = d0.mean().item(), d1.mean().item()
            gap = abs(D0 - D1)
            eps_stab = 0.04 / (1.0 + 5.0 * H_avg)
            tau = (gap / 3.0) + eps_stab
            w1, w0 = stable_softmax(D0, D1, tau)
            
        elif method == "Method D (CP-AM)":
            routing_type = 1
            d0 = get_distances(feats_mnist, proto_mnist_norm, "Angular")
            d1 = get_distances(feats_fashion, proto_fashion_norm, "Angular")
            D0, D1 = d0.mean().item(), d1.mean().item()
            tau = 0.1 # Fixed temperature
            w1, w0 = stable_softmax(D0, D1, tau)
            
        elif method == "Method E (Original BK-AHR, Scale-Mismatched)":
            if S_hoyer >= 0.50:
                routing_type = 0
                d0 = get_distances(feats_mnist, proto_mnist_unnorm, "Euclidean")
                d1 = get_distances(feats_fashion, proto_fashion_unnorm, "Euclidean")
                D0, D1 = d0.mean().item(), d1.mean().item()
                gap = abs(D0 - D1)
                eps_stab = 0.08 / (1.0 + 5.0 * H_avg)
                tau = (gap / 3.0) + eps_stab
            else:
                routing_type = 1
                d0 = get_distances(feats_mnist, proto_mnist_norm, "Angular")
                d1 = get_distances(feats_fashion, proto_fashion_norm, "Angular")
                D0, D1 = d0.mean().item(), d1.mean().item()
                gap = abs(D0 - D1)
                eps_stab = 0.04 / (1.0 + 5.0 * H_avg)
                tau = (gap / 3.0) + eps_stab
            w1, w0 = stable_softmax(D0, D1, tau)

        elif method == "Method F (BK-AHR with Normalized L2)":
            if S_hoyer >= 0.50:
                routing_type = 0
                feats_mnist_norm = feats_mnist / (feats_mnist.norm(p=2, dim=-1, keepdim=True) + 1e-9)
                feats_fashion_norm = feats_fashion / (feats_fashion.norm(p=2, dim=-1, keepdim=True) + 1e-9)
                d0 = get_distances(feats_mnist_norm, proto_mnist_norm, "Euclidean")
                d1 = get_distances(feats_fashion_norm, proto_fashion_norm, "Euclidean")
                D0, D1 = d0.mean().item(), d1.mean().item()
                gap = abs(D0 - D1)
                eps_stab = 0.08 / (1.0 + 5.0 * H_avg)
                tau = (gap / 3.0) + eps_stab
            else:
                routing_type = 1
                d0 = get_distances(feats_mnist, proto_mnist_norm, "Angular")
                d1 = get_distances(feats_fashion, proto_fashion_norm, "Angular")
                D0, D1 = d0.mean().item(), d1.mean().item()
                gap = abs(D0 - D1)
                eps_stab = 0.04 / (1.0 + 5.0 * H_avg)
                tau = (gap / 3.0) + eps_stab
            w1, w0 = stable_softmax(D0, D1, tau)
            
        elif method in ["Method G (CSAIR, Ours)", "Method H (CSASAM, Ours)"]:
            # Compute both and smoothly interpolate the priors
            # 1. Euclidean SCTS (normalized to prevent scale mismatch under TTBN)
            feats_mnist_norm = feats_mnist / (feats_mnist.norm(p=2, dim=-1, keepdim=True) + 1e-9)
            feats_fashion_norm = feats_fashion / (feats_fashion.norm(p=2, dim=-1, keepdim=True) + 1e-9)
            d0_E = get_distances(feats_mnist_norm, proto_mnist_norm, "Euclidean")
            d1_E = get_distances(feats_fashion_norm, proto_fashion_norm, "Euclidean")
            D0_E, D1_E = d0_E.mean().item(), d1_E.mean().item()
            gap_E = abs(D0_E - D1_E)
            eps_stab_E = 0.08 / (1.0 + 5.0 * H_avg)
            tau_E = (gap_E / 3.0) + eps_stab_E
            w1_E, _ = stable_softmax(D0_E, D1_E, tau_E)
            
            # 2. Angular SCTS
            d0_A = get_distances(feats_mnist, proto_mnist_norm, "Angular")
            d1_A = get_distances(feats_fashion, proto_fashion_norm, "Angular")
            D0_A, D1_A = d0_A.mean().item(), d1_A.mean().item()
            gap_A = abs(D0_A - D1_A)
            eps_stab_A = 0.04 / (1.0 + 5.0 * H_avg)
            tau_A = (gap_A / 3.0) + eps_stab_A
            w1_A, _ = stable_softmax(D0_A, D1_A, tau_A)
            
            # 3. Soft blend using Hoyer sparsity sigmoid
            # S_hoyer is between 0 and 1. MNIST has sparsity ~0.56, FashionMNIST has ~0.44
            # We map S_hoyer to sigmoid blend: lambda = sigmoid(50 * (S_hoyer - 0.50))
            lambda_blend = 1.0 / (1.0 + np.exp(-50.0 * (S_hoyer - 0.50)))
            w1 = lambda_blend * w1_E + (1.0 - lambda_blend) * w1_A
            w0 = 1.0 - w1
            routing_type = 0 if lambda_blend >= 0.5 else 1
            
        phase_hoyers[phase_idx].append(S_hoyer)
        phase_w1s[phase_idx].append(w1)
            
        # Define TTA hyperparameters
        use_ttbn = (method in [
            "Method E (Original BK-AHR, Scale-Mismatched)",
            "Method F (BK-AHR with Normalized L2)",
            "Method G (CSAIR, Ours)",
            "Method H (CSASAM, Ours)"
        ])
        use_coherence = (method in [
            "Method B (CL W-Fisher + SCTS L2)",
            "Method C (CL W-Fisher + A-SCTS)", 
            "Method E (Original BK-AHR, Scale-Mismatched)",
            "Method F (BK-AHR with Normalized L2)",
            "Method G (CSAIR, Ours)",
            "Method H (CSASAM, Ours)"
        ])
        optimize_params = (method != "Method D (CP-AM)")
        
        # Enable TTBN or set to Eval mode
        if use_ttbn:
            set_bn_mode(merged_model, train=True)
            set_bn_mode(model_mnist, train=True)
            set_bn_mode(model_fashion, train=True)
        else:
            set_bn_mode(merged_model, train=False)
            set_bn_mode(model_mnist, train=False)
            set_bn_mode(model_fashion, train=False)
            
        # Reconstruct initial merged buffers (running stats)
        fused_buffers = fuse_bn_buffers(model_mnist, model_fashion, w1, w0)
        # Apply running buffers to merged template model
        merged_model.load_state_dict(model_mnist.state_dict(), strict=False) # load arch and placeholders
        for k, v in fused_buffers.items():
            merged_model.state_dict()[k].copy_(v)
            
        # 4. Sensitivity tracking (On-the-fly preconditioning)
        sensitivities = None
        if use_coherence:
            sensitivities = estimate_sensitivities(merged_model, params_mnist, params_fashion, w1, w0, x_batch)
            
        # 5. Execute TTA
        initial_w_global = np.log(w1 / (w0 + 1e-9) + 1e-9)
        initial_deltas = {name: 0.0 for name in params_mnist.keys()}
        
        lr_base = 0.05
        # Entropy-Adaptive Learning Rate (EALR) scaling
        if method in [
            "Method E (Original BK-AHR, Scale-Mismatched)",
            "Method F (BK-AHR with Normalized L2)",
            "Method G (CSAIR, Ours)",
            "Method H (CSASAM, Ours)"
        ]:
            lr_t = lr_base / (1.0 + 5.0 * H_avg)
        else:
            lr_t = lr_base
            
        if optimize_params:
            use_sam = (method == "Method H (CSASAM, Ours)")
            final_params = run_tta(merged_model, params_mnist, params_fashion, initial_w_global, initial_deltas,
                                   x_batch, w1, w0, preconditioning_sensitivities=sensitivities, 
                                   use_coherence=use_coherence, lr=lr_t, use_sam=use_sam, rho=0.05)
        else:
            # No optimization, just evaluate with initial merge
            final_params = {}
            for name in params_mnist.keys():
                final_params[name] = w1 * params_fashion[name] + w0 * params_mnist[name]
                
        # 6. Evaluation
        # During final evaluation, keep BN in test mode (eval) to use the fused running buffers,
        # UNLESS TTBN is active, in which case we use training mode to evaluate on the batch statistics.
        if use_ttbn:
            set_bn_mode(merged_model, train=True)
        else:
            set_bn_mode(merged_model, train=False)
            
        with torch.no_grad():
            out = functional_call(merged_model, final_params, x_batch)
            _, predicted = out.max(1)
            correct = predicted.eq(y_batch).sum().item()
            
        phase_correct[phase_idx] += correct
        phase_total[phase_idx] += x_batch.size(0)
        
    # Summarize results
    accuracies = []
    print(f"Diagnostics for {method}:")
    for i in range(5):
        acc = 100.0 * phase_correct[i] / phase_total[i]
        accuracies.append(acc)
        avg_h = np.mean(phase_hoyers[i])
        avg_w = np.mean(phase_w1s[i])
        print(f"  Phase {i+1}: Acc={acc:.2f}%, Avg Hoyer={avg_h:.4f}, Avg w1 (Fashion weight)={avg_w:.4f}")
        
    overall_acc = np.mean(accuracies)
    return accuracies, overall_acc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    # Load expert models
    model_mnist = SimpleCNN().to(device)
    model_mnist.load_state_dict(torch.load("expert_mnist.pth", map_location=device))
    model_mnist.eval()
    
    model_fashion = SimpleCNN().to(device)
    model_fashion.load_state_dict(torch.load("expert_fashion.pth", map_location=device))
    model_fashion.eval()
    
    # Load prototypes
    protos = torch.load("prototypes.pth", map_location=device)
    
    # Prepare datasets and non-stationary stream
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=False, transform=transform)
    
    # Construct 50 sequential batches (size 64)
    test_batches = []
    
    # Phase 1: Clean MNIST (batches 0-9)
    # Phase 2: Noisy MNIST (batches 10-19)
    # We take the first 640 for clean, and the next 640 for noisy.
    loader_mnist_clean = DataLoader(Subset(mnist_test, list(range(0, 640))), batch_size=64, shuffle=False)
    for x, y in loader_mnist_clean:
        test_batches.append((x, y))
        
    loader_mnist_noisy = DataLoader(Subset(mnist_test, list(range(640, 1280))), batch_size=64, shuffle=False)
    for x, y in loader_mnist_noisy:
        # Add Gaussian noise sigma = 0.6
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        test_batches.append((x_noisy, y))
        
    # Phase 3: Clean FashionMNIST (batches 20-29)
    # Phase 4: Noisy FashionMNIST (batches 30-39)
    loader_fashion_clean = DataLoader(Subset(fmnist_test, list(range(0, 640))), batch_size=64, shuffle=False)
    for x, y in loader_fashion_clean:
        test_batches.append((x, y))
        
    loader_fashion_noisy = DataLoader(Subset(fmnist_test, list(range(640, 1280))), batch_size=64, shuffle=False)
    for x, y in loader_fashion_noisy:
        noise = torch.randn_like(x) * 0.6
        x_noisy = torch.clamp(x + noise, -1.0, 1.0)
        test_batches.append((x_noisy, y))
        
    # Phase 5: Novel KMNIST (batches 40-49)
    loader_kmnist = DataLoader(Subset(kmnist_test, list(range(0, 640))), batch_size=64, shuffle=False)
    for x, y in loader_kmnist:
        test_batches.append((x, y))
        
    print(f"Prepared stream with {len(test_batches)} batches.")
    
    # Define methods to evaluate
    methods = [
        "Method A (Fixed TTA + Reset)",
        "Method B (CL W-Fisher + SCTS L2)",
        "Method C (CL W-Fisher + A-SCTS)",
        "Method D (CP-AM)",
        "Method E (Original BK-AHR, Scale-Mismatched)",
        "Method F (BK-AHR with Normalized L2)",
        "Method G (CSAIR, Ours)",
        "Method H (CSASAM, Ours)"
    ]
    
    results = {}
    for m in methods:
        accs, overall = evaluate_method(m, test_batches, model_mnist, model_fashion, protos, device)
        results[m] = (accs, overall)
        print(f"Accs: C-MN={accs[0]:.2f}%, N-MN={accs[1]:.2f}%, C-FN={accs[2]:.2f}%, N-FN={accs[3]:.2f}%, Nov-K={accs[4]:.2f}% | Overall={overall:.2f}%")
        
    # Print nice final markdown table
    print("\n\n" + "="*50)
    print("FINAL RESULTS TABLE")
    print("="*50)
    print("| Method | C-MN | N-MN | C-FN | N-FN | Nov-K | Overall |")
    print("|---|---|---|---|---|---|---|")
    for m in methods:
        accs, overall = results[m]
        print(f"| {m} | {accs[0]:.2f}% | {accs[1]:.2f}% | {accs[2]:.2f}% | {accs[3]:.2f}% | {accs[4]:.2f}% | {overall:.2f}% |")
