import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False

set_seed(42)

# SimpleCNN Architecture as specified in papers
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(3136, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, stride=2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, stride=2)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Hook definitions for monitoring activations and gradients
inputs_dict = {}
grads_dict = {}

def register_hooks(model):
    handles = []
    # We register hooks on the weight-carrying layers
    layers_to_monitor = {
        'conv1': model.conv1,
        'conv2': model.conv2,
        'fc1': model.fc1,
        'fc2': model.fc2
    }
    
    def get_forward_hook(name):
        def hook(module, inp, out):
            inputs_dict[name] = inp[0].detach().clone()
        return hook

    def get_backward_hook(name):
        def hook(module, grad_inp, grad_out):
            grads_dict[name] = grad_out[0].detach().clone()
        return hook

    for name, layer in layers_to_monitor.items():
        h_f = layer.register_forward_hook(get_forward_hook(name))
        h_b = layer.register_full_backward_hook(get_backward_hook(name))
        handles.extend([h_f, h_b])
    return handles

# Data-free Batch Normalization Fusion strategies
def fuse_bn_statistics_mog(model, expert0, expert1, w):
    """
    Standard Mixture of Gaussians (MoG) BN statistic fusion.
    Suffers from variance inflation due to distance in task means.
    """
    w0, w1 = w[0].item(), w[1].item()
    with torch.no_grad():
        for bn_name in ['bn1', 'bn2']:
            m0 = expert0.state_dict()[f'{bn_name}.running_mean']
            v0 = expert0.state_dict()[f'{bn_name}.running_var']
            m1 = expert1.state_dict()[f'{bn_name}.running_mean']
            v1 = expert1.state_dict()[f'{bn_name}.running_var']
            
            # MoG moment matching mean
            m_fused = w0 * m0 + w1 * m1
            # MoG moment matching variance (inflated by the difference in means)
            v_fused = w0 * (v0 + (m0 - m_fused)**2) + w1 * (v1 + (m1 - m_fused)**2)
            
            model.state_dict()[f'{bn_name}.running_mean'].copy_(m_fused)
            model.state_dict()[f'{bn_name}.running_var'].copy_(v_fused)

def fuse_bn_statistics_wb(model, expert0, expert1, w):
    """
    Proposed Wasserstein-Barycenter (WB) BN statistic fusion.
    No variance inflation, scales activations correctly.
    """
    w0, w1 = w[0].item(), w[1].item()
    with torch.no_grad():
        for bn_name in ['bn1', 'bn2']:
            m0 = expert0.state_dict()[f'{bn_name}.running_mean']
            v0 = expert0.state_dict()[f'{bn_name}.running_var']
            m1 = expert1.state_dict()[f'{bn_name}.running_mean']
            v1 = expert1.state_dict()[f'{bn_name}.running_var']
            
            # Means interpolate linearly
            m_fused = w0 * m0 + w1 * m1
            # Standard deviations interpolate linearly (Wasserstein Barycenter geodesic)
            s0 = torch.sqrt(torch.clamp(v0, min=0.0))
            s1 = torch.sqrt(torch.clamp(v1, min=0.0))
            s_fused = w0 * s0 + w1 * s1
            v_fused = s_fused ** 2
            
            model.state_dict()[f'{bn_name}.running_mean'].copy_(m_fused)
            model.state_dict()[f'{bn_name}.running_var'].copy_(v_fused)

def fuse_bn_statistics_dwb(model, expert0, expert1, w, X_batch, alpha=0.15):
    """
    Dynamic Wasserstein-Barycenter (D-WB) BN statistic fusion.
    Blends the expert Wasserstein-Barycenter statistics with the current batch's empirical statistics.
    """
    w0, w1 = w[0].item(), w[1].item()
    with torch.no_grad():
        # Get empirical stats of X_batch from the experts
        expert0.eval()
        expert1.eval()
        
        feat0_1 = expert0.conv1(X_batch)
        feat1_1 = expert1.conv1(X_batch)
        mean_emp1 = w0 * feat0_1.mean(dim=(0, 2, 3)) + w1 * feat1_1.mean(dim=(0, 2, 3))
        var_emp1 = w0 * feat0_1.var(dim=(0, 2, 3), unbiased=False) + w1 * feat1_1.var(dim=(0, 2, 3), unbiased=False)
        
        act0_1 = F.max_pool2d(F.relu(expert0.bn1(feat0_1)), 2, stride=2)
        act1_1 = F.max_pool2d(F.relu(expert1.bn1(feat1_1)), 2, stride=2)
        feat0_2 = expert0.conv2(act0_1)
        feat1_2 = expert1.conv2(act1_1)
        mean_emp2 = w0 * feat0_2.mean(dim=(0, 2, 3)) + w1 * feat1_2.mean(dim=(0, 2, 3))
        var_emp2 = w0 * feat0_2.var(dim=(0, 2, 3), unbiased=False) + w1 * feat1_2.var(dim=(0, 2, 3), unbiased=False)
        
        emp_means = {'bn1': mean_emp1, 'bn2': mean_emp2}
        emp_vars = {'bn1': var_emp1, 'bn2': var_emp2}
        
        for bn_name in ['bn1', 'bn2']:
            m0 = expert0.state_dict()[f'{bn_name}.running_mean']
            v0 = expert0.state_dict()[f'{bn_name}.running_var']
            m1 = expert1.state_dict()[f'{bn_name}.running_mean']
            v1 = expert1.state_dict()[f'{bn_name}.running_var']
            
            # Base WB mean and std
            m_wb = w0 * m0 + w1 * m1
            s0 = torch.sqrt(torch.clamp(v0, min=0.0))
            s1 = torch.sqrt(torch.clamp(v1, min=0.0))
            s_wb = w0 * s0 + w1 * s1
            
            # Empirical mean and std
            m_emp = emp_means[bn_name]
            s_emp = torch.sqrt(torch.clamp(emp_vars[bn_name], min=0.0))
            
            # Dynamic blend
            m_fused = (1.0 - alpha) * m_wb + alpha * m_emp
            s_fused = (1.0 - alpha) * s_wb + alpha * s_emp
            v_fused = s_fused ** 2
            
            model.state_dict()[f'{bn_name}.running_mean'].copy_(m_fused)
            model.state_dict()[f'{bn_name}.running_var'].copy_(v_fused)

def differentiable_blended_batch_norm(x, bn_module, mu_wb, sigma_wb, alpha, eps=1e-5):
    """
    Differentiable Blended Batch Normalization.
    Computes empirical batch statistics and blends them with the precomputed
    Wasserstein-Barycenter statistics (mu_wb, sigma_wb) on-the-fly.
    Maintains full differentiability back to the network inputs and BN weights/biases.
    """
    if x.ndim == 4:
        mean_emp = x.mean(dim=(0, 2, 3), keepdim=True)
        var_emp = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
    elif x.ndim == 2:
        mean_emp = x.mean(dim=0, keepdim=True)
        var_emp = x.var(dim=0, unbiased=False, keepdim=True)
    else:
        raise ValueError(f"Unsupported input dimension: {x.ndim}")
        
    sigma_emp = torch.sqrt(torch.clamp(var_emp, min=1e-12))
    
    C = x.shape[1]
    if x.ndim == 4:
        mu_wb_r = mu_wb.view(1, C, 1, 1)
        sigma_wb_r = sigma_wb.view(1, C, 1, 1)
    else:
        mu_wb_r = mu_wb.view(1, C)
        sigma_wb_r = sigma_wb.view(1, C)
        
    mean_fused = (1.0 - alpha) * mu_wb_r + alpha * mean_emp
    sigma_fused = (1.0 - alpha) * sigma_wb_r + alpha * sigma_emp
    
    x_norm = (x - mean_fused) / (sigma_fused + eps)
    
    if bn_module.weight is not None:
        if x.ndim == 4:
            w = bn_module.weight.view(1, C, 1, 1)
            b = bn_module.bias.view(1, C, 1, 1)
        else:
            w = bn_module.weight.view(1, C)
            b = bn_module.bias.view(1, C)
        return x_norm * w + b
    else:
        return x_norm

# Layer-wise parameter groups
layer_groups = {
    'conv1': ['conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias'],
    'conv2': ['conv2.weight', 'conv2.bias', 'bn2.weight', 'bn2.bias'],
    'fc1': ['fc1.weight', 'fc1.bias'],
    'fc2': ['fc2.weight', 'fc2.bias']
}

def apply_differentiable_merging(model, expert0, expert1, w_global, offsets):
    """
    Dynamically merges parameters in a differentiable way using MAML-like delattr/setattr trick,
    enabling backpropagation to flow to w_global and layer-wise offsets.
    """
    e0_state = expert0.state_dict()
    e1_state = expert1.state_dict()
    
    for group_name, param_names in layer_groups.items():
        # Compute lambda_j = sigmoid(w_global + offset_j)
        l_j = torch.sigmoid(w_global + offsets[group_name])
        
        for p_name in param_names:
            p0 = e0_state[p_name]
            p1 = e1_state[p_name]
            p_merged = l_j * p0 + (1.0 - l_j) * p1
            
            # Traverse submodule path to delete and set parameter as tensor attribute
            parts = p_name.split('.')
            submodule = model
            for part in parts[:-1]:
                submodule = getattr(submodule, part)
            attr_name = parts[-1]
            
            if hasattr(submodule, attr_name):
                delattr(submodule, attr_name)
            setattr(submodule, attr_name, p_merged)

def restore_model_parameters(model, original_parameters):
    """
    Cleans up the model after adaptation by converting tensor attributes back to nn.Parameter
    and restoring the original module state structure.
    """
    for p_name, original_param in original_parameters.items():
        parts = p_name.split('.')
        submodule = model
        for part in parts[:-1]:
            submodule = getattr(submodule, part)
        attr_name = parts[-1]
        
        # Ensure we delete any custom tensor attributes
        if hasattr(submodule, attr_name):
            delattr(submodule, attr_name)
        # Re-register parameter
        submodule.register_parameter(attr_name, original_param)

# Main Test-Time Adaptation Loop
def run_test_time_adaptation(expert0, expert1, stream_batches, method="wb", device="cpu", eta=0.05, gamma_c=0.02, alpha=0.15, routing_mode="entropy"):
    """
    Runs Test-Time Model Merging on a sequential unlabeled stream.
    method can be:
        "static": Uniform 0.5/0.5 merge, no adaptation.
        "mog": BK-CoMerge (SOTA, MoG BN Fusion, Kronecker sensitivity preconditioning, coherence regularization).
        "wb": WB-CoMerge (Ours, Wasserstein-Barycenter BN Fusion, Kronecker preconditioning, coherence regularization).
        "dwb": D-WB-CoMerge (Ours, Dynamic Wasserstein-Barycenter BN Fusion).
    """
    # Instantiate the base merged model
    merged_model = SimpleCNN().to(device)
    original_state = {k: v for k, v in merged_model.named_parameters()}
    
    # Pre-register hooks
    hook_handles = register_hooks(merged_model)
    
    # Sensitivity tracking for Kronecker trace preconditioning
    F_running = {name: 1.0 for name in layer_groups}
    g_running = {name: 1.0 for name in layer_groups}
    
    # Temporal smoothing for entropy gap
    smoothed_gap = None
    gamma_s = 0.9 # EMA smoothing factor
    
    # Evaluation metrics
    accuracies = []
    routing_coefs = []
    
    # SCTS constants
    s_scale = 3.0
    epsilon_stab = 0.1
    tau_N = 0.70 # Novelty entropy threshold
    
    # Adaptation hyperparameters
    N_steps = 5      # Steps per batch
    beta = 1.5       # KL routing regularization weight
    
    for b_idx, (X_batch, y_batch) in enumerate(stream_batches):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # --- Step 1: Compute Dynamic Soft Routing Prior ---
        # Evaluate expert properties on current batch
        with torch.no_grad():
            expert0.eval()
            expert1.eval()
            out0 = expert0(X_batch)
            out1 = expert1(X_batch)
            
            p0 = F.softmax(out0, dim=1)
            p1 = F.softmax(out1, dim=1)
            
            H0_avg = -torch.mean(torch.sum(p0 * torch.log(p0 + 1e-12), dim=1)).item()
            H1_avg = -torch.mean(torch.sum(p1 * torch.log(p1 + 1e-12), dim=1)).item()
            H_mean = 0.5 * (H0_avg + H1_avg)
            
            if H_mean > tau_N:
                # Novel domain detected: uniform routing
                w = torch.tensor([0.5, 0.5], device=device)
            else:
                if routing_mode == "entropy":
                    gap = abs(H0_avg - H1_avg)
                    if smoothed_gap is None:
                        smoothed_gap = gap
                    else:
                        smoothed_gap = gamma_s * smoothed_gap + (1.0 - gamma_s) * gap
                    
                    tau_self = (smoothed_gap / s_scale) + epsilon_stab
                    w0 = np.exp(-H0_avg / tau_self)
                    w1 = np.exp(-H1_avg / tau_self)
                else: # "mi" routing mode
                    p0_mean = torch.mean(p0, dim=0)
                    p1_mean = torch.mean(p1, dim=0)
                    H0_marg = -torch.sum(p0_mean * torch.log(p0_mean + 1e-12)).item()
                    H1_marg = -torch.sum(p1_mean * torch.log(p1_mean + 1e-12)).item()
                    MI0 = H0_marg - H0_avg
                    MI1 = H1_marg - H1_avg
                    
                    gap = abs(MI0 - MI1)
                    if smoothed_gap is None:
                        smoothed_gap = gap
                    else:
                        smoothed_gap = gamma_s * smoothed_gap + (1.0 - gamma_s) * gap
                    
                    tau_self = (smoothed_gap / s_scale) + epsilon_stab
                    w0 = np.exp(MI0 / tau_self)
                    w1 = np.exp(MI1 / tau_self)
                
                w_sum = w0 + w1
                w = torch.tensor([w0 / w_sum, w1 / w_sum], device=device)
        
        routing_coefs.append(w[0].item())
        
        # Precompute sr_dwb stats if needed
        sr_dwb_stats = {}
        if method == "sr_dwb":
            w0, w1 = w[0].item(), w[1].item()
            with torch.no_grad():
                for bn_name in ['bn1', 'bn2']:
                    m0 = expert0.state_dict()[f'{bn_name}.running_mean']
                    v0 = expert0.state_dict()[f'{bn_name}.running_var']
                    m1 = expert1.state_dict()[f'{bn_name}.running_mean']
                    v1 = expert1.state_dict()[f'{bn_name}.running_var']
                    
                    m_wb = w0 * m0 + w1 * m1
                    s0 = torch.sqrt(torch.clamp(v0, min=0.0))
                    s1 = torch.sqrt(torch.clamp(v1, min=0.0))
                    s_wb = w0 * s0 + w1 * s1
                    sr_dwb_stats[bn_name] = (m_wb, s_wb)

        # --- Step 2: Fuse BN Running Statistics ---
        if method == "static" or method == "mog":
            fuse_bn_statistics_mog(merged_model, expert0, expert1, w)
        elif method == "wb":
            fuse_bn_statistics_wb(merged_model, expert0, expert1, w)
        elif method == "dwb":
            fuse_bn_statistics_dwb(merged_model, expert0, expert1, w, X_batch, alpha=alpha)
        elif method == "sr_dwb":
            pass
        
        # --- Step 3: Test-Time Adaptation Steps ---
        if method == "static":
            # No adaptation: directly evaluate with uniform weights
            with torch.no_grad():
                # Apply 0.5/0.5 weights
                w_glob_zero = torch.tensor(0.0, device=device)
                offsets_zero = {name: torch.tensor(0.0, device=device) for name in layer_groups}
                apply_differentiable_merging(merged_model, expert0, expert1, w_glob_zero, offsets_zero)
                
                merged_model.eval()
                preds = merged_model(X_batch)
                acc = (preds.argmax(dim=1) == y_batch).float().mean().item()
                accuracies.append(acc)
                
                # Restore base model parameters
                restore_model_parameters(merged_model, original_state)
            continue
            
        # For optimized setups: perform optimization
        original_forwards = {}
        if method == "sr_dwb":
            for bn_name in ['bn1', 'bn2']:
                bn_module = getattr(merged_model, bn_name)
                original_forwards[bn_name] = bn_module.forward
                mu_wb, sigma_wb = sr_dwb_stats[bn_name]
                bn_module.forward = (lambda bn_mod, m_wb=mu_wb, s_wb=sigma_wb: 
                                     lambda x: differentiable_blended_batch_norm(x, bn_mod, m_wb, s_wb, alpha, eps=bn_mod.eps))(bn_module)

        # Initialize optimization parameters for this batch
        w_global = torch.tensor(0.0, requires_grad=True, device=device)
        offsets = {name: torch.tensor(0.0, requires_grad=True, device=device) for name in layer_groups}
        
        for step in range(N_steps):
            # Clear hook dicts
            inputs_dict.clear()
            grads_dict.clear()
            
            # Differentiable merge
            apply_differentiable_merging(merged_model, expert0, expert1, w_global, offsets)
            
            # Forward pass
            merged_model.train() # Set to train so hooks and backprop are active
            outputs = merged_model(X_batch)
            
            # Compute adaptation losses
            p_outputs = F.softmax(outputs, dim=1)
            L_entropy = -torch.mean(torch.sum(p_outputs * torch.log(p_outputs + 1e-12), dim=1))
            
            # KL routing regularization (keep dynamic weights close to prior)
            # mean of Sigmoid(w_global + offsets)
            lambdas = [torch.sigmoid(w_global + offsets[name]) for name in layer_groups]
            mean_lambda = torch.stack(lambdas).mean()
            # KL divergence between Bernoulli(mean_lambda) and Bernoulli(w[0])
            eps_kl = 1e-8
            L_KL = w[0] * torch.log((w[0] + eps_kl) / (mean_lambda + eps_kl)) + \
                   (1.0 - w[0]) * torch.log((1.0 - w[0] + eps_kl) / (1.0 - mean_lambda + eps_kl))
            
            # Adaptive Consensus Coherence Regularization
            L_coherence = 0.0
            for name in layer_groups:
                # F_tilde preconditioning coefficient
                F_tilde = F_running[name]
                L_coherence += F_tilde * (offsets[name] ** 2)
            
            # Total Loss
            loss = L_entropy + beta * L_KL + gamma_c * L_coherence
            
            # Backward pass
            loss.backward()
            
            # Update running gradients and sensitivities on-the-fly
            with torch.no_grad():
                F_step = {}
                for name in layer_groups:
                    # Retrieve tracked activations and pre-activation gradients from hooks
                    act_sq = torch.mean(inputs_dict[name] ** 2).item() if name in inputs_dict else 1.0
                    grad_sq = torch.mean(grads_dict[name] ** 2).item() if name in grads_dict else 1.0
                    
                    # Update running stats
                    g_running[name] = grad_sq
                    # Kronecker trace sensitivity
                    F_step[name] = act_sq * grad_sq
                
                # Global normalization of sensitivities
                max_F = max(F_step.values()) if len(F_step) > 0 else 1.0
                for name in layer_groups:
                    F_running[name] = F_step[name] / (max_F + 1e-12)
                
                # Execute Kronecker-preconditioned SGD updates
                # 1. Update global consensus
                w_global_grad = w_global.grad if w_global.grad is not None else 0.0
                w_global.copy_(w_global - eta * w_global_grad)
                w_global.grad = None
                
                # 2. Update layer-wise offsets (preconditioned by running Fisher sensitivity)
                for name in layer_groups:
                    offset_grad = offsets[name].grad if offsets[name].grad is not None else 0.0
                    precond = 1.0 / (F_running[name] + epsilon_stab)
                    offsets[name].copy_(offsets[name] - eta * precond * offset_grad)
                    offsets[name].grad = None
        
        # Final classification forward pass with adapted weights on the batch
        with torch.no_grad():
            apply_differentiable_merging(merged_model, expert0, expert1, w_global, offsets)
            merged_model.eval()
            final_outputs = merged_model(X_batch)
            acc = (final_outputs.argmax(dim=1) == y_batch).float().mean().item()
            accuracies.append(acc)
            
            # Restore BN forward paths if they were patched
            if method == "sr_dwb":
                for bn_name, orig_fwd in original_forwards.items():
                    bn_mod = getattr(merged_model, bn_name)
                    bn_mod.forward = orig_fwd
            
            # Restore model structure back to clean state
            restore_model_parameters(merged_model, original_state)

    # Clean up hooks
    for handle in hook_handles:
        handle.remove()
        
    return accuracies, routing_coefs

# Main execution logic for training and evaluation
def main():
    print("="*60)
    print("Starting WB-CoMerge (Wasserstein-Barycenter) Research Cycle")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Setup Datasets (MNIST, FashionMNIST, KMNIST)
    print("Downloading and preparing datasets...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Load KMNIST from downloaded NPZ files
    kmnist_imgs = np.load('kmnist-test-imgs.npz')['arr_0']
    kmnist_labels = np.load('kmnist-test-labels.npz')['arr_0']
    kmnist_imgs_t = torch.from_numpy(kmnist_imgs).float() / 255.0
    kmnist_imgs_t = (kmnist_imgs_t - 0.1307) / 0.3081
    kmnist_imgs_t = kmnist_imgs_t.unsqueeze(1)
    kmnist_labels_t = torch.from_numpy(kmnist_labels).long()
    kmnist_test = torch.utils.data.TensorDataset(kmnist_imgs_t, kmnist_labels_t)
    
    print("Datasets loaded successfully.")
    
    # 2. Expert Pre-training or loading
    expert0_path = "expert_mnist.pt"
    expert1_path = "expert_fmnist.pt"
    
    expert0 = SimpleCNN().to(device)
    expert1 = SimpleCNN().to(device)
    
    # We check if pretrained experts exist
    if os.path.exists(expert0_path) and os.path.exists(expert1_path):
        print("Pretrained experts found. Loading weights...")
        expert0.load_state_dict(torch.load(expert0_path, map_location=device, weights_only=True))
        expert1.load_state_dict(torch.load(expert1_path, map_location=device, weights_only=True))
    else:
        print("Pretrained experts not found. Starting training...")
        # Train MNIST expert (Expert 0)
        loader_mnist = DataLoader(mnist_train, batch_size=64, shuffle=True)
        opt0 = torch.optim.Adam(expert0.parameters(), lr=1e-3, weight_decay=1e-4)
        expert0.train()
        print("Training MNIST Expert (2 epochs)...")
        for epoch in range(2):
            for X_b, y_b in loader_mnist:
                X_b, y_b = X_b.to(device), y_b.to(device)
                opt0.zero_grad()
                loss = F.cross_entropy(expert0(X_b), y_b)
                loss.backward()
                opt0.step()
        torch.save(expert0.state_dict(), expert0_path)
        print("MNIST Expert trained and saved.")
        
        # Train FashionMNIST expert (Expert 1)
        loader_fmnist = DataLoader(fmnist_train, batch_size=64, shuffle=True)
        opt1 = torch.optim.Adam(expert1.parameters(), lr=1e-3, weight_decay=1e-4)
        expert1.train()
        print("Training FashionMNIST Expert (2 epochs)...")
        for epoch in range(2):
            for X_b, y_b in loader_fmnist:
                X_b, y_b = X_b.to(device), y_b.to(device)
                opt1.zero_grad()
                loss = F.cross_entropy(expert1(X_b), y_b)
                loss.backward()
                opt1.step()
        torch.save(expert1.state_dict(), expert1_path)
        print("FashionMNIST Expert trained and saved.")
    
    # Validate standalone expert accuracies
    expert0.eval()
    expert1.eval()
    
    mnist_test_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)
    fmnist_test_loader = DataLoader(fmnist_test, batch_size=256, shuffle=False)
    
    acc0_m = sum((expert0(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().item() for X, y in mnist_test_loader) / len(mnist_test)
    acc1_f = sum((expert1(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().item() for X, y in fmnist_test_loader) / len(fmnist_test)
    
    print(f"Standalone Expert 0 (MNIST) Accuracy: {acc0_m * 100:.2f}%")
    print(f"Standalone Expert 1 (FashionMNIST) Accuracy: {acc1_f * 100:.2f}%")
    
    # 3. Construct the Test Stream
    print("Constructing the sequential test stream...")
    # Stream composition:
    # 0-9: Clean MNIST
    # 10-19: Noisy MNIST
    # 20-29: Clean FashionMNIST
    # 30-39: Noisy FashionMNIST
    # 40-49: Novel KMNIST
    
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)
    fmnist_loader = DataLoader(fmnist_test, batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True)
    
    mnist_iter = iter(mnist_loader)
    fmnist_iter = iter(fmnist_loader)
    kmnist_iter = iter(kmnist_loader)
    
    stream_batches = []
    
    # Batches 0-9: Clean MNIST
    for _ in range(10):
        stream_batches.append(next(mnist_iter))
        
    # Batches 10-19: Noisy MNIST
    for _ in range(10):
        X, y = next(mnist_iter)
        # Add Gaussian noise (sigma = 0.6)
        noise = torch.randn_like(X) * 0.6
        stream_batches.append((X + noise, y))
        
    # Batches 20-29: Clean FashionMNIST
    for _ in range(10):
        stream_batches.append(next(fmnist_iter))
        
    # Batches 30-39: Noisy FashionMNIST
    for _ in range(10):
        X, y = next(fmnist_iter)
        # Add Gaussian noise (sigma = 0.6)
        noise = torch.randn_like(X) * 0.6
        stream_batches.append((X + noise, y))
        
    # Batches 40-49: Novel KMNIST
    for _ in range(10):
        stream_batches.append(next(kmnist_iter))
        
    print(f"Created a stream of {len(stream_batches)} batches of size 64.")
    
    # 4. Evaluate Merging Methods
    methods = {
        "static": "Static Merging (Uniform)",
        "mog": "BK-CoMerge (SOTA baseline)",
        "wb": "WB-CoMerge (Ours)",
        "dwb": "D-WB-CoMerge (Ours, Dynamic)",
        "sr_dwb": "SR-DWB-CoMerge (Ours, Self-Referential)"
    }
    
    results = {}
    routing_curves = {}
    
    optimal_params = {
        "static": {"eta": 0.05, "gamma_c": 0.02, "alpha": 0.15},
        "mog": {"eta": 0.10, "gamma_c": 0.01, "alpha": 0.15},
        "wb": {"eta": 0.10, "gamma_c": 0.05, "alpha": 0.15},
        "dwb": {"eta": 0.10, "gamma_c": 0.02, "alpha": 0.05},
        "sr_dwb": {"eta": 0.10, "gamma_c": 0.01, "alpha": 0.25}
    }
    
    for method_key, method_name in methods.items():
        print(f"Running evaluation of: {method_name}...")
        p = optimal_params[method_key]
        accuracies, routing = run_test_time_adaptation(
            expert0, expert1, stream_batches, 
            method=method_key, device=device, 
            eta=p["eta"], gamma_c=p["gamma_c"], alpha=p["alpha"],
            routing_mode="mi"
        )
        results[method_key] = accuracies
        routing_curves[method_key] = routing
        
        # Segment-wise accuracies
        seq_accs = [
            np.mean(accuracies[0:10]),   # Clean MNIST
            np.mean(accuracies[10:20]),  # Noisy MNIST
            np.mean(accuracies[20:30]),  # Clean FMNIST
            np.mean(accuracies[30:40]),  # Noisy FMNIST
            np.mean(accuracies[40:50])   # Novel KMNIST
        ]
        print(f"  -> {method_name} Segment Accuracies:")
        print(f"     Clean MNIST:  {seq_accs[0]*100:.2f}%")
        print(f"     Noisy MNIST:  {seq_accs[1]*100:.2f}%")
        print(f"     Clean FMNIST: {seq_accs[2]*100:.2f}%")
        print(f"     Noisy FMNIST: {seq_accs[3]*100:.2f}%")
        print(f"     Novel KMNIST: {seq_accs[4]*100:.2f}%")
        print(f"     Overall:      {np.mean(accuracies)*100:.2f}%")
    
    # 5. Save Results Table and Generate Plots
    print("Generating figures and plots...")
    
    # Plot 1: Accuracy over the stream
    plt.figure(figsize=(10, 5))
    x_batches = np.arange(50)
    for method_key, method_name in methods.items():
        plt.plot(x_batches, results[method_key], label=method_name, linewidth=2)
    
    plt.axvline(x=10, color='gray', linestyle='--')
    plt.axvline(x=20, color='gray', linestyle='--')
    plt.axvline(x=30, color='gray', linestyle='--')
    plt.axvline(x=40, color='gray', linestyle='--')
    
    plt.text(3, 0.05, 'Clean\nMNIST', fontsize=8, ha='center')
    plt.text(14, 0.05, 'Noisy\nMNIST', fontsize=8, ha='center')
    plt.text(24, 0.05, 'Clean\nFMNIST', fontsize=8, ha='center')
    plt.text(34, 0.05, 'Noisy\nFMNIST', fontsize=8, ha='center')
    plt.text(44, 0.05, 'Novel\nKMNIST', fontsize=8, ha='center')
    
    plt.xlabel("Stream Batch Index")
    plt.ylabel("Classification Accuracy")
    plt.title("Test-Time Model Merging Performance Comparison")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig("stream_accuracy_comparison.png", dpi=300)
    plt.close()
    
    # Plot 2: Routing Coef lambda0 over the stream
    plt.figure(figsize=(10, 4))
    for method_key, method_name in [("mog", "BK-CoMerge (SOTA)"), ("wb", "WB-CoMerge (Ours)"), ("dwb", "D-WB-CoMerge (Ours)"), ("sr_dwb", "SR-DWB-CoMerge (Ours)")]:
        plt.plot(x_batches, routing_curves[method_key], label=f"{method_name} Routing prior", linewidth=2)
    
    plt.axvline(x=10, color='gray', linestyle='--')
    plt.axvline(x=20, color='gray', linestyle='--')
    plt.axvline(x=30, color='gray', linestyle='--')
    plt.axvline(x=40, color='gray', linestyle='--')
    
    plt.xlabel("Stream Batch Index")
    plt.ylabel("Routing Prior weight for Expert 0 (MNIST)")
    plt.title("Dynamic Soft-Routing Prior Trajectory")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig("routing_trajectory.png", dpi=300)
    plt.close()
    
    # Save the text table of results
    print("Writing results table...")
    with open("experimental_results.txt", "w") as f:
        f.write("Method | Clean MNIST | Noisy MNIST | Clean Fashion | Noisy Fashion | Novel KMNIST | Overall\n")
        f.write("---|---|---|---|---|---|---\n")
        for m_key, m_name in methods.items():
            accs = results[m_key]
            f.write(f"{m_name} | {np.mean(accs[0:10])*100:.2f}% | {np.mean(accs[10:20])*100:.2f}% | {np.mean(accs[20:30])*100:.2f}% | {np.mean(accs[30:40])*100:.2f}% | {np.mean(accs[40:50])*100:.2f}% | {np.mean(accs)*100:.2f}%\n")
    
    print("Research cycle experiments successfully completed and plotted!")

if __name__ == "__main__":
    main()
