import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

# Set seeds
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(42)

# SimpleCNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2, stride=2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2, stride=2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Layer-wise parameter groups
layer_groups = {
    'conv1': ['conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias'],
    'conv2': ['conv2.weight', 'conv2.bias', 'bn2.weight', 'bn2.bias'],
    'fc1': ['fc1.weight', 'fc1.bias'],
    'fc2': ['fc2.weight', 'fc2.bias']
}

# Hooks for tracking sensitivity
inputs_dict = {}
grads_dict = {}

def register_hooks(model):
    handles = []
    
    def get_forward_hook(name):
        def hook(module, inp, out):
            inputs_dict[name] = inp[0].detach()
        return hook
        
    def get_backward_hook(name):
        def hook(module, grad_inp, grad_out):
            grads_dict[name] = grad_out[0].detach()
        return hook

    # Register hooks on layers
    handles.append(model.conv1.register_forward_hook(get_forward_hook('conv1')))
    handles.append(model.conv1.register_full_backward_hook(get_backward_hook('conv1')))
    
    handles.append(model.conv2.register_forward_hook(get_forward_hook('conv2')))
    handles.append(model.conv2.register_full_backward_hook(get_backward_hook('conv2')))
    
    handles.append(model.fc1.register_forward_hook(get_forward_hook('fc1')))
    handles.append(model.fc1.register_full_backward_hook(get_backward_hook('fc1')))
    
    handles.append(model.fc2.register_forward_hook(get_forward_hook('fc2')))
    handles.append(model.fc2.register_full_backward_hook(get_backward_hook('fc2')))
    
    return handles

# BN statistic fusion functions
def fuse_bn_statistics_mog(model, expert0, expert1, w):
    w0, w1 = w[0].item(), w[1].item()
    with torch.no_grad():
        for bn_name in ['bn1', 'bn2']:
            m0 = expert0.state_dict()[f'{bn_name}.running_mean']
            v0 = expert0.state_dict()[f'{bn_name}.running_var']
            m1 = expert1.state_dict()[f'{bn_name}.running_mean']
            v1 = expert1.state_dict()[f'{bn_name}.running_var']
            
            m_fused = w0 * m0 + w1 * m1
            v_fused = w0 * v0 + w1 * v1 + w0 * w1 * ((m0 - m1) ** 2)
            
            model.state_dict()[f'{bn_name}.running_mean'].copy_(m_fused)
            model.state_dict()[f'{bn_name}.running_var'].copy_(v_fused)

def fuse_bn_statistics_wb(model, expert0, expert1, w):
    w0, w1 = w[0].item(), w[1].item()
    with torch.no_grad():
        for bn_name in ['bn1', 'bn2']:
            m0 = expert0.state_dict()[f'{bn_name}.running_mean']
            v0 = expert0.state_dict()[f'{bn_name}.running_var']
            m1 = expert1.state_dict()[f'{bn_name}.running_mean']
            v1 = expert1.state_dict()[f'{bn_name}.running_var']
            
            m_fused = w0 * m0 + w1 * m1
            s0 = torch.sqrt(torch.clamp(v0, min=0.0))
            s1 = torch.sqrt(torch.clamp(v1, min=0.0))
            s_fused = w0 * s0 + w1 * s1
            v_fused = s_fused ** 2
            
            model.state_dict()[f'{bn_name}.running_mean'].copy_(m_fused)
            model.state_dict()[f'{bn_name}.running_var'].copy_(v_fused)

def fuse_bn_statistics_dwb(model, expert0, expert1, w, X_batch, alpha=0.15):
    w0, w1 = w[0].item(), w[1].item()
    with torch.no_grad():
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
            
            m_wb = w0 * m0 + w1 * m1
            s0 = torch.sqrt(torch.clamp(v0, min=0.0))
            s1 = torch.sqrt(torch.clamp(v1, min=0.0))
            s_wb = w0 * s0 + w1 * s1
            
            m_emp = emp_means[bn_name]
            s_emp = torch.sqrt(torch.clamp(emp_vars[bn_name], min=0.0))
            
            m_fused = (1.0 - alpha) * m_wb + alpha * m_emp
            s_fused = (1.0 - alpha) * s_wb + alpha * s_emp
            v_fused = s_fused ** 2
            
            model.state_dict()[f'{bn_name}.running_mean'].copy_(m_fused)
            model.state_dict()[f'{bn_name}.running_var'].copy_(v_fused)

def differentiable_blended_batch_norm(x, bn_module, mu_wb, sigma_wb, alpha, eps=1e-5):
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

def apply_differentiable_merging(model, expert0, expert1, w_global, offsets):
    e0_state = expert0.state_dict()
    e1_state = expert1.state_dict()
    
    for group_name, param_names in layer_groups.items():
        l_j = torch.sigmoid(w_global + offsets[group_name])
        
        for p_name in param_names:
            p0 = e0_state[p_name]
            p1 = e1_state[p_name]
            p_merged = l_j * p0 + (1.0 - l_j) * p1
            
            parts = p_name.split('.')
            submodule = model
            for part in parts[:-1]:
                submodule = getattr(submodule, part)
            attr_name = parts[-1]
            
            if hasattr(submodule, attr_name):
                delattr(submodule, attr_name)
            setattr(submodule, attr_name, p_merged)

def restore_model_parameters(model, original_parameters):
    for p_name, original_param in original_parameters.items():
        parts = p_name.split('.')
        submodule = model
        for part in parts[:-1]:
            submodule = getattr(submodule, part)
        attr_name = parts[-1]
        
        if hasattr(submodule, attr_name):
            delattr(submodule, attr_name)
        submodule.register_parameter(attr_name, original_param)

# Main TTA Loop with flexible routing mode ("entropy" or "mi")
def run_test_time_adaptation(expert0, expert1, stream_batches, method="wb", device="cpu", eta=0.05, gamma_c=0.02, alpha=0.15, routing_mode="entropy"):
    merged_model = SimpleCNN().to(device)
    original_state = {k: v for k, v in merged_model.named_parameters()}
    
    hook_handles = register_hooks(merged_model)
    
    F_running = {name: 1.0 for name in layer_groups}
    g_running = {name: 1.0 for name in layer_groups}
    
    smoothed_gap = None
    gamma_s = 0.9
    
    accuracies = []
    routing_coefs = []
    
    s_scale = 3.0
    epsilon_stab = 0.1
    tau_N = 0.70
    
    N_steps = 5
    beta = 1.5
    
    for b_idx, (X_batch, y_batch) in enumerate(stream_batches):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # --- Step 1: Compute Dynamic Soft Routing Prior ---
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
            with torch.no_grad():
                w_glob_zero = torch.tensor(0.0, device=device)
                offsets_zero = {name: torch.tensor(0.0, device=device) for name in layer_groups}
                apply_differentiable_merging(merged_model, expert0, expert1, w_glob_zero, offsets_zero)
                
                merged_model.eval()
                preds = merged_model(X_batch)
                acc = (preds.argmax(dim=1) == y_batch).float().mean().item()
                accuracies.append(acc)
                restore_model_parameters(merged_model, original_state)
            continue
            
        original_forwards = {}
        if method == "sr_dwb":
            for bn_name in ['bn1', 'bn2']:
                bn_module = getattr(merged_model, bn_name)
                original_forwards[bn_name] = bn_module.forward
                mu_wb, sigma_wb = sr_dwb_stats[bn_name]
                bn_module.forward = (lambda bn_mod, m_wb=mu_wb, s_wb=sigma_wb: 
                                     lambda x: differentiable_blended_batch_norm(x, bn_mod, m_wb, s_wb, alpha, eps=bn_mod.eps))(bn_module)

        w_global = torch.tensor(0.0, requires_grad=True, device=device)
        offsets = {name: torch.tensor(0.0, requires_grad=True, device=device) for name in layer_groups}
        
        for step in range(N_steps):
            inputs_dict.clear()
            grads_dict.clear()
            
            apply_differentiable_merging(merged_model, expert0, expert1, w_global, offsets)
            
            merged_model.train()
            outputs = merged_model(X_batch)
            
            p_outputs = F.softmax(outputs, dim=1)
            L_entropy = -torch.mean(torch.sum(p_outputs * torch.log(p_outputs + 1e-12), dim=1))
            
            lambdas = [torch.sigmoid(w_global + offsets[name]) for name in layer_groups]
            mean_lambda = torch.stack(lambdas).mean()
            eps_kl = 1e-8
            L_KL = w[0] * torch.log((w[0] + eps_kl) / (mean_lambda + eps_kl)) + \
                   (1.0 - w[0]) * torch.log((1.0 - w[0] + eps_kl) / (1.0 - mean_lambda + eps_kl))
            
            L_coherence = 0.0
            for name in layer_groups:
                F_tilde = F_running[name]
                L_coherence += F_tilde * (offsets[name] ** 2)
            
            loss = L_entropy + beta * L_KL + gamma_c * L_coherence
            loss.backward()
            
            with torch.no_grad():
                F_step = {}
                for name in layer_groups:
                    act_sq = torch.mean(inputs_dict[name] ** 2).item() if name in inputs_dict else 1.0
                    grad_sq = torch.mean(grads_dict[name] ** 2).item() if name in grads_dict else 1.0
                    g_running[name] = grad_sq
                    F_step[name] = act_sq * grad_sq
                
                max_F = max(F_step.values()) if len(F_step) > 0 else 1.0
                for name in layer_groups:
                    F_running[name] = F_step[name] / (max_F + 1e-12)
                
                w_global_grad = w_global.grad if w_global.grad is not None else 0.0
                w_global.copy_(w_global - eta * w_global_grad)
                w_global.grad = None
                
                for name in layer_groups:
                    offset_grad = offsets[name].grad if offsets[name].grad is not None else 0.0
                    precond = 1.0 / (F_running[name] + epsilon_stab)
                    offsets[name].copy_(offsets[name] - eta * precond * offset_grad)
                    offsets[name].grad = None
        
        with torch.no_grad():
            apply_differentiable_merging(merged_model, expert0, expert1, w_global, offsets)
            merged_model.eval()
            final_outputs = merged_model(X_batch)
            acc = (final_outputs.argmax(dim=1) == y_batch).float().mean().item()
            accuracies.append(acc)
            
            if method == "sr_dwb":
                for bn_name, orig_fwd in original_forwards.items():
                    bn_mod = getattr(merged_model, bn_name)
                    bn_mod.forward = orig_fwd
            
            restore_model_parameters(merged_model, original_state)

    for handle in hook_handles:
        handle.remove()
        
    return accuracies, routing_coefs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    kmnist_imgs = np.load('kmnist-test-imgs.npz')['arr_0']
    kmnist_labels = np.load('kmnist-test-labels.npz')['arr_0']
    kmnist_imgs_t = torch.from_numpy(kmnist_imgs).float() / 255.0
    kmnist_imgs_t = (kmnist_imgs_t - 0.1307) / 0.3081
    kmnist_imgs_t = kmnist_imgs_t.unsqueeze(1)
    kmnist_labels_t = torch.from_numpy(kmnist_labels).long()
    kmnist_test = TensorDataset(kmnist_imgs_t, kmnist_labels_t)
    
    expert0 = SimpleCNN().to(device)
    expert1 = SimpleCNN().to(device)
    expert0.load_state_dict(torch.load('expert_mnist.pt', map_location=device, weights_only=True))
    expert1.load_state_dict(torch.load('expert_fmnist.pt', map_location=device, weights_only=True))
    
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)
    fmnist_loader = DataLoader(fmnist_test, batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True)
    
    # Run 5 independent trials to average results and remove batch randomness
    all_results = {"entropy": {}, "mi": {}}
    n_trials = 5
    
    for trial in range(n_trials):
        print(f"\n--- TRIAL {trial+1} ---")
        # Build stream
        mnist_iter = iter(mnist_loader)
        fmnist_iter = iter(fmnist_loader)
        kmnist_iter = iter(kmnist_loader)
        
        stream_batches = []
        for _ in range(10): stream_batches.append(next(mnist_iter))
        for _ in range(10):
            X, y = next(mnist_iter)
            stream_batches.append((X + torch.randn_like(X) * 0.6, y))
        for _ in range(10): stream_batches.append(next(fmnist_iter))
        for _ in range(10):
            X, y = next(fmnist_iter)
            stream_batches.append((X + torch.randn_like(X) * 0.6, y))
        for _ in range(10): stream_batches.append(next(kmnist_iter))
        
        configs = {
            "mog": {"eta": 0.10, "gamma_c": 0.01, "alpha": 0.15},
            "wb": {"eta": 0.10, "gamma_c": 0.05, "alpha": 0.15},
            "dwb": {"eta": 0.10, "gamma_c": 0.02, "alpha": 0.05},
            "sr_dwb": {"eta": 0.10, "gamma_c": 0.01, "alpha": 0.25}
        }
        
        for r_mode in ["entropy", "mi"]:
            print(f"Evaluating {r_mode.upper()} routing mode...")
            for method, p in configs.items():
                accs, routing = run_test_time_adaptation(
                    expert0, expert1, stream_batches,
                    method=method, device=device,
                    eta=p["eta"], gamma_c=p["gamma_c"], alpha=p["alpha"],
                    routing_mode=r_mode
                )
                if method not in all_results[r_mode]:
                    all_results[r_mode][method] = []
                all_results[r_mode][method].append(accs)
                
    # Average results
    print("\n" + "="*80)
    print("AVERAGED ACCURACIES COMPARING ROUTING MODES")
    print("="*80)
    print("Method | Mode | Clean MNIST | Noisy MNIST | Clean FMNIST | Noisy FMNIST | Novel KMNIST | Overall")
    print("-" * 105)
    
    latex_rows = []
    
    for method in ["mog", "wb", "dwb", "sr_dwb"]:
        method_name_map = {
            "mog": "BK-CoMerge (MoG)",
            "wb": "\\textbf{WB-CoMerge (Ours)}",
            "dwb": "\\textbf{D-WB-CoMerge (Ours)}",
            "sr_dwb": "\\textbf{SR-DWB-CoMerge (Ours)}"
        }
        method_latex = method_name_map[method]
        
        for r_mode_idx, r_mode in enumerate(["entropy", "mi"]):
            accs_trial_list = all_results[r_mode][method] # length 5, each is (50,)
            accs_trial = np.mean(accs_trial_list, axis=0) # shape (50,)
            
            # Compute overall overall mean & std across trials
            trial_overalls = [np.mean(trial_acc) * 100 for trial_acc in accs_trial_list]
            mean_overall = np.mean(trial_overalls)
            std_overall = np.std(trial_overalls)
            
            seg_accs = [
                np.mean(accs_trial[0:10]) * 100,
                np.mean(accs_trial[10:20]) * 100,
                np.mean(accs_trial[20:30]) * 100,
                np.mean(accs_trial[30:40]) * 100,
                np.mean(accs_trial[40:50]) * 100
            ]
            print(f"{method:<6} | {r_mode:<7} | {seg_accs[0]:.2f}% | {seg_accs[1]:.2f}% | {seg_accs[2]:.2f}% | {seg_accs[3]:.2f}% | {seg_accs[4]:.2f}% | {mean_overall:.2f}% +- {std_overall:.2f}%")
            
            r_mode_latex = "Entropy" if r_mode == "entropy" else "MI (Ours)"
            row_prefix = method_latex if r_mode_idx == 0 else ""
            
            latex_row = f"        {row_prefix:<30} & {r_mode_latex:<10} & ${seg_accs[0]:.2f}\\%$ & ${seg_accs[1]:.2f}\\%$ & ${seg_accs[2]:.2f}\\%$ & ${seg_accs[3]:.2f}\\%$ & ${seg_accs[4]:.2f}\\%$ & ${mean_overall:.2f}\\% \\pm {std_overall:.2f}\\%$ \\\\"
            latex_rows.append(latex_row)
            
    print("\n" + "="*80)
    print("LATEX TABLE CODE FOR TABLE 3")
    print("="*80)
    print("""\\begin{table*}[t]
  \\caption{Comparison of averaged accuracies over 5 independent sequential multi-task stream trials between Entropy-based routing and our proposed Mutual Information (MI)-guided routing. We report the mean and standard deviation of overall accuracy across trials.}
  \\label{tab:routing_comp}
  \\vskip 0.15in
  \\begin{center}
    \\begin{scriptsize}
    \\setlength{\\tabcolsep}{3.0pt}
      \\begin{tabular}{llcccccc}
        \\toprule
        Method & Routing Mode & Clean MNIST & Noisy MNIST & Clean FMNIST & Noisy FMNIST & Novel KMNIST & Overall \\\\
        \\midrule""")
    
    for i, row in enumerate(latex_rows):
        print(row)
        if i % 2 == 1 and i < len(latex_rows) - 1:
            print("        \\midrule")
            
    print("""        \\bottomrule
      \\end{tabular}
    \\end{scriptsize}
  \\end{center}
  \\vskip -0.1in
\\end{table*}""")

if __name__ == "__main__":
    main()
