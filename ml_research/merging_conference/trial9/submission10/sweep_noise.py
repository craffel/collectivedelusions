import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.func import functional_call

# --- Define Model Architecture ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_layer_group_index(param_name):
    if 'conv1' in param_name or 'bn1' in param_name:
        return 0
    elif 'conv2' in param_name or 'bn2' in param_name:
        return 1
    elif 'fc1' in param_name:
        return 2
    elif 'fc2' in param_name:
        return 3
    else:
        return 0

def run_noise_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Noise Robustness Sweep on: {device}")
    
    # 1. Prepare Datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Loading datasets...")
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=False, transform=transform)
    
    print("Loading expert checkpoints...")
    state0 = torch.load("checkpoints/expert0_mnist.pt", map_location=device)
    state1 = torch.load("checkpoints/expert1_fmnist.pt", map_location=device)
    
    expert0_eval = SimpleCNN().to(device)
    expert0_eval.load_state_dict(state0)
    expert0_eval.eval()
    
    expert1_eval = SimpleCNN().to(device)
    expert1_eval.load_state_dict(state1)
    expert1_eval.eval()
    
    # Methods to evaluate
    methods = [
        "Static Merging",
        "Fixed TTA",
        "BK-CoMerge (eta=0.05)",
        "TS-BK-CoMerge (eta=0.005)",
        "SAM-TTMM (rho=0.05, eta=0.005)",
        "SAM-TTMM (rho=0.10, eta=0.005)"
    ]
    
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sweep_results = {m: [] for m in methods}
    
    for sigma in noise_levels:
        print(f"\n--- Evaluating Noise Level sigma = {sigma} ---")
        
        # Construct the stream for this noise level
        np.random.seed(42)
        torch.manual_seed(42)
        
        mnist_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True)
        fmnist_loader = torch.utils.data.DataLoader(fmnist_test, batch_size=64, shuffle=True)
        kmnist_loader = torch.utils.data.DataLoader(kmnist_test, batch_size=64, shuffle=True)
        
        mnist_iter = iter(mnist_loader)
        fmnist_iter = iter(fmnist_loader)
        kmnist_iter = iter(kmnist_loader)
        
        stream_batches = []
        for _ in range(10):
            images, labels = next(mnist_iter)
            stream_batches.append((images, labels))
        for _ in range(10):
            images, labels = next(mnist_iter)
            noisy_images = images + torch.randn_like(images) * sigma
            stream_batches.append((noisy_images, labels))
        for _ in range(10):
            images, labels = next(fmnist_iter)
            stream_batches.append((images, labels))
        for _ in range(10):
            images, labels = next(fmnist_iter)
            noisy_images = images + torch.randn_like(images) * sigma
            stream_batches.append((noisy_images, labels))
        for _ in range(10):
            images, labels = next(kmnist_iter)
            stream_batches.append((images, labels))
            
        # Helper for BN running statistics fusion
        def get_fused_bn_buffers(state0, state1, w):
            fused_buffers = {}
            for name in state0.keys():
                if 'running_mean' in name:
                    mean0 = state0[name]
                    mean1 = state1[name]
                    fused_buffers[name] = w[0] * mean0 + w[1] * mean1
                elif 'running_var' in name:
                    var0 = state0[name]
                    var1 = state1[name]
                    mean0_name = name.replace('running_var', 'running_mean')
                    mean0 = state0[mean0_name]
                    mean1 = state1[mean0_name]
                    mean_fused = w[0] * mean0 + w[1] * mean1
                    fused_buffers[name] = w[0] * (var0 + mean0**2) + w[1] * (var1 + mean1**2) - mean_fused**2
                elif 'num_batches_tracked' in name:
                    fused_buffers[name] = state0[name]
            return fused_buffers

        for method in methods:
            w_global = torch.tensor(0.0, device=device)
            deltas = [torch.tensor(0.0, device=device) for _ in range(4)]
            g_running = [torch.tensor(1.0, device=device) for _ in range(4)]
            
            model_base = SimpleCNN().to(device)
            model_base.eval()
            
            activations = {}
            gradients = {}
            
            def make_fw_hook(idx):
                def hook(module, input, output):
                    activations[idx] = input[0].detach()
                return hook
                
            def make_bw_hook(idx):
                def hook(module, grad_input, grad_output):
                    gradients[idx] = grad_output[0].detach()
                return hook
                
            h_f1 = model_base.conv1.register_forward_hook(make_fw_hook(0))
            h_b1 = model_base.conv1.register_full_backward_hook(make_bw_hook(0))
            h_f2 = model_base.conv2.register_forward_hook(make_fw_hook(1))
            h_b2 = model_base.conv2.register_full_backward_hook(make_bw_hook(1))
            h_f3 = model_base.fc1.register_forward_hook(make_fw_hook(2))
            h_b3 = model_base.fc1.register_full_backward_hook(make_bw_hook(2))
            h_f4 = model_base.fc2.register_forward_hook(make_fw_hook(3))
            h_b4 = model_base.fc2.register_full_backward_hook(make_bw_hook(3))
            
            method_accs = []
            smoothed_gap = None
            
            for batch_idx, (images, labels) in enumerate(stream_batches):
                images, labels = images.to(device), labels.to(device)
                
                with torch.no_grad():
                    out0 = expert0_eval(images)
                    out1 = expert1_eval(images)
                    p0 = F.softmax(out0, dim=1)
                    p1 = F.softmax(out1, dim=1)
                    H0 = -torch.sum(p0 * torch.log(p0 + 1e-9), dim=1).mean().item()
                    H1 = -torch.sum(p1 * torch.log(p1 + 1e-9), dim=1).mean().item()
                    H_avg = 0.5 * (H0 + H1)
                    
                tau_N = 1.5
                s_scale = 2.0
                eps_stab = 0.1
                
                if H_avg > tau_N:
                    w = [0.5, 0.5]
                else:
                    gap = abs(H0 - H1)
                    if "TS-" in method or "SAM-" in method:
                        gamma_s = 0.9
                        if smoothed_gap is None:
                            smoothed_gap = gap
                        else:
                            smoothed_gap = gamma_s * smoothed_gap + (1.0 - gamma_s) * gap
                        tau_self = smoothed_gap / s_scale + eps_stab
                    else:
                        tau_self = gap / s_scale + eps_stab
                    
                    v0 = -H0 / tau_self
                    v1 = -H1 / tau_self
                    max_v = max(v0, v1)
                    exp0 = np.exp(v0 - max_v)
                    exp1 = np.exp(v1 - max_v)
                    sum_exp = exp0 + exp1
                    w = [exp0 / sum_exp, exp1 / sum_exp]
                    
                fused_bn = get_fused_bn_buffers(state0, state1, w)
                
                # Parameters based on method
                Nstep = 5
                gamma_c = 0.02
                eta = 0.05
                beta_kl = 0.1
                rho = 0.05
                
                if "eta=0.005" in method:
                    eta = 0.005
                    beta_kl = 0.01
                if "rho=0.10" in method:
                    rho = 0.10
                    
                w_global_adapted = w_global.clone().detach().requires_grad_(True)
                deltas_adapted = [d.clone().detach().requires_grad_(True) for d in deltas]
                
                for step in range(Nstep):
                    lambdas = [torch.sigmoid(w_global_adapted + deltas_adapted[j]) for j in range(4)]
                    
                    params = {}
                    for name, param in model_base.named_parameters():
                        idx = get_layer_group_index(name)
                        lam = lambdas[idx]
                        params[name] = (1.0 - lam) * state0[name] + lam * state1[name]
                    for name in fused_bn.keys():
                        params[name] = fused_bn[name]
                        
                    out = functional_call(model_base, params, images)
                    probs = F.softmax(out, dim=1)
                    L_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean()
                    
                    mean_lambda = torch.stack(lambdas).mean()
                    p_dist = torch.stack([1.0 - mean_lambda, mean_lambda])
                    q_dist = torch.tensor(w, device=device)
                    L_KL = torch.sum(p_dist * torch.log((p_dist + 1e-9) / (q_dist + 1e-9)))
                    
                    L_coherence = 0.0
                    for j in range(4):
                        act_mean = torch.mean(activations[j] ** 2) if j in activations else 1.0
                        L_coherence += gamma_c * act_mean * g_running[j] * (deltas_adapted[j] ** 2)
                        
                    loss = L_entropy + beta_kl * L_KL + L_coherence
                    
                    if method == "Static Merging":
                        break
                    elif method == "Fixed TTA":
                        loss_tta = L_entropy
                        w_global_adapted.grad = None
                        for d in deltas_adapted:
                            d.grad = None
                        loss_tta.backward()
                        with torch.no_grad():
                            w_global_adapted -= 0.01 * w_global_adapted.grad if w_global_adapted.grad is not None else 0.0
                            for d in deltas_adapted:
                                d -= 0.01 * d.grad if d.grad is not None else 0.0
                    elif "BK-CoMerge" in method:
                        w_global_adapted.grad = None
                        for d in deltas_adapted:
                            d.grad = None
                        loss.backward()
                        
                        with torch.no_grad():
                            for j in range(4):
                                if j in gradients:
                                    mean_g2 = torch.mean(gradients[j] ** 2)
                                    g_running[j] = 0.9 * g_running[j] + 0.1 * mean_g2
                                    
                            for j in range(4):
                                Fj = torch.mean(activations[j] ** 2) * torch.mean(gradients[j] ** 2) if j in activations and j in gradients else 1.0
                                d_grad = deltas_adapted[j].grad if deltas_adapted[j].grad is not None else 0.0
                                deltas_adapted[j] -= eta * (1.0 / (Fj + eps_stab)) * d_grad
                                
                            w_grad = w_global_adapted.grad if w_global_adapted.grad is not None else 0.0
                            w_global_adapted -= eta * w_grad
                    elif "SAM-TTMM" in method:
                        # 1. First backward
                        w_global_adapted.grad = None
                        for d in deltas_adapted:
                            d.grad = None
                        loss.backward()
                        
                        with torch.no_grad():
                            F_sens = []
                            for j in range(4):
                                if j in gradients:
                                    mean_g2 = torch.mean(gradients[j] ** 2)
                                    g_running[j] = 0.9 * g_running[j] + 0.1 * mean_g2
                                Fj = torch.mean(activations[j] ** 2) * torch.mean(gradients[j] ** 2) if j in activations and j in gradients else 1.0
                                F_sens.append(Fj)
                                
                            d_w = w_global_adapted.grad if w_global_adapted.grad is not None else torch.tensor(0.0, device=device)
                            d_deltas = []
                            for j in range(4):
                                d_d = deltas_adapted[j].grad if deltas_adapted[j].grad is not None else torch.tensor(0.0, device=device)
                                d_deltas.append(d_d / (F_sens[j] + eps_stab))
                                
                            dir_norm = torch.sqrt(d_w**2 + sum([torch.sum(dd**2) for dd in d_deltas]) + eps_stab)
                            epsilon_w = rho * d_w / dir_norm
                            epsilon_deltas = [rho * dd / dir_norm for dd in d_deltas]
                            
                            w_perturbed = w_global_adapted + epsilon_w
                            deltas_perturbed = [deltas_adapted[j] + epsilon_deltas[j] for j in range(4)]
                            
                        w_perturbed.requires_grad_(True)
                        for dp in deltas_perturbed:
                            dp.requires_grad_(True)
                            
                        # 2. Second forward
                        lambdas_p = [torch.sigmoid(w_perturbed + deltas_perturbed[j]) for j in range(4)]
                        params_p = {}
                        for name, param in model_base.named_parameters():
                            idx = get_layer_group_index(name)
                            lam = lambdas_p[idx]
                            params_p[name] = (1.0 - lam) * state0[name] + lam * state1[name]
                        for name in fused_bn.keys():
                            params_p[name] = fused_bn[name]
                            
                        out_p = functional_call(model_base, params_p, images)
                        probs_p = F.softmax(out_p, dim=1)
                        L_entropy_p = -torch.sum(probs_p * torch.log(probs_p + 1e-9), dim=1).mean()
                        
                        mean_lambda_p = torch.stack(lambdas_p).mean()
                        p_dist_p = torch.stack([1.0 - mean_lambda_p, mean_lambda_p])
                        L_KL_p = torch.sum(p_dist_p * torch.log((p_dist_p + 1e-9) / (q_dist + 1e-9)))
                        
                        L_coherence_p = 0.0
                        for j in range(4):
                            act_mean = torch.mean(activations[j] ** 2) if j in activations else 1.0
                            L_coherence_p += gamma_c * act_mean * g_running[j] * (deltas_perturbed[j] ** 2)
                            
                        loss_perturbed = L_entropy_p + beta_kl * L_KL_p + L_coherence_p
                        
                        w_perturbed.grad = None
                        for dp in deltas_perturbed:
                            dp.grad = None
                        loss_perturbed.backward()
                        
                        # 3. Update using perturbed gradients
                        with torch.no_grad():
                            for j in range(4):
                                dp_grad = deltas_perturbed[j].grad if deltas_perturbed[j].grad is not None else 0.0
                                deltas_adapted[j] -= eta * (1.0 / (F_sens[j] + eps_stab)) * dp_grad
                                
                            wp_grad = w_perturbed.grad if w_perturbed.grad is not None else 0.0
                            w_global_adapted -= eta * wp_grad

                # --- Evaluation ---
                with torch.no_grad():
                    final_lambdas = [torch.sigmoid(w_global_adapted + deltas_adapted[j]) for j in range(4)]
                    
                    final_params = {}
                    for name, param in model_base.named_parameters():
                        idx = get_layer_group_index(name)
                        lam = final_lambdas[idx]
                        final_params[name] = (1.0 - lam) * state0[name] + lam * state1[name]
                    for name in fused_bn.keys():
                        final_params[name] = fused_bn[name]
                        
                    final_out = functional_call(model_base, final_params, images)
                    _, predicted = final_out.max(1)
                    correct = predicted.eq(labels).sum().item()
                    acc = 100. * correct / labels.size(0)
                    method_accs.append(acc)
                    
                if method != "Static Merging":
                    with torch.no_grad():
                        w_global = w_global_adapted.clone().detach()
                        deltas = [d.clone().detach() for d in deltas_adapted]
                        
            h_f1.remove()
            h_b1.remove()
            h_f2.remove()
            h_b2.remove()
            h_f3.remove()
            h_b3.remove()
            h_f4.remove()
            h_b4.remove()
            
            overall_acc = np.mean(method_accs)
            sweep_results[method].append(overall_acc)
            print(f"Method: {method:<35} | Overall Accuracy: {overall_acc:.4f}%")
            
    # Save the JSON results
    results_to_save = {
        "noise_levels": noise_levels,
        "results": sweep_results
    }
    with open("noise_sweep_results.json", "w") as f:
        json.dump(results_to_save, f, indent=4)
        
    print("\nSweep completed! Results saved to noise_sweep_results.json")
    
    # Generate the noise robustness plot
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'v', 'D', '*']
    for i, m in enumerate(methods):
        plt.plot(noise_levels, sweep_results[m], label=m, marker=markers[i % len(markers)], linewidth=2, markersize=8)
        
    plt.title("Noise Robustness Curve: Accuracy vs. Noise Level $\sigma$ (Gaussian)", fontsize=14)
    plt.xlabel("Gaussian Noise Standard Deviation ($\sigma$)", fontsize=12)
    plt.ylabel("Overall Stream Classification Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig("noise_robustness.png", dpi=300)
    plt.close()
    print("Plot saved as noise_robustness.png")

if __name__ == "__main__":
    run_noise_sweep()
