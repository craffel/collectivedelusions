import os
import json
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    import random
    random.seed(seed)

# ---------------------------------------------------------
# SAM Optimizer Implementation
# ---------------------------------------------------------
class SAM:
    def __init__(self, optimizer, rho=0.05, eta=1e-12):
        self.optimizer = optimizer
        self.rho = rho
        self.eta = eta
        self.state = {}

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + self.eta)

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                self.state[p] = p.data.clone()
                p.add_(e_w)

    @torch.no_grad()
    def second_step(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.copy_(self.state[p])
        self.optimizer.step()

    def _grad_norm(self):
        shared_device = next(iter(self.optimizer.param_groups[0]["params"])).device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

# ---------------------------------------------------------
# Adapter Wrappers (LoRA and SATA-OP)
# ---------------------------------------------------------
class LoRA_Conv2d(nn.Module):
    def __init__(self, original_conv, r):
        super().__init__()
        self.original_conv = original_conv
        self.original_conv.weight.requires_grad = False
        if original_conv.bias is not None:
            self.original_conv.bias.requires_grad = False
            
        d_out, d_in, kh, kw = original_conv.weight.shape
        self.r = r
        
        # Trainable matrices
        self.A = nn.Parameter(torch.randn(r, d_in * kh * kw) * (1.0 / r))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        
    def forward(self, x):
        d_out, d_in, kh, kw = self.original_conv.weight.shape
        delta_W = self.B @ self.A
        delta_W = delta_W.view(d_out, d_in, kh, kw)
        weight = self.original_conv.weight + delta_W
        return F.conv2d(x, weight, self.original_conv.bias, 
                        self.original_conv.stride, self.original_conv.padding, 
                        self.original_conv.dilation, self.original_conv.groups)

    def get_delta_W(self):
        d_out, d_in, kh, kw = self.original_conv.weight.shape
        return (self.B @ self.A).view(d_out, d_in, kh, kw)

    def set_delta_W(self, delta_W):
        # Used for instantiating the merged weight back into standard LoRA parameterization
        # or we can just keep the merged weight as a buffer during evaluation.
        pass


class SATA_OP_Conv2d(nn.Module):
    def __init__(self, original_conv, r, task_id, num_tasks, seed=42):
        super().__init__()
        self.original_conv = original_conv
        self.original_conv.weight.requires_grad = False
        if original_conv.bias is not None:
            self.original_conv.bias.requires_grad = False
            
        d_out, d_in, kh, kw = original_conv.weight.shape
        self.r = r
        self.task_id = task_id
        
        effective_r = min(r, d_out // num_tasks, (d_in * kh * kw) // num_tasks)
        if effective_r < 1:
            effective_r = 1
            
        # Orthonormal projection matrices P and Q generated reproducibly
        rng = torch.Generator()
        rng.manual_seed(seed)
        
        P_large = torch.randn(d_out, num_tasks * effective_r, generator=rng)
        P_large, _ = torch.linalg.qr(P_large)
        start_idx = task_id * effective_r
        end_idx = (task_id + 1) * effective_r
        P = P_large[:, start_idx:end_idx]
        self.register_buffer("P", P)
        
        Q_large = torch.randn(d_in * kh * kw, num_tasks * effective_r, generator=rng)
        Q_large, _ = torch.linalg.qr(Q_large)
        Q = Q_large[:, start_idx:end_idx]
        self.register_buffer("Q", Q)
        
        # Trainable theta of size (effective_r, effective_r)
        self.theta = nn.Parameter(torch.zeros(effective_r, effective_r))
        nn.init.normal_(self.theta, std=1e-3)
        
    def forward(self, x):
        d_out, d_in, kh, kw = self.original_conv.weight.shape
        delta_W = self.P @ self.theta @ self.Q.t()
        delta_W = delta_W.view(d_out, d_in, kh, kw)
        weight = self.original_conv.weight + delta_W
        return F.conv2d(x, weight, self.original_conv.bias, 
                        self.original_conv.stride, self.original_conv.padding, 
                        self.original_conv.dilation, self.original_conv.groups)

    def get_delta_W(self):
        d_out, d_in, kh, kw = self.original_conv.weight.shape
        return (self.P @ self.theta @ self.Q.t()).view(d_out, d_in, kh, kw)


class SATA_OP_HC_Conv2d(nn.Module):
    def __init__(self, original_conv, r, task_id, num_tasks, seed=42):
        super().__init__()
        self.original_conv = original_conv
        self.original_conv.weight.requires_grad = False
        if original_conv.bias is not None:
            self.original_conv.bias.requires_grad = False
            
        d_out, d_in, kh, kw = original_conv.weight.shape
        self.r = r
        self.task_id = task_id
        
        effective_r = min(r, d_out // num_tasks)
        if effective_r < 1:
            effective_r = 1
            
        # Orthonormal projection matrix P generated reproducibly
        rng = torch.Generator()
        rng.manual_seed(seed)
        
        P_large = torch.randn(d_out, num_tasks * effective_r, generator=rng)
        P_large, _ = torch.linalg.qr(P_large)
        start_idx = task_id * effective_r
        end_idx = (task_id + 1) * effective_r
        P = P_large[:, start_idx:end_idx]
        self.register_buffer("P", P)
        
        # Trainable theta of size (effective_r, d_in * kh * kw)
        # Initializes with small random normal weights to break symmetry while starting close to zero
        self.theta = nn.Parameter(torch.randn(effective_r, d_in * kh * kw) * 1e-3)
        
    def forward(self, x):
        d_out, d_in, kh, kw = self.original_conv.weight.shape
        delta_W = self.P @ self.theta
        delta_W = delta_W.view(d_out, d_in, kh, kw)
        weight = self.original_conv.weight + delta_W
        return F.conv2d(x, weight, self.original_conv.bias, 
                        self.original_conv.stride, self.original_conv.padding, 
                        self.original_conv.dilation, self.original_conv.groups)

    def get_delta_W(self):
        d_out, d_in, kh, kw = self.original_conv.weight.shape
        return (self.P @ self.theta).view(d_out, d_in, kh, kw)


class SATA_OP_SVD_Conv2d(nn.Module):
    def __init__(self, original_conv, r, task_id, num_tasks):
        super().__init__()
        self.original_conv = original_conv
        self.original_conv.weight.requires_grad = False
        if original_conv.bias is not None:
            self.original_conv.bias.requires_grad = False
            
        d_out, d_in, kh, kw = original_conv.weight.shape
        self.r = r
        self.task_id = task_id
        
        # SVD on original_conv.weight to find principal directions
        with torch.no_grad():
            W_flat = original_conv.weight.detach().view(d_out, -1) # (d_out, d_in * kh * kw)
            U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)
            V = Vh.t() # (d_in * kh * kw, k)
            
            k = U.shape[1]
            effective_r = min(r, k // num_tasks)
            if effective_r < 1:
                effective_r = 1
                
            start_idx = task_id * effective_r
            end_idx = (task_id + 1) * effective_r
            
            P = U[:, start_idx:end_idx].clone()
            Q = V[:, start_idx:end_idx].clone()
            
        self.register_buffer("P", P)
        self.register_buffer("Q", Q)
        
        # Trainable theta of size (effective_r, effective_r)
        self.theta = nn.Parameter(torch.zeros(effective_r, effective_r))
        nn.init.normal_(self.theta, std=1e-3)
        
    def forward(self, x):
        d_out, d_in, kh, kw = self.original_conv.weight.shape
        delta_W = self.P @ self.theta @ self.Q.t()
        delta_W = delta_W.view(d_out, d_in, kh, kw)
        weight = self.original_conv.weight + delta_W
        return F.conv2d(x, weight, self.original_conv.bias, 
                        self.original_conv.stride, self.original_conv.padding, 
                        self.original_conv.dilation, self.original_conv.groups)

    def get_delta_W(self):
        d_out, d_in, kh, kw = self.original_conv.weight.shape
        return (self.P @ self.theta @ self.Q.t()).view(d_out, d_in, kh, kw)


class SATA_OP_SVD_HC_Conv2d(nn.Module):
    def __init__(self, original_conv, r, task_id, num_tasks):
        super().__init__()
        self.original_conv = original_conv
        self.original_conv.weight.requires_grad = False
        if original_conv.bias is not None:
            self.original_conv.bias.requires_grad = False
            
        d_out, d_in, kh, kw = original_conv.weight.shape
        self.r = r
        self.task_id = task_id
        
        # SVD on original_conv.weight to get principal left singular vectors P
        with torch.no_grad():
            W_flat = original_conv.weight.detach().view(d_out, -1)
            U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)
            
            k = U.shape[1]
            effective_r = min(r, k // num_tasks)
            if effective_r < 1:
                effective_r = 1
                
            start_idx = task_id * effective_r
            end_idx = (task_id + 1) * effective_r
            
            P = U[:, start_idx:end_idx].clone()
            
        self.register_buffer("P", P)
        
        # Trainable theta of size (effective_r, d_in * kh * kw)
        self.theta = nn.Parameter(torch.randn(effective_r, d_in * kh * kw) * 1e-3)
        
    def forward(self, x):
        d_out, d_in, kh, kw = self.original_conv.weight.shape
        delta_W = self.P @ self.theta
        delta_W = delta_W.view(d_out, d_in, kh, kw)
        weight = self.original_conv.weight + delta_W
        return F.conv2d(x, weight, self.original_conv.bias, 
                        self.original_conv.stride, self.original_conv.padding, 
                        self.original_conv.dilation, self.original_conv.groups)

    def get_delta_W(self):
        d_out, d_in, kh, kw = self.original_conv.weight.shape
        return (self.P @ self.theta).view(d_out, d_in, kh, kw)


# Helper function to recursively apply adapters
def apply_adapters(model, adapter_type, r, task_id, num_tasks, seed=42):
    for name, module in model.named_children():
        if name == 'fc':
            continue
        if isinstance(module, nn.Conv2d):
            if adapter_type == 'sata_op':
                setattr(model, name, SATA_OP_Conv2d(module, r, task_id, num_tasks, seed))
            elif adapter_type == 'sata_op_hc':
                setattr(model, name, SATA_OP_HC_Conv2d(module, r, task_id, num_tasks, seed))
            elif adapter_type == 'sata_op_svd':
                setattr(model, name, SATA_OP_SVD_Conv2d(module, r, task_id, num_tasks))
            elif adapter_type == 'sata_op_svd_hc':
                setattr(model, name, SATA_OP_SVD_HC_Conv2d(module, r, task_id, num_tasks))
            elif adapter_type == 'lora':
                setattr(model, name, LoRA_Conv2d(module, r))
        else:
            apply_adapters(module, adapter_type, r, task_id, num_tasks, seed)

# Helper function to freeze batch normalization layers in eval mode
def set_bn_eval(module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()

# Helper function to extract all delta_Ws
def get_all_delta_ws(model):
    delta_ws = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoRA_Conv2d, SATA_OP_Conv2d, SATA_OP_HC_Conv2d)):
            delta_ws[name] = module.get_delta_W().detach().cpu()
    return delta_ws

# Helper function to inject merged delta_Ws as frozen evaluation weights
class MergedConv2d(nn.Module):
    def __init__(self, original_conv, delta_W):
        super().__init__()
        self.original_conv = original_conv
        self.original_conv.weight.requires_grad = False
        self.register_buffer("delta_W", delta_W.to(original_conv.weight.device))
        
    def forward(self, x):
        weight = self.original_conv.weight + self.delta_W
        return F.conv2d(x, weight, self.original_conv.bias, 
                        self.original_conv.stride, self.original_conv.padding, 
                        self.original_conv.dilation, self.original_conv.groups)

def apply_merged_weights(model, merged_delta_ws, prefix=""):
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if name == 'fc':
            continue
        if isinstance(module, nn.Conv2d):
            # This is the original_conv. If it was wrapped, we extract the original_conv
            if hasattr(module, "original_conv"):
                original_conv = module.original_conv
            else:
                original_conv = module
            if full_name in merged_delta_ws:
                setattr(model, name, MergedConv2d(original_conv, merged_delta_ws[full_name]))
        else:
            apply_merged_weights(module, merged_delta_ws, prefix=full_name)

# ---------------------------------------------------------
# OOD Corruptions
# ---------------------------------------------------------
def apply_corruption(images, corruption_type, severity=1):
    if corruption_type == 'clean':
        return images
    elif corruption_type == 'gaussian_noise':
        noise = torch.randn_like(images) * (0.2 * severity)
        return torch.clamp(images + noise, -1.0, 1.0)
    elif corruption_type == 'gaussian_blur':
        kernel_size = int(2 * severity) | 1
        return transforms.functional.gaussian_blur(images, kernel_size=[kernel_size, kernel_size])
    elif corruption_type == 'contrast_reduction':
        factor = 1.0 - (0.25 * severity)
        mean = images.mean(dim=[-1, -2], keepdim=True)
        return torch.clamp((images - mean) * factor + mean, -1.0, 1.0)
    elif corruption_type == 'rotation':
        angle = 20.0 * severity
        return transforms.functional.rotate(images, angle)
    return images

# ---------------------------------------------------------
# Plotting and Visualization Helper
# ---------------------------------------------------------
def plot_all_results(results, task_names, corruptions):
    # Filter core methods for the main plot results_plot.png
    core_methods = [
        "Standard LoRA",
        "LoRA + SAM",
        "SATA-OP (Original)",
        "SATA-OP-HC (Ours, r=16)",
        "SATA-OP-SVD (Ours, r=8)",
        "SATA-OP-SVD-HC (Ours, r=16)"
    ]
    methods = [m for m in core_methods if m in results]
    if not methods:
        methods = list(results.keys())[:6]
        
    # Plotting main results
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    clean_means = [np.mean([results[m][t]["clean"] for t in task_names]) * 100 for m in methods]
    corr_means = [np.mean([[results[m][t][c] for c in corruptions if c != 'clean'] for t in task_names]) * 100 for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    rects1 = ax[0].bar(x - width/2, clean_means, width, label='Clean', color='#4C72B0')
    rects2 = ax[0].bar(x + width/2, corr_means, width, label='Corrupted (Avg)', color='#C44E52')
    
    ax[0].set_ylabel('Accuracy (%)')
    ax[0].set_title('Multi-Task Merged Performance on Clean vs Corrupted Data')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(methods, rotation=15, ha='right')
    ax[0].set_ylim(0, 100)
    ax[0].legend()
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    for rect in rects1:
        height = rect.get_height()
        ax[0].annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        height = rect.get_height()
        ax[0].annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
                    
    for m_idx, m in enumerate(methods):
        m_corr_avg = []
        for c in corruptions:
            m_corr_avg.append(np.mean([results[m][t][c] for t in task_names]) * 100)
        ax[1].plot(corruptions, m_corr_avg, marker='o', label=m, linewidth=2)
        
    ax[1].set_ylabel('Average Accuracy (%)')
    ax[1].set_xlabel('Corruption Type')
    ax[1].set_title('Robustness Profile Across Corruptions')
    ax[1].set_ylim(0, 100)
    ax[1].legend()
    ax[1].grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("results_plot.png", dpi=150)
    print("\nSaved comparison plots to results_plot.png!")
    
    # ---------------------------------------------------------
    # Generate ablation_rho.png if keys are present
    # ---------------------------------------------------------
    ablation_keys = [
        "SATA-OP (r=8, rho=0.01)", "SATA-OP (r=8, rho=0.1)", "SATA-OP (r=8, rho=0.2)",
        "LoRA + SAM (rho=0.01)", "LoRA + SAM (rho=0.1)", "LoRA + SAM (rho=0.2)"
    ]
    if all(k in results for k in ablation_keys):
        print("\nGenerating ablation_rho.png plot...")
        rhos = [0.01, 0.05, 0.1, 0.2]
        
        # Extract for SATA-OP
        sata_op_clean = []
        sata_op_corr = []
        for r_val in rhos:
            if r_val == 0.05:
                key = "SATA-OP (Original)"
            else:
                key = f"SATA-OP (r=8, rho={r_val})"
            
            clean_accs = [results[key][t]["clean"] for t in task_names]
            corr_accs = [[results[key][t][c] for c in corruptions if c != 'clean'] for t in task_names]
            sata_op_clean.append(np.mean(clean_accs) * 100)
            sata_op_corr.append(np.mean(corr_accs) * 100)
            
        # Extract for LoRA + SAM
        lora_sam_clean = []
        lora_sam_corr = []
        for r_val in rhos:
            if r_val == 0.05:
                key = "LoRA + SAM"
            else:
                key = f"LoRA + SAM (rho={r_val})"
            
            clean_accs = [results[key][t]["clean"] for t in task_names]
            corr_accs = [[results[key][t][c] for c in corruptions if c != 'clean'] for t in task_names]
            lora_sam_clean.append(np.mean(clean_accs) * 100)
            lora_sam_corr.append(np.mean(corr_accs) * 100)
            
        fig_ab, ax_ab = plt.subplots(1, 1, figsize=(8, 6))
        ax_ab.plot(rhos, sata_op_clean, marker='o', color='#4C72B0', linestyle='-', linewidth=2.5, label='SATA-OP (Clean)')
        ax_ab.plot(rhos, sata_op_corr, marker='^', color='#4C72B0', linestyle='--', linewidth=2.5, label='SATA-OP (Corrupted Avg)')
        
        ax_ab.plot(rhos, lora_sam_clean, marker='o', color='#C44E52', linestyle='-', linewidth=2.5, label='LoRA+SAM (Clean)')
        ax_ab.plot(rhos, lora_sam_corr, marker='^', color='#C44E52', linestyle='--', linewidth=2.5, label='LoRA+SAM (Corrupted Avg)')
        
        ax_ab.set_xlabel('SAM Perturbation Radius ($\\rho$)', fontsize=12)
        ax_ab.set_ylabel('Accuracy (%)', fontsize=12)
        ax_ab.set_title('Ablation of SAM Perturbation Radius ($\\rho$) on Merging Accuracy', fontsize=14)
        ax_ab.set_xscale('log')
        ax_ab.set_xticks(rhos)
        ax_ab.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax_ab.set_ylim(40, 95)
        ax_ab.legend(fontsize=10)
        ax_ab.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig("ablation_rho.png", dpi=150)
        print("Saved ablation plots to ablation_rho.png!")

# ---------------------------------------------------------
# Training and Evaluation Main Loop
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_idx", type=int, default=-1, help="Index of configuration to run (0-13), -1 for all")
    parser.add_argument("--only_plot", action="store_true", help="Only perform plotting from results.json")
    args = parser.parse_args()

    task_names = ["MNIST", "FashionMNIST", "KMNIST"]
    corruptions = ["clean", "gaussian_noise", "gaussian_blur", "contrast_reduction", "rotation"]

    if args.only_plot:
        print("Running in ONLY_PLOT mode...")
        with open("results.json", "r") as f:
            results = json.load(f)
        
        print("\n" + "=" * 60)
        print("FINAL SUMMARY OF MULTI-TASK MERGING PERFORMANCE")
        print("=" * 60)
        
        # Compute and print average performance across tasks and corruptions
        for name in results:
            print(f"\nMethod: {name}")
            clean_accs = [results[name][t]["clean"] for t in task_names]
            noise_accs = [results[name][t]["gaussian_noise"] for t in task_names]
            blur_accs = [results[name][t]["gaussian_blur"] for t in task_names]
            contrast_accs = [results[name][t]["contrast_reduction"] for t in task_names]
            rot_accs = [results[name][t]["rotation"] for t in task_names]
            
            avg_clean = np.mean(clean_accs) * 100
            avg_corrupted = np.mean([noise_accs, blur_accs, contrast_accs, rot_accs]) * 100
            avg_overall = np.mean([clean_accs, noise_accs, blur_accs, contrast_accs, rot_accs]) * 100
            
            print(f"  Average Clean Accuracy:     {avg_clean:.2f}%")
            print(f"  Average Corrupted Accuracy: {avg_corrupted:.2f}%")
            print(f"  Average Overall Accuracy:   {avg_overall:.2f}%")
            
        plot_all_results(results, task_names, corruptions)
        return
        for name in results:
            print(f"\nMethod: {name}")
            clean_accs = [results[name][t]["clean"] for t in task_names]
            noise_accs = [results[name][t]["gaussian_noise"] for t in task_names]
            blur_accs = [results[name][t]["gaussian_blur"] for t in task_names]
            contrast_accs = [results[name][t]["contrast_reduction"] for t in task_names]
            rot_accs = [results[name][t]["rotation"] for t in task_names]
            
            avg_clean = np.mean(clean_accs) * 100
            avg_corrupted = np.mean([noise_accs, blur_accs, contrast_accs, rot_accs]) * 100
            avg_overall = np.mean([clean_accs, noise_accs, blur_accs, contrast_accs, rot_accs]) * 100
            
            print(f"  Average Clean Accuracy:     {avg_clean:.2f}%")
            print(f"  Average Corrupted Accuracy: {avg_corrupted:.2f}%")
            print(f"  Average Overall Accuracy:   {avg_overall:.2f}%")
            
        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        methods = list(results.keys())
        clean_means = [np.mean([results[m][t]["clean"] for t in task_names]) * 100 for m in methods]
        corr_means = [np.mean([[results[m][t][c] for c in corruptions if c != 'clean'] for t in task_names]) * 100 for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        rects1 = ax[0].bar(x - width/2, clean_means, width, label='Clean', color='#4C72B0')
        rects2 = ax[0].bar(x + width/2, corr_means, width, label='Corrupted (Avg)', color='#C44E52')
        
        ax[0].set_ylabel('Accuracy (%)')
        ax[0].set_title('Multi-Task Merged Performance on Clean vs Corrupted Data')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(methods)
        ax[0].set_ylim(0, 100)
        ax[0].legend()
        ax[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        for rect in rects1:
            height = rect.get_height()
            ax[0].annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        for rect in rects2:
            height = rect.get_height()
            ax[0].annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
        for m_idx, m in enumerate(methods):
            m_corr_avg = []
            for c in corruptions:
                m_corr_avg.append(np.mean([results[m][t][c] for t in task_names]) * 100)
            ax[1].plot(corruptions, m_corr_avg, marker='o', label=m, linewidth=2)
            
        ax[1].set_ylabel('Average Accuracy (%)')
        ax[1].set_xlabel('Corruption Type')
        ax[1].set_title('Robustness Profile Across Corruptions')
        ax[1].set_ylim(0, 100)
        ax[1].legend()
        ax[1].grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("results_plot.png", dpi=150)
        print("\nSaved comparison plots to results_plot.png!")
        return

    print("=" * 60)
    print("SATA-OP RESEARCH EXPERIMENT PIPELINE")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Datasets
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("Loading datasets...")
    train_mnist = MNIST(root='./data', train=True, download=True, transform=transform)
    train_fmnist = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_kmnist = KMNIST(root='./data', train=True, download=True, transform=transform)
    
    test_mnist = MNIST(root='./data', train=False, download=True, transform=transform)
    test_fmnist = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_kmnist = KMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loaders = [
        DataLoader(train_mnist, batch_size=256, shuffle=True, num_workers=2),
        DataLoader(train_fmnist, batch_size=256, shuffle=True, num_workers=2),
        DataLoader(train_kmnist, batch_size=256, shuffle=True, num_workers=2)
    ]
    
    test_loaders = [
        DataLoader(test_mnist, batch_size=256, shuffle=False, num_workers=2),
        DataLoader(test_fmnist, batch_size=256, shuffle=False, num_workers=2),
        DataLoader(test_kmnist, batch_size=256, shuffle=False, num_workers=2)
    ]
    
    task_names = ["MNIST", "FashionMNIST", "KMNIST"]
    corruptions = ["clean", "gaussian_noise", "gaussian_blur", "contrast_reduction", "rotation"]
    
    # Define experiment configurations
    all_configs = [
        {"name": "Standard LoRA", "adapter": "lora", "use_sam": False, "r": 8, "rho": 0.05},
        {"name": "LoRA + SAM", "adapter": "lora", "use_sam": True, "r": 8, "rho": 0.05},
        {"name": "SATA-OP (Original)", "adapter": "sata_op", "use_sam": True, "r": 8, "rho": 0.05},
        {"name": "SATA-OP-HC (Ours, r=8)", "adapter": "sata_op_hc", "use_sam": True, "r": 8, "rho": 0.05},
        {"name": "SATA-OP-HC (Ours, r=16)", "adapter": "sata_op_hc", "use_sam": True, "r": 16, "rho": 0.05},
        {"name": "SATA-OP (Original, r=16)", "adapter": "sata_op", "use_sam": True, "r": 16, "rho": 0.05},
        {"name": "SATA-OP (r=8, rho=0.01)", "adapter": "sata_op", "use_sam": True, "r": 8, "rho": 0.01},
        {"name": "SATA-OP (r=8, rho=0.1)", "adapter": "sata_op", "use_sam": True, "r": 8, "rho": 0.1},
        {"name": "SATA-OP (r=8, rho=0.2)", "adapter": "sata_op", "use_sam": True, "r": 8, "rho": 0.2},
        {"name": "LoRA + SAM (rho=0.01)", "adapter": "lora", "use_sam": True, "r": 8, "rho": 0.01},
        {"name": "LoRA + SAM (rho=0.1)", "adapter": "lora", "use_sam": True, "r": 8, "rho": 0.1},
        {"name": "LoRA + SAM (rho=0.2)", "adapter": "lora", "use_sam": True, "r": 8, "rho": 0.2},
        {"name": "SATA-OP-SVD (Ours, r=8)", "adapter": "sata_op_svd", "use_sam": True, "r": 8, "rho": 0.05},
        {"name": "SATA-OP-SVD-HC (Ours, r=16)", "adapter": "sata_op_svd_hc", "use_sam": True, "r": 16, "rho": 0.05}
    ]
    
    if args.config_idx != -1:
        configs = [all_configs[args.config_idx]]
    else:
        configs = all_configs
    
    results = {}
    
    # Iterate over configurations
    for config in configs:
        name = config["name"]
        adapter_type = config["adapter"]
        use_sam = config["use_sam"]
        r = config["r"]
        rho = config.get("rho", 0.05)
        
        print("\n" + "-" * 50)
        print(f"RUNNING CONFIGURATION: {name}")
        print("-" * 50)
        
        # Keep track of independent task heads and delta_ws
        task_heads = {}
        task_delta_ws = []
        
        # Train on each task independently
        for task_idx, task_name in enumerate(task_names):
            set_seed(42) # Ensure identical initialization for fair comparison
            
            # Load pre-trained ResNet-18
            base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            base_model.fc = nn.Identity() # backbone output is 512-dim feature
            
            # Explicitly freeze all backbone parameters
            for p in base_model.parameters():
                p.requires_grad = False
            
            # Apply adapters
            apply_adapters(base_model, adapter_type, r, task_idx, len(task_names))
            base_model = base_model.to(device)
            
            # Classification head for this task
            head = nn.Linear(512, 10).to(device)
            
            # Define parameters to optimize (only adapter parameters + head)
            trainable_params = []
            for n, p in base_model.named_parameters():
                if p.requires_grad:
                    trainable_params.append(p)
            for n, p in head.named_parameters():
                trainable_params.append(p)
                
            optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
            if use_sam:
                sam_opt = SAM(optimizer, rho=rho)
                
            criterion = nn.CrossEntropyLoss()
            
            # Train for 2 epochs
            epochs = 2
            print(f"Training {name} on {task_name}...")
            base_model.train()
            base_model.apply(set_bn_eval)
            head.train()
            
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                for images, labels in train_loaders[task_idx]:
                    images, labels = images.to(device), labels.to(device)
                    
                    if use_sam:
                        # First pass
                        features = base_model(images)
                        features = torch.flatten(features, 1)
                        outputs = head(features)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        sam_opt.first_step()
                        optimizer.zero_grad()
                        
                        # Second pass
                        features_p = base_model(images)
                        features_p = torch.flatten(features_p, 1)
                        outputs_p = head(features_p)
                        loss_p = criterion(outputs_p, labels)
                        loss_p.backward()
                        sam_opt.second_step()
                        optimizer.zero_grad()
                    else:
                        features = base_model(images)
                        features = torch.flatten(features, 1)
                        outputs = head(features)
                        loss = criterion(outputs, labels)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    running_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                epoch_loss = running_loss / total
                epoch_acc = correct / total
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%")
                
            # Store trained head and delta_W
            task_heads[task_idx] = head.cpu()
            delta_ws = get_all_delta_ws(base_model)
            task_delta_ws.append(delta_ws)
            
        print(f"\nMerging expert backbones for {name}...")
        # Average the delta_Ws across tasks
        merged_delta_ws = {}
        first_delta_ws = task_delta_ws[0]
        for layer_name in first_delta_ws:
            # Stack and average
            layer_deltas = [task_delta_ws[t][layer_name] for t in range(len(task_names))]
            merged_delta_ws[layer_name] = torch.stack(layer_deltas, dim=0).mean(dim=0)
            
        # Re-initialize clean base model and apply merged weights
        set_seed(42)
        base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        base_model.fc = nn.Identity()
        
        apply_merged_weights(base_model, merged_delta_ws)
        base_model = base_model.to(device)
        base_model.eval()
        
        # Evaluate merged model on each task and corruption
        config_results = {}
        for task_idx, task_name in enumerate(task_names):
            head = task_heads[task_idx].to(device)
            head.eval()
            
            task_corr_results = {}
            for corr in corruptions:
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in test_loaders[task_idx]:
                        images, labels = images.to(device), labels.to(device)
                        # Apply OOD corruption
                        images_corr = apply_corruption(images, corr, severity=1.5)
                        
                        features = base_model(images_corr)
                        features = torch.flatten(features, 1)
                        outputs = head(features)
                        
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                        
                acc = correct / total
                task_corr_results[corr] = acc
                print(f"  Eval on {task_name} | Corruption: {corr:20s} | Acc: {acc*100:.2f}%")
                
            config_results[task_name] = task_corr_results
            
        results[name] = config_results
        
    # Write results to file
    if args.config_idx != -1:
        output_filename = f"results_config_{args.config_idx}.json"
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nConfiguration results written to {output_filename}!")
        return
        
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "=" * 60)
    print("FINAL SUMMARY OF MULTI-TASK MERGING PERFORMANCE")
    print("=" * 60)
    
    # Compute and print average performance across tasks and corruptions
    for name in results:
        print(f"\nMethod: {name}")
        clean_accs = [results[name][t]["clean"] for t in task_names]
        noise_accs = [results[name][t]["gaussian_noise"] for t in task_names]
        blur_accs = [results[name][t]["gaussian_blur"] for t in task_names]
        contrast_accs = [results[name][t]["contrast_reduction"] for t in task_names]
        rot_accs = [results[name][t]["rotation"] for t in task_names]
        
        avg_clean = np.mean(clean_accs) * 100
        avg_corrupted = np.mean([noise_accs, blur_accs, contrast_accs, rot_accs]) * 100
        avg_overall = np.mean([clean_accs, noise_accs, blur_accs, contrast_accs, rot_accs]) * 100
        
        print(f"  Average Clean Accuracy:     {avg_clean:.2f}%")
        print(f"  Average Corrupted Accuracy: {avg_corrupted:.2f}%")
        print(f"  Average Overall Accuracy:   {avg_overall:.2f}%")
        
    plot_all_results(results, task_names, corruptions)

if __name__ == "__main__":
    main()
