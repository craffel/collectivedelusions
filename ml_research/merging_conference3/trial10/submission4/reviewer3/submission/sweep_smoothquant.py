"""
QA-Merge: SmoothQuant Alpha Migration Strength Parameter Sweep
This script performs a rigorous empirical sensitivity sweep of the SmoothQuant migration
strength parameter alpha in [0.0, 1.0] under activation outlier conditions.
It quantizes activations to INT8 and ensembling gating weights to INT4, and evaluates
the impact on Gating Logit MSE and Gating Decision Match Rate.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
torch.manual_seed(2026)
np.random.seed(2026)

def quantize_symmetric(x, bits=8):
    """Symmetric Uniform Quantization to low-precision INT representation"""
    limit = 2 ** (bits - 1)
    max_val = torch.max(torch.abs(x))
    scale = max_val / (limit - 1)
    scale = torch.clamp(scale, min=1e-8)
    
    # Quantize and dequantize
    q = torch.clamp(torch.round(x / scale), -limit, limit - 1)
    x_tilde = q * scale
    return x_tilde

def run_smoothquant_sweep():
    # Simulation Parameters
    B = 1000       # Batch size
    D = 192        # Hidden dimension
    K = 3          # Number of experts
    
    # 1. Generate baseline activations and weights
    X_base = torch.randn(B, D) * 0.1
    W_g = torch.randn(K, D) * 0.05
    
    # 2. Inject heavy-tailed activation outliers in 3 random channels
    # Outlier factor of 40.0, mimicking real-world attention sinks
    outlier_channels = [12, 85, 147]
    X_outliers = X_base.clone()
    for col in outlier_channels:
        X_outliers[:, col] = X_outliers[:, col] * 40.0
        
    # Unquantized Floating-Point Gating Logits (Ideal Baseline)
    logits_float = torch.matmul(X_outliers, W_g.t())
    float_decisions = torch.argmax(logits_float, dim=-1)
    
    # Sweeping alpha from 0.0 to 1.0
    alphas = np.linspace(0.0, 1.0, 11)
    logit_mses = []
    match_rates = []
    
    print("=" * 80)
    print("SMOOTHQUANT ALPHA SENSITIVITY SWEEP UNDER HEAVY OUTLIER CONDITIONS")
    print("-" * 80)
    print(f"{'Alpha':^8} | {'Max Act Range':^15} | {'Max Weight Range':^18} | {'Logit MSE':^12} | {'Match Rate (%)':^15}")
    print("-" * 80)
    
    for alpha in alphas:
        # Compute smoothquant scales: s_i = (max(|X_i|)^alpha) / (max(|W_i|)^(1-alpha))
        max_act = torch.max(torch.abs(X_outliers), dim=0)[0]
        max_wt = torch.max(torch.abs(W_g), dim=0)[0]
        
        # Avoid division by zero
        max_act = torch.clamp(max_act, min=1e-8)
        max_wt = torch.clamp(max_wt, min=1e-8)
        
        # Scale factors
        scales = (max_act ** alpha) / (max_wt ** (1.0 - alpha))
        scales = torch.clamp(scales, min=1e-8)
        
        # Apply scaling matrix S
        X_scaled = X_outliers / scales
        W_scaled = W_g * scales
        
        # Quantize activations to INT8, and gating weights to INT4
        X_quant = quantize_symmetric(X_scaled, bits=8)
        W_quant = quantize_symmetric(W_scaled, bits=4)
        
        # Compute quantized logits
        logits_quant = torch.matmul(X_quant, W_quant.t())
        quant_decisions = torch.argmax(logits_quant, dim=-1)
        
        # Calculate evaluation metrics
        mse = torch.mean((logits_float - logits_quant) ** 2).item()
        match_rate = (quant_decisions == float_decisions).float().mean().item() * 100.0
        
        logit_mses.append(mse)
        match_rates.append(match_rate)
        
        # Max values for logging
        max_scaled_act = torch.max(torch.abs(X_scaled)).item()
        max_scaled_wt = torch.max(torch.abs(W_scaled)).item()
        
        print(f"{alpha:8.1f} | {max_scaled_act:15.4f} | {max_scaled_wt:18.4f} | {mse:12.6f} | {match_rate:14.2f}%")
        
    print("=" * 80)
    
    # Generate Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Plot Logit MSE on primary y-axis
    color = '#d62728'
    ax1.set_xlabel(r'SmoothQuant Migration Strength Parameter ($\alpha$)', fontsize=11)
    ax1.set_ylabel('Gating Logit MSE (Lower is Better)', color=color, fontsize=11)
    line1 = ax1.plot(alphas, logit_mses, color=color, marker='o', linewidth=2.5, label='Logit MSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Plot Gating Match Rate on secondary y-axis
    ax2 = ax1.twinx()
    color = '#1f77b4'
    ax2.set_ylabel('Gating Decision Match Rate (%)', color=color, fontsize=11)
    line2 = ax2.plot(alphas, match_rates, color=color, marker='s', linewidth=2.5, label='Gating Match Rate')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Consolidated legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left')
    
    plt.title(r'SmoothQuant $\alpha$ Sweep under Heavy-Tailed Activation Outliers', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    # Save chart
    os.makedirs('submission/results', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    plt.savefig('submission/results/fig4_smoothquant_sweep.png', dpi=300)
    plt.savefig('results/fig4_smoothquant_sweep.png', dpi=300)
    plt.close()
    print("Chart saved successfully to submission/results/fig4_smoothquant_sweep.png and results/fig4_smoothquant_sweep.png")

if __name__ == "__main__":
    run_smoothquant_sweep()
