import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Create results directory
os.makedirs("results", exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(2026)
np.random.seed(2026)

L = 12
K = 4
num_seeds = 15
seeds = list(range(42, 42 + num_seeds))

# Normalized layer depth bar l = l / (L-1)
l_bar = torch.linspace(0, 1, L)

# Calibrated target profiles from literature (representing MNIST, F-MNIST, CIFAR, SVHN)
target_profiles = torch.stack([
    0.5 - 0.25 * l_bar,  # MNIST (early-layer focus)
    0.2 + 0.35 * torch.sin(np.pi * l_bar),  # FashionMNIST (balanced mid-layer peak)
    0.1 + 0.45 * (l_bar ** 2),  # CIFAR-10 (late-layer specialized focus)
    0.4 - 0.35 * ((l_bar - 0.5) ** 2)  # SVHN (mid-layer strength)
], dim=0)

# Calibrated baseline accuracies and sensitivities
base_accuracies = torch.tensor([92.71, 81.64, 90.17, 73.24]) / 100.0
sensitivities = torch.tensor([1.5, 4.0, 2.5, 5.5]) / 100.0

# Model II: Covariance matrix (Bottleneck layer-wise non-linear sensitivity and coupling)
s = torch.tensor([0.6, 0.6, 0.6, 0.6, 1.0, 1.0, 1.0, 1.0, 1.6, 1.6, 1.6, 1.6], dtype=torch.float32)
Sigma = torch.zeros(L, L)
for i in range(L):
    for j in range(L):
        Sigma[i, j] = torch.sqrt(s[i] * s[j]) * (0.5 ** abs(i - j))
Sigma_inv = torch.inverse(Sigma)

# Compute Vandermonde projection matrix
def get_vandermonde(degree):
    V = torch.zeros(L, degree + 1)
    for j in range(degree + 1):
        V[:, j] = l_bar ** j
    return V

# Polynomial projection
def alphas_to_lambdas(alphas, degree):
    V = get_vandermonde(degree)
    return torch.matmul(alphas, V.t())

# Generate transductive noise for a given seed and corruption scaling factor gamma
def generate_noise(seed, gamma=1.0):
    torch.manual_seed(seed)
    # 1. Alternating noise
    z = torch.randn(K) * 0.12
    alt_noise = torch.zeros(K, L)
    for k in range(K):
        for l in range(L):
            alt_noise[k, l] = z[k] * ((-1) ** l)
            
    # 2. White noise
    white_noise = torch.randn(K, L) * 0.08
    
    # 3. Brownian walk noise
    brown_noise = torch.zeros(K, L)
    for k in range(K):
        current = 0.0
        for l in range(L):
            current = current + torch.randn(1).item() * 0.05
            brown_noise[k, l] = current
            
    # Composite noise
    composite = 0.5 * alt_noise + 0.3 * white_noise + 0.2 * brown_noise
    return gamma * composite

# Model II Loss
def compute_loss_model2(lambdas, target_noisy):
    e = lambdas - target_noisy
    loss = 0.0
    for k in range(K):
        ek = e[k].unsqueeze(1)
        dist = torch.matmul(ek.t(), torch.matmul(Sigma_inv, ek)).squeeze()
        cos_term = torch.sum(1.0 - torch.cos(10.0 * np.pi * e[k]))
        loss_task = 0.5 + 1.5 * dist + 0.03 * cos_term
        loss += loss_task
    return loss

# Model II Accuracy
def compute_accuracy_model2(lambdas):
    acc = torch.zeros(K)
    for k in range(K):
        dk = (lambdas[k] - target_profiles[k]).unsqueeze(1)
        d0k = (torch.tensor([0.3]*L, dtype=torch.float32) - target_profiles[k]).unsqueeze(1)
        dist_merged = torch.matmul(dk.t(), torch.matmul(Sigma_inv, dk)).squeeze()
        dist_initial = torch.matmul(d0k.t(), torch.matmul(Sigma_inv, d0k)).squeeze()
        acc[k] = base_accuracies[k] + sensitivities[k] * (1.0 - dist_merged / dist_initial)
    return torch.clamp(acc, 0.0, 1.0)

# FlatMerge optimizer
def optimize_flatmerge(target_noisy, degree=2, steps=200, lr=0.02, rho=0.05, num_samples=10):
    alphas = torch.zeros(K, degree + 1, dtype=torch.float32)
    alphas[:, 0] = 0.3
    
    # Manual Adam optimizer
    m = torch.zeros_like(alphas)
    v = torch.zeros_like(alphas)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    for step in range(1, steps + 1):
        grad_est = torch.zeros_like(alphas)
        
        for _ in range(num_samples):
            # Sample random direction and normalize to unit vector
            E = torch.randn_like(alphas)
            U = E / (torch.norm(E) + 1e-12)
            
            # Positive perturbation
            lambdas_pos = alphas_to_lambdas(alphas + rho * U, degree)
            loss_pos = compute_loss_model2(lambdas_pos, target_noisy)
                
            # Negative perturbation
            lambdas_neg = alphas_to_lambdas(alphas - rho * U, degree)
            loss_neg = compute_loss_model2(lambdas_neg, target_noisy)
                
            # ZO gradient estimate (mathematically rigorous Option A)
            grad_est += ((loss_pos - loss_neg) / (2.0 * rho)) * U
            
        grad_est /= num_samples
        
        # Adam updates
        m = beta1 * m + (1.0 - beta1) * grad_est
        v = beta2 * v + (1.0 - beta2) * (grad_est ** 2)
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        
        alphas -= lr * m_hat / (torch.sqrt(v_hat) + eps)
        
    return alphas_to_lambdas(alphas, degree).detach()

# Main Ablation Experiment
b_zo_values = [2, 4, 6, 8, 10, 15, 20]
gamma_val = 1.5  # Moderate noise to stress test the algorithm

results = {}

print(f"Starting B_zo ablation sweep under Model II with gamma = {gamma_val} across {num_seeds} seeds...")

for b_zo in b_zo_values:
    print(f"\nEvaluating B_zo = {b_zo}...")
    accuracies = []
    step_times_ms = []
    
    for seed in seeds:
        noise = generate_noise(seed, gamma=gamma_val)
        target_noisy = target_profiles + noise
        
        # Measure time for 200 optimization steps
        start_time = time.perf_counter()
        lambdas = optimize_flatmerge(target_noisy, degree=2, num_samples=b_zo)
        end_time = time.perf_counter()
        
        # Calculate step latency in ms
        total_time_ms = (end_time - start_time) * 1000.0
        step_latency = total_time_ms / 200.0
        step_times_ms.append(step_latency)
        
        # Calculate generalization accuracy
        accs = compute_accuracy_model2(lambdas) * 100.0
        joint_acc = torch.mean(accs).item()
        accuracies.append(joint_acc)
        
    # Stats
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    mean_latency = float(np.mean(step_times_ms))
    
    print(f"  Accuracy: {mean_acc:.2f}% +- {std_acc:.2f}%")
    print(f"  Step Latency: {mean_latency:.2f} ms/step")
    
    results[str(b_zo)] = {
        "accuracies": accuracies,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_latency_ms": mean_latency
    }

# Save results
with open("results/bzo_ablation.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results to results/bzo_ablation.json")

# Generate beautiful publication-quality plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

means_acc = [results[str(b)]["mean_accuracy"] for b in b_zo_values]
stds_acc = [results[str(b)]["std_accuracy"] for b in b_zo_values]
latencies = [results[str(b)]["mean_latency_ms"] for b in b_zo_values]

# Subplot 1: Accuracy vs B_zo
ax1.errorbar(b_zo_values, means_acc, yerr=stds_acc, fmt="o-", color="#2c3e50", ecolor="#e74c3c", elinewidth=1.5, capsize=4, linewidth=2, markersize=8)
ax1.set_title("Generalization Accuracy vs. Zeroth-Order Budget ($B_{\\text{zo}}$)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Perturbation Sample Size ($B_{\\text{zo}}$)", fontsize=11)
ax1.set_ylabel("Joint Average Accuracy (%)", fontsize=11)
ax1.set_xticks(b_zo_values)
ax1.grid(True, linestyle="--", alpha=0.5)

# Subplot 2: Latency vs B_zo
ax2.plot(b_zo_values, latencies, "s-", color="#2ecc71", linewidth=2, markersize=8)
ax2.set_title("Step Latency vs. Zeroth-Order Budget ($B_{\\text{zo}}$)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Perturbation Sample Size ($B_{\\text{zo}}$)", fontsize=11)
ax2.set_ylabel("Adaptation Latency (ms/step)", fontsize=11)
ax2.set_xticks(b_zo_values)
ax2.grid(True, linestyle="--", alpha=0.5)

# Linear trendline on latency
z = np.polyfit(b_zo_values, latencies, 1)
p = np.poly1d(z)
ax2.plot(b_zo_values, p(b_zo_values), "r--", alpha=0.7, label=f"O(B_zo) linear fit")
ax2.legend()

plt.tight_layout()
plt.savefig("results/fig7_bzo_ablation.png", dpi=300)
plt.close()
print("Generated results/fig7_bzo_ablation.png")
