import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from run_experiments import SimpleCNN, get_datasets, compute_fisher_and_prototypes
import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt

def run_multi_expert_simulation():
    print("=== Starting Multi-Expert Routing Simulation & Benchmark ===")
    device = torch.device("cpu")
    train_mnist, test_mnist, train_fashion, test_fashion, test_kmnist = get_datasets()
    
    # 1. Load primary experts
    expert0 = SimpleCNN(use_cosface=True).to(device)
    expert0.load_state_dict(torch.load("./models/mnist_cosface.pt", map_location=device))
    expert1 = SimpleCNN(use_cosface=True).to(device)
    expert1.load_state_dict(torch.load("./models/fashion_cosface.pt", map_location=device))
    
    prototypes0, _ = compute_fisher_and_prototypes(expert0, test_mnist, device, is_cosface=True)
    prototypes1, _ = compute_fisher_and_prototypes(expert1, test_fashion, device, is_cosface=True)
    
    # Create extra simulated experts (up to M=10)
    experts = [expert0, expert1]
    prototypes = [prototypes0, prototypes1]
    
    # Initialize additional experts with randomized weights (simulating non-specialized or alternative-task experts)
    for m in range(2, 10):
        exp = SimpleCNN(use_cosface=True).to(device)
        # Load a perturbed state dict of expert0/1 to simulate different degrees of expertise
        base_state = expert0.state_dict() if m % 2 == 0 else expert1.state_dict()
        new_state = {}
        for k, v in base_state.items():
            if v.is_floating_point() and ('weight' in k or 'bias' in k):
                # Add noise to simulate perturbed representations
                noise = torch.randn_like(v) * 0.15
                new_state[k] = v + noise
            else:
                new_state[k] = v.clone()
        exp.load_state_dict(new_state, strict=False)
        exp.eval()
        
        # Precompute prototypes for additional experts on Fashion dataset to ensure they are defined
        prot, _ = compute_fisher_and_prototypes(exp, test_fashion if m % 2 == 0 else test_mnist, device, is_cosface=True)
        experts.append(exp)
        prototypes.append(prot)
        
    print(f"Loaded {len(experts)} simulated and real experts.")
    
    # 2. Generate target vision stream
    mnist_loader = DataLoader(test_mnist, batch_size=64, shuffle=False)
    fashion_loader = DataLoader(test_fashion, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(test_kmnist, batch_size=64, shuffle=False)
    
    mnist_iter = iter(mnist_loader)
    fashion_iter = iter(fashion_loader)
    kmnist_iter = iter(kmnist_loader)
    
    stream_batches = []
    for _ in range(5):
        images, labels = next(mnist_iter)
        stream_batches.append((images, labels, "Clean MNIST"))
    for _ in range(5):
        images, labels = next(mnist_iter)
        noise = torch.randn_like(images) * 0.6
        noisy_images = torch.clamp(images + noise, -1.0, 1.0)
        stream_batches.append((noisy_images, labels, "Noisy MNIST"))
    for _ in range(5):
        images, labels = next(fashion_iter)
        stream_batches.append((images, labels, "Clean Fashion"))
    for _ in range(5):
        images, labels = next(fashion_iter)
        noise = torch.randn_like(images) * 0.6
        noisy_images = torch.clamp(images + noise, -1.0, 1.0)
        stream_batches.append((noisy_images, labels, "Noisy Fashion"))
    for _ in range(5):
        images, labels = next(kmnist_iter)
        stream_batches.append((images, labels, "Novel KMNIST"))
        
    # 3. Dynamic Multi-Expert Routing Function
    def multi_expert_route(X_t, active_experts, active_prototypes):
        M = len(active_experts)
        with torch.no_grad():
            # Compute average distances D_m and entropies
            D_m = []
            entropies = []
            for m in range(M):
                exp = active_experts[m]
                prot = active_prototypes[m]
                
                # Forward pass for entropy
                logits = exp(X_t)
                probs = F.softmax(logits, dim=1)
                h = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1)).item()
                entropies.append(h)
                
                # Features for routing
                feat = exp.get_features(X_t)
                dist_list = []
                for i in range(X_t.size(0)):
                    f_i_norm = F.normalize(feat[i], p=2, dim=0)
                    # Normalized L2 distance
                    d = min(torch.sum((f_i_norm - prot[c])**2).item() for c in range(10))
                    dist_list.append(d)
                D_m.append(np.mean(dist_list))
                
            D_m = np.array(D_m)
            entropies = np.array(entropies)
            h_avg = np.mean(entropies)
            
            # Identify top 2 experts
            sorted_indices = np.argsort(D_m)
            m_star = sorted_indices[0]
            m_prime = sorted_indices[1]
            gap = D_m[m_prime] - D_m[m_star]
            
            # Stability factor
            eps_stab = 0.08 / (1.0 + 2.0 * h_avg)
            tau = (gap / M) + eps_stab
            
            # Compute softmax routing weights
            weights = np.exp(-D_m / tau)
            weights = weights / np.sum(weights)
            
            # DEBUG PRINT
            print(f"DEBUG | D_m: {D_m} | h_avg: {h_avg} | gap: {gap} | eps_stab: {eps_stab} | tau: {tau}")
            
            return weights
            
    # 4. Run routing simulation for M=5
    print("\nRunning routing trajectory simulation with M=5 experts...")
    selected_M = 5
    m5_experts = experts[:selected_M]
    m5_prototypes = prototypes[:selected_M]
    
    weights_history = {m: [] for m in range(selected_M)}
    stream_labels = []
    
    for batch_idx, (X_t, Y_t, domain) in enumerate(stream_batches):
        w = multi_expert_route(X_t, m5_experts, m5_prototypes)
        for m in range(selected_M):
            weights_history[m].append(w[m])
        stream_labels.append(domain)
        print(f"Batch {batch_idx+1:2d} ({domain:13s}) | Routing Weights: " + 
              " ".join([f"Exp{m}:{w[m]:.3f}" for m in range(selected_M)]))
              
    # 5. Scaling Benchmark: Time vs. M
    print("\nBenchmarking routing computation time vs. Number of Experts M...")
    X_sample, _, _ = stream_batches[0]
    M_values = list(range(2, 11))
    times = []
    
    for M in M_values:
        sub_experts = experts[:M]
        sub_prototypes = prototypes[:M]
        
        # Warm-up
        for _ in range(3):
            _ = multi_expert_route(X_sample, sub_experts, sub_prototypes)
            
        start_time = time.perf_counter()
        runs = 15
        for _ in range(runs):
            _ = multi_expert_route(X_sample, sub_experts, sub_prototypes)
        elapsed = (time.perf_counter() - start_time) / runs * 1000.0 # ms per batch
        times.append(elapsed)
        print(f"  M = {M:2d} experts | Avg time per batch: {elapsed:.3f} ms")
        
    # 6. Save Plot
    os.makedirs("plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Plot 1: Routing Trajectory
    x_axis = np.arange(1, len(stream_batches) + 1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    labels = ["Expert 0 (MNIST)", "Expert 1 (Fashion)", "Expert 2 (Simulated)", "Expert 3 (Simulated)", "Expert 4 (Simulated)"]
    
    for m in range(selected_M):
        ax1.plot(x_axis, weights_history[m], label=labels[m], color=colors[m], linewidth=2.5, marker='o', markersize=4)
        
    ax1.set_title("Multi-Expert Dynamic Routing Trajectory (M=5)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Stream Batch Index", fontsize=11)
    ax1.set_ylabel("Routing Weight", fontsize=11)
    ax1.set_xticks(x_axis)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Draw domain boundary lines and labels
    boundaries = [5, 10, 15, 20]
    for b in boundaries:
        ax1.axvline(x=b+0.5, color='gray', linestyle=':', alpha=0.8)
    
    # Add domain labels
    ax1.text(3, 0.85, "Clean\nMNIST", ha='center', fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    ax1.text(8, 0.85, "Noisy\nMNIST", ha='center', fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    ax1.text(13, 0.85, "Clean\nFashion", ha='center', fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    ax1.text(18, 0.85, "Noisy\nFashion", ha='center', fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    ax1.text(23, 0.85, "Novel\nKMNIST", ha='center', fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower left", fontsize=9, framealpha=0.9)
    
    # Plot 2: Scaling Benchmark
    ax2.plot(M_values, times, color='#d62728', linewidth=2.5, marker='s', markersize=8, label="Routing Execution Time")
    
    # Add linear fit line to highlight linear O(M) complexity
    fit = np.polyfit(M_values, times, 1)
    fit_fn = np.poly1d(fit)
    ax2.plot(M_values, fit_fn(M_values), color='#1f77b4', linestyle='--', alpha=0.7, label=f"Linear Fit (O(M))")
    
    ax2.set_title("Routing Computation Time vs. Number of Experts", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Number of Experts (M)", fontsize=11)
    ax2.set_ylabel("Execution Time (ms per batch)", fontsize=11)
    ax2.set_xticks(M_values)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc="upper left", fontsize=10)
    
    plt.tight_layout()
    plot_path = "plots/multi_expert_scaling.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved scaling benchmark visualization to: {plot_path}")
    
    # Save text report
    with open("plots/multi_expert_metrics.txt", "w") as f:
        f.write("Multi-Expert Routing Benchmark Report\n")
        f.write("======================================\n\n")
        f.write("1. Dynamic Routing Trajectory (M=5):\n")
        for i, domain in enumerate(stream_labels):
            w_str = ", ".join([f"Exp{m}: {weights_history[m][i]*100.0:.2f}%" for m in range(selected_M)])
            f.write(f"  Batch {i+1:2d} ({domain:13s}) | {w_str}\n")
        f.write("\n2. Complexity Scaling Benchmark (Time vs. M):\n")
        for j, M in enumerate(M_values):
            f.write(f"  M = {M:2d} experts | Time: {times[j]:.3f} ms per batch\n")
            
    print("Saved multi_expert_metrics.txt report.")
    print("=== Multi-Expert Benchmark Complete ===")

if __name__ == "__main__":
    run_multi_expert_simulation()
