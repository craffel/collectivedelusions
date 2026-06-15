import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

# Sandbox configuration parameters
D = 192
K = 4
L = 14
d = 48
sigmas = [0.05, 0.15, 0.40, 1.20]
biases = [0.0, 0.0, -0.90, -2.30]
gamma_val = 0.05

def get_signatures(rho=0.0):
    v_orth = np.zeros((K, D))
    for k in range(K):
        v_orth[k, k*d:(k+1)*d] = 1.0 / np.sqrt(d)
        
    if rho > 0.0:
        Sigma = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                Sigma[i, j] = rho ** abs(i - j)
        U, S, Vt = np.linalg.svd(Sigma)
        Sigma_half = U @ np.diag(np.sqrt(S)) @ Vt
        
        v = np.zeros((K, D))
        for k in range(K):
            v[k] = Sigma_half @ v_orth[k]
            v[k] /= np.linalg.norm(v[k])
    else:
        v = v_orth.copy()
    return v

def generate_dataset(v, num_samples_per_task=250, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    samples = []
    labels = []
    for k in range(K):
        for _ in range(num_samples_per_task):
            eps = np.random.normal(0, sigmas[k], D)
            samples.append(v[k] + eps)
            labels.append(k)
            
    return torch.tensor(np.array(samples), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

class SandboxModule(nn.Module):
    def __init__(self, v, biases, gating='softmax', zero_init=True):
        super().__init__()
        self.gating = gating
        self.register_buffer('v', torch.tensor(v, dtype=torch.float32))
        self.register_buffer('biases', torch.tensor(biases, dtype=torch.float32))
        
        self.weight = nn.Parameter(torch.empty(D, K))
        self.bias = nn.Parameter(torch.empty(K))
        if zero_init:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)
        else:
            nn.init.normal_(self.weight, std=0.01)
            nn.init.zeros_(self.bias)
            
    def forward(self, h0):
        logits_g = h0 @ self.weight + self.bias
        if self.gating == 'softmax':
            alpha = torch.softmax(logits_g, dim=-1)
        else:
            alpha = torch.sigmoid(logits_g)
            
        h = h0
        for l in range(4, L + 1):
            update = torch.zeros_like(h)
            for k in range(K):
                update += alpha[:, k:k+1] * gamma_val * (self.v[k] - h)
            h = h + update
            
        h_expanded = h.unsqueeze(1)
        v_expanded = self.v.unsqueeze(0)
        dists = torch.sum((h_expanded - v_expanded)**2, dim=-1)
        logits = -dists + self.biases
        return logits, alpha

def run_sable(h0, v_tensor, biases_tensor, tau=0.05):
    h_norm = h0 / torch.norm(h0, dim=-1, keepdim=True).clamp(min=1e-8)
    v_norm = v_tensor / torch.norm(v_tensor, dim=-1, keepdim=True).clamp(min=1e-8)
    sims = h_norm @ v_norm.T
    alpha = torch.softmax(sims / tau, dim=-1)
    
    h = h0
    for l in range(4, L + 1):
        update = torch.zeros_like(h)
        for k in range(K):
            update += alpha[:, k:k+1] * gamma_val * (v_tensor[k] - h)
        h = h + update
        
    h_expanded = h.unsqueeze(1)
    v_expanded = v_tensor.unsqueeze(0)
    dists = torch.sum((h_expanded - v_expanded)**2, dim=-1)
    logits = -dists + biases_tensor
    return logits, alpha

def run_chemmerge(h0, v_tensor, biases_tensor, tau=0.01, dt=1.5, k_decay=0.3):
    C = torch.ones(h0.shape[0], K) * 0.25
    h = h0
    alphas = []
    
    for l in range(4, L + 1):
        h_norm = h / torch.norm(h, dim=-1, keepdim=True).clamp(min=1e-8)
        v_norm = v_tensor / torch.norm(v_tensor, dim=-1, keepdim=True).clamp(min=1e-8)
        sims = h_norm @ v_norm.T
        
        rates = torch.softmax(sims / tau, dim=-1)
        C_next = C + dt * (rates * (1.0 - C) - k_decay * C)
        C = torch.clamp(C_next, 0.0, 1.0)
        
        alpha = C / torch.sum(C, dim=-1, keepdim=True).clamp(min=1e-8)
        alphas.append(alpha.clone())
        
        update = torch.zeros_like(h)
        for k in range(K):
            update += alpha[:, k:k+1] * gamma_val * (v_tensor[k] - h)
        h = h + update
        
    h_expanded = h.unsqueeze(1)
    v_expanded = v_tensor.unsqueeze(0)
    dists = torch.sum((h_expanded - v_expanded)**2, dim=-1)
    logits = -dists + biases_tensor
    return logits, alphas

def run_sweeps():
    print("Starting Optimized Sweeps for Refinement...")
    os.makedirs("results", exist_ok=True)
    
    seeds = [42, 43, 44] # Use 3 seeds for rapid but statistically sound complexity sweep
    
    # ---------------- 1. Multi-Curve Sample Complexity Sweep ----------------
    print("Running Multi-Curve Sample Complexity Sweep...")
    N_task_list = [8, 16, 32, 64, 128, 256, 512, 1000]
    total_samples_list = [n * K for n in N_task_list]
    
    rho_list = [0.0, 0.3, 0.5]
    colors = {0.0: 'green', 0.3: 'blue', 0.5: 'purple'}
    sable_colors = {0.0: 'orange', 0.3: 'red', 0.5: 'brown'}
    
    plt.figure(figsize=(11, 7))
    
    for rho in rho_list:
        print(f"Sweeping Sample Complexity for rho = {rho}...")
        v = get_signatures(rho)
        v_tensor = torch.tensor(v, dtype=torch.float32)
        biases_tensor = torch.tensor(biases, dtype=torch.float32)
        
        # Trackers
        unreg_accs = {n: [] for n in N_task_list}
        reg_accs = {n: [] for n in N_task_list}
        sable_accs = []
        chem_accs = []
        
        # References (using full 5 seeds for high-precision baseline reference lines)
        all_seeds = [42, 43, 44, 45, 46]
        for seed in all_seeds:
            test_samples, test_labels = generate_dataset(v, num_samples_per_task=250, seed=seed)
            
            logits_sable, _ = run_sable(test_samples, v_tensor, biases_tensor)
            sable_accs.append((torch.argmax(logits_sable, dim=-1) == test_labels).float().mean().item())
            
            logits_chem, _ = run_chemmerge(test_samples, v_tensor, biases_tensor)
            chem_accs.append((torch.argmax(logits_chem, dim=-1) == test_labels).float().mean().item())
            
        mean_sable = np.mean(sable_accs)
        mean_chem = np.mean(chem_accs)
        
        for n in N_task_list:
            print(f"  rho={rho}, N_cal={n*K}...")
            for seed in seeds:
                train_samples, train_labels = generate_dataset(v, num_samples_per_task=n, seed=seed)
                test_samples, test_labels = generate_dataset(v, num_samples_per_task=250, seed=seed)
                
                # Using 45 epochs and higher LR 0.04 for rapid convergence in sweeps
                def train_router(zero_init, wd, epochs=45, lr=0.04):
                    model = SandboxModule(v, biases, gating='softmax', zero_init=zero_init)
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                    criterion = nn.CrossEntropyLoss()
                    
                    model.train()
                    for _ in range(epochs):
                        optimizer.zero_grad()
                        logits_train, _ = model(train_samples)
                        loss = criterion(logits_train, train_labels)
                        loss.backward()
                        optimizer.step()
                    
                    model.eval()
                    with torch.no_grad():
                        logits_test, _ = model(test_samples)
                        acc = (torch.argmax(logits_test, dim=-1) == test_labels).float().mean().item()
                    return acc
                
                # Unregularized Softmax Router
                unreg_accs[n].append(train_router(zero_init=False, wd=0.0))
                # Proposed Zero-Init Softmax Router (WD=1e-2)
                reg_accs[n].append(train_router(zero_init=True, wd=1e-2))
                
        # Compile stats
        unreg_means = [np.mean(unreg_accs[n]) for n in N_task_list]
        unreg_stds = [np.std(unreg_accs[n]) for n in N_task_list]
        reg_means = [np.mean(reg_accs[n]) for n in N_task_list]
        reg_stds = [np.std(reg_accs[n]) for n in N_task_list]
        
        # Plot curves for this rho
        plt.errorbar(total_samples_list, reg_means, yerr=reg_stds, 
                     label=f"Proposed Softmax (rho={rho}, WD=1e-2)", 
                     fmt='-^', capsize=3, color=colors[rho])
        plt.errorbar(total_samples_list, unreg_means, yerr=unreg_stds, 
                     label=f"Unreg. Softmax (rho={rho})", 
                     fmt='--v', capsize=3, color=colors[rho], alpha=0.6)
        
        plt.axhline(y=mean_sable, color=sable_colors[rho], linestyle=':', 
                    label=f"SABLE (rho={rho}, Constant: {mean_sable*100:.1f}%)")
        plt.axhline(y=mean_chem, color=colors[rho], linestyle='-.', alpha=0.5,
                    label=f"ChemMerge (rho={rho}, Constant: {mean_chem*100:.1f}%)")
        
    plt.xscale('log')
    plt.xticks(total_samples_list, total_samples_list)
    plt.minorticks_off()
    plt.title("Multi-Curve Sample Complexity Sweep across Entanglement Levels (rho)")
    plt.xlabel("Total Calibration Sample Size (N_cal)")
    plt.ylabel("Joint Mean Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig("results/fig2.png")
    plt.close()
    print("Saved Multi-Curve Sample Complexity Plot to results/fig2.png")
    
    # ---------------- 2. Hyperparameter Sensitivity Sweep ----------------
    print("Running Hyperparameter Sensitivity Sweep...")
    rho = 0.3
    v = get_signatures(rho)
    v_tensor = torch.tensor(v, dtype=torch.float32)
    biases_tensor = torch.tensor(biases, dtype=torch.float32)
    
    tau_list = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    sable_sens = {tau: [] for tau in tau_list}
    chem_sens = {tau: [] for tau in tau_list}
    
    for tau in tau_list:
        print(f"  tau={tau}...")
        for seed in all_seeds: # Use full 5 seeds for smooth curves
            test_samples, test_labels = generate_dataset(v, num_samples_per_task=250, seed=seed)
            
            logits_sable, _ = run_sable(test_samples, v_tensor, biases_tensor, tau=tau)
            sable_sens[tau].append((torch.argmax(logits_sable, dim=-1) == test_labels).float().mean().item())
            
            logits_chem, _ = run_chemmerge(test_samples, v_tensor, biases_tensor, tau=tau)
            chem_sens[tau].append((torch.argmax(logits_chem, dim=-1) == test_labels).float().mean().item())
            
    sable_means = [np.mean(sable_sens[tau]) for tau in tau_list]
    sable_stds = [np.std(sable_sens[tau]) for tau in tau_list]
    
    chem_means = [np.mean(chem_sens[tau]) for tau in tau_list]
    chem_stds = [np.std(chem_sens[tau]) for tau in tau_list]
    
    # Plot Hyperparameter Sensitivity Sweep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.errorbar(tau_list, sable_means, yerr=sable_stds, label="SABLE", fmt='-o', color='orange', capsize=3)
    ax1.set_xscale('log')
    ax1.set_title("SABLE Routing Temperature (tau) Sensitivity")
    ax1.set_xlabel("Temperature (tau)")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    
    ax2.errorbar(tau_list, chem_means, yerr=chem_stds, label="ChemMerge", fmt='-s', color='blue', capsize=3)
    ax2.set_xscale('log')
    ax2.set_title("ChemMerge Routing Temperature (tau) Sensitivity")
    ax2.set_xlabel("Temperature (tau)")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    
    plt.suptitle("Hyperparameter Sensitivity Analysis of Training-Free Priors (rho=0.3)", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/fig4.png")
    plt.close()
    print("Saved Hyperparameter Sensitivity Plot to results/fig4.png")
    
    print("All sweeps completed successfully!")

if __name__ == "__main__":
    run_sweeps()
