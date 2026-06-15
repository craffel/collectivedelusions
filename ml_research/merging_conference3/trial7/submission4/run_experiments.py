import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Set random seed helper
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Define High-Fidelity Analytical Coordinate Sandbox
class RepresentationSandbox:
    def __init__(self, seed, rho=0.0):
        set_seed(seed)
        self.D = 192
        self.K = 4
        self.C = 10
        self.sigmas = [0.05, 0.25, 0.40, 1.95] # Perfectly calibrated noise scales
        self.block_size = self.D // self.K  # 48
        
        # Generate disjoint orthogonal prototype subspaces
        self.prototypes = torch.zeros(self.K, self.C, self.D)
        for k in range(self.K):
            for c in range(self.C):
                v = torch.zeros(self.D)
                v[k*self.block_size : (k+1)*self.block_size] = torch.randn(self.block_size)
                self.prototypes[k, c] = v / torch.norm(v)
                
        # Support task correlation (subspace overlap)
        if rho > 0.0:
            shared = torch.randn(self.C, self.D)
            for c in range(self.C):
                shared[c] = shared[c] / torch.norm(shared[c])
            for k in range(self.K):
                for c in range(self.C):
                    mixed = (1.0 - rho) * self.prototypes[k, c] + rho * shared[c]
                    self.prototypes[k, c] = mixed / torch.norm(mixed)
                    
    def generate_split(self, num_samples_per_task, is_calibration=False):
        X = []
        Y = []
        task_labels = []
        
        for k in range(self.K):
            if is_calibration:
                classes = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9]
                assert len(classes) == num_samples_per_task
            else:
                classes = []
                samples_per_class = num_samples_per_task // self.C
                for c in range(self.C):
                    classes.extend([c] * samples_per_class)
                    
            for c in classes:
                prototype = self.prototypes[k, c].clone()
                # Subspace-isolated noise to prevent random dimensional leakage
                noise = torch.zeros(self.D)
                noise[k*self.block_size : (k+1)*self.block_size] = torch.randn(self.block_size) * self.sigmas[k]
                
                z = prototype + noise
                X.append(z)
                Y.append(c)
                task_labels.append(k)
                
        X = torch.stack(X)
        Y = torch.tensor(Y, dtype=torch.long)
        task_labels = torch.tensor(task_labels, dtype=torch.long)
        return X, Y, task_labels

# Instantiate Oracle Experts (perfectly aligned with prototypes)
def get_oracle_experts(sandbox):
    experts = []
    for k in range(sandbox.K):
        expert = nn.Linear(sandbox.D, sandbox.C, bias=True)
        with torch.no_grad():
            expert.weight.zero_()
            for c in range(sandbox.C):
                # Copy the entire prototype vector to reflect task correlations in weight space
                expert.weight[c, :] = sandbox.prototypes[k, c, :]
                expert.bias[c] = 0.0
        experts.append(expert)
    return experts

# Evaluate Expert ceilings on test split
def evaluate_ceilings(sandbox, experts, X_test, Y_test, task_test):
    ceilings = []
    for k in range(sandbox.K):
        mask = (task_test == k)
        X_k = X_test[mask]
        Y_k = Y_test[mask]
        
        with torch.no_grad():
            logits = experts[k](X_k)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == Y_k).float().mean().item()
        ceilings.append(acc)
    return ceilings

# Orthonormal Projection Matrix helper
def get_projection_matrix(D, d):
    P = torch.randn(D, d)
    q, r = torch.linalg.qr(P)
    return q

# Define Trainable Router
class ParametricRouter(nn.Module):
    def __init__(self, d, K, mode='linear', init_zero=False):
        super().__init__()
        self.mode = mode
        self.W = nn.Parameter(torch.randn(K, d) * 0.01)
        self.B = nn.Parameter(torch.randn(K) * 0.01)
        
        if init_zero:
            self.W.data.fill_(0.0)
            self.B.data.fill_(0.0)
            
    def forward(self, psi):
        logits = F.linear(psi, self.W, self.B)
        
        if self.mode == 'linear':
            return logits
        elif self.mode == 'softmax':
            return F.softmax(logits, dim=-1)
        elif self.mode == 'qws':
            cos_sq = torch.cos(logits) ** 2
            return cos_sq / (torch.sum(cos_sq, dim=-1, keepdim=True) + 1.0e-8)

# Train Parametric Router
def train_router(router, sandbox, experts, X_cal, Y_cal, task_cal, P, epochs=100, lr=1.0e-3, wd=1.0e-3):
    with torch.no_grad():
        psi_cal = X_cal / (torch.norm(X_cal, dim=-1, keepdim=True) + 1.0e-8)
        
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=wd)
    
    for epoch in range(epochs):
        if router.mode == 'linear':
            logits = router(psi_cal)
            loss = F.cross_entropy(logits, task_cal)
        else:
            alpha = router(psi_cal)
            alpha = torch.clamp(alpha, 1.0e-8, 1.0)
            loss = -torch.mean(torch.log(alpha[torch.arange(len(task_cal)), task_cal]))
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Non-Parametric PFSR Router
class PFSRRouter:
    def __init__(self, experts, tau=0.001):
        self.K = len(experts)
        self.D = experts[0].weight.shape[1]
        self.tau = tau
        
        v = torch.zeros(self.K, self.D)
        for k in range(self.K):
            W_k = experts[k].weight.data
            _, _, Vh = torch.linalg.svd(W_k, full_matrices=False)
            v[k] = Vh[0]
            
        self.v_bar = v / (torch.norm(v, dim=1, keepdim=True) + 1.0e-8)
        
    def __call__(self, X):
        tilde_z = X / (torch.norm(X, dim=-1, keepdim=True) + 1.0e-8)
        u = torch.matmul(tilde_z, self.v_bar.t())
        u_abs = torch.abs(u)
        alpha = F.softmax(u_abs / self.tau, dim=-1)
        return alpha

# Non-Parametric OTSP Router with Löwdin Orthogonalization
class OTSPRouter:
    def __init__(self, experts, tau=0.001):
        self.K = len(experts)
        self.D = experts[0].weight.shape[1]
        self.tau = tau
        
        v = torch.zeros(self.K, self.D)
        for k in range(self.K):
            W_k = experts[k].weight.data
            _, _, Vh = torch.linalg.svd(W_k, full_matrices=False)
            v[k] = Vh[0]
            
        self.v_bar = v / (torch.norm(v, dim=1, keepdim=True) + 1.0e-8)
        
        # Compute Overlap Matrix
        S = torch.matmul(self.v_bar, self.v_bar.t())
        
        # Löwdin Symmetric Orthogonalization
        eigenvalues, eigenvectors = torch.linalg.eigh(S)
        inv_sqrt_eigenvalues = 1.0 / torch.sqrt(eigenvalues + 1.0e-6)
        S_inv_sqrt = torch.matmul(eigenvectors, torch.matmul(torch.diag(inv_sqrt_eigenvalues), eigenvectors.t()))
        
        self.Q = torch.matmul(S_inv_sqrt, self.v_bar)
        
    def __call__(self, X):
        tilde_z = X / (torch.norm(X, dim=-1, keepdim=True) + 1.0e-8)
        u_prime = torch.matmul(tilde_z, self.Q.t())
        u_prime_abs = torch.abs(u_prime)
        alpha = F.softmax(u_prime_abs / self.tau, dim=-1)
        return alpha

# Evaluate Router and return metrics
def evaluate_router_detailed(router, sandbox, experts, X_test, Y_test, task_test, P):
    with torch.no_grad():
        if hasattr(router, 'forward') or isinstance(router, nn.Module):
            psi_test = X_test / (torch.norm(X_test, dim=-1, keepdim=True) + 1.0e-8)
            alpha = router(psi_test)
        else:
            alpha = router(X_test)
            
        expert_logits = torch.stack([experts[j](X_test) for j in range(sandbox.K)], dim=1)
        
        # Compute Routing accuracy: percentage of samples where argmax of alpha matches task_test
        preds_routing = torch.argmax(alpha, dim=1)
        routing_acc = (preds_routing == task_test).float().mean().item()
        
        # Homogeneous
        accuracies_hom = []
        for k in range(sandbox.K):
            mask = (task_test == k)
            alpha_k = alpha[mask]
            bar_alpha_k = torch.mean(alpha_k, dim=0)
            
            expert_logits_k = expert_logits[mask]
            logits_merged = torch.sum(bar_alpha_k.view(1, sandbox.K, 1) * expert_logits_k, dim=1)
            Y_k = Y_test[mask]
            
            preds = torch.argmax(logits_merged, dim=1)
            acc = (preds == Y_k).float().mean().item()
            accuracies_hom.append(acc)
        homogeneous_acc = np.mean(accuracies_hom)
        
        # Heterogeneous B=256
        num_samples = len(X_test)
        indices = torch.arange(num_samples)
        batch_size = 256
        correct_het256 = 0
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            alpha_batch = alpha[batch_indices]
            bar_alpha = torch.mean(alpha_batch, dim=0)
            
            expert_logits_batch = expert_logits[batch_indices]
            logits_merged = torch.sum(bar_alpha.view(1, sandbox.K, 1) * expert_logits_batch, dim=1)
            Y_batch = Y_test[batch_indices]
            
            preds = torch.argmax(logits_merged, dim=1)
            correct_het256 += torch.sum(preds == Y_batch).item()
        heterogeneous_256_acc = correct_het256 / num_samples
        
        # Heterogeneous B=1
        logits_merged_het1 = torch.sum(alpha.unsqueeze(-1) * expert_logits, dim=1)
        preds_het1 = torch.argmax(logits_merged_het1, dim=1)
        heterogeneous_1_acc = (preds_het1 == Y_test).float().mean().item()
        
        return homogeneous_acc, heterogeneous_256_acc, heterogeneous_1_acc, routing_acc

# Main Execution Script
def main():
    seeds = list(range(42, 52)) # 10 seeds (42 to 51)
    
    methods = [
        'Uniform',
        'LinearRouter',
        'QWS_Merge',
        'L3_Softmax',
        'L3_Softmax_WellReg',
        'PFSR_Baseline',
        'OTSP_Ours'
    ]
    
    configs = ['homogeneous', 'heterogeneous_256', 'heterogeneous_1', 'routing_acc']
    
    results = {m: {c: [] for c in configs} for m in methods}
    ceilings_all = []
    
    print(f"Running Statistical Significance sweep over {len(seeds)} seeds...")
    
    for seed in seeds:
        set_seed(seed)
        
        # Instantiate sandbox
        sandbox = RepresentationSandbox(seed, rho=0.0)
        
        # Load uncorrupted oracle experts
        experts = get_oracle_experts(sandbox)
        
        # Generate Calibration and Test splits
        X_cal, Y_cal, task_cal = sandbox.generate_split(16, is_calibration=True)
        X_test, Y_test, task_test = sandbox.generate_split(250)
        
        # Evaluate Expert ceilings on test split
        ceilings = evaluate_ceilings(sandbox, experts, X_test, Y_test, task_test)
        ceilings_all.append(np.mean(ceilings))
        
        # Random projection matrix
        P = get_projection_matrix(sandbox.D, sandbox.K)
        
        # 1. Uniform
        class UniformRouter:
            def __call__(self, X):
                return torch.ones(len(X), 4) * 0.25
        uniform_router = UniformRouter()
        h, h256, h1, r = evaluate_router_detailed(uniform_router, sandbox, experts, X_test, Y_test, task_test, P)
        results['Uniform']['homogeneous'].append(h)
        results['Uniform']['heterogeneous_256'].append(h256)
        results['Uniform']['heterogeneous_1'].append(h1)
        results['Uniform']['routing_acc'].append(r)
        
        # 2. LinearRouter
        lin_router = ParametricRouter(sandbox.D, sandbox.K, mode='linear', init_zero=False)
        train_router(lin_router, sandbox, experts, X_cal, Y_cal, task_cal, P)
        h, h256, h1, r = evaluate_router_detailed(lin_router, sandbox, experts, X_test, Y_test, task_test, P)
        results['LinearRouter']['homogeneous'].append(h)
        results['LinearRouter']['heterogeneous_256'].append(h256)
        results['LinearRouter']['heterogeneous_1'].append(h1)
        results['LinearRouter']['routing_acc'].append(r)
        
        # 3. QWS_Merge
        qws_router = ParametricRouter(sandbox.D, sandbox.K, mode='qws', init_zero=False)
        train_router(qws_router, sandbox, experts, X_cal, Y_cal, task_cal, P)
        h, h256, h1, r = evaluate_router_detailed(qws_router, sandbox, experts, X_test, Y_test, task_test, P)
        results['QWS_Merge']['homogeneous'].append(h)
        results['QWS_Merge']['heterogeneous_256'].append(h256)
        results['QWS_Merge']['heterogeneous_1'].append(h1)
        results['QWS_Merge']['routing_acc'].append(r)
        
        # 4. L3_Softmax
        softmax_router = ParametricRouter(sandbox.D, sandbox.K, mode='softmax', init_zero=False)
        train_router(softmax_router, sandbox, experts, X_cal, Y_cal, task_cal, P)
        h, h256, h1, r = evaluate_router_detailed(softmax_router, sandbox, experts, X_test, Y_test, task_test, P)
        results['L3_Softmax']['homogeneous'].append(h)
        results['L3_Softmax']['heterogeneous_256'].append(h256)
        results['L3_Softmax']['heterogeneous_1'].append(h1)
        results['L3_Softmax']['routing_acc'].append(r)
        
        # 5. L3_Softmax_WellReg
        softmax_wellreg = ParametricRouter(sandbox.D, sandbox.K, mode='softmax', init_zero=True)
        train_router(softmax_wellreg, sandbox, experts, X_cal, Y_cal, task_cal, P)
        h, h256, h1, r = evaluate_router_detailed(softmax_wellreg, sandbox, experts, X_test, Y_test, task_test, P)
        results['L3_Softmax_WellReg']['homogeneous'].append(h)
        results['L3_Softmax_WellReg']['heterogeneous_256'].append(h256)
        results['L3_Softmax_WellReg']['heterogeneous_1'].append(h1)
        results['L3_Softmax_WellReg']['routing_acc'].append(r)
        
        # 6. PFSR_Baseline
        pfsr_router = PFSRRouter(experts)
        h, h256, h1, r = evaluate_router_detailed(pfsr_router, sandbox, experts, X_test, Y_test, task_test, P)
        results['PFSR_Baseline']['homogeneous'].append(h)
        results['PFSR_Baseline']['heterogeneous_256'].append(h256)
        results['PFSR_Baseline']['heterogeneous_1'].append(h1)
        results['PFSR_Baseline']['routing_acc'].append(r)
        
        # 7. OTSP_Ours
        otsp_router = OTSPRouter(experts)
        h, h256, h1, r = evaluate_router_detailed(otsp_router, sandbox, experts, X_test, Y_test, task_test, P)
        results['OTSP_Ours']['homogeneous'].append(h)
        results['OTSP_Ours']['heterogeneous_256'].append(h256)
        results['OTSP_Ours']['heterogeneous_1'].append(h1)
        results['OTSP_Ours']['routing_acc'].append(r)
        
    print("\n--- RESULTS SUMMARY ---")
    print(f"Expert Ceiling Reference: {np.mean(ceilings_all)*100:.2f}%")
    
    # Format markdown table
    print("\n| Router Method | Homogeneous (B=256) | Heterogeneous (B=256) | Heterogeneous (B=1) | Routing Accuracy |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for m in methods:
        row = f"| {m} "
        for c in ['homogeneous', 'heterogeneous_256', 'heterogeneous_1', 'routing_acc']:
            mean = np.mean(results[m][c]) * 100
            std = np.std(results[m][c]) * 100
            row += f"| {mean:.2f}% ± {std:.2f}% "
        row += "|"
        print(row)
        
    # Generate bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.25
    
    means_hom = [np.mean(results[m]['homogeneous'])*100 for m in methods]
    stds_hom = [np.std(results[m]['homogeneous'])*100 for m in methods]
    
    means_het256 = [np.mean(results[m]['heterogeneous_256'])*100 for m in methods]
    stds_het256 = [np.std(results[m]['heterogeneous_256'])*100 for m in methods]
    
    means_het1 = [np.mean(results[m]['heterogeneous_1'])*100 for m in methods]
    stds_het1 = [np.std(results[m]['heterogeneous_1'])*100 for m in methods]
    
    rects1 = ax.bar(x - width, means_hom, width, yerr=stds_hom, label='Homogeneous (B=256)', capsize=4, color='#1f77b4')
    rects2 = ax.bar(x, means_het256, width, yerr=stds_het256, label='Heterogeneous (B=256)', capsize=4, color='#ff7f0e')
    rects3 = ax.bar(x + width, means_het1, width, yerr=stds_het1, label='Heterogeneous (B=1)', capsize=4, color='#2ca02c')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Merging Performance across Deployment Streams (10 Seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=300)
    print("\nPlot saved successfully to comparison_plot.png")

if __name__ == '__main__':
    main()
