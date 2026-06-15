import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class AsymmetricSandbox:
    def __init__(self, seed, skew_type='asymmetric'):
        set_seed(seed)
        self.D = 192
        self.K = 4
        self.C = 10
        self.sigmas = [0.05, 0.25, 0.40, 1.95] # Noise scales
        self.block_size = self.D // self.K  # 48
        
        # Generate disjoint orthogonal prototype subspaces
        self.prototypes = torch.zeros(self.K, self.C, self.D)
        for k in range(self.K):
            for c in range(self.C):
                v = torch.zeros(self.D)
                v[k*self.block_size : (k+1)*self.block_size] = torch.randn(self.block_size)
                self.prototypes[k, c] = v / torch.norm(v)
                
        if skew_type == 'asymmetric':
            # Create a highly skewed, asymmetric overlap:
            # Task 0 and Task 1 overlap heavily (rho = 0.7)
            # Task 1 and Task 2 overlap moderately (rho = 0.4)
            # Task 2 and Task 3 overlap slightly (rho = 0.1)
            # Other pairs have zero direct overlap
            
            # We mix the prototypes to achieve this
            # Task 0 remains unmixed
            # Task 1 mixes with Task 0
            rho_01 = 0.7
            for c in range(self.C):
                mixed = (1.0 - rho_01) * self.prototypes[1, c] + rho_01 * self.prototypes[0, c]
                self.prototypes[1, c] = mixed / torch.norm(mixed)
                
            # Task 2 mixes with Task 1
            rho_12 = 0.4
            for c in range(self.C):
                mixed = (1.0 - rho_12) * self.prototypes[2, c] + rho_12 * self.prototypes[1, c]
                self.prototypes[2, c] = mixed / torch.norm(mixed)
                
            # Task 3 mixes with Task 2
            rho_23 = 0.1
            for c in range(self.C):
                mixed = (1.0 - rho_23) * self.prototypes[3, c] + rho_23 * self.prototypes[2, c]
                self.prototypes[3, c] = mixed / torch.norm(mixed)
                
    def generate_split(self, num_samples_per_task):
        X = []
        Y = []
        task_labels = []
        for k in range(self.K):
            classes = []
            samples_per_class = num_samples_per_task // self.C
            for c in range(self.C):
                classes.extend([c] * samples_per_class)
                
            for c in classes:
                prototype = self.prototypes[k, c].clone()
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

def evaluate_router_detailed(router, sandbox, experts, X_test, Y_test, task_test):
    with torch.no_grad():
        alpha = router(X_test)
        expert_logits = torch.stack([experts[j](X_test) for j in range(sandbox.K)], dim=1)
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
        
        # Heterogeneous B=1
        logits_merged_het1 = torch.sum(alpha.unsqueeze(-1) * expert_logits, dim=1)
        preds_het1 = torch.argmax(logits_merged_het1, dim=1)
        heterogeneous_1_acc = (preds_het1 == Y_test).float().mean().item()
        
        return homogeneous_acc, heterogeneous_1_acc, routing_acc

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
        self.S = torch.matmul(self.v_bar, self.v_bar.t())
        
        # Löwdin Symmetric Orthogonalization
        eigenvalues, eigenvectors = torch.linalg.eigh(self.S)
        inv_sqrt_eigenvalues = 1.0 / torch.sqrt(eigenvalues + 1.0e-6)
        S_inv_sqrt = torch.matmul(eigenvectors, torch.matmul(torch.diag(inv_sqrt_eigenvalues), eigenvectors.t()))
        
        self.Q = torch.matmul(S_inv_sqrt, self.v_bar)
        
    def __call__(self, X):
        tilde_z = X / (torch.norm(X, dim=-1, keepdim=True) + 1.0e-8)
        u_prime = torch.matmul(tilde_z, self.Q.t())
        u_prime_abs = torch.abs(u_prime)
        alpha = F.softmax(u_prime_abs / self.tau, dim=-1)
        return alpha

# Run comparison
seed = 42
sandbox = AsymmetricSandbox(seed, skew_type='asymmetric')
experts = get_oracle_experts(sandbox)
X_test, Y_test, task_test = sandbox.generate_split(250)

pfsr = PFSRRouter(experts)
otsp = OTSPRouter(experts)

print("Centroid overlap matrix S:")
print(otsp.S)

h_pfsr, h1_pfsr, r_pfsr = evaluate_router_detailed(pfsr, sandbox, experts, X_test, Y_test, task_test)
h_otsp, h1_otsp, r_otsp = evaluate_router_detailed(otsp, sandbox, experts, X_test, Y_test, task_test)

print("\nPFSR Results:")
print(f"  Homogeneous Accuracy: {h_pfsr*100:.2f}%")
print(f"  Heterogeneous (B=1) Accuracy: {h1_pfsr*100:.2f}%")
print(f"  Routing Accuracy: {r_pfsr*100:.2f}%")

print("\nOTSP Results:")
print(f"  Homogeneous Accuracy: {h_otsp*100:.2f}%")
print(f"  Heterogeneous (B=1) Accuracy: {h1_otsp*100:.2f}%")
print(f"  Routing Accuracy: {r_otsp*100:.2f}%")
