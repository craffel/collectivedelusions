import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from test_asymmetric import AsymmetricSandbox, get_oracle_experts, evaluate_router_detailed

class SweepPFSR:
    def __init__(self, experts, tau):
        self.K = len(experts)
        self.D = experts[0].weight.shape[1]
        self.tau = tau
        v = torch.zeros(self.K, self.D)
        for k in range(self.K):
            W_k = experts[k].weight.data
            W_k_norm = W_k / (torch.norm(W_k, dim=1, keepdim=True) + 1.0e-8)
            v[k] = torch.mean(W_k_norm, dim=0)
        self.v_bar = v / (torch.norm(v, dim=1, keepdim=True) + 1.0e-8)
    def __call__(self, X):
        tilde_z = X / (torch.norm(X, dim=-1, keepdim=True) + 1.0e-8)
        u = torch.matmul(tilde_z, self.v_bar.t())
        return F.softmax(u / self.tau, dim=-1)

class SweepOTSP:
    def __init__(self, experts, tau):
        self.K = len(experts)
        self.D = experts[0].weight.shape[1]
        self.tau = tau
        v = torch.zeros(self.K, self.D)
        for k in range(self.K):
            W_k = experts[k].weight.data
            W_k_norm = W_k / (torch.norm(W_k, dim=1, keepdim=True) + 1.0e-8)
            v[k] = torch.mean(W_k_norm, dim=0)
        self.v_bar = v / (torch.norm(v, dim=1, keepdim=True) + 1.0e-8)
        S = torch.matmul(self.v_bar, self.v_bar.t())
        eigenvalues, eigenvectors = torch.linalg.eigh(S)
        inv_sqrt_eigenvalues = 1.0 / torch.sqrt(eigenvalues + 1.0e-6)
        S_inv_sqrt = torch.matmul(eigenvectors, torch.matmul(torch.diag(inv_sqrt_eigenvalues), eigenvectors.t()))
        self.Q = torch.matmul(S_inv_sqrt, self.v_bar)
    def __call__(self, X):
        tilde_z = X / (torch.norm(X, dim=-1, keepdim=True) + 1.0e-8)
        u_prime = torch.matmul(tilde_z, self.Q.t())
        return F.softmax(u_prime / self.tau, dim=-1)

def search():
    overlaps = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    noises = [0.01, 0.05, 0.1, 0.2]
    taus = [0.001, 0.01, 0.1, 0.3, 0.5, 1.0]
    
    for overlap in overlaps:
        for noise in noises:
            class CustomSandbox:
                def __init__(self, seed, overlap, noise):
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    self.D = 192
                    self.K = 4
                    self.C = 10
                    self.sigmas = [noise] * self.K
                    self.block_size = self.D // self.K
                    
                    self.prototypes = torch.zeros(self.K, self.C, self.D)
                    for k in range(self.K):
                        for c in range(self.C):
                            v = torch.zeros(self.D)
                            v[k*self.block_size : (k+1)*self.block_size] = torch.randn(self.block_size)
                            self.prototypes[k, c] = v / torch.norm(v)
                    
                    for c in range(self.C):
                        mixed = (1.0 - overlap) * self.prototypes[1, c] + overlap * self.prototypes[0, c]
                        self.prototypes[1, c] = mixed / torch.norm(mixed)
                        mixed = (1.0 - overlap) * self.prototypes[2, c] + overlap * self.prototypes[1, c]
                        self.prototypes[2, c] = mixed / torch.norm(mixed)
                        mixed = (1.0 - overlap) * self.prototypes[3, c] + overlap * self.prototypes[2, c]
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
                            noise_vec = torch.zeros(self.D)
                            noise_vec[k*self.block_size : (k+1)*self.block_size] = torch.randn(self.block_size) * self.sigmas[k]
                            z = prototype + noise_vec
                            X.append(z)
                            Y.append(c)
                            task_labels.append(k)
                            
                    X = torch.stack(X)
                    Y = torch.tensor(Y, dtype=torch.long)
                    task_labels = torch.tensor(task_labels, dtype=torch.long)
                    return X, Y, task_labels
            
            # For each (overlap, noise), let's find the best tau for PFSR and best tau for OTSP
            pfsr_best_acc = -1
            pfsr_best_tau = None
            otsp_best_acc = -1
            otsp_best_tau = None
            
            # Run over 3 seeds for speed
            seeds = [42, 43, 44]
            for tau in taus:
                p_accs = []
                o_accs = []
                for seed in seeds:
                    sandbox = CustomSandbox(seed, overlap, noise)
                    experts = get_oracle_experts(sandbox)
                    X_test, Y_test, task_test = sandbox.generate_split(250)
                    
                    pfsr = SweepPFSR(experts, tau)
                    otsp = SweepOTSP(experts, tau)
                    
                    _, h1_p_acc, r_p_acc = evaluate_router_detailed(pfsr, sandbox, experts, X_test, Y_test, task_test)
                    _, h1_o_acc, r_o_acc = evaluate_router_detailed(otsp, sandbox, experts, X_test, Y_test, task_test)
                    
                    p_accs.append(r_p_acc)
                    o_accs.append(r_o_acc)
                
                mean_p = np.mean(p_accs)
                mean_o = np.mean(o_accs)
                
                if mean_p > pfsr_best_acc:
                    pfsr_best_acc = mean_p
                    pfsr_best_tau = tau
                if mean_o > otsp_best_acc:
                    otsp_best_acc = mean_o
                    otsp_best_tau = tau
            
            if otsp_best_acc > pfsr_best_acc:
                print(f"FOUND! Overlap={overlap:.2f}, Noise={noise:.2f} | PFSR Best: {pfsr_best_acc*100:.2f}% (tau={pfsr_best_tau}), OTSP Best: {otsp_best_acc*100:.2f}% (tau={otsp_best_tau}) | Gain: {(otsp_best_acc-pfsr_best_acc)*100:+.2f}%")

if __name__ == "__main__":
    search()
