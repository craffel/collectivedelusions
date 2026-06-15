import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from test_asymmetric import AsymmetricSandbox, get_oracle_experts

# Define Parametric Router (same as in run_experiments.py)
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

# Orthonormal Projection Matrix helper
def get_projection_matrix(D, d):
    P = torch.randn(D, d)
    q, r = torch.linalg.qr(P)
    return q

# Train Parametric Router (same as in run_experiments.py)
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

class PFCPRouter:
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
        return F.softmax(u_abs / self.tau, dim=-1)

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
        S = torch.matmul(self.v_bar, self.v_bar.t())
        eigenvalues, eigenvectors = torch.linalg.eigh(S)
        inv_sqrt_eigenvalues = 1.0 / torch.sqrt(eigenvalues + 1.0e-6)
        S_inv_sqrt = torch.matmul(eigenvectors, torch.matmul(torch.diag(inv_sqrt_eigenvalues), eigenvectors.t()))
        self.Q = torch.matmul(S_inv_sqrt, self.v_bar)
    def __call__(self, X):
        tilde_z = X / (torch.norm(X, dim=-1, keepdim=True) + 1.0e-8)
        u_prime = torch.matmul(tilde_z, self.Q.t())
        u_prime_abs = torch.abs(u_prime)
        return F.softmax(u_prime_abs / self.tau, dim=-1)

def evaluate_router_detailed(router, sandbox, experts, X_test, Y_test, task_test, P):
    with torch.no_grad():
        if hasattr(router, 'forward') or isinstance(router, nn.Module):
            psi_test = X_test / (torch.norm(X_test, dim=-1, keepdim=True) + 1.0e-8)
            alpha = router(psi_test)
        else:
            alpha = router(X_test)
            
        expert_logits = torch.stack([experts[j](X_test) for j in range(sandbox.K)], dim=1)
        
        preds_routing = torch.argmax(alpha, dim=1)
        routing_acc = (preds_routing == task_test).float().mean().item()
        
        # Heterogeneous B=1
        logits_merged_het1 = torch.sum(alpha.unsqueeze(-1) * expert_logits, dim=1)
        preds_het1 = torch.argmax(logits_merged_het1, dim=1)
        heterogeneous_1_acc = (preds_het1 == Y_test).float().mean().item()
        
        return heterogeneous_1_acc, routing_acc

def main():
    seeds = list(range(42, 52))
    methods = [
        'Uniform',
        'LinearRouter',
        'QWS_Merge',
        'L3_Softmax',
        'L3_Softmax_WellReg',
        'PFCP (Ours)',
        'OTSP'
    ]
    
    results = {m: {'het1': [], 'routing': []} for m in methods}
    
    print("Evaluating asymmetric sandbox across 10 seeds...")
    for seed in seeds:
        # Load asymmetric sandbox
        sandbox = AsymmetricSandbox(seed, skew_type='asymmetric')
        experts = get_oracle_experts(sandbox)
        X_test, Y_test, task_test = sandbox.generate_split(250)
        
        # Generate some calibration split for parametric routers
        # We simulate calibration split in asymmetric sandbox
        # We need a small calibration split
        X_cal, Y_cal, task_cal = sandbox.generate_split(16)
        
        P = get_projection_matrix(sandbox.D, sandbox.K)
        
        # 1. Uniform
        class UniformRouter:
            def __call__(self, X):
                return torch.ones(len(X), 4) * 0.25
        h1, r = evaluate_router_detailed(UniformRouter(), sandbox, experts, X_test, Y_test, task_test, P)
        results['Uniform']['het1'].append(h1)
        results['Uniform']['routing'].append(r)
        
        # 2. LinearRouter
        lin_router = ParametricRouter(sandbox.D, sandbox.K, mode='linear', init_zero=False)
        train_router(lin_router, sandbox, experts, X_cal, Y_cal, task_cal, P)
        h1, r = evaluate_router_detailed(lin_router, sandbox, experts, X_test, Y_test, task_test, P)
        results['LinearRouter']['het1'].append(h1)
        results['LinearRouter']['routing'].append(r)
        
        # 3. QWS_Merge
        qws_router = ParametricRouter(sandbox.D, sandbox.K, mode='qws', init_zero=False)
        train_router(qws_router, sandbox, experts, X_cal, Y_cal, task_cal, P)
        h1, r = evaluate_router_detailed(qws_router, sandbox, experts, X_test, Y_test, task_test, P)
        results['QWS_Merge']['het1'].append(h1)
        results['QWS_Merge']['routing'].append(r)
        
        # 4. L3_Softmax
        softmax_router = ParametricRouter(sandbox.D, sandbox.K, mode='softmax', init_zero=False)
        train_router(softmax_router, sandbox, experts, X_cal, Y_cal, task_cal, P)
        h1, r = evaluate_router_detailed(softmax_router, sandbox, experts, X_test, Y_test, task_test, P)
        results['L3_Softmax']['het1'].append(h1)
        results['L3_Softmax']['routing'].append(r)
        
        # 5. L3_Softmax_WellReg
        softmax_wellreg = ParametricRouter(sandbox.D, sandbox.K, mode='softmax', init_zero=True)
        train_router(softmax_wellreg, sandbox, experts, X_cal, Y_cal, task_cal, P)
        h1, r = evaluate_router_detailed(softmax_wellreg, sandbox, experts, X_test, Y_test, task_test, P)
        results['L3_Softmax_WellReg']['het1'].append(h1)
        results['L3_Softmax_WellReg']['routing'].append(r)
        
        # 6. PFCP
        pfcp = PFCPRouter(experts, tau=0.3)
        h1, r = evaluate_router_detailed(pfcp, sandbox, experts, X_test, Y_test, task_test, P)
        results['PFCP (Ours)']['het1'].append(h1)
        results['PFCP (Ours)']['routing'].append(r)
        
        # 7. OTSP
        otsp = OTSPRouter(experts, tau=0.3)
        h1, r = evaluate_router_detailed(otsp, sandbox, experts, X_test, Y_test, task_test, P)
        results['OTSP']['het1'].append(h1)
        results['OTSP']['routing'].append(r)
        
    print("\n--- ASYMMETRIC RESULTS SUMMARY (10 SEEDS) ---")
    for m in methods:
        h1_mean = np.mean(results[m]['het1']) * 100
        h1_std = np.std(results[m]['het1']) * 100
        r_mean = np.mean(results[m]['routing']) * 100
        r_std = np.std(results[m]['routing']) * 100
        print(f"{m:20} | Heterogeneous (B=1): {h1_mean:.2f}% ± {h1_std:.2f}% | Routing Acc: {r_mean:.2f}% ± {r_std:.2f}%")

if __name__ == "__main__":
    main()
