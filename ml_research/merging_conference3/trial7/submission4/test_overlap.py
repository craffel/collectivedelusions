import torch
import torch.nn as nn
import numpy as np
from run_experiments import RepresentationSandbox, get_oracle_experts, PFSRRouter, OTSPRouter, evaluate_router_detailed

seed = 42
sandbox = RepresentationSandbox(seed, rho=0.33) # Overlap rho = 0.33
experts = get_oracle_experts(sandbox)
X_test, Y_test, task_test = sandbox.generate_split(250)

# Evaluate Expert ceilings on test split
ceilings = []
for k in range(sandbox.K):
    mask = (task_test == k)
    logits = experts[k](X_test[mask])
    acc = (torch.argmax(logits, dim=1) == Y_test[mask]).float().mean().item()
    ceilings.append(acc)
print("Expert Ceilings under rho=0.33:")
print("  mnist, f-mnist, cifar, svhn:", [f"{c*100:.2f}%" for c in ceilings])
print(f"  Joint Mean: {np.mean(ceilings)*100:.2f}%")

P = torch.randn(sandbox.D, sandbox.K)
q, r = torch.linalg.qr(P)

# 1. Uniform
class UniformRouter:
    def __call__(self, X):
        return torch.ones(len(X), 4) * 0.25
uniform_router = UniformRouter()
h, h256, h1, r_val = evaluate_router_detailed(uniform_router, sandbox, experts, X_test, Y_test, task_test, q)
print("\nUniform Merging:")
print(f"  Homogeneous: {h*100:.2f}%")
print(f"  Heterogeneous (B=256): {h256*100:.2f}%")
print(f"  Heterogeneous (B=1): {h1*100:.2f}%")

# 2. PFSR
pfsr = PFSRRouter(experts)
h, h256, h1, r_val = evaluate_router_detailed(pfsr, sandbox, experts, X_test, Y_test, task_test, q)
print("\nPFSR Baseline:")
print(f"  Homogeneous: {h*100:.2f}%")
print(f"  Heterogeneous (B=256): {h256*100:.2f}%")
print(f"  Heterogeneous (B=1): {h1*100:.2f}%")
print(f"  Routing Accuracy: {r_val*100:.2f}%")

# 3. OTSP (Ours)
otsp = OTSPRouter(experts)
h, h256, h1, r_val = evaluate_router_detailed(otsp, sandbox, experts, X_test, Y_test, task_test, q)
print("\nOTSP (Ours):")
print(f"  Homogeneous: {h*100:.2f}%")
print(f"  Heterogeneous (B=256): {h256*100:.2f}%")
print(f"  Heterogeneous (B=1): {h1*100:.2f}%")
print(f"  Routing Accuracy: {r_val*100:.2f}%")
