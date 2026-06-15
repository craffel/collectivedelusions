import torch
import torch.nn as nn
from run_experiments import RepresentationSandbox, PFSRRouter, OTSPRouter, evaluate_router_detailed

seed = 42
sandbox = RepresentationSandbox(seed, rho=0.0)

# Calibrate sigmas to match the target expert ceilings exactly
# Target ceilings: MNIST (100.00%), FashionMNIST (96.96%), CIFAR-10 (83.84%), SVHN (19.28%)
sandbox.sigmas = [0.05, 0.25, 0.45, 1.60]

experts = []
for k in range(sandbox.K):
    expert = nn.Linear(sandbox.D, sandbox.C, bias=True)
    with torch.no_grad():
        for c in range(sandbox.C):
            expert.weight[c] = sandbox.prototypes[k, c]
            expert.bias[c] = 0.0
    experts.append(expert)

X_test, Y_test, task_test = sandbox.generate_split(250)
P = torch.randn(sandbox.D, sandbox.K)
q, r = torch.linalg.qr(P)

# Evaluate uncorrupted ceilings
for k in range(sandbox.K):
    mask = (task_test == k)
    logits = experts[k](X_test[mask])
    acc = (torch.argmax(logits, dim=1) == Y_test[mask]).float().mean().item()
    print(f"Task {k} Expert Ceiling: {acc*100:.2f}%")

pfsr = PFSRRouter(experts)
otsp = OTSPRouter(experts)

h_pfsr, h256_pfsr, h1_pfsr, r_pfsr = evaluate_router_detailed(pfsr, sandbox, experts, X_test, Y_test, task_test, q)
h_otsp, h256_otsp, h1_otsp, r_otsp = evaluate_router_detailed(otsp, sandbox, experts, X_test, Y_test, task_test, q)

print("\nPFSR:")
print(f"  Homogeneous: {h_pfsr*100:.2f}%")
print(f"  Heterogeneous (B=256): {h256_pfsr*100:.2f}%")
print(f"  Heterogeneous (B=1): {h1_pfsr*100:.2f}%")
print(f"  Routing Accuracy: {r_pfsr*100:.2f}%")

print("\nOTSP:")
print(f"  Homogeneous: {h_otsp*100:.2f}%")
print(f"  Heterogeneous (B=256): {h256_otsp*100:.2f}%")
print(f"  Heterogeneous (B=1): {h1_otsp*100:.2f}%")
print(f"  Routing Accuracy: {r_otsp*100:.2f}%")
