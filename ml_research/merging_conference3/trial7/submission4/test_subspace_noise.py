import torch
import torch.nn as nn
import torch.nn.functional as F

class RepresentationSandbox:
    def __init__(self, seed, rho=0.0):
        torch.manual_seed(seed)
        self.D = 192
        self.K = 4
        self.C = 10
        self.sigmas = [0.05, 0.25, 0.65, 1.30] # Calibrated sigmas
        
        # Generate disjoint orthogonal prototype subspaces
        self.prototypes = torch.zeros(self.K, self.C, self.D)
        self.block_size = self.D // self.K  # 48
        
        for k in range(self.K):
            for c in range(self.C):
                v = torch.zeros(self.D)
                v[k*self.block_size : (k+1)*self.block_size] = torch.randn(self.block_size)
                self.prototypes[k, c] = v / torch.norm(v)
                
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
                # Generate noise ONLY in task k's subspace!
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

seed = 42
sandbox = RepresentationSandbox(seed, rho=0.0)
X_test, Y_test, task_test = sandbox.generate_split(250)

# Evaluate expert ceilings
experts = []
for k in range(sandbox.K):
    expert = nn.Linear(sandbox.D, sandbox.C, bias=True)
    with torch.no_grad():
        for c in range(sandbox.C):
            expert.weight[c] = sandbox.prototypes[k, c]
            expert.bias[c] = 0.0
    experts.append(expert)

for k in range(sandbox.K):
    mask = (task_test == k)
    logits = experts[k](X_test[mask])
    acc = (torch.argmax(logits, dim=1) == Y_test[mask]).float().mean().item()
    print(f"Task {k} Expert Ceiling: {acc*100:.2f}%")

from run_experiments import PFSRRouter, OTSPRouter, evaluate_router_detailed
pfsr = PFSRRouter(experts)
otsp = OTSPRouter(experts)

P = torch.randn(sandbox.D, sandbox.K)
q, r = torch.linalg.qr(P)

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
