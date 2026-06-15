import torch
import torch.nn as nn
import numpy as np
from test_asymmetric import AsymmetricSandbox, get_oracle_experts, OTSPRouter, PFSRRouter

seed = 42
sandbox = AsymmetricSandbox(seed, skew_type='asymmetric')
experts = get_oracle_experts(sandbox)
X_test, Y_test, task_test = sandbox.generate_split(250)

expert_logits = torch.stack([experts[j](X_test) for j in range(sandbox.K)], dim=1)

# Uniform Merging
logits_uniform = torch.mean(expert_logits, dim=1)
preds_uniform = torch.argmax(logits_uniform, dim=1)

print("Per-task Accuracies (Uniform Merging):")
for k in range(sandbox.K):
    mask = (task_test == k)
    acc = (preds_uniform[mask] == Y_test[mask]).float().mean().item()
    print(f"  Task {k}: {acc*100:.2f}%")
print(f"Joint Mean: { (preds_uniform == Y_test).float().mean().item()*100:.2f}%")

# PFSR / OTSP at tau=0.3
for name, router_cls in [("PFSR", PFSRRouter), ("OTSP", OTSPRouter)]:
    for tau in [0.001, 0.3]:
        router = router_cls(experts, tau=tau)
        alpha = router(X_test)
        logits = torch.sum(alpha.unsqueeze(-1) * expert_logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        preds_routing = torch.argmax(alpha, dim=1)
        
        print(f"\n{name} Results (tau={tau}):")
        print(f"  Joint Mean Acc: { (preds == Y_test).float().mean().item()*100:.2f}%")
        print("  Per-task Accuracies:")
        for k in range(sandbox.K):
            mask = (task_test == k)
            acc = (preds[mask] == Y_test[mask]).float().mean().item()
            print(f"    Task {k}: {acc*100:.2f}%")
        print("  Per-task Routing:")
        for k in range(sandbox.K):
            mask = (task_test == k)
            acc = (preds_routing[mask] == k).float().mean().item()
            print(f"    Task {k}: {acc*100:.2f}%")
