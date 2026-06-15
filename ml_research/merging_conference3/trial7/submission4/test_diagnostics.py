import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from run_experiments import RepresentationSandbox, get_oracle_experts, get_projection_matrix, PFSRRouter, OTSPRouter, evaluate_router_detailed

seed = 42
sandbox = RepresentationSandbox(seed, rho=0.0)
experts = get_oracle_experts(sandbox)
X_test, Y_test, task_test = sandbox.generate_split(250)

# Check the properties of task centroids and features
pfsr = PFSRRouter(experts)
otsp = OTSPRouter(experts)

# Print v_bar norms and overlaps
print("v_bar shapes:", pfsr.v_bar.shape)
overlaps = torch.matmul(pfsr.v_bar, pfsr.v_bar.t())
print("v_bar mutual overlaps:\n", overlaps)

# Check some test samples of task 0 (MNIST)
task_0_indices = (task_test == 0).nonzero().squeeze()
X_0 = X_test[task_0_indices]
tilde_X_0 = X_0 / torch.norm(X_0, dim=-1, keepdim=True)

# Project task 0 samples onto v_bar
u_0 = torch.matmul(tilde_X_0, pfsr.v_bar.t())
print("First 10 task 0 projection u:\n", u_0[:10])

# Compute soft probabilities
alpha_0 = F.softmax(u_0 / pfsr.tau, dim=-1)
print("First 10 task 0 routing coefficients alpha:\n", alpha_0[:10])

preds = torch.argmax(alpha_0, dim=-1)
print("First 10 predictions:", preds[:10])
print("Task 0 routing accuracy:", (preds == 0).float().mean().item())
