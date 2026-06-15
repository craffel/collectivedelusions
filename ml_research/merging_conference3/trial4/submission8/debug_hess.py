import os
import torch
import torch.nn as nn
import sys

# Prepend env_packages
sys.path.insert(0, "./env_packages")

import timm
import numpy as np
from contextlib import nullcontext

# Devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# We'll copy the minimal setup from run_merging.py
from experiments.run_merging import (
    get_layer_idx, load_eval_data, datasets_config, evaluate_model,
    compute_entropy_loss, params_pre, expert_heads, task_vectors, model, K, L
)

print("Running one epoch optimization debug...")

# Initialize raw theta values for adamerging
theta_ada = torch.ones(L, K, device=device) * -0.847  # sigmoid(-0.847) = 0.3
theta_ada.requires_grad_(True)
optimizer_ada = torch.optim.Adam([theta_ada], lr=1e-2)

# AdaMerging step
lambdas_ada = torch.sigmoid(theta_ada)
loss_ada = compute_entropy_loss(lambdas_ada, params_pre, task_vectors, model, calib_batches, K, q_schema=0, use_ste=False)
optimizer_ada.zero_grad()
loss_ada.backward()
print(f"AdaMerging loss: {loss_ada.item():.6f}")
print(f"AdaMerging theta grad (mean of abs): {theta_ada.grad.abs().mean().item():.8f}")

# Initialize raw theta values for hessmerge
theta_hess = torch.ones(L, K, device=device) * -0.847  # sigmoid(-0.847) = 0.3
theta_hess.requires_grad_(True)
optimizer_hess = torch.optim.Adam([theta_hess], lr=1e-2)

# HessMerge step
lambdas_hess = torch.sigmoid(theta_hess)
loss_hess = compute_entropy_loss(lambdas_hess, params_pre, task_vectors, model, calib_batches, K, q_schema=0, use_ste=False)

# Compute first-order grads of loss_hess with respect to lambdas_hess
grads = torch.autograd.grad(loss_hess, lambdas_hess, create_graph=True)[0]

# Compute exact Hessian trace
hessian_trace = 0.0
for i in range(L):
    for j in range(K):
        g = grads[i, j]
        g2 = torch.autograd.grad(g, lambdas_hess, retain_graph=True)[0][i, j]
        hessian_trace += g2

print(f"Hessian trace value: {hessian_trace}")

# Total loss
total_loss_hess = loss_hess + 0.5 * hessian_trace

optimizer_hess.zero_grad()
total_loss_hess.backward()
print(f"HessMerge loss before trace: {loss_hess.item():.6f}")
print(f"HessMerge total loss: {total_loss_hess.item():.6f}")
print(f"HessMerge theta grad (mean of abs): {theta_hess.grad.abs().mean().item():.8f}")
