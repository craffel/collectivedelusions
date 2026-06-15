import os
import torch
import timm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.run_merging import (
    get_layer_idx, load_eval_data, datasets_config,
    compute_entropy_loss
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

loaders, calib_batches = load_eval_data()
base_state = torch.load("checkpoints/vit_tiny_pretrained.pt", map_location=device)

expert_states = []
for d in datasets_config.keys():
    expert_states.append(torch.load(f"checkpoints/vit_tiny_{d}.pt", map_location=device))
    
K = len(expert_states)
L = 14  # 14 layer groups

params_pre = {k: v for k, v in base_state.items() if not k.startswith("head.")}
expert_heads = []
task_vectors = []

for k in range(K):
    expert_heads.append({
        "head.weight": expert_states[k]["head.weight"],
        "head.bias": expert_states[k]["head.bias"]
    })
    tv = {}
    for name, p in expert_states[k].items():
        if not name.startswith("head."):
            tv[name] = p - base_state[name]
    task_vectors.append(tv)
    
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
model = model.to(device)

import experiments.run_merging
experiments.run_merging.expert_heads = expert_heads
experiments.run_merging.device = device

# Compute L2 norm of task vectors per layer group
N = torch.zeros(L, K, device=device)
for l in range(L):
    for k in range(K):
        sq_norm = 0.0
        for name, tv_p in task_vectors[k].items():
            if get_layer_idx(name) == l:
                sq_norm += torch.sum(tv_p ** 2).item()
        N[l, k] = np.sqrt(sq_norm)

# Optimization loop to track convergence
def run_and_log(objective_type):
    poly_coeffs = torch.zeros(3, K, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([poly_coeffs], lr=1e-2)
    
    losses = []
    sharpnesses = []
    
    for epoch in range(40):
        # Compute layer-wise lambdas from polynomial
        lambdas = torch.zeros(L, K, device=device)
        for l in range(L):
            depth = l / L
            for k in range(K):
                logits = poly_coeffs[0, k] + poly_coeffs[1, k] * depth + poly_coeffs[2, k] * (depth ** 2)
                lambdas[l, k] = torch.sigmoid(logits)
                
        loss = compute_entropy_loss(lambdas, params_pre, task_vectors, model, calib_batches, K, q_schema=0, use_ste=False)
        losses.append(loss.item())
        print(f"  Step {epoch:02d} | loss: {loss.item():.4f}")
        
        if objective_type == "polysacm":
            # Compute CR-SACM perturbation
            grads = torch.autograd.grad(loss, lambdas, retain_graph=True)[0]
            N_clipped = torch.clamp(N, min=0.10)
            grads_hat = grads / N_clipped
            grad_hat_norm = torch.norm(grads_hat) + 1e-12
            rho = 0.15
            epsilon = rho * grads_hat / (grad_hat_norm * N_clipped)
            perturbed_lambdas = torch.clamp(lambdas + epsilon, 0.0, 1.0)
            perturbed_loss = compute_entropy_loss(perturbed_lambdas, params_pre, task_vectors, model, calib_batches, K, q_schema=0, use_ste=False)
            
            sharpness = perturbed_loss.item() - loss.item()
            sharpnesses.append(sharpness)
            loss_to_backprop = perturbed_loss
        else:
            sharpnesses.append(0.0)
            loss_to_backprop = loss
            
        optimizer.zero_grad()
        loss_to_backprop.backward()
        optimizer.step()
        
    return losses, sharpnesses

print("Running PolyMerge (unregularized baseline)...")
poly_losses, _ = run_and_log("poly")

print("Running CR-PolySACM (sharpness-aware ours)...")
sacm_losses, sharpnesses = run_and_log("polysacm")

print("Generating plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Entropy Loss Convergence
ax1.plot(poly_losses, label="PolyMerge (Baseline)", color="tab:blue", linestyle="--", linewidth=2)
ax1.plot(sacm_losses, label="CR-PolySACM (Ours)", color="tab:red", linewidth=2)
ax1.set_xlabel("Optimization Step", fontsize=12)
ax1.set_ylabel("Multi-Task Entropy Loss", fontsize=12)
ax1.set_title("Multi-Task Entropy Loss Convergence", fontsize=13, fontweight='bold')
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(fontsize=11)

# Plot 2: Sharpness Convergence (CR-PolySACM)
ax2.plot(sharpnesses, label="Local Sharpness ($\Delta \mathcal{L}$)", color="tab:orange", linewidth=2)
ax2.set_xlabel("Optimization Step", fontsize=12)
ax2.set_ylabel("Local Sharpness ($\mathcal{L}(\Lambda + \epsilon) - \mathcal{L}(\Lambda)$)", fontsize=12)
ax2.set_title("Local Landscape Flatness Trajectory", fontsize=13, fontweight='bold')
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(fontsize=11)

plt.tight_layout()
os.makedirs("submission", exist_ok=True)
plt.savefig("submission/convergence_plot.png", dpi=300)
print("Convergence plot saved to submission/convergence_plot.png")
