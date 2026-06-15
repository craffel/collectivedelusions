import sys
sys.path.insert(0, "./env_packages")
import torch
import timm
import time
from experiments.run_merging import (
    get_layer_idx, load_eval_data, datasets_config,
    compute_entropy_loss
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

print("Loading evaluation data...")
loaders, calib_batches = load_eval_data()

print("Loading checkpoints...")
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

print("Testing SACM step...")
theta = torch.ones(L, K, device=device) * -0.847
theta.requires_grad_(True)
optimizer = torch.optim.Adam([theta], lr=1e-2)

t0 = time.time()
# Standard forward
lambdas = torch.sigmoid(theta)
loss = compute_entropy_loss(lambdas, params_pre, task_vectors, model, calib_batches, K, q_schema=0, use_ste=False)

# Compute first-order gradient with respect to theta
loss.backward()
theta_grad = theta.grad.clone()
optimizer.zero_grad()

# Compute SAM perturbation
rho = 0.05
grad_norm = torch.norm(theta_grad) + 1e-12
epsilon = rho * theta_grad / grad_norm

# Perturb
perturbed_theta = theta + epsilon
perturbed_lambdas = torch.sigmoid(perturbed_theta)

# Compute perturbed loss
perturbed_loss = compute_entropy_loss(perturbed_lambdas, params_pre, task_vectors, model, calib_batches, K, q_schema=0, use_ste=False)

# Final step
optimizer.zero_grad()
perturbed_loss.backward()
optimizer.step()
t1 = time.time()

print(f"SACM step successfully completed in {t1 - t0:.4f} seconds!")
print("Initial loss:", loss.item())
print("Perturbed loss:", perturbed_loss.item())
print("Theta grad (mean of abs):", theta_grad.abs().mean().item())
