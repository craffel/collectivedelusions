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

# Separate head parameters from backbone parameters
params_pre = {k: v for k, v in base_state.items() if not k.startswith("head.")}
expert_heads = []
task_vectors = []

for k in range(K):
    expert_heads.append({
        "head.weight": expert_states[k]["head.weight"],
        "head.bias": expert_states[k]["head.bias"]
    })
    # Extract task vector
    tv = {}
    for name, p in expert_states[k].items():
        if not name.startswith("head."):
            tv[name] = p - base_state[name]
    task_vectors.append(tv)
    
# Standard model setup for functional calling
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
model = model.to(device)

import experiments.run_merging
experiments.run_merging.expert_heads = expert_heads
experiments.run_merging.device = device

print("Computing entropy loss...")
theta_hess = torch.ones(L, K, device=device) * -0.847  # sigmoid(-0.847) = 0.3
theta_hess.requires_grad_(True)
lambdas_hess = torch.sigmoid(theta_hess)

loss_hess = compute_entropy_loss(lambdas_hess, params_pre, task_vectors, model, calib_batches, K, q_schema=0, use_ste=False)

print("Computing first-order grads...")
t0 = time.time()
grads = torch.autograd.grad(loss_hess, lambdas_hess, create_graph=True)[0]
t1 = time.time()
print(f"First-order grads computed in {t1 - t0:.4f} seconds.")
print(f"grads requires_grad: {grads.requires_grad}")
print(f"grads shape: {grads.shape}")
print(f"grads value (first few elements): {grads[0]}")

print("Computing one second-order grad element...")
t0 = time.time()
g = grads[0, 0]
g2 = torch.autograd.grad(g, lambdas_hess, retain_graph=True)[0]
t1 = time.time()
print(f"Second-order grad computed in {t1 - t0:.4f} seconds.")
print(f"g2 shape: {g2.shape}")
print(f"g2[0,0] value (diagonal second derivative): {g2[0, 0].item()}")
