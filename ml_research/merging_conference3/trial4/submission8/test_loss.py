import sys
sys.path.insert(0, "./env_packages")
import torch
import timm
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
theta_ada = torch.ones(L, K, device=device) * -0.847  # sigmoid(-0.847) = 0.3
theta_ada.requires_grad_(True)
lambdas_ada = torch.sigmoid(theta_ada)

loss_ada = compute_entropy_loss(lambdas_ada, params_pre, task_vectors, model, calib_batches, K, q_schema=0, use_ste=False)
print("Loss computed:", loss_ada.item())
