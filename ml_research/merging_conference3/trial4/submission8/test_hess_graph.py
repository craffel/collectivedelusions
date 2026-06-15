import sys
sys.path.insert(0, "./env_packages")
import torch
import timm
from experiments.run_merging import (
    get_layer_idx, load_eval_data, datasets_config,
    compute_entropy_loss
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Check case 1: without create_graph=True on the second grad call
print("--- CASE 1: Without create_graph=True on second grad call ---")
theta_1 = torch.ones(L, K, device=device) * -0.847
theta_1.requires_grad_(True)
lambdas_1 = torch.sigmoid(theta_1)
loss_1 = compute_entropy_loss(lambdas_1, params_pre, task_vectors, model, calib_batches, K, q_schema=0, use_ste=False)
grads_1 = torch.autograd.grad(loss_1, lambdas_1, create_graph=True)[0]
g_1 = grads_1[0, 0]
g2_1 = torch.autograd.grad(g_1, lambdas_1, retain_graph=True)[0][0, 0]
print("g2_1 requires_grad:", g2_1.requires_grad)

# Check case 2: with create_graph=True on the second grad call
print("\n--- CASE 2: With create_graph=True on second grad call ---")
theta_2 = torch.ones(L, K, device=device) * -0.847
theta_2.requires_grad_(True)
lambdas_2 = torch.sigmoid(theta_2)
loss_2 = compute_entropy_loss(lambdas_2, params_pre, task_vectors, model, calib_batches, K, q_schema=0, use_ste=False)
grads_2 = torch.autograd.grad(loss_2, lambdas_2, create_graph=True)[0]
g_2 = grads_2[0, 0]
g2_2 = torch.autograd.grad(g_2, lambdas_2, create_graph=True)[0][0, 0]
print("g2_2 requires_grad:", g2_2.requires_grad)
