import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

checkpoint = torch.load("data/real_world_features.pt")
features = checkpoint["features"]
tasks = checkpoint["tasks"]
classes = checkpoint["classes"]

K = 4
D = 192
L = 14
seed = 42

class ParametricRouter(nn.Module):
    def __init__(self, use_contraction_regularizer=False):
        super().__init__()
        self.use_contraction_regularizer = use_contraction_regularizer
        self.W_route = nn.ParameterList([
            nn.Parameter(torch.randn(K, D) * 0.001) for _ in range(L)
        ])
        self.log_tau = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(0.05))) for _ in range(L)
        ])

    def get_coefficients(self, h, layer_idx, temperature_scale=1.0):
        tau = torch.exp(self.log_tau[layer_idx]) * temperature_scale
        logits = torch.matmul(h, self.W_route[layer_idx].t())
        return torch.softmax(logits / tau, dim=-1)

def propagate_layers_with_scale(inputs, sweep_router, prototypes, temperature_scale=1.0, use_soft_coordinates=True, tau_c=0.05):
    h = inputs.clone()
    alpha_history = []
    gamma_l = [0.1 + 0.9 * (l / L) for l in range(1, L + 1)]
    for l in range(L):
        # Apply temperature scale during inference
        alpha_l = sweep_router.get_coefficients(h, l, temperature_scale=temperature_scale)
        alpha_history.append(alpha_l)
        expert_updates_list = []
        for k in range(K):
            scores = torch.matmul(h, prototypes[k].t())
            S_kc = torch.softmax(scores / tau_c, dim=-1)
            best_proto = torch.matmul(S_kc, prototypes[k])
            expert_updates_list.append(gamma_l[l] * (best_proto - h))
        expert_updates = torch.stack(expert_updates_list, dim=1)
        blended_update = torch.sum(alpha_l.unsqueeze(-1) * expert_updates, dim=1)
        h = h + blended_update
    return h, torch.stack(alpha_history, dim=1)

def evaluate_model(final_h, targets_task, targets_class, prototypes):
    B = final_h.shape[0]
    scores = torch.zeros(B, K, 10)
    for k in range(K):
        for c in range(10):
            scores[:, k, c] = torch.cosine_similarity(final_h, prototypes[k, c].unsqueeze(0), dim=-1)
    scores_flat = scores.view(B, K * 10)
    preds_flat = torch.argmax(scores_flat, dim=-1)
    pred_tasks = preds_flat // 10
    pred_classes = preds_flat % 10
    correct_class = torch.sum((pred_tasks == targets_task) & (pred_classes == targets_class)).item()
    correct_routing = torch.sum(pred_tasks == targets_task).item()
    return (correct_class / B) * 100.0, (correct_routing / B) * 100.0

# Set seed
torch.manual_seed(seed)
np.random.seed(seed)

# Stable class prototypes
prototypes = torch.zeros(K, 10, D)
for k in range(K):
    for c in range(10):
        proto = features[(tasks == k) & (classes == c)].mean(dim=0)
        prototypes[k, c] = proto / proto.norm()

# Sample splits
cal_indices = []
test_indices = []
for k in range(K):
    task_indices = torch.where(tasks == k)[0]
    perm = torch.randperm(len(task_indices))
    cal_indices.append(task_indices[perm[:16]])
    test_indices.append(task_indices[perm[16:116]])
cal_indices = torch.cat(cal_indices)
test_indices = torch.cat(test_indices)

cal_inputs = features[cal_indices]
cal_targets_task = tasks[cal_indices]
cal_targets_class = classes[cal_indices]

test_inputs = features[test_indices]
test_targets_task = tasks[test_indices]
test_targets_class = classes[test_indices]

# Train the model with a fixed lambda that provides strong regularization
lam = 0.010
print(f"Training CR-Router with lambda = {lam}...")

router = ParametricRouter(use_contraction_regularizer=True)
optimizer = optim.Adam(router.parameters(), lr=0.01)

for epoch in range(50):
    optimizer.zero_grad()
    h_prop = cal_inputs.clone()
    cal_loss = 0.0
    gamma_l = [0.1 + 0.9 * (l / L) for l in range(1, L + 1)]
    for l in range(L):
        alpha_l = router.get_coefficients(h_prop, l)
        if l < 4:
            target_dist = torch.ones(alpha_l.shape[0], K) / K
            cal_loss += -torch.sum(target_dist * torch.log(alpha_l + 1e-10), dim=-1).mean()
        else:
            cal_loss += nn.CrossEntropyLoss()(alpha_l, cal_targets_task)
        expert_updates = torch.stack([
            gamma_l[l] * (prototypes[k, cal_targets_class] - h_prop) for k in range(K)
        ], dim=1)
        blended_update = torch.sum(alpha_l.unsqueeze(-1) * expert_updates, dim=1)
        h_prop = h_prop + blended_update
    cal_loss = cal_loss / L
    
    reg_spec = sum(torch.sum(W ** 2) for W in router.W_route)
    reg_temp = sum(1.0 / (torch.exp(log_tau) ** 2) for log_tau in router.log_tau)
    total_loss = cal_loss + lam * reg_spec + lam * reg_temp
        
    total_loss.backward()
    optimizer.step()

print("\nEvaluating Adaptive Test-Time Temperature Annealing:")
print(f"{'Scale Factor':<15} | {'Test Acc':<10} | {'Routing Acc':<11} | {'Avg Test Temp':<13} | {'Entropy'}")
print("-" * 70)

scales = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
for scale in scales:
    final_h_scale, alphas_scale = propagate_layers_with_scale(test_inputs, router, prototypes, temperature_scale=scale)
    acc_class, acc_route = evaluate_model(final_h_scale, test_targets_task, test_targets_class, prototypes)
    entropy = -torch.mean(torch.sum(alphas_scale * torch.log(alphas_scale + 1e-10), dim=-1)).item()
    
    with torch.no_grad():
        avg_temp = np.mean([torch.exp(router.log_tau[l]).item() * scale for l in range(L)])
        
    print(f"{scale:<15.2f} | {acc_class:<9.2f}% | {acc_route:<10.2f}% | {avg_temp:<13.4f} | {entropy:.4f}")

# Let's also do a baseline check for unregularized router
print("\nFor comparison, training Unregularized Linear Router...")
router_unreg = ParametricRouter(use_contraction_regularizer=False)
optimizer_unreg = optim.Adam(router_unreg.parameters(), lr=0.01)
for epoch in range(50):
    optimizer_unreg.zero_grad()
    h_prop = cal_inputs.clone()
    cal_loss = 0.0
    gamma_l = [0.1 + 0.9 * (l / L) for l in range(1, L + 1)]
    for l in range(L):
        alpha_l = router_unreg.get_coefficients(h_prop, l)
        if l < 4:
            target_dist = torch.ones(alpha_l.shape[0], K) / K
            cal_loss += -torch.sum(target_dist * torch.log(alpha_l + 1e-10), dim=-1).mean()
        else:
            cal_loss += nn.CrossEntropyLoss()(alpha_l, cal_targets_task)
        expert_updates = torch.stack([
            gamma_l[l] * (prototypes[k, cal_targets_class] - h_prop) for k in range(K)
        ], dim=1)
        blended_update = torch.sum(alpha_l.unsqueeze(-1) * expert_updates, dim=1)
        h_prop = h_prop + blended_update
    cal_loss = cal_loss / L
    cal_loss.backward()
    optimizer_unreg.step()

final_h_unreg, alphas_unreg = propagate_layers_with_scale(test_inputs, router_unreg, prototypes, temperature_scale=1.0)
acc_class_unreg, acc_route_unreg = evaluate_model(final_h_unreg, test_targets_task, test_targets_class, prototypes)
print(f"Unregularized Baseline: Test Acc = {acc_class_unreg:.2f}%, Routing Acc = {acc_route_unreg:.2f}%")
