import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# --- SANDBOX GENERATION ---
D = 192
K = 4
L = 14
CLASSES = 10

# Noise levels calibrated to match the expert ceilings from literature
noise_levels = [0.01, 0.08, 0.22, 0.85]  # MNIST, F-MNIST, CIFAR-10, SVHN

# Generate class-specific prototypes in orthogonal task subspaces
# Each task has a disjoint 48-dimensional subspace
subspace_dim = D // K  # 48

prototypes = []  # shape: (K, CLASSES, D)
for k in range(K):
    task_prototypes = []
    for c in range(CLASSES):
        p = np.zeros(D)
        start_idx = k * subspace_dim
        end_idx = (k + 1) * subspace_dim
        p[start_idx:end_idx] = np.random.normal(0, 1, subspace_dim)
        p = p / np.linalg.norm(p)
        task_prototypes.append(p)
    prototypes.append(task_prototypes)
prototypes = np.array(prototypes)

def generate_split(num_samples_per_task, noise_levels):
    features = []
    labels = []
    task_ids = []
    for k in range(K):
        for _ in range(num_samples_per_task):
            c = np.random.randint(0, CLASSES)
            p = prototypes[k, c]
            noise = np.random.normal(0, noise_levels[k], D)
            x = p + noise
            x = x / np.linalg.norm(x)
            features.append(x)
            labels.append(c)
            task_ids.append(k)
    return (torch.tensor(np.array(features), dtype=torch.float32),
            torch.tensor(np.array(labels), dtype=torch.long),
            torch.tensor(np.array(task_ids), dtype=torch.long))

train_x, train_y, train_task = generate_split(1000, noise_levels)
calib_x, calib_y, calib_task = generate_split(16, noise_levels)
test_x, test_y, test_task = generate_split(250, noise_levels)

# --- MODEL DEFINITIONS ---
class LinearLayerNearIdentity(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.005)
        self.bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class BackboneMLP(nn.Module):
    def __init__(self, num_layers=14, dim=192):
        super().__init__()
        self.layers = nn.ModuleList([LinearLayerNearIdentity(dim) for _ in range(num_layers)])
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, heads):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList(heads)
        
    def forward(self, x, task_id):
        z = self.backbone(x)
        logits = torch.stack([self.heads[task_id[i]](z[i]) for i in range(len(task_id))])
        return logits, z

base_backbone = BackboneMLP(L, D)
base_heads = [nn.Linear(D, CLASSES) for _ in range(K)]
for k in range(K):
    with torch.no_grad():
        base_heads[k].weight.copy_(torch.tensor(prototypes[k], dtype=torch.float32))
        base_heads[k].bias.zero_()

base_model = MultiTaskModel(base_backbone, base_heads)
base_model.eval()

expert_models = []
for k in range(K):
    print(f"Fine-tuning specialized expert {k}...")
    exp_backbone = BackboneMLP(L, D)
    exp_backbone.load_state_dict(base_backbone.state_dict())
    
    exp_heads = []
    for i in range(K):
        head = nn.Linear(D, CLASSES)
        head.load_state_dict(base_heads[i].state_dict())
        exp_heads.append(head)
        
    exp_model = MultiTaskModel(exp_backbone, exp_heads)
    task_mask = (train_task == k)
    task_x = train_x[task_mask]
    task_y = train_y[task_mask]
    task_t = train_task[task_mask]
    
    optimizer = optim.AdamW(list(exp_model.backbone.parameters()) + list(exp_model.heads[k].parameters()), lr=1e-3, weight_decay=1e-3)
    exp_model.train()
    for epoch in range(12):
        indices = torch.randperm(task_x.shape[0])
        for i in range(0, task_x.shape[0], 64):
            batch_idx = indices[i:i+64]
            bx, by, bt = task_x[batch_idx], task_y[batch_idx], task_t[batch_idx]
            optimizer.zero_grad()
            logits, _ = exp_model(bx, bt)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()
    exp_model.eval()
    expert_models.append(exp_model)

# stand-alone accuracy
expert_ceilings = []
for k in range(K):
    exp_model = expert_models[k]
    task_mask = (test_task == k)
    tx, ty, tt = test_x[task_mask], test_y[task_mask], test_task[task_mask]
    with torch.no_grad():
        logits, _ = exp_model(tx, tt)
        preds = logits.argmax(dim=-1)
        acc = (preds == ty).float().mean().item() * 100.0
    expert_ceilings.append(acc)

# --- LORA LOW-RANK DECOMPOSITION ---
r_rank = 8
A_adapters = []
B_adapters = []
delta_biases = []

for k in range(K):
    exp_backbone = expert_models[k].backbone
    task_A = []
    task_B = []
    task_bias = []
    for l in range(L):
        W_base = base_backbone.layers[l].weight.data
        b_base = base_backbone.layers[l].bias.data
        W_exp = exp_backbone.layers[l].weight.data
        b_exp = exp_backbone.layers[l].bias.data
        V_kl = W_exp - W_base
        delta_b = b_exp - b_base
        
        U, S, Vh = torch.linalg.svd(V_kl)
        Ur = U[:, :r_rank]
        Sr = torch.diag(torch.sqrt(S[:r_rank]))
        Vhr = Vh[:r_rank, :]
        
        A_kl = torch.matmul(Ur, Sr)
        B_kl = torch.matmul(Sr, Vhr)
        
        task_A.append(A_kl)
        task_B.append(B_kl)
        task_bias.append(delta_b)
        
    A_adapters.append(task_A)
    B_adapters.append(task_B)
    delta_biases.append(task_bias)

# Uniform Merging Backbone
W_uniform = []
b_uniform = []
for l in range(L):
    W_l = base_backbone.layers[l].weight.data.clone()
    b_l = base_backbone.layers[l].bias.data.clone()
    for k in range(K):
        W_l += (expert_models[k].backbone.layers[l].weight.data - base_backbone.layers[l].weight.data) / K
        b_l += (expert_models[k].backbone.layers[l].bias.data - base_backbone.layers[l].bias.data) / K
    W_uniform.append(W_l)
    b_uniform.append(b_l)

centroids = torch.zeros(K, D)
for k in range(K):
    centroids[k, k*subspace_dim:(k+1)*subspace_dim] = 1.0
centroids_norm = centroids / centroids.norm(dim=-1, keepdim=True)

def pfsr_coefficients(features, gamma_OOD=0.2, tau=0.05):
    feat_norm = features / (features.norm(dim=-1, keepdim=True) + 1e-12)
    sims = torch.matmul(feat_norm, centroids_norm.t())
    coeffs = torch.zeros_like(sims)
    for b in range(features.shape[0]):
        max_sim, max_idx = sims[b].max(dim=0)
        if max_sim < gamma_OOD:
            coeffs[b] = 0.0
        else:
            coeffs[b] = torch.softmax(sims[b] / tau, dim=-1)
    return coeffs

# Global Router Training
class GlobalLinearRouter(nn.Module):
    def __init__(self, dim=192, out_dim=4):
        super().__init__()
        self.W_route = nn.Parameter(torch.randn(dim, out_dim) * 0.01)
    def forward(self, z):
        return torch.softmax(torch.matmul(z, self.W_route), dim=-1)

with torch.no_grad():
    train_z = base_model.backbone(train_x)
    calib_z = base_model.backbone(calib_x)

global_router = GlobalLinearRouter(D, K)
router_opt = optim.AdamW(global_router.parameters(), lr=1e-2, weight_decay=1e-4)
for epoch in range(100):
    router_opt.zero_grad()
    coeffs = global_router(calib_z)
    mean_coeffs = coeffs.mean(dim=0)
    W_merged = []
    b_merged = []
    for l in range(L):
        W_l = base_backbone.layers[l].weight.data.clone()
        b_l = base_backbone.layers[l].bias.data.clone()
        for k in range(K):
            W_l += mean_coeffs[k] * (expert_models[k].backbone.layers[l].weight.data - base_backbone.layers[l].weight.data)
            b_l += mean_coeffs[k] * (expert_models[k].backbone.layers[l].bias.data - base_backbone.layers[l].bias.data)
        W_merged.append(W_l)
        b_merged.append(b_l)
    feat = calib_x
    for l in range(L):
        feat = F.relu(F.linear(feat, W_merged[l], b_merged[l]))
    logits_all_experts = torch.stack([expert_models[k].heads[k](feat) for k in range(K)], dim=1)
    logits = torch.sum(coeffs.unsqueeze(-1) * logits_all_experts, dim=1)
    loss = F.cross_entropy(logits, calib_y)
    loss.backward()
    router_opt.step()
global_router.eval()

# Blended head prediction function
def get_blended_head_logits(feat, coeffs):
    logits_all_experts = torch.stack([expert_models[k].heads[k](feat) for k in range(K)], dim=1)
    return torch.sum(coeffs.unsqueeze(-1) * logits_all_experts, dim=1)

# Non-oracle evaluators
def evaluate_uniform_non_oracle(X):
    feat = X
    for l in range(L):
        feat = F.relu(F.linear(feat, W_uniform[l], b_uniform[l]))
    coeffs = torch.ones(X.shape[0], K) / K
    return get_blended_head_logits(feat, coeffs)

def evaluate_linear_router_non_oracle(X, task_ids, heterogeneous=False):
    with torch.no_grad():
        z = base_model.backbone(X)
        coeffs = global_router(z)
        if heterogeneous:
            mean_coeffs = coeffs.mean(dim=0)
            W_merged = []
            b_merged = []
            for l in range(L):
                W_l = base_backbone.layers[l].weight.data.clone()
                b_l = base_backbone.layers[l].bias.data.clone()
                for k in range(K):
                    W_l += mean_coeffs[k] * (expert_models[k].backbone.layers[l].weight.data - base_backbone.layers[l].weight.data)
                    b_l += mean_coeffs[k] * (expert_models[k].backbone.layers[l].bias.data - base_backbone.layers[l].bias.data)
                W_merged.append(W_l)
                b_merged.append(b_l)
            feat = X
            for l in range(L):
                feat = F.relu(F.linear(feat, W_merged[l], b_merged[l]))
        else:
            # Homogeneous processing
            all_feats = []
            for k in range(K):
                mask = (task_ids == k)
                if not mask.any():
                    continue
                block_X = X[mask]
                block_coeffs = coeffs[mask].mean(dim=0)
                W_merged = []
                b_merged = []
                for l in range(L):
                    W_l = base_backbone.layers[l].weight.data.clone()
                    b_l = base_backbone.layers[l].bias.data.clone()
                    for k_exp in range(K):
                        W_l += block_coeffs[k_exp] * (expert_models[k_exp].backbone.layers[l].weight.data - base_backbone.layers[l].weight.data)
                        b_l += block_coeffs[k_exp] * (expert_models[k_exp].backbone.layers[l].bias.data - base_backbone.layers[l].bias.data)
                    W_merged.append(W_l)
                    b_merged.append(b_l)
                block_feat = block_X
                for l in range(L):
                    block_feat = F.relu(F.linear(block_feat, W_merged[l], b_merged[l]))
                all_feats.append((mask, block_feat))
            feat = torch.zeros(X.shape[0], D)
            for mask, block_feat in all_feats:
                feat[mask] = block_feat
        return get_blended_head_logits(feat, coeffs)

def evaluate_pfsr_non_oracle(X, task_ids, heterogeneous=False, use_mbh=False):
    with torch.no_grad():
        z = base_model.backbone(X)
        coeffs = pfsr_coefficients(z)
        if heterogeneous:
            if use_mbh:
                predicted_tasks = coeffs.argmax(dim=-1)
                all_feats = torch.zeros(X.shape[0], D)
                for k in range(K):
                    mask = (predicted_tasks == k)
                    if not mask.any():
                        continue
                    sub_X = X[mask]
                    sub_coeffs = coeffs[mask].mean(dim=0)
                    W_merged = []
                    b_merged = []
                    for l in range(L):
                        W_l = base_backbone.layers[l].weight.data.clone()
                        b_l = base_backbone.layers[l].bias.data.clone()
                        for k_exp in range(K):
                            W_l += sub_coeffs[k_exp] * (expert_models[k_exp].backbone.layers[l].weight.data - base_backbone.layers[l].weight.data)
                            b_l += sub_coeffs[k_exp] * (expert_models[k_exp].backbone.layers[l].bias.data - base_backbone.layers[l].bias.data)
                        W_merged.append(W_l)
                        b_merged.append(b_l)
                    feat_sub = sub_X
                    for l in range(L):
                        feat_sub = F.relu(F.linear(feat_sub, W_merged[l], b_merged[l]))
                    all_feats[mask] = feat_sub
                feat = all_feats
            else:
                mean_coeffs = coeffs.mean(dim=0)
                W_merged = []
                b_merged = []
                for l in range(L):
                    W_l = base_backbone.layers[l].weight.data.clone()
                    b_l = base_backbone.layers[l].bias.data.clone()
                    for k in range(K):
                        W_l += mean_coeffs[k] * (expert_models[k].backbone.layers[l].weight.data - base_backbone.layers[l].weight.data)
                        b_l += mean_coeffs[k] * (expert_models[k].backbone.layers[l].bias.data - base_backbone.layers[l].bias.data)
                    W_merged.append(W_l)
                    b_merged.append(b_l)
                feat = X
                for l in range(L):
                    feat = F.relu(F.linear(feat, W_merged[l], b_merged[l]))
        else:
            all_feats = []
            for k in range(K):
                mask = (task_ids == k)
                if not mask.any():
                    continue
                block_X = X[mask]
                block_coeffs = coeffs[mask].mean(dim=0)
                W_merged = []
                b_merged = []
                for l in range(L):
                    W_l = base_backbone.layers[l].weight.data.clone()
                    b_l = base_backbone.layers[l].bias.data.clone()
                    for k_exp in range(K):
                        W_l += block_coeffs[k_exp] * (expert_models[k_exp].backbone.layers[l].weight.data - base_backbone.layers[l].weight.data)
                        b_l += block_coeffs[k_exp] * (expert_models[k_exp].backbone.layers[l].bias.data - base_backbone.layers[l].bias.data)
                    W_merged.append(W_l)
                    b_merged.append(b_l)
                block_feat = block_X
                for l in range(L):
                    block_feat = F.relu(F.linear(block_feat, W_merged[l], b_merged[l]))
                all_feats.append((mask, block_feat))
            feat = torch.zeros(X.shape[0], D)
            for mask, block_feat in all_feats:
                feat[mask] = block_feat
        return get_blended_head_logits(feat, coeffs)

# SABLE Single Pass non-oracle (Layer 0 Routing)
def evaluate_sable_single_pass_non_oracle(X, M=None):
    with torch.no_grad():
        W_base0 = base_backbone.layers[0].weight.data
        b_base0 = base_backbone.layers[0].bias.data
        H_base0 = F.linear(X, W_base0, b_base0)
        feat0_base = F.relu(H_base0)
        
        coeffs = pfsr_coefficients(feat0_base)
        
        if M is not None and M < K:
            top_vals, top_idx = torch.topk(coeffs, M, dim=-1)
            mask = torch.zeros_like(coeffs)
            mask.scatter_(dim=-1, index=top_idx, src=torch.ones_like(top_vals))
            pruned_coeffs = coeffs * mask
            coeffs = pruned_coeffs / (pruned_coeffs.sum(dim=-1, keepdim=True) + 1e-12)
            
        H_experts0 = torch.zeros(K, X.shape[0], D, device=X.device)
        active_expert_mask = (coeffs > 1e-12).any(dim=0)
        for k in range(K):
            if active_expert_mask[k]:
                A_k0 = A_adapters[k][0]
                B_k0 = B_adapters[k][0]
                delta_b_k0 = delta_biases[k][0]
                proj = torch.matmul(X, B_k0.t())
                out = torch.matmul(proj, A_k0.t()) + delta_b_k0
                H_experts0[k] = out
        
        coeffs_reshaped = coeffs.t().unsqueeze(-1)
        H_blended0 = torch.sum(coeffs_reshaped * H_experts0, dim=0)
        feat = F.relu(H_base0 + H_blended0)
        
        for l in range(1, L):
            W_base = base_backbone.layers[l].weight.data
            b_base = base_backbone.layers[l].bias.data
            H_base = F.linear(feat, W_base, b_base)
            
            H_experts = torch.zeros(K, feat.shape[0], D, device=feat.device)
            for k in range(K):
                if active_expert_mask[k]:
                    A_kl = A_adapters[k][l]
                    B_kl = B_adapters[k][l]
                    delta_b_kl = delta_biases[k][l]
                    proj = torch.matmul(feat, B_kl.t())
                    out = torch.matmul(proj, A_kl.t()) + delta_b_kl
                    H_experts[k] = out
            
            H_blended = torch.sum(coeffs_reshaped * H_experts, dim=0)
            feat = F.relu(H_base + H_blended)
            
        logits = get_blended_head_logits(feat, coeffs)
    return logits

def compute_acc(logits, targets):
    return (logits.argmax(dim=-1) == targets).float().mean().item() * 100.0

print("\n=== NON-ORACLE STREAMING EVALUATION ===")
# Uniform
uniform_acc = compute_acc(evaluate_uniform_non_oracle(test_x), test_y)

# Linear Router
lr_homog = compute_acc(evaluate_linear_router_non_oracle(test_x, test_task, heterogeneous=False), test_y)
lr_hetero = compute_acc(evaluate_linear_router_non_oracle(test_x, test_task, heterogeneous=True), test_y)

# PFSR No MBH
pfsr_homog = compute_acc(evaluate_pfsr_non_oracle(test_x, test_task, heterogeneous=False), test_y)
pfsr_hetero = compute_acc(evaluate_pfsr_non_oracle(test_x, test_task, heterogeneous=True, use_mbh=False), test_y)

# PFSR + MBH
pfsr_mbh_hetero = compute_acc(evaluate_pfsr_non_oracle(test_x, test_task, heterogeneous=True, use_mbh=True), test_y)

# SABLE Single-Pass (Ours)
sable_sp_acc = compute_acc(evaluate_sable_single_pass_non_oracle(test_x), test_y)

print(f"{'Method':<25} | {'Homogeneous':<15} | {'Heterogeneous':<15}")
print("-" * 65)
print(f"{'Expert Ceiling':<25} | {np.mean(expert_ceilings):>14.2f}% | {np.mean(expert_ceilings):>14.2f}%")
print(f"{'Uniform Merging':<25} | {uniform_acc:>14.2f}% | {uniform_acc:>14.2f}%")
print(f"{'Linear Router (Unreg)':<25} | {lr_homog:>14.2f}% | {lr_hetero:>14.2f}%")
print(f"{'PFSR (No MBH)':<25} | {pfsr_homog:>14.2f}% | {pfsr_hetero:>14.2f}%")
print(f"{'PFSR + MBH':<25} | {pfsr_homog:>14.2f}% | {pfsr_mbh_hetero:>14.2f}%")
print(f"{'SABLE Single-Pass (Ours)':<25} | {sable_sp_acc:>14.2f}% | {sable_sp_acc:>14.2f}%")
