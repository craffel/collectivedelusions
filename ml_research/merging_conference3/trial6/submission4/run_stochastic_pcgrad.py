import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import random

# Replicate representation sandbox from run_experiments.py exactly
D = 192  # High-dimensional feature dimension
K = 4    # Number of tasks (MNIST, FashionMNIST, CIFAR-10, SVHN)
C = 10   # Classes per task
L = 14   # Layers in the router
D_PROJ = 4  # Projection dimension d
SEEDS = [10, 11, 12, 13, 14]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_sandbox_data(seed, num_train=1000, num_test=250, num_cal=16, leakage=0.0):
    set_seed(seed)
    sigmas = [0.01, 0.12, 0.18, 0.95]
    subspace_dim = D // K  # 48
    prototypes = torch.zeros(K, C, D)
    for k in range(K):
        for c in range(C):
            base_coords = torch.randn(subspace_dim)
            base_coords = base_coords / (torch.norm(base_coords) + 1e-8)
            prototypes[k, c, k * subspace_dim : (k + 1) * subspace_dim] = (1.0 - leakage) * base_coords
            if leakage > 0.0:
                leak_energy_per_task = leakage / (K - 1)
                for other_k in range(K):
                    if other_k != k:
                        leak_coords = torch.randn(subspace_dim)
                        leak_coords = leak_coords / (torch.norm(leak_coords) + 1e-8)
                        prototypes[k, c, other_k * subspace_dim : (other_k + 1) * subspace_dim] = leak_energy_per_task * leak_coords
            prototypes[k, c] = prototypes[k, c] / (torch.norm(prototypes[k, c]) + 1e-8)
            
    splits = {"train": [], "test": [], "cal": []}
    for k in range(K):
        sigma = sigmas[k]
        # Train split
        train_feats = []
        train_labels = []
        for _ in range(num_train):
            c = random.randint(0, C - 1)
            noise = torch.randn(D) * sigma
            x = prototypes[k, c] + noise
            train_feats.append(x)
            train_labels.append(c)
        splits["train"].append((torch.stack(train_feats), torch.tensor(train_labels)))
        
        # Test split
        test_feats = []
        test_labels = []
        for _ in range(num_test):
            c = random.randint(0, C - 1)
            noise = torch.randn(D) * sigma
            x = prototypes[k, c] + noise
            test_feats.append(x)
            test_labels.append(c)
        splits["test"].append((torch.stack(test_feats), torch.tensor(test_labels)))
        
        # Calibration split
        cal_feats = []
        cal_labels = []
        for _ in range(num_cal):
            c = random.randint(0, C - 1)
            noise = torch.randn(D) * sigma
            x = prototypes[k, c] + noise
            cal_feats.append(x)
            cal_labels.append(c)
        splits["cal"].append((torch.stack(cal_feats), torch.tensor(cal_labels)))
    return splits

def train_experts(train_splits, test_splits, epochs=40, lr=1e-2):
    experts = []
    expert_ceilings = []
    for k in range(K):
        X_train, y_train = train_splits[k]
        X_test, y_test = test_splits[k]
        
        model = nn.Linear(D, C)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            preds = torch.argmax(test_logits, dim=1)
            acc = (preds == y_test).float().mean().item()
            expert_ceilings.append(acc)
            experts.append(model)
    return experts

# Router architecture
class L3LinearRouter(nn.Module):
    def __init__(self, L=L, K=K, d=D_PROJ):
        super(L3LinearRouter, self).__init__()
        self.W = nn.Parameter(torch.zeros(L, K, d))
        self.B = nn.Parameter(torch.zeros(L, K))
    def forward(self, psi):
        alpha = torch.einsum('bd,lkd->blk', psi, self.W) + self.B.unsqueeze(0)
        return alpha

def compute_pca_matrix(X_cal, d=D_PROJ):
    mean = X_cal.mean(dim=0, keepdim=True)
    X_centered = X_cal - mean
    U, S, V = torch.linalg.svd(X_centered, full_matrices=False)
    P = V[:d, :].t()
    return P

def project_states(X, P):
    X_proj = torch.matmul(X, P)
    norms = torch.norm(X_proj, dim=-1, keepdim=True)
    return X_proj / (norms + 1e-8)

def compute_task_anchors(cal_splits, P):
    anchors = []
    for k in range(K):
        X_cal, _ = cal_splits[k]
        psi = project_states(X_cal, P)
        anchors.append(psi.mean(dim=0))
    return torch.stack(anchors)

# Stochastic PCGrad Router optimization
def train_router_stochastic_pcgrad(cal_splits, experts, router, P, lambda_wd=1e-3, lambda_anchor=0.1, anchors=None, epochs=100, lr=1e-2, M_tasks=2):
    # Prepare combined calibration batch
    all_cal_z = []
    all_cal_y = []
    all_cal_tasks = []
    for k in range(K):
        X_cal, y_cal = cal_splits[k]
        all_cal_z.append(X_cal)
        all_cal_y.append(y_cal)
        all_cal_tasks.append(torch.ones(X_cal.shape[0], dtype=torch.long) * k)
        
    all_cal_z = torch.cat(all_cal_z, dim=0)
    all_cal_y = torch.cat(all_cal_y, dim=0)
    all_cal_tasks = torch.cat(all_cal_tasks, dim=0)
    
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    router.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        psi = project_states(all_cal_z, P)
        alpha = router(psi)
        
        alpha_avg = alpha.mean(dim=1)
        bar_alpha = alpha_avg.mean(dim=0)
        
        W_merged = torch.zeros(C, D)
        b_merged = torch.zeros(C)
        for k in range(K):
            W_merged += bar_alpha[k] * experts[k].weight
            b_merged += bar_alpha[k] * experts[k].bias
            
        logits = torch.matmul(all_cal_z, W_merged.t()) + b_merged
        
        # Manual WD
        loss_wd = 0.0
        for p in router.parameters():
            loss_wd += lambda_wd * torch.sum(p**2)
            
        # TSAR anchor penalty
        loss_anchor = 0.0
        if lambda_anchor > 0.0 and anchors is not None:
            loss_anchor += lambda_anchor * torch.sum((router.W - anchors.unsqueeze(0))**2)
            
        # Stochastic Task Sampling of M_tasks
        active_tasks = random.sample(list(range(K)), M_tasks)
        
        grads = []
        params = list(router.parameters())
        for k in active_tasks:
            optimizer.zero_grad()
            mask_k = (all_cal_tasks == k)
            loss_k = criterion(logits[mask_k], all_cal_y[mask_k])
            # Scale WD and Anchor loss proportionally
            loss_k_total = loss_k + (loss_wd + loss_anchor) / len(active_tasks)
            loss_k_total.backward(retain_graph=True)
            
            g_k = []
            for p in params:
                if p.grad is not None:
                    g_k.append(p.grad.clone())
                else:
                    g_k.append(torch.zeros_like(p))
            grads.append(g_k)
            
        # Flatten and project
        flat_grads = []
        for g in grads:
            flat_grads.append(torch.cat([tensor.flatten() for tensor in g]))
            
        projected_flat_grads = []
        for i in range(len(active_tasks)):
            g_i = flat_grads[i].clone()
            other_indices = list(range(len(active_tasks)))
            other_indices.remove(i)
            random.shuffle(other_indices)
            for j in other_indices:
                g_j = flat_grads[j]
                dot_prod = torch.dot(g_i, g_j)
                if dot_prod < 0:
                    g_i = g_i - (dot_prod / (torch.norm(g_j)**2 + 1e-8)) * g_j
            projected_flat_grads.append(g_i)
            
        summed_flat_grad = torch.stack(projected_flat_grads).sum(dim=0)
        
        # Set gradients back to parameters
        optimizer.zero_grad()
        idx = 0
        for p in params:
            numel = p.numel()
            g_slice = summed_flat_grad[idx : idx + numel].view_as(p)
            p.grad = g_slice
            idx += numel
            
        optimizer.step()

def evaluate_merged_model_homogeneous(test_splits, experts, router, P, B_batch=256):
    router.eval()
    correct_by_task = {k: 0 for k in range(K)}
    total_by_task = {k: 0 for k in range(K)}
    
    for k in range(K):
        X_test, y_test = test_splits[k]
        num_task_samples = X_test.shape[0]
        
        for start_idx in range(0, num_task_samples, B_batch):
            end_idx = min(start_idx + B_batch, num_task_samples)
            batch_z = X_test[start_idx:end_idx]
            batch_y = y_test[start_idx:end_idx]
            
            with torch.no_grad():
                psi = project_states(batch_z, P)
                alpha = router(psi)
                alpha_avg = alpha.mean(dim=1)  # (B_sub, K)
                bar_alpha = alpha_avg.mean(dim=0)  # (K,)
                
                # Merge weights for this batch
                W_merged = torch.zeros(C, D)
                b_merged = torch.zeros(C)
                for t in range(K):
                    W_merged += bar_alpha[t] * experts[t].weight
                    b_merged += bar_alpha[t] * experts[t].bias
                    
                logits = torch.matmul(batch_z, W_merged.t()) + b_merged
                preds = torch.argmax(logits, dim=1)
                correct_by_task[k] += (preds == batch_y).sum().item()
                total_by_task[k] += batch_z.shape[0]
                
    task_accs = [correct_by_task[k] / total_by_task[k] for k in range(K)]
    return task_accs, np.mean(task_accs)

def run_sweep():
    # Sweep M_tasks \in [1, 2, 3, 4]
    sweep_results = {}
    for M in [1, 2, 3, 4]:
        sweep_results[M] = []
        
    for seed in SEEDS:
        print(f"Running seed {seed}...")
        splits = generate_sandbox_data(seed, num_train=1000, num_test=250, num_cal=16) # B_cal = 64
        experts = train_experts(splits["train"], splits["test"])
        
        # Combined calibration z for PCA
        all_cal_z = []
        for k in range(K):
            all_cal_z.append(splits["cal"][k][0])
        all_cal_z = torch.cat(all_cal_z, dim=0)
        P = compute_pca_matrix(all_cal_z, d=D_PROJ)
        anchors = compute_task_anchors(splits["cal"], P)
        
        for M in [1, 2, 3, 4]:
            router = L3LinearRouter()
            train_router_stochastic_pcgrad(
                splits["cal"], experts, router, P, 
                lambda_wd=1e-3, lambda_anchor=0.1, anchors=anchors, 
                epochs=100, lr=1e-2, M_tasks=M
            )
            _, joint_mean = evaluate_merged_model_homogeneous(splits["test"], experts, router, P)
            sweep_results[M].append(joint_mean)
            
    print("\nSWEEP RESULTS FOR STOCHASTIC PCGRAD (HOMOGENEOUS EVAL):")
    for M in [1, 2, 3, 4]:
        means = np.mean(sweep_results[M]) * 100
        stds = np.std(sweep_results[M]) * 100
        print(f"M={M} active tasks: {means:.2f} \pm {stds:.2f}%")
        
    # Save to json file
    with open("results/stochastic_pcgrad_sweep.json", "w") as f:
        json.dump(sweep_results, f)

if __name__ == "__main__":
    run_sweep()
