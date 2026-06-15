import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import random

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

def generate_sandbox_data_with_drift(seed, num_train=1000, num_test=250, num_cal=16, drift_magnitude=0.3):
    set_seed(seed)
    sigmas = [0.01, 0.12, 0.18, 0.95]
    subspace_dim = D // K  # 48
    prototypes = torch.zeros(K, C, D)
    for k in range(K):
        for c in range(C):
            base_coords = torch.randn(subspace_dim)
            base_coords = base_coords / (torch.norm(base_coords) + 1e-8)
            prototypes[k, c, k * subspace_dim : (k + 1) * subspace_dim] = base_coords
            prototypes[k, c] = prototypes[k, c] / (torch.norm(prototypes[k, c]) + 1e-8)
            
    splits = {"train": [], "test": [], "cal": []}
    for k in range(K):
        sigma = sigmas[k]
        train_feats = []
        train_labels = []
        for _ in range(num_train):
            c = random.randint(0, C - 1)
            noise = torch.randn(D) * sigma
            x = prototypes[k, c] + noise
            train_feats.append(x)
            train_labels.append(c)
        splits["train"].append((torch.stack(train_feats), torch.tensor(train_labels)))
        
        test_feats = []
        test_labels = []
        for _ in range(num_test):
            c = random.randint(0, C - 1)
            noise = torch.randn(D) * sigma
            x = prototypes[k, c] + noise
            test_feats.append(x)
            test_labels.append(c)
        splits["test"].append((torch.stack(test_feats), torch.tensor(test_labels)))
        
        cal_feats = []
        cal_labels = []
        for _ in range(num_cal):
            c = random.randint(0, C - 1)
            noise = torch.randn(D) * sigma
            x = prototypes[k, c] + noise
            cal_feats.append(x)
            cal_labels.append(c)
        splits["cal"].append((torch.stack(cal_feats), torch.tensor(cal_labels)))
    return splits, prototypes, sigmas

def train_experts(train_splits):
    experts = []
    for k in range(K):
        X_train, y_train = train_splits[k]
        model = nn.Linear(D, C)
        optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(40):
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
        experts.append(model)
    return experts

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

def train_router(cal_splits, experts, router, P, lambda_wd=1e-3, lambda_anchor=0.1, anchors=None):
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
    
    optimizer = optim.AdamW(router.parameters(), lr=1e-2, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    router.train()
    for epoch in range(100):
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
        
        loss_wd = 0.0
        for p in router.parameters():
            loss_wd += lambda_wd * torch.sum(p**2)
        loss_anchor = 0.0
        if lambda_anchor > 0.0 and anchors is not None:
            loss_anchor += lambda_anchor * torch.sum((router.W - anchors.unsqueeze(0))**2)
            
        grads = []
        params = list(router.parameters())
        for k in range(K):
            optimizer.zero_grad()
            mask_k = (all_cal_tasks == k)
            loss_k = criterion(logits[mask_k], all_cal_y[mask_k])
            loss_k_total = loss_k + (loss_wd + loss_anchor) / K
            loss_k_total.backward(retain_graph=True)
            g_k = []
            for p in params:
                if p.grad is not None:
                    g_k.append(p.grad.clone())
                else:
                    g_k.append(torch.zeros_like(p))
            grads.append(g_k)
            
        flat_grads = []
        for g in grads:
            flat_grads.append(torch.cat([tensor.flatten() for tensor in g]))
            
        projected_flat_grads = []
        for i in range(K):
            g_i = flat_grads[i].clone()
            other_tasks = list(range(K))
            other_tasks.remove(i)
            random.shuffle(other_tasks)
            for j in other_tasks:
                g_j = flat_grads[j]
                dot_prod = torch.dot(g_i, g_j)
                if dot_prod < 0:
                    g_i = g_i - (dot_prod / (torch.norm(g_j)**2 + 1e-8)) * g_j
            projected_flat_grads.append(g_i)
            
        summed_flat_grad = torch.stack(projected_flat_grads).sum(dim=0)
        optimizer.zero_grad()
        idx = 0
        for p in params:
            numel = p.numel()
            p.grad = summed_flat_grad[idx : idx + numel].view_as(p).clone()
            idx += numel
        optimizer.step()

def run_streaming_drift_experiment():
    # We simulate a streaming non-stationary sequence of size T = 2000 steps
    # We introduce representational drift where the prototypes experience coordinate shift over time
    T = 1000
    betas = [0.01, 0.05, 0.1, 0.2]
    drift_scale = 0.4  # Maximum coordinate drift magnitude at t = T
    
    static_results = []
    ema_results = {beta: [] for beta in betas}
    
    for seed in SEEDS:
        splits, prototypes, sigmas = generate_sandbox_data_with_drift(seed)
        experts = train_experts(splits["train"])
        
        # Combined calibration splits
        all_cal_z = []
        for k in range(K):
            all_cal_z.append(splits["cal"][k][0])
        all_cal_z = torch.cat(all_cal_z, dim=0)
        P = compute_pca_matrix(all_cal_z, d=D_PROJ)
        anchors = compute_task_anchors(splits["cal"], P)
        
        # Train router once on clean, non-drifted calibration set
        router_static = L3LinearRouter()
        train_router(splits["cal"], experts, router_static, P, lambda_wd=1e-3, lambda_anchor=0.1, anchors=anchors)
        
        # Generate sequential, non-stationary test stream
        # Sudden label transitions:
        # t in [0, 250): MNIST (k=0)
        # t in [250, 500): FashionMNIST (k=1)
        # t in [500, 750): CIFAR-10 (k=2)
        # t in [750, 1000): SVHN (k=3)
        stream_correct_static = 0
        for t in range(T):
            k_t = t // 250
            sigma_t = sigmas[k_t]
            c_t = random.randint(0, C - 1)
            
            # Base prototype
            proto = prototypes[k_t, c_t].clone()
            # Apply systematic temporal coordinate drift: shifting features linearly over time
            drift_direction = torch.ones(D) / np.sqrt(D)
            drift_magnitude_factor = drift_scale * (t / T)
            proto_drifted = proto + drift_direction * drift_magnitude_factor
            
            noise = torch.randn(D) * sigma_t
            x_t = proto_drifted + noise
            y_t = torch.tensor(c_t)
            
            # Project to unit sphere
            with torch.no_grad():
                psi_t = project_states(x_t.unsqueeze(0), P)
                
                # --- Evaluate Static Router (No on-the-fly tracking) ---
                alpha_static = router_static(psi_t)
                bar_alpha_static = alpha_static.mean(dim=1).squeeze(0)  # (K,)
                
                W_merged_static = torch.zeros(C, D)
                b_merged_static = torch.zeros(C)
                for k in range(K):
                    W_merged_static += bar_alpha_static[k] * experts[k].weight
                    b_merged_static += bar_alpha_static[k] * experts[k].bias
                logits_static = torch.matmul(x_t, W_merged_static.t()) + b_merged_static
                pred_static = torch.argmax(logits_static)
                if pred_static == y_t:
                    stream_correct_static += 1
        acc_static = stream_correct_static / T
        static_results.append(acc_static)
                    
        # --- Evaluate EMA Tracking Router for each beta ---
        for beta in betas:
            router_ema = L3LinearRouter()
            with torch.no_grad():
                router_ema.W.copy_(router_static.W)
                router_ema.B.copy_(router_static.B)
            ema_anchors = anchors.clone()
            
            stream_correct_ema = 0
            # Reset random seed for reproducible sequence across betas
            set_seed(seed)
            for t in range(T):
                k_t = t // 250
                sigma_t = sigmas[k_t]
                c_t = random.randint(0, C - 1)
                
                proto = prototypes[k_t, c_t].clone()
                drift_direction = torch.ones(D) / np.sqrt(D)
                drift_magnitude_factor = drift_scale * (t / T)
                proto_drifted = proto + drift_direction * drift_magnitude_factor
                
                noise = torch.randn(D) * sigma_t
                x_t = proto_drifted + noise
                y_t = torch.tensor(c_t)
                
                with torch.no_grad():
                    psi_t = project_states(x_t.unsqueeze(0), P)
                    
                    # 1. Update active task's centroid anchor using EMA tracking
                    old_anchor_k = ema_anchors[k_t].clone()
                    ema_anchors[k_t] = (1 - beta) * ema_anchors[k_t] + beta * psi_t.squeeze(0)
                    
                    # 2. Closed-form weight adaptation: shift the active task's weights
                    delta_anchor = ema_anchors[k_t] - old_anchor_k
                    router_ema.W.data[:, k_t, :] += delta_anchor.unsqueeze(0)
                    
                    alpha_ema = router_ema(psi_t)
                    bar_alpha_ema = alpha_ema.mean(dim=1).squeeze(0)
                    
                    W_merged_ema = torch.zeros(C, D)
                    b_merged_ema = torch.zeros(C)
                    for k in range(K):
                        W_merged_ema += bar_alpha_ema[k] * experts[k].weight
                        b_merged_ema += bar_alpha_ema[k] * experts[k].bias
                    logits_ema = torch.matmul(x_t, W_merged_ema.t()) + b_merged_ema
                    pred_ema = torch.argmax(logits_ema)
                    if pred_ema == y_t:
                        stream_correct_ema += 1
                        
            acc_ema = stream_correct_ema / T
            ema_results[beta].append(acc_ema)
        
    print("\nSWEEP RESULTS FOR ONLINE EMA ANCHOR TRACKING UNDER STREAM DRIFT:")
    print(f"Static Router Accuracy: {np.mean(static_results)*100:.2f} \\pm {np.std(static_results)*100:.2f}%")
    for beta in betas:
        print(f"EMA-Tracking Router Accuracy (beta={beta}): {np.mean(ema_results[beta])*100:.2f} \\pm {np.std(ema_results[beta])*100:.2f}%")
    
    # Save results to JSON
    with open("results/ema_tracking_sweep.json", "w") as f:
        # Convert keys of dict to str for JSON serializability
        serialized_ema = {str(k): v for k, v in ema_results.items()}
        json.dump({"static": static_results, "ema": serialized_ema}, f)

if __name__ == "__main__":
    run_streaming_drift_experiment()
