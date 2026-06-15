import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import random

# Massive Multi-Task Sandbox Configuration
D = 200    # High-dimensional feature dimension
K = 20     # 20 independent tasks!
C = 10     # 10 classes per task
L = 14     # 14 layers in the router
D_PROJ = 20 # Projection dimension d
SEEDS = [10, 11] # Run over 2 independent seeds for efficiency and statistical validity

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_massive_sandbox_data(seed, num_train=500, num_test=100, num_cal=8):
    set_seed(seed)
    # Generate noisy task-specific expert distributions (some clean, some noisy, some extremely noisy)
    sigmas = []
    for k in range(K):
        if k % 4 == 0:
            sigmas.append(0.01) # Clean (like MNIST)
        elif k % 4 == 1:
            sigmas.append(0.12) # Low noise
        elif k % 4 == 2:
            sigmas.append(0.20) # Medium noise
        else:
            sigmas.append(0.80) # High noise (like SVHN)
            
    subspace_dim = D // K # 10 dimensions per task
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
        
    return splits, sigmas

def train_experts(train_splits):
    experts = []
    for k in range(K):
        X_train, y_train = train_splits[k]
        model = nn.Linear(D, C)
        optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(20): # Fast train for experts
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

def evaluate_merged_model(test_splits, experts, router, P, B_batch=128):
    router.eval()
    task_accs = []
    for k in range(K):
        X_test, y_test = test_splits[k]
        num_task_samples = X_test.shape[0]
        correct = 0
        total = 0
        for start_idx in range(0, num_task_samples, B_batch):
            end_idx = min(start_idx + B_batch, num_task_samples)
            batch_z = X_test[start_idx:end_idx]
            batch_y = y_test[start_idx:end_idx]
            
            with torch.no_grad():
                psi = project_states(batch_z, P)
                alpha = router(psi)
                alpha_avg = alpha.mean(dim=1)
                bar_alpha = alpha_avg.mean(dim=0)
                
                W_merged = torch.zeros(C, D)
                b_merged = torch.zeros(C)
                for t in range(K):
                    W_merged += bar_alpha[t] * experts[t].weight
                    b_merged += bar_alpha[t] * experts[t].bias
                logits = torch.matmul(batch_z, W_merged.t()) + b_merged
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_z.shape[0]
        task_accs.append(correct / total)
    return np.mean(task_accs)

# General Optimization Routine with support for different PCGrad schemes
def train_router_scalability(cal_splits, experts, router, P, scheme="none", anchors=None, epochs=50, lr=1e-2):
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
    
    # Task grouping configuration (group 20 tasks into G=4 groups of 5 tasks each)
    G = 4
    groups = {g: list(range(g*5, (g+1)*5)) for g in range(G)}
    
    router.train()
    start_time = time.time()
    
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
        
        loss_wd = 0.0
        for p in router.parameters():
            loss_wd += 1e-3 * torch.sum(p**2)
        loss_anchor = 0.0
        if anchors is not None:
            loss_anchor += 0.1 * torch.sum((router.W - anchors.unsqueeze(0))**2)
            
        params = list(router.parameters())
        
        if scheme == "none":
            # Simple joint SGD loss
            loss_ce = criterion(logits, all_cal_y)
            loss_total = loss_ce + loss_wd + loss_anchor
            loss_total.backward()
            optimizer.step()
            
        elif scheme == "full":
            # Standard PCGrad over all K=20 tasks (O(K) backprops)
            grads = []
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
                
            # Flatten and project standard O(K^2) pairwise
            flat_grads = [torch.cat([tensor.flatten() for tensor in g]) for g in grads]
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
            
        elif scheme == "stochastic":
            # Stochastic Task Sampling (M=2 active tasks per step, O(1) constant scaling)
            M = 2
            active_tasks = random.sample(list(range(K)), M)
            grads = []
            for k in active_tasks:
                optimizer.zero_grad()
                mask_k = (all_cal_tasks == k)
                loss_k = criterion(logits[mask_k], all_cal_y[mask_k])
                loss_k_total = loss_k + (loss_wd + loss_anchor) / M
                loss_k_total.backward(retain_graph=True)
                
                g_k = []
                for p in params:
                    if p.grad is not None:
                        g_k.append(p.grad.clone())
                    else:
                        g_k.append(torch.zeros_like(p))
                grads.append(g_k)
                
            flat_grads = [torch.cat([tensor.flatten() for tensor in g]) for g in grads]
            projected_flat_grads = []
            for i in range(M):
                g_i = flat_grads[i].clone()
                other_indices = list(range(M))
                other_indices.remove(i)
                random.shuffle(other_indices)
                for j in other_indices:
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
            
        elif scheme == "grouping":
            # Task Grouping (G=4 semantic groups, O(G) backprops)
            grads = []
            for g in range(G):
                optimizer.zero_grad()
                mask_g = torch.zeros(all_cal_tasks.shape[0], dtype=torch.bool)
                for k in groups[g]:
                    mask_g = mask_g | (all_cal_tasks == k)
                
                loss_g = criterion(logits[mask_g], all_cal_y[mask_g])
                loss_g_total = loss_g + (loss_wd + loss_anchor) / G
                loss_g_total.backward(retain_graph=True)
                
                g_g = []
                for p in params:
                    if p.grad is not None:
                        g_g.append(p.grad.clone())
                    else:
                        g_g.append(torch.zeros_like(p))
                grads.append(g_g)
                
            flat_grads = [torch.cat([tensor.flatten() for tensor in g]) for g in grads]
            projected_flat_grads = []
            for i in range(G):
                g_i = flat_grads[i].clone()
                other_groups = list(range(G))
                other_groups.remove(i)
                random.shuffle(other_groups)
                for j in other_groups:
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
            
    end_time = time.time()
    time_per_epoch_ms = ((end_time - start_time) / epochs) * 1000
    return time_per_epoch_ms

def run_scalability_audit():
    schemes = ["none", "full", "stochastic", "grouping"]
    results = {s: {"acc": [], "time": []} for s in schemes}
    
    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        splits, sigmas = generate_massive_sandbox_data(seed)
        experts = train_experts(splits["train"])
        
        # PCA matrix projection
        all_cal_z = []
        for k in range(K):
            all_cal_z.append(splits["cal"][k][0])
        all_cal_z = torch.cat(all_cal_z, dim=0)
        P = compute_pca_matrix(all_cal_z, d=D_PROJ)
        anchors = compute_task_anchors(splits["cal"], P)
        
        for scheme in schemes:
            router = L3LinearRouter(L=L, K=K, d=D_PROJ)
            print(f"Optimizing router with scheme: {scheme}...")
            time_per_epoch = train_router_scalability(
                splits["cal"], experts, router, P, 
                scheme=scheme, anchors=anchors, epochs=50
            )
            joint_acc = evaluate_merged_model(splits["test"], experts, router, P)
            print(f"[{scheme}] Accuracy: {joint_acc*100:.2f}%, Time/epoch: {time_per_epoch:.2f} ms")
            
            results[scheme]["acc"].append(joint_acc)
            results[scheme]["time"].append(time_per_epoch)
            
    print("\n" + "="*50)
    print("FINAL 20-TASK PCGRAD SCALABILITY SWEEP RESULTS:")
    print("="*50)
    scheme_names = {
        "none": "L3-Linear + TSAR (No PCGrad)",
        "full": "TSAR + Full PCGrad (O(K))",
        "stochastic": "TSAR + Stochastic PCGrad (O(1), M=2)",
        "grouping": "TSAR + Task Grouping PCGrad (O(G), G=4)"
    }
    for scheme in schemes:
        mean_acc = np.mean(results[scheme]["acc"]) * 100
        std_acc = np.std(results[scheme]["acc"]) * 100
        mean_time = np.mean(results[scheme]["time"])
        print(f"{scheme_names[scheme]:<45} | Accuracy: {mean_acc:.2f} \\pm {std_acc:.2f}% | Wall-Time/Epoch: {mean_time:.1f} ms")
        
    # Save results to JSON
    with open("results/pcgrad_scalability_sweep.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_scalability_audit()
