import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random

# Set up results directory
os.makedirs("results", exist_ok=True)

# Define constants
D = 192  # High-dimensional feature dimension
K = 4    # Number of tasks (MNIST, FashionMNIST, CIFAR-10, SVHN)
C = 10   # Classes per task
L = 14   # Layers in the router
D_PROJ = 4  # Projection dimension d

# Helper function to set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Synthetic representation sandbox data generator
def generate_sandbox_data(seed, num_train=1000, num_test=250, num_cal=16, leakage=0.0):
    set_seed(seed)
    
    # Noise levels to achieve realistic expert ceilings:
    # MNIST (99.8% - 100%), FashionMNIST (95% - 96.8%), CIFAR-10 (89% - 90.4%), SVHN (30% - 32%)
    sigmas = [0.01, 0.12, 0.18, 0.95]
    
    # Partition D=192 dimensions into K=4 disjoint subspaces of size 48 each
    subspace_dim = D // K  # 48
    
    # Generate task/class prototypes with potential leakage (overlapping task manifolds)
    prototypes = torch.zeros(K, C, D)
    for k in range(K):
        for c in range(C):
            # Base task coordinate
            base_coords = torch.randn(subspace_dim)
            base_coords = base_coords / (torch.norm(base_coords) + 1e-8)
            
            # Non-zero coordinates can leak into other task subspaces
            prototypes[k, c, k * subspace_dim : (k + 1) * subspace_dim] = (1.0 - leakage) * base_coords
            
            if leakage > 0.0:
                # Distribute leakage energy evenly across other task subspaces
                leak_energy_per_task = leakage / (K - 1)
                for other_k in range(K):
                    if other_k != k:
                        leak_coords = torch.randn(subspace_dim)
                        leak_coords = leak_coords / (torch.norm(leak_coords) + 1e-8)
                        prototypes[k, c, other_k * subspace_dim : (other_k + 1) * subspace_dim] = leak_energy_per_task * leak_coords
                        
            # Normalize the full prototype vector to unit norm to ensure constant prototype scale
            prototypes[k, c] = prototypes[k, c] / (torch.norm(prototypes[k, c]) + 1e-8)
            
    # Generate splits
    splits = {"train": [], "test": [], "cal": []}
    
    # Generate datasets for each task
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

# Train expert classifiers
def train_experts(train_splits, test_splits, epochs=40, lr=1e-2):
    experts = []
    expert_ceilings = []
    
    for k in range(K):
        X_train, y_train = train_splits[k]
        X_test, y_test = test_splits[k]
        
        # Define a linear expert classifier mapping R^192 -> R^10
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
            
    print(f"Expert ceilings trained: {expert_ceilings}")
    return experts, expert_ceilings

# PCA Projection setup
def compute_pca_matrix(X_cal, d=D_PROJ):
    # Centering
    mean = X_cal.mean(dim=0, keepdim=True)
    X_centered = X_cal - mean
    # SVD
    U, S, V = torch.linalg.svd(X_centered, full_matrices=False)
    # The top d right singular vectors
    P = V[:d, :].t()  # Shape: (D, d)
    return P

# Helper to project and normalize states to the unit sphere
def project_states(X, P):
    # Project
    X_proj = torch.matmul(X, P)  # Shape: (B, d)
    # Normalize onto the unit sphere
    norms = torch.norm(X_proj, dim=-1, keepdim=True)
    X_sphere = X_proj / (norms + 1e-8)
    return X_sphere

# Precompute Task Space Anchors
def compute_task_anchors(cal_splits, P):
    anchors = []
    for k in range(K):
        X_cal_k, _ = cal_splits[k]
        X_sphere_k = project_states(X_cal_k, P)
        anchor_k = X_sphere_k.mean(dim=0)
        anchors.append(anchor_k)
    return torch.stack(anchors)  # Shape: (K, d)

# Dynamic Routers definition
class L3LinearRouter(nn.Module):
    def __init__(self, L, K, d, activation="identity"):
        super().__init__()
        self.L = L
        self.K = K
        self.d = d
        self.activation = activation
        # Trainable weights and biases
        self.W = nn.Parameter(torch.randn(L, K, d) * 0.1)
        self.B = nn.Parameter(torch.zeros(L, K))
        
    def forward(self, psi):
        # psi: Shape (B, d)
        # Compute alpha[b, l, k] = dot(psi[b], W[l, k]) + B[l, k]
        alpha = torch.einsum('bd,lkd->blk', psi, self.W) + self.B.unsqueeze(0)  # (B, L, K)
        
        if self.activation == "softmax":
            alpha = torch.softmax(alpha, dim=-1)
        elif self.activation == "tanh":
            alpha = torch.tanh(alpha)
        elif self.activation == "relu":
            alpha = torch.relu(alpha)
        elif self.activation == "sigmoid":
            alpha = 1.5 * torch.sigmoid(alpha)
        # B=1 is sample-wise. In batch mode we average coefficients over batch.
        return alpha

class TrainingFreeCentroidRouter(nn.Module):
    def __init__(self, L, K, d, anchors):
        super().__init__()
        self.L = L
        self.K = K
        self.d = d
        self.register_buffer("centroids", anchors.clone())
        
    def forward(self, psi):
        B = psi.shape[0]
        alpha = torch.einsum('bd,kd->bk', psi, self.centroids)  # (B, K)
        return alpha.unsqueeze(1).repeat(1, self.L, 1)

class GlobalLinearRouter(nn.Module):
    def __init__(self, D, K):
        super().__init__()
        self.linear = nn.Linear(D, K)
        
    def forward(self, z):
        # z: Shape (B, D)
        # Directly map to task scores
        scores = self.linear(z)  # (B, K)
        # Repeat across L layers to match architecture
        alpha = scores.unsqueeze(1).repeat(1, L, 1)  # (B, L, K)
        return alpha

class QWSRouter(nn.Module):
    def __init__(self, L, K, d):
        super().__init__()
        self.L = L
        self.K = K
        self.d = d
        # Initialization matches standard formulation:
        # basis states Phi initialized around identity matrix with minor perturbation
        # R initialized to 0.3, phi_bias initialized to -pi
        Phi_init = torch.eye(K, d).unsqueeze(0).repeat(L, 1, 1) + torch.randn(L, K, d) * 0.1
        self.Phi = nn.Parameter(Phi_init)
        self.R = nn.Parameter(torch.ones(L, K) * 0.3)
        self.phi_bias = nn.Parameter(torch.ones(L, K) * (-np.pi))
        
    def forward(self, psi):
        # Normalize Phi onto unit sphere
        hat_Phi = self.Phi / (torch.norm(self.Phi, dim=-1, keepdim=True) + 1e-8)
        # Cosine phase-overlap
        cos_term = torch.einsum('bd,lkd->blk', psi, hat_Phi)
        alpha = self.R.unsqueeze(0) * torch.cos(np.pi * cos_term + self.phi_bias.unsqueeze(0))
        return alpha

class AdaMergingRouter(nn.Module):
    def __init__(self, L, K):
        super().__init__()
        self.L = L
        self.K = K
        # Initialize to uniform 1/K
        self.theta = nn.Parameter(torch.ones(L, K) / K)
        
    def forward(self, psi):
        # Ignores input, just returns theta replicated across batch dimension
        B = psi.shape[0]
        # Return shape (B, L, K)
        return self.theta.unsqueeze(0).repeat(B, 1, 1)

# Evaluation functions
def evaluate_merged_model(test_splits, experts, router, P, stream_type="homogeneous", B_batch=256, top1_gating=False, cluster_partitioning=False, anchors=None):
    router.eval()
    
    # Collect all test samples
    all_z = []
    all_y = []
    all_tasks = []
    for k in range(K):
        X_test, y_test = test_splits[k]
        all_z.append(X_test)
        all_y.append(y_test)
        all_tasks.append(torch.ones(X_test.shape[0], dtype=torch.long) * k)
        
    all_z = torch.cat(all_z, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_tasks = torch.cat(all_tasks, dim=0)
    num_samples = all_z.shape[0]
    
    correct_by_task = {k: 0 for k in range(K)}
    total_by_task = {k: 0 for k in range(K)}
    
    # Process according to stream type
    if stream_type == "sample_wise":
        # B=1 (Sample-wise)
        # Compute all coefficients
        with torch.no_grad():
            if isinstance(router, GlobalLinearRouter):
                alpha = router(all_z)  # (B, L, K)
            else:
                psi = project_states(all_z, P)
                alpha = router(psi)  # (B, L, K)
                
            # Average over layers: (B, K)
            alpha_avg = alpha.mean(dim=1)
            
            if top1_gating:
                max_idx = torch.argmax(alpha_avg, dim=-1)
                gated = torch.zeros_like(alpha_avg)
                gated[torch.arange(alpha_avg.shape[0]), max_idx] = alpha_avg[torch.arange(alpha_avg.shape[0]), max_idx]
                alpha_avg = gated
            
            # Sample-wise forward
            for b in range(num_samples):
                z_b = all_z[b]
                y_b = all_y[b]
                task_k = all_tasks[b].item()
                bar_alpha = alpha_avg[b]  # (K,)
                
                # Dynamic weight assembly
                W_merged = torch.zeros(C, D)
                b_merged = torch.zeros(C)
                for k_idx in range(K):
                    W_merged += bar_alpha[k_idx] * experts[k_idx].weight
                    b_merged += bar_alpha[k_idx] * experts[k_idx].bias
                    
                logits = torch.matmul(W_merged, z_b) + b_merged
                pred = torch.argmax(logits).item()
                if pred == y_b.item():
                    correct_by_task[task_k] += 1
                total_by_task[task_k] += 1
                
    elif stream_type == "homogeneous":
        # Task-wise deployment (homogeneous batches)
        # Group by task
        for k in range(K):
            X_test, y_test = test_splits[k]
            num_task_samples = X_test.shape[0]
            
            # Batch loop
            for start_idx in range(0, num_task_samples, B_batch):
                end_idx = min(start_idx + B_batch, num_task_samples)
                batch_z = X_test[start_idx:end_idx]
                batch_y = y_test[start_idx:end_idx]
                
                with torch.no_grad():
                    if isinstance(router, GlobalLinearRouter):
                        alpha = router(batch_z)  # (B_sub, L, K)
                    else:
                        psi = project_states(batch_z, P)
                        alpha = router(psi)
                        
                    # Average over layers and then average over batch
                    alpha_avg = alpha.mean(dim=1)  # (B_sub, K)
                    
                    if top1_gating:
                        max_idx = torch.argmax(alpha_avg, dim=-1)
                        gated = torch.zeros_like(alpha_avg)
                        gated[torch.arange(alpha_avg.shape[0]), max_idx] = alpha_avg[torch.arange(alpha_avg.shape[0]), max_idx]
                        alpha_avg = gated
                        
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
                    
    elif stream_type == "heterogeneous":
        # Mixed-task deployment (heterogeneous batches)
        # Shuffle indices
        shuffled_indices = list(range(num_samples))
        # Keep consistent shuffling per evaluation but let it be randomized
        # Set small local seed for reproducible test stream shuffling
        rand_gen = random.Random(42)
        rand_gen.shuffle(shuffled_indices)
        
        # Batch loop
        for start_idx in range(0, num_samples, B_batch):
            end_idx = min(start_idx + B_batch, num_samples)
            batch_idxs = shuffled_indices[start_idx:end_idx]
            batch_z = all_z[batch_idxs]
            batch_y = all_y[batch_idxs]
            batch_tasks = all_tasks[batch_idxs]
            
            with torch.no_grad():
                if cluster_partitioning:
                    psi = project_states(batch_z, P)
                    if anchors is not None:
                        # Centroid-Guided Online Batch Partitioning: use nearest anchors
                        # Compute L2 distances from each sample's psi to each task's anchor
                        dists = torch.cdist(psi, anchors)  # Shape (B_batch, K)
                        cluster_labels = torch.argmin(dists, dim=1).cpu().numpy()
                    else:
                        # Fallback to online unsupervised K-Means
                        from sklearn.cluster import KMeans
                        psi_np = psi.cpu().numpy()
                        kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
                        cluster_labels = kmeans.fit_predict(psi_np)
                    
                    for c in range(K):
                        cluster_idxs = np.where(cluster_labels == c)[0]
                        if len(cluster_idxs) == 0:
                            continue
                        cluster_z = batch_z[cluster_idxs]
                        cluster_y = batch_y[cluster_idxs]
                        cluster_tasks = batch_tasks[cluster_idxs]
                        cluster_psi = psi[cluster_idxs]
                        
                        cluster_alpha = router(cluster_psi)
                        cluster_alpha_avg = cluster_alpha.mean(dim=1)
                        
                        if top1_gating:
                            max_idx = torch.argmax(cluster_alpha_avg, dim=-1)
                            gated = torch.zeros_like(cluster_alpha_avg)
                            gated[torch.arange(cluster_alpha_avg.shape[0]), max_idx] = cluster_alpha_avg[torch.arange(cluster_alpha_avg.shape[0]), max_idx]
                            cluster_alpha_avg = gated
                            
                        cluster_bar_alpha = cluster_alpha_avg.mean(dim=0)
                        
                        # Merge weights
                        W_merged = torch.zeros(C, D)
                        b_merged = torch.zeros(C)
                        for t in range(K):
                            W_merged += cluster_bar_alpha[t] * experts[t].weight
                            b_merged += cluster_bar_alpha[t] * experts[t].bias
                            
                        logits = torch.matmul(cluster_z, W_merged.t()) + b_merged
                        preds = torch.argmax(logits, dim=1)
                        
                        for b_i, task_k in enumerate(cluster_tasks):
                            if preds[b_i] == cluster_y[b_i]:
                                correct_by_task[task_k.item()] += 1
                            total_by_task[task_k.item()] += 1
                else:
                    if isinstance(router, GlobalLinearRouter):
                        alpha = router(batch_z)  # (B_sub, L, K)
                    else:
                        psi = project_states(batch_z, P)
                        alpha = router(psi)
                        
                    # Average over layers and then average over batch
                    alpha_avg = alpha.mean(dim=1)  # (B_sub, K)
                    
                    if top1_gating:
                        max_idx = torch.argmax(alpha_avg, dim=-1)
                        gated = torch.zeros_like(alpha_avg)
                        gated[torch.arange(alpha_avg.shape[0]), max_idx] = alpha_avg[torch.arange(alpha_avg.shape[0]), max_idx]
                        alpha_avg = gated
                        
                    bar_alpha = alpha_avg.mean(dim=0)  # (K,)
                    
                    # Merge weights
                    W_merged = torch.zeros(C, D)
                    b_merged = torch.zeros(C)
                    for t in range(K):
                        W_merged += bar_alpha[t] * experts[t].weight
                        b_merged += bar_alpha[t] * experts[t].bias
                        
                    logits = torch.matmul(batch_z, W_merged.t()) + b_merged
                    preds = torch.argmax(logits, dim=1)
                    
                    for b_i, task_k in enumerate(batch_tasks):
                        if preds[b_i] == batch_y[b_i]:
                            correct_by_task[task_k.item()] += 1
                        total_by_task[task_k.item()] += 1
                    
    # Compute task-specific accuracies and mean
    accuracies = {}
    for k in range(K):
        accuracies[k] = correct_by_task[k] / (total_by_task[k] + 1e-8)
    mean_acc = np.mean(list(accuracies.values()))
    
    return accuracies, mean_acc

# Evaluate Static Uniform Merging (Task Arithmetic)
def evaluate_uniform_merging(test_splits, experts):
    correct_by_task = {k: 0 for k in range(K)}
    total_by_task = {k: 0 for k in range(K)}
    
    for k in range(K):
        X_test, y_test = test_splits[k]
        
        # Merge weights uniformly (alpha = 1/K for all tasks)
        W_merged = torch.zeros(C, D)
        b_merged = torch.zeros(C)
        for t in range(K):
            W_merged += (1.0 / K) * experts[t].weight
            b_merged += (1.0 / K) * experts[t].bias
            
        with torch.no_grad():
            logits = torch.matmul(X_test, W_merged.t()) + b_merged
            preds = torch.argmax(logits, dim=1)
            correct_by_task[k] = (preds == y_test).sum().item()
            total_by_task[k] = X_test.shape[0]
            
    accuracies = {}
    for k in range(K):
        accuracies[k] = correct_by_task[k] / (total_by_task[k] + 1e-8)
    mean_acc = np.mean(list(accuracies.values()))
    return accuracies, mean_acc

# Calibration (Router optimization with optional task gradient masking / PCGrad)
def train_router(cal_splits, experts, router, P, lambda_wd=1e-3, lambda_anchor=0.0, anchors=None, epochs=100, lr=1e-2, clip_grad=False, grad_masking=False, pcgrad=False):
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
    
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0)  # We handle WD manually
    criterion = nn.CrossEntropyLoss()
    
    router.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass through router
        if isinstance(router, GlobalLinearRouter):
            alpha = router(all_cal_z)  # (B, L, K)
        else:
            psi = project_states(all_cal_z, P)
            alpha = router(psi)
            
        # Apply task-specific gradient masking to prevent gradient cross-talk
        if grad_masking:
            mask = torch.zeros_like(alpha)
            for b in range(alpha.shape[0]):
                mask[b, :, all_cal_tasks[b]] = 1.0
            # Retain gradient flow only for the true task's routing parameters
            alpha = alpha * mask + alpha.detach() * (1.0 - mask)
            
        # Average over layers and average over calibration batch
        alpha_avg = alpha.mean(dim=1)  # (B_cal, K)
        bar_alpha = alpha_avg.mean(dim=0)  # (K,)
        
        # Dynamically assemble merged weights and biases
        W_merged = torch.zeros(C, D)
        b_merged = torch.zeros(C)
        for k in range(K):
            W_merged += bar_alpha[k] * experts[k].weight
            b_merged += bar_alpha[k] * experts[k].bias
            
        logits = torch.matmul(all_cal_z, W_merged.t()) + b_merged
        
        # Compute standard cross-entropy loss
        loss_ce = criterion(logits, all_cal_y)
        
        # Manual L2 weight decay on router parameters
        loss_wd = 0.0
        for p in router.parameters():
            loss_wd += lambda_wd * torch.sum(p**2)
            
        # TSAR anchor regularization penalty
        loss_anchor = 0.0
        if lambda_anchor > 0.0 and anchors is not None:
            # Anchoring the layer weights W to precomputed task anchors
            # Compute squared L2 norm: sum_{l, k} || W_{l,k} - \bar{\psi}_k ||_2^2
            loss_anchor += lambda_anchor * torch.sum((router.W - anchors.unsqueeze(0))**2)
            
        if pcgrad:
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
                
            # Flatten each task's gradients for easier projection
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
            
            # Assign gradients back to parameters
            idx = 0
            optimizer.zero_grad()
            for p in params:
                numel = p.numel()
                p.grad = summed_flat_grad[idx : idx + numel].view_as(p).clone()
                idx += numel
        else:
            total_loss = loss_ce + loss_wd + loss_anchor
            total_loss.backward()
        
        if clip_grad:
            nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
            
        optimizer.step()

# Main Experiment Runner across Seeds
def run_all_experiments():
    seeds = [10, 11, 12, 13, 14]
    
    # Storage for results
    results = {
        "expert_ceiling": [],
        "uniform_merging": [],
        "adamerging": [],
        "linear_router": [],
        "qws_merge": [],
        "l1_linear_l2": [],
        "l1_linear_tsar": [],
        "l3_linear_unreg": [],
        "l3_linear_l2": [],
        "l3_softmax_l2": [],
        "l3_softmax_tsar": [],
        "tf_centroid_router": [],
        "tsar_sweep": {},  # lambda_anchor -> results
        "cal_size_sweep": {},  # B_cal -> results
        "cal_size_sweep_masked": {},  # B_cal -> results with gradient masking
        "cal_size_sweep_pcgrad": {},  # B_cal -> results with PCGrad
        "leakage_sweep": {},  # leakage -> results for various methods
    }
    
    for seed in seeds:
        print(f"\n================ Running Seed {seed} ================")
        splits = generate_sandbox_data(seed)
        experts, ceilings = train_experts(splits["train"], splits["test"])
        results["expert_ceiling"].append(ceilings)
        
        # Uniform model merging
        uni_accs, uni_mean = evaluate_uniform_merging(splits["test"], experts)
        results["uniform_merging"].append([uni_accs[0], uni_accs[1], uni_accs[2], uni_accs[3], uni_mean])
        
        # Compute projection matrix P
        all_cal_z = torch.cat([splits["cal"][k][0] for k in range(K)], dim=0)
        P = compute_pca_matrix(all_cal_z, d=D_PROJ)
        
        # Task Anchors
        anchors = compute_task_anchors(splits["cal"], P)
        
        # 1. AdaMerging (Static blending coefficients optimized on calibration set)
        ada_router = AdaMergingRouter(L, K)
        train_router(splits["cal"], experts, ada_router, P, lambda_wd=0.0, epochs=100, lr=1e-2)
        ada_accs, ada_mean = evaluate_merged_model(splits["test"], experts, ada_router, P, "homogeneous")
        results["adamerging"].append([ada_accs[0], ada_accs[1], ada_accs[2], ada_accs[3], ada_mean])
        
        # 2. Global Classical Linear Router
        g_router = GlobalLinearRouter(D, K)
        train_router(splits["cal"], experts, g_router, P, lambda_wd=1e-3, epochs=100, lr=1e-2)
        g_accs, g_mean = evaluate_merged_model(splits["test"], experts, g_router, P, "homogeneous")
        results["linear_router"].append([g_accs[0], g_accs[1], g_accs[2], g_accs[3], g_mean])
        
        # 3. QWS-Merge SOTA
        q_router = QWSRouter(L, K, D_PROJ)
        train_router(splits["cal"], experts, q_router, P, lambda_wd=1e-3, epochs=100, lr=1e-2, clip_grad=True)
        q_accs, q_mean = evaluate_merged_model(splits["test"], experts, q_router, P, "homogeneous")
        results["qws_merge"].append([q_accs[0], q_accs[1], q_accs[2], q_accs[3], q_mean])
        
        # 4. L1-Linear Router (L2 Reg)
        l1_lin_l2 = L3LinearRouter(1, K, D_PROJ, activation="identity")
        train_router(splits["cal"], experts, l1_lin_l2, P, lambda_wd=1e-3, epochs=100, lr=1e-2)
        l1_lin_l2_accs, l1_lin_l2_mean = evaluate_merged_model(splits["test"], experts, l1_lin_l2, P, "homogeneous")
        results["l1_linear_l2"].append([l1_lin_l2_accs[0], l1_lin_l2_accs[1], l1_lin_l2_accs[2], l1_lin_l2_accs[3], l1_lin_l2_mean])
        
        # 5. L1-Linear Router + TSAR
        l1_lin_tsar = L3LinearRouter(1, K, D_PROJ, activation="identity")
        train_router(splits["cal"], experts, l1_lin_tsar, P, lambda_wd=1e-3, lambda_anchor=1e-1, anchors=anchors, epochs=100, lr=1e-2)
        l1_lin_tsar_accs, l1_lin_tsar_mean = evaluate_merged_model(splits["test"], experts, l1_lin_tsar, P, "homogeneous")
        results["l1_linear_tsar"].append([l1_lin_tsar_accs[0], l1_lin_tsar_accs[1], l1_lin_tsar_accs[2], l1_lin_tsar_accs[3], l1_lin_tsar_mean])
        
        # 6. L3-Linear Router (Unregularized)
        l3_lin_unreg = L3LinearRouter(L, K, D_PROJ, activation="identity")
        train_router(splits["cal"], experts, l3_lin_unreg, P, lambda_wd=0.0, epochs=100, lr=1e-2)
        l3_lin_unreg_accs, l3_lin_unreg_mean = evaluate_merged_model(splits["test"], experts, l3_lin_unreg, P, "homogeneous")
        results["l3_linear_unreg"].append([l3_lin_unreg_accs[0], l3_lin_unreg_accs[1], l3_lin_unreg_accs[2], l3_lin_unreg_accs[3], l3_lin_unreg_mean])
        
        # 7. L3-Linear Router (L2 Reg)
        l3_lin_l2 = L3LinearRouter(L, K, D_PROJ, activation="identity")
        train_router(splits["cal"], experts, l3_lin_l2, P, lambda_wd=1e-3, epochs=100, lr=1e-2)
        l3_lin_l2_accs, l3_lin_l2_mean = evaluate_merged_model(splits["test"], experts, l3_lin_l2, P, "homogeneous")
        results["l3_linear_l2"].append([l3_lin_l2_accs[0], l3_lin_l2_accs[1], l3_lin_l2_accs[2], l3_lin_l2_accs[3], l3_lin_l2_mean])
        
        # 8. L3-Softmax (L2 Reg)
        l3_soft_l2 = L3LinearRouter(L, K, D_PROJ, activation="softmax")
        train_router(splits["cal"], experts, l3_soft_l2, P, lambda_wd=1e-3, epochs=100, lr=1e-2)
        l3_soft_l2_accs, l3_soft_l2_mean = evaluate_merged_model(splits["test"], experts, l3_soft_l2, P, "homogeneous")
        results["l3_softmax_l2"].append([l3_soft_l2_accs[0], l3_soft_l2_accs[1], l3_soft_l2_accs[2], l3_soft_l2_accs[3], l3_soft_l2_mean])
        
        # 9. L3-Softmax + TSAR
        l3_soft_tsar = L3LinearRouter(L, K, D_PROJ, activation="softmax")
        train_router(splits["cal"], experts, l3_soft_tsar, P, lambda_wd=1e-3, lambda_anchor=1e-1, anchors=anchors, epochs=100, lr=1e-2)
        l3_soft_tsar_accs, l3_soft_tsar_mean = evaluate_merged_model(splits["test"], experts, l3_soft_tsar, P, "homogeneous")
        results["l3_softmax_tsar"].append([l3_soft_tsar_accs[0], l3_soft_tsar_accs[1], l3_soft_tsar_accs[2], l3_soft_tsar_accs[3], l3_soft_tsar_mean])
        
        # 9b. Training-Free Centroid Router
        tf_router = TrainingFreeCentroidRouter(L, K, D_PROJ, anchors)
        tf_accs, tf_mean = evaluate_merged_model(splits["test"], experts, tf_router, P, "homogeneous")
        results["tf_centroid_router"].append([tf_accs[0], tf_accs[1], tf_accs[2], tf_accs[3], tf_mean])
        
        # 10. TSAR Sweep over lambda_anchor
        lambda_anchors = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        for lam in lambda_anchors:
            if lam not in results["tsar_sweep"]:
                results["tsar_sweep"][lam] = []
            tsar_router = L3LinearRouter(L, K, D_PROJ, activation="identity")
            train_router(splits["cal"], experts, tsar_router, P, lambda_wd=1e-3, lambda_anchor=lam, anchors=anchors, epochs=100, lr=1e-2)
            tsar_accs, tsar_mean = evaluate_merged_model(splits["test"], experts, tsar_router, P, "homogeneous")
            results["tsar_sweep"][lam].append([tsar_accs[0], tsar_accs[1], tsar_accs[2], tsar_accs[3], tsar_mean])
            
        # 11. Calibration set size sweep for TSAR (using optimal lam=1e-1)
        cal_sizes = [16, 32, 64, 128]
        for sz in cal_sizes:
            if sz not in results["cal_size_sweep"]:
                results["cal_size_sweep"][sz] = []
            if sz not in results["cal_size_sweep_masked"]:
                results["cal_size_sweep_masked"][sz] = []
            if sz not in results["cal_size_sweep_pcgrad"]:
                results["cal_size_sweep_pcgrad"][sz] = []
                
            num_samples_per_task = sz // K
            splits_sz = generate_sandbox_data(seed, num_cal=num_samples_per_task)
            all_cal_z_sz = torch.cat([splits_sz["cal"][k][0] for k in range(K)], dim=0)
            P_sz = compute_pca_matrix(all_cal_z_sz, d=D_PROJ)
            anchors_sz = compute_task_anchors(splits_sz["cal"], P_sz)
            
            # Standard complexity sweep
            tsar_sz = L3LinearRouter(L, K, D_PROJ, activation="identity")
            train_router(splits_sz["cal"], experts, tsar_sz, P_sz, lambda_wd=1e-3, lambda_anchor=1e-1, anchors=anchors_sz, epochs=100, lr=1e-2)
            tsar_sz_accs, tsar_sz_mean = evaluate_merged_model(splits_sz["test"], experts, tsar_sz, P_sz, "homogeneous")
            results["cal_size_sweep"][sz].append([tsar_sz_accs[0], tsar_sz_accs[1], tsar_sz_accs[2], tsar_sz_accs[3], tsar_sz_mean])
            
            # Gradient-masked complexity sweep
            tsar_sz_masked = L3LinearRouter(L, K, D_PROJ, activation="identity")
            train_router(splits_sz["cal"], experts, tsar_sz_masked, P_sz, lambda_wd=1e-3, lambda_anchor=1e-1, anchors=anchors_sz, epochs=100, lr=1e-2, grad_masking=True)
            tsar_sz_masked_accs, tsar_sz_masked_mean = evaluate_merged_model(splits_sz["test"], experts, tsar_sz_masked, P_sz, "homogeneous")
            results["cal_size_sweep_masked"][sz].append([tsar_sz_masked_accs[0], tsar_sz_masked_accs[1], tsar_sz_masked_accs[2], tsar_sz_masked_accs[3], tsar_sz_masked_mean])

            # PCGrad complexity sweep
            tsar_sz_pcgrad = L3LinearRouter(L, K, D_PROJ, activation="identity")
            train_router(splits_sz["cal"], experts, tsar_sz_pcgrad, P_sz, lambda_wd=1e-3, lambda_anchor=1e-1, anchors=anchors_sz, epochs=100, lr=1e-2, pcgrad=True)
            tsar_sz_pcgrad_accs, tsar_sz_pcgrad_mean = evaluate_merged_model(splits_sz["test"], experts, tsar_sz_pcgrad, P_sz, "homogeneous")
            results["cal_size_sweep_pcgrad"][sz].append([tsar_sz_pcgrad_accs[0], tsar_sz_pcgrad_accs[1], tsar_sz_pcgrad_accs[2], tsar_sz_pcgrad_accs[3], tsar_sz_pcgrad_mean])

        # 12. Subspace Leakage Sweep (Overlapping task manifolds)
        if seed in seeds[:3]:
            leakages = [0.0, 0.1, 0.2, 0.3, 0.4]
            for leak in leakages:
                if leak not in results["leakage_sweep"]:
                    results["leakage_sweep"][leak] = {
                        "uniform_merging": [],
                        "l3_linear_l2": [],
                        "qws_merge": [],
                        "tsar_pcgrad": []
                    }
                
                # Generate overlapping data
                splits_leak = generate_sandbox_data(seed, leakage=leak)
                experts_leak, _ = train_experts(splits_leak["train"], splits_leak["test"])
                
                all_cal_z_leak = torch.cat([splits_leak["cal"][k][0] for k in range(K)], dim=0)
                P_leak = compute_pca_matrix(all_cal_z_leak, d=D_PROJ)
                anchors_leak = compute_task_anchors(splits_leak["cal"], P_leak)
                
                # (a) Uniform Merging
                uni_accs, uni_mean = evaluate_uniform_merging(splits_leak["test"], experts_leak)
                results["leakage_sweep"][leak]["uniform_merging"].append([uni_accs[0], uni_accs[1], uni_accs[2], uni_accs[3], uni_mean])
                
                # (b) L3-Linear (L2 Reg)
                l3_lin_leak = L3LinearRouter(L, K, D_PROJ, activation="identity")
                train_router(splits_leak["cal"], experts_leak, l3_lin_leak, P_leak, lambda_wd=1e-3, epochs=100, lr=1e-2)
                l3_accs, l3_mean = evaluate_merged_model(splits_leak["test"], experts_leak, l3_lin_leak, P_leak, "homogeneous")
                results["leakage_sweep"][leak]["l3_linear_l2"].append([l3_accs[0], l3_accs[1], l3_accs[2], l3_accs[3], l3_mean])
                
                # (c) QWS-Merge
                qws_leak = QWSRouter(L, K, D_PROJ)
                train_router(splits_leak["cal"], experts_leak, qws_leak, P_leak, lambda_wd=1e-3, epochs=100, lr=1e-2, clip_grad=True)
                qws_accs, qws_mean = evaluate_merged_model(splits_leak["test"], experts_leak, qws_leak, P_leak, "homogeneous")
                results["leakage_sweep"][leak]["qws_merge"].append([qws_accs[0], qws_accs[1], qws_accs[2], qws_accs[3], qws_mean])
                
                # (d) TSAR + PCGrad
                tsar_leak = L3LinearRouter(L, K, D_PROJ, activation="identity")
                train_router(splits_leak["cal"], experts_leak, tsar_leak, P_leak, lambda_wd=1e-3, lambda_anchor=1e-1, anchors=anchors_leak, epochs=100, lr=1e-2, pcgrad=True)
                tsar_accs, tsar_mean = evaluate_merged_model(splits_leak["test"], experts_leak, tsar_leak, P_leak, "homogeneous")
                results["leakage_sweep"][leak]["tsar_pcgrad"].append([tsar_accs[0], tsar_accs[1], tsar_accs[2], tsar_accs[3], tsar_mean])

    # Now let's run deployment stream audit for the optimal seed and seed averages
    # We will evaluate under: Homogeneous (B=1), Homogeneous (B=256), Heterogeneous (B=256)
    stream_results = {
        "Linear Router": {"homog_1": [], "homog_256": [], "hetero_256": []},
        "QWS-Merge SOTA": {"homog_1": [], "homog_256": [], "hetero_256": []},
        "L3-Linear (L2 Reg)": {"homog_1": [], "homog_256": [], "hetero_256": []},
        "L3-Softmax (L2 Reg)": {"homog_1": [], "homog_256": [], "hetero_256": []},
        "TSAR (Ours)": {"homog_1": [], "homog_256": [], "hetero_256": []},
        "Softmax-bounded TSAR (Ours)": {"homog_1": [], "homog_256": [], "hetero_256": []},
        "TSAR + ReLU Activation (Ours)": {"homog_1": [], "homog_256": [], "hetero_256": []},
        "TSAR + Sigmoid Activation (Ours)": {"homog_1": [], "homog_256": [], "hetero_256": []},
        "TSAR + Top-1 Gating (Ours)": {"homog_1": [], "homog_256": [], "hetero_256": []},
        "TSAR + Batch Partitioning (Ours)": {"homog_1": [], "homog_256": [], "hetero_256": []}
    }
    
    for seed in seeds:
        splits = generate_sandbox_data(seed)
        experts, _ = train_experts(splits["train"], splits["test"])
        all_cal_z = torch.cat([splits["cal"][k][0] for k in range(K)], dim=0)
        P = compute_pca_matrix(all_cal_z, d=D_PROJ)
        anchors = compute_task_anchors(splits["cal"], P)
        
        # Re-train models
        g_router = GlobalLinearRouter(D, K)
        train_router(splits["cal"], experts, g_router, P, lambda_wd=1e-3)
        
        q_router = QWSRouter(L, K, D_PROJ)
        train_router(splits["cal"], experts, q_router, P, lambda_wd=1e-3, clip_grad=True)
        
        l3_lin_l2 = L3LinearRouter(L, K, D_PROJ, activation="identity")
        train_router(splits["cal"], experts, l3_lin_l2, P, lambda_wd=1e-3)
        
        l3_soft_l2 = L3LinearRouter(L, K, D_PROJ, activation="softmax")
        train_router(splits["cal"], experts, l3_soft_l2, P, lambda_wd=1e-3)
        
        tsar_router = L3LinearRouter(L, K, D_PROJ, activation="identity")
        train_router(splits["cal"], experts, tsar_router, P, lambda_wd=1e-3, lambda_anchor=1e-1, anchors=anchors)
        
        l3_soft_tsar = L3LinearRouter(L, K, D_PROJ, activation="softmax")
        train_router(splits["cal"], experts, l3_soft_tsar, P, lambda_wd=1e-3, lambda_anchor=1e-1, anchors=anchors)
        
        l3_relu_tsar = L3LinearRouter(L, K, D_PROJ, activation="relu")
        train_router(splits["cal"], experts, l3_relu_tsar, P, lambda_wd=1e-3, lambda_anchor=1e-1, anchors=anchors)
        
        l3_sigmoid_tsar = L3LinearRouter(L, K, D_PROJ, activation="sigmoid")
        train_router(splits["cal"], experts, l3_sigmoid_tsar, P, lambda_wd=1e-3, lambda_anchor=1e-1, anchors=anchors)
        
        # Stream evaluations
        for name, model, t1, cp in [
            ("Linear Router", g_router, False, False),
            ("QWS-Merge SOTA", q_router, False, False),
            ("L3-Linear (L2 Reg)", l3_lin_l2, False, False),
            ("L3-Softmax (L2 Reg)", l3_soft_l2, False, False),
            ("TSAR (Ours)", tsar_router, False, False),
            ("Softmax-bounded TSAR (Ours)", l3_soft_tsar, False, False),
            ("TSAR + ReLU Activation (Ours)", l3_relu_tsar, False, False),
            ("TSAR + Sigmoid Activation (Ours)", l3_sigmoid_tsar, False, False),
            ("TSAR + Top-1 Gating (Ours)", tsar_router, True, False),
            ("TSAR + Batch Partitioning (Ours)", tsar_router, False, True)
        ]:
            _, homog_1_m = evaluate_merged_model(splits["test"], experts, model, P, "sample_wise", 256, t1, cp, anchors=anchors)
            _, homog_256_m = evaluate_merged_model(splits["test"], experts, model, P, "homogeneous", 256, t1, cp, anchors=anchors)
            _, hetero_256_m = evaluate_merged_model(splits["test"], experts, model, P, "heterogeneous", 256, t1, cp, anchors=anchors)
            
            stream_results[name]["homog_1"].append(homog_1_m)
            stream_results[name]["homog_256"].append(homog_256_m)
            stream_results[name]["hetero_256"].append(hetero_256_m)

    # Let's write output stats
    print("\n\n################ EXPERIMENTAL RESULTS ################")
    
    # Process seed stats helper
    def get_stats(arr):
        arr = np.array(arr)
        # If it's a 2D array of shapes (seeds, metrics)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        return mean, std
        
    print("\nMAIN RESULT TABLE (Averages across 5 seeds):")
    for key in ["expert_ceiling", "uniform_merging", "adamerging", "linear_router", "qws_merge", "l1_linear_l2", "l1_linear_tsar", "l3_linear_unreg", "l3_linear_l2", "l3_softmax_l2", "l3_softmax_tsar", "tf_centroid_router"]:
        mean, std = get_stats(results[key])
        if key == "expert_ceiling":
            print(f"Expert Ceiling: MNIST={mean[0]*100:.2f}%, F-MNIST={mean[1]*100:.2f}%, CIFAR={mean[2]*100:.2f}%, SVHN={mean[3]*100:.2f}% | Mean={np.mean(mean)*100:.2f}%")
        else:
            print(f"{key}: MNIST={mean[0]*100:.2f}±{std[0]*100:.2f}%, F-MNIST={mean[1]*100:.2f}±{std[1]*100:.2f}%, CIFAR={mean[2]*100:.2f}±{std[2]*100:.2f}%, SVHN={mean[3]*100:.2f}±{std[3]*100:.2f}% | Mean={mean[4]*100:.2f}±{std[4]*100:.2f}%")

    print("\nTSAR LAMBDA SWEEP:")
    for lam in sorted(results["tsar_sweep"].keys()):
        mean, std = get_stats(results["tsar_sweep"][lam])
        print(f"lam={lam}: MNIST={mean[0]*100:.2f}±{std[0]*100:.2f}%, F-MNIST={mean[1]*100:.2f}±{std[1]*100:.2f}%, CIFAR={mean[2]*100:.2f}±{std[2]*100:.2f}%, SVHN={mean[3]*100:.2f}±{std[3]*100:.2f}% | Mean={mean[4]*100:.2f}±{std[4]*100:.2f}%")

    print("\nCALIBRATION SIZE SWEEP FOR TSAR:")
    for sz in sorted(results["cal_size_sweep"].keys()):
        mean, std = get_stats(results["cal_size_sweep"][sz])
        print(f"B_cal={sz}: MNIST={mean[0]*100:.2f}±{std[0]*100:.2f}%, F-MNIST={mean[1]*100:.2f}±{std[1]*100:.2f}%, CIFAR={mean[2]*100:.2f}±{std[2]*100:.2f}%, SVHN={mean[3]*100:.2f}±{std[3]*100:.2f}% | Mean={mean[4]*100:.2f}±{std[4]*100:.2f}%")

    print("\nCALIBRATION SIZE SWEEP WITH GRADIENT MASKING:")
    for sz in sorted(results["cal_size_sweep_masked"].keys()):
        mean, std = get_stats(results["cal_size_sweep_masked"][sz])
        print(f"B_cal={sz}: MNIST={mean[0]*100:.2f}±{std[0]*100:.2f}%, F-MNIST={mean[1]*100:.2f}±{std[1]*100:.2f}%, CIFAR={mean[2]*100:.2f}±{std[2]*100:.2f}%, SVHN={mean[3]*100:.2f}±{std[3]*100:.2f}% | Mean={mean[4]*100:.2f}±{std[4]*100:.2f}%")

    print("\nCALIBRATION SIZE SWEEP WITH PCGRAD:")
    for sz in sorted(results["cal_size_sweep_pcgrad"].keys()):
        mean, std = get_stats(results["cal_size_sweep_pcgrad"][sz])
        print(f"B_cal={sz}: MNIST={mean[0]*100:.2f}±{std[0]*100:.2f}%, F-MNIST={mean[1]*100:.2f}±{std[1]*100:.2f}%, CIFAR={mean[2]*100:.2f}±{std[2]*100:.2f}%, SVHN={mean[3]*100:.2f}±{std[3]*100:.2f}% | Mean={mean[4]*100:.2f}±{std[4]*100:.2f}%")

    print("\nSTREAM DEPLOYMENT AUDIT (Averages across 5 seeds):")
    for name in stream_results:
        m1 = np.mean(stream_results[name]["homog_1"]) * 100
        s1 = np.std(stream_results[name]["homog_1"]) * 100
        m256_hom = np.mean(stream_results[name]["homog_256"]) * 100
        s256_hom = np.std(stream_results[name]["homog_256"]) * 100
        m256_het = np.mean(stream_results[name]["hetero_256"]) * 100
        s256_het = np.std(stream_results[name]["hetero_256"]) * 100
        print(f"{name}: Homog B=1: {m1:.2f}±{s1:.2f}% | Homog B=256: {m256_hom:.2f}±{s256_hom:.2f}% | Hetero B=256: {m256_het:.2f}±{s256_het:.2f}%")

    # Let's generate plots
    # Plot 1: TSAR Lambda Sweep (Joint Mean and SVHN Accuracy)
    lams = sorted(results["tsar_sweep"].keys())
    joint_means = [np.mean([x[4] for x in results["tsar_sweep"][l]]) * 100 for l in lams]
    svhn_accs = [np.mean([x[3] for x in results["tsar_sweep"][l]]) * 100 for l in lams]
    joint_means_stds = [np.std([x[4] for x in results["tsar_sweep"][l]]) * 100 for l in lams]
    svhn_accs_stds = [np.std([x[3] for x in results["tsar_sweep"][l]]) * 100 for l in lams]
    
    # Include lambda=0.0 from L3-Linear (L2 Reg)
    lams_all = [0.0] + lams
    joint_means_all = [np.mean([x[4] for x in results["l3_linear_l2"]]) * 100] + joint_means
    svhn_accs_all = [np.mean([x[3] for x in results["l3_linear_l2"]]) * 100] + svhn_accs
    joint_means_stds_all = [np.std([x[4] for x in results["l3_linear_l2"]]) * 100] + joint_means_stds
    svhn_accs_stds_all = [np.std([x[3] for x in results["l3_linear_l2"]]) * 100] + svhn_accs_stds

    plt.figure(figsize=(8, 5))
    plt.errorbar(range(len(lams_all)), joint_means_all, yerr=joint_means_stds_all, fmt='-o', capsize=5, label='Joint Mean Accuracy', color='blue')
    plt.errorbar(range(len(lams_all)), svhn_accs_all, yerr=svhn_accs_stds_all, fmt='--s', capsize=5, label='SVHN (OOD Task) Accuracy', color='red')
    plt.xticks(range(len(lams_all)), [f"0.0 (L2 Only)"] + [f"{l}" for l in lams])
    plt.xlabel('Anchor Regularization Coefficient ($\lambda_{anchor}$)')
    plt.ylabel('Accuracy (%)')
    plt.title('TSAR Parameter Sensitivity Sweep')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/tsar_sensitivity_sweep.png", dpi=150)
    plt.close()

    # Plot 2: Stream Deployment Audit (Heterogeneity Collapse and Mitigations)
    labels = list(stream_results.keys())
    homog_1_means = [np.mean(stream_results[name]["homog_1"]) * 100 for name in labels]
    homog_256_means = [np.mean(stream_results[name]["homog_256"]) * 100 for name in labels]
    hetero_256_means = [np.mean(stream_results[name]["hetero_256"]) * 100 for name in labels]
    
    x_positions = np.arange(len(labels))
    width = 0.25
    
    plt.figure(figsize=(15, 6))
    plt.bar(x_positions - width, homog_1_means, width, label='Homogeneous (B=1)', color='#4CAF50')
    plt.bar(x_positions, homog_256_means, width, label='Homogeneous (B=256)', color='#2196F3')
    plt.bar(x_positions + width, hetero_256_means, width, label='Heterogeneous (B=256)', color='#FF5722')
    
    plt.xticks(x_positions, labels, rotation=25, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Deployment Stream Audit: Heterogeneity Collapse & Algorithmic Mitigations')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/heterogeneity_collapse_audit.png", dpi=150)
    plt.close()

    # Plot 3: Sample Complexity Sweep for TSAR (Standard vs Gradient Masking vs PCGrad)
    sizes = sorted(results["cal_size_sweep"].keys())
    sz_means = [np.mean([x[4] for x in results["cal_size_sweep"][s]]) * 100 for s in sizes]
    sz_stds = [np.std([x[4] for x in results["cal_size_sweep"][s]]) * 100 for s in sizes]
    
    sz_means_masked = [np.mean([x[4] for x in results["cal_size_sweep_masked"][s]]) * 100 for s in sizes]
    sz_stds_masked = [np.std([x[4] for x in results["cal_size_sweep_masked"][s]]) * 100 for s in sizes]

    sz_means_pcgrad = [np.mean([x[4] for x in results["cal_size_sweep_pcgrad"][s]]) * 100 for s in sizes]
    sz_stds_pcgrad = [np.std([x[4] for x in results["cal_size_sweep_pcgrad"][s]]) * 100 for s in sizes]
    
    # Also add standard L3-Linear L2 only at sz=64 for comparison
    l2_sz64_mean = np.mean([x[4] for x in results["l3_linear_l2"]]) * 100
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(sizes, sz_means, yerr=sz_stds, fmt='-o', color='purple', capsize=5, label=r'TSAR (Standard Calibration)', linewidth=2)
    plt.errorbar(sizes, sz_means_masked, yerr=sz_stds_masked, fmt='--s', color='green', capsize=5, label=r'TSAR + Gradient Masking (Ours)', linewidth=2)
    plt.errorbar(sizes, sz_means_pcgrad, yerr=sz_stds_pcgrad, fmt='-^', color='blue', capsize=5, label=r'TSAR + PCGrad (Ours)', linewidth=2)
    plt.axhline(y=l2_sz64_mean, color='gray', linestyle='--', label='L2 Regularization (at $B_{cal}=64$)')
    plt.xlabel('Calibration Set Size ($B_{cal}$)')
    plt.ylabel('Joint Mean Accuracy (%)')
    plt.title('Sample Complexity: Standard vs Gradient-Masked vs PCGrad TSAR')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/sample_complexity_sweep.png", dpi=150)
    plt.close()

    # Plot 4: Subspace Leakage Sweep (Overlapping task manifolds)
    leaks = sorted(results["leakage_sweep"].keys())
    uni_leak_means = [np.mean([x[4] for x in results["leakage_sweep"][lk]["uniform_merging"]]) * 100 for lk in leaks]
    l3_leak_means = [np.mean([x[4] for x in results["leakage_sweep"][lk]["l3_linear_l2"]]) * 100 for lk in leaks]
    qws_leak_means = [np.mean([x[4] for x in results["leakage_sweep"][lk]["qws_merge"]]) * 100 for lk in leaks]
    tsar_leak_means = [np.mean([x[4] for x in results["leakage_sweep"][lk]["tsar_pcgrad"]]) * 100 for lk in leaks]
    
    uni_leak_stds = [np.std([x[4] for x in results["leakage_sweep"][lk]["uniform_merging"]]) * 100 for lk in leaks]
    l3_leak_stds = [np.std([x[4] for x in results["leakage_sweep"][lk]["l3_linear_l2"]]) * 100 for lk in leaks]
    qws_leak_stds = [np.std([x[4] for x in results["leakage_sweep"][lk]["qws_merge"]]) * 100 for lk in leaks]
    tsar_leak_stds = [np.std([x[4] for x in results["leakage_sweep"][lk]["tsar_pcgrad"]]) * 100 for lk in leaks]

    plt.figure(figsize=(8, 5))
    plt.errorbar(leaks, tsar_leak_means, yerr=tsar_leak_stds, fmt='-^', color='blue', capsize=5, label='TSAR + PCGrad (Ours)', linewidth=2)
    plt.errorbar(leaks, uni_leak_means, yerr=uni_leak_stds, fmt='-s', color='green', capsize=5, label='Static Uniform Merging', linewidth=2)
    plt.errorbar(leaks, l3_leak_means, yerr=l3_leak_stds, fmt='-o', color='purple', capsize=5, label='L3-Linear (L2 Reg)', linewidth=2)
    plt.errorbar(leaks, qws_leak_means, yerr=qws_leak_stds, fmt='-x', color='red', capsize=5, label='QWS-Merge SOTA', linewidth=2)
    plt.xlabel('Subspace Leakage Factor ($\eta$)')
    plt.ylabel('Joint Mean Accuracy (%)')
    plt.title('Robustness to Overlapping Task Manifolds (Subspace Leakage Sweep)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/subspace_leakage_sweep.png", dpi=150)
    plt.close()

    print("\nSUBSPACE LEAKAGE SWEEP (Averages across seeds):")
    for leak in leaks:
        print(f"leak={leak}:")
        for key in ["uniform_merging", "l3_linear_l2", "qws_merge", "tsar_pcgrad"]:
            arr = np.array(results["leakage_sweep"][leak][key])
            mean_acc = np.mean(arr[:, 4]) * 100
            std_acc = np.std(arr[:, 4]) * 100
            print(f"  {key}: {mean_acc:.2f}±{std_acc:.2f}%")

    # Save raw json results
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int16)):
            return int(obj)
        else:
            return obj

    final_save_dict = {
        "results": make_serializable(results),
        "stream_results": make_serializable(stream_results)
    }
    with open("results/all_results.json", "w") as f:
        json.dump(final_save_dict, f, indent=2)
    print("\nAll experiments run and saved successfully!")

if __name__ == "__main__":
    run_all_experiments()
