# Highly Optimized Test Residual-EHPB (Sparsity Sweep)
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

D = 192
K = 4
num_classes = 10

# Generate distinct class prototypes
prototypes = {}
for k in range(K):
    prototypes[k] = []
    for c in range(num_classes):
        proto = np.random.normal(0, 1, D)
        proto /= (np.linalg.norm(proto) + 1e-8)
        prototypes[k].append(proto)

noise_scales = {0: 0.01, 1: 0.15, 2: 0.30, 3: 0.90}

def generate_samples(num_samples_per_task, noise_dict):
    X_list = []
    Y_task_list = []
    Y_class_list = []
    for k in range(K):
        for _ in range(num_samples_per_task):
            c = np.random.randint(0, num_classes)
            proto = prototypes[k][c]
            noise = np.random.normal(0, noise_dict[k], D)
            feat = proto + noise
            X_list.append(feat)
            Y_task_list.append(k)
            Y_class_list.append(c)
    return (torch.tensor(np.array(X_list), dtype=torch.float32), 
            torch.tensor(np.array(Y_task_list), dtype=torch.long), 
            torch.tensor(np.array(Y_class_list), dtype=torch.long))

# Generate dataset splits
X_train, Y_task_train, Y_class_train = generate_samples(1000, noise_scales)
X_cal, Y_task_cal, Y_class_cal = generate_samples(16, noise_scales)
X_test, Y_task_test, Y_class_test = generate_samples(250, noise_scales)

# Train specialized experts (ceilings)
experts = {}
for k in range(K):
    mask = (Y_task_train == k)
    X_k = X_train[mask]
    Y_k = Y_class_train[mask]
    head = nn.Linear(D, num_classes)
    optimizer = optim.Adam(head.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(40): # Reduced from 80 for speed
        optimizer.zero_grad()
        outputs = head(X_k)
        loss = criterion(outputs, Y_k)
        loss.backward()
        optimizer.step()
    experts[k] = head

# Setup 14-layer backbone simulation
L = 14
W_base_layers = []
V_task_layers = {k: [] for k in range(K)}
for l in range(L - 1):
    W_base_layers.append(torch.eye(D))
    for k in range(K):
        V_k = torch.randn(D, D) / math.sqrt(D) * 0.45
        V_task_layers[k].append(V_k)

W_base_head = torch.zeros(num_classes, D)
V_task_head = []
for k in range(K):
    V_task_head.append(experts[k].weight.data - W_base_head)

# Unsupervised PCA projection matrix P onto d=4 subspace
X_cal_centered = X_cal - X_cal.mean(dim=0)
_, _, V = torch.pca_lowrank(X_cal_centered, q=K)
P_proj = V[:, :K]

def get_psi(x):
    z = torch.matmul(x, P_proj)
    return z / (torch.norm(z, p=2, dim=-1, keepdim=True) + 1e-8)

Psi_cal = get_psi(X_cal)
Psi_test = get_psi(X_test)

class L3_Router_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(L, K, K) * 0.1)
        self.B = nn.Parameter(torch.zeros(L, K))
    def forward(self, psi):
        coeffs = []
        for l in range(L):
            logits = torch.matmul(psi, self.W[l].t()) + self.B[l]
            coeffs.append(logits)
        return coeffs

def run_residual_ehpb(p_sparsity=0.05):
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Key generation (factored rank-1 carrier keys)
    frozen_keys = {}
    for l in range(L - 1):
        frozen_keys[l] = []
        for k in range(K):
            r_k = torch.sign(torch.randn(D))
            c_k = torch.sign(torch.randn(D))
            frozen_keys[l].append(torch.outer(r_k, c_k))
            
    frozen_keys[L - 1] = []
    for k in range(K):
        r_head = torch.sign(torch.randn(num_classes))
        c_head = torch.sign(torch.randn(D))
        frozen_keys[L - 1].append(torch.outer(r_head, c_head))
        
    # 2. Compute Sparse Coordinate-wise Importance Masks M^(l)
    masks = {}
    for l in range(L - 1):
        task_sum = torch.zeros(D, D)
        for k in range(K):
            task_sum += torch.abs(V_task_layers[k][l])
        if p_sparsity == 0.0:
            masks[l] = torch.zeros(D, D)
        else:
            threshold = torch.quantile(task_sum, 1.0 - p_sparsity)
            masks[l] = (task_sum >= threshold).float()
            
    task_sum_head = torch.zeros(num_classes, D)
    for k in range(K):
        task_sum_head += torch.abs(V_task_head[k])
    if p_sparsity == 0.0:
        masks[L - 1] = torch.zeros(num_classes, D)
    else:
        threshold_head = torch.quantile(task_sum_head, 1.0 - p_sparsity)
        masks[L - 1] = (task_sum_head >= threshold_head).float()

    # 3. Holographic Superposition with Residual Bypassing
    W_holo_layers = []
    for l in range(L - 1):
        W_holo_res = torch.zeros(D, D)
        for k in range(K):
            W_holo_res += V_task_layers[k][l] * (1.0 - masks[l]) * frozen_keys[l][k]
        W_holo_layers.append(W_holo_res)
        
    W_holo_head = torch.zeros(num_classes, D)
    for k in range(K):
        W_holo_head += V_task_head[k] * (1.0 - masks[L - 1]) * frozen_keys[L - 1][k]
        
    # Stack the keys and task vectors for vectorization
    # frozen_keys_l: shape [K, D, D]
    keys_stacked = {l: torch.stack(frozen_keys[l]) for l in range(L)}
    V_task_stacked = {l: torch.stack([V_task_layers[k][l] for k in range(K)]) for l in range(L - 1)}
    V_task_stacked[L - 1] = torch.stack(V_task_head)

    # 4. Router Training (very fast 15 steps)
    router = L3_Router_Network()
    optimizer = optim.AdamW(router.parameters(), lr=0.01, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(15): # Reduced for extreme speed
        optimizer.zero_grad()
        coeffs = router(Psi_cal)
        logits_list = []
        for b in range(Psi_cal.shape[0]):
            h_b = X_cal[b].unsqueeze(0)
            for l in range(L - 1):
                # Vectorized U_bl computation:
                # coeffs[l][b] is shape [K]
                # keys_stacked[l] is shape [K, D, D]
                # U_bl is sum of coeffs * keys -> shape [D, D]
                c_b_l = coeffs[l][b]
                U_bl = torch.sum(c_b_l.view(K, 1, 1) * keys_stacked[l], dim=0)
                
                # Vectorized clean_path computation:
                # clean_path is sum of coeffs * V_k * M
                clean_path = torch.sum(c_b_l.view(K, 1, 1) * V_task_stacked[l] * masks[l].unsqueeze(0), dim=0)
                
                W_bl = W_base_layers[l] + W_holo_layers[l] * U_bl + clean_path
                h_b = torch.matmul(h_b, W_bl.t())
                
            c_b_head = coeffs[L - 1][b]
            U_head_b = torch.sum(c_b_head.view(K, 1, 1) * keys_stacked[L - 1], dim=0)
            clean_path_head = torch.sum(c_b_head.view(K, 1, 1) * V_task_stacked[L - 1] * masks[L - 1].unsqueeze(0), dim=0)
            
            W_head_b = W_base_head + W_holo_head * U_head_b + clean_path_head
            logits_b = torch.matmul(h_b, W_head_b.t())
            logits_list.append(logits_b)
            
        logits_all = torch.cat(logits_list, dim=0)
        loss = criterion(logits_all, Y_class_cal)
        loss.backward()
        optimizer.step()
        
    # 5. Evaluation
    task_correct = {k: 0 for k in range(K)}
    task_counts = {k: 0 for k in range(K)}
    with torch.no_grad():
        coeffs_test = router(Psi_test)
        for b in range(X_test.shape[0]):
            h_b = X_test[b].unsqueeze(0)
            for l in range(L - 1):
                c_b_l = coeffs_test[l][b]
                U_bl = torch.sum(c_b_l.view(K, 1, 1) * keys_stacked[l], dim=0)
                clean_path = torch.sum(c_b_l.view(K, 1, 1) * V_task_stacked[l] * masks[l].unsqueeze(0), dim=0)
                W_bl = W_base_layers[l] + W_holo_layers[l] * U_bl + clean_path
                h_b = torch.matmul(h_b, W_bl.t())
                
            c_b_head = coeffs_test[L - 1][b]
            U_head_b = torch.sum(c_b_head.view(K, 1, 1) * keys_stacked[L - 1], dim=0)
            clean_path_head = torch.sum(c_b_head.view(K, 1, 1) * V_task_stacked[L - 1] * masks[L - 1].unsqueeze(0), dim=0)
            W_head_b = W_base_head + W_holo_head * U_head_b + clean_path_head
            logits_b = torch.matmul(h_b, W_head_b.t())
            pred_b = logits_b.argmax(dim=-1).item()
            
            t_label = Y_task_test[b].item()
            is_correct = (pred_b == Y_class_test[b].item())
            if is_correct:
                task_correct[t_label] += 1
            task_counts[t_label] += 1
            
    accs = []
    for k in range(K):
        acc = (task_correct[k] / task_counts[k]) * 100.0
        accs.append(acc)
    return accs, np.mean(accs)

# Sweep sparsity values
sparsities = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]
print("Starting Optimized Residual-EHPB Sparsity Sweep:")
for p in sparsities:
    t0 = time.time()
    accs, mean_acc = run_residual_ehpb(p_sparsity=p)
    dt = time.time() - t0
    print(f"  p = {p:<4} ({int(p*100):>2}% uncompressed) | MNIST: {accs[0]:.1f}% | Fashion: {accs[1]:.1f}% | CIFAR: {accs[2]:.1f}% | SVHN: {accs[3]:.1f}% | Joint Mean: {mean_acc:.2f}% | Time: {dt:.2f}s")
