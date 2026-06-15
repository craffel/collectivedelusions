# Empirical Validation of Non-Linear Bottleneck CCN
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
    for epoch in range(40):
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

class NonLinearCCN(nn.Module):
    def __init__(self, D, bottleneck_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, D)
        )
    def forward(self, x):
        return self.net(x)

def run_ccn_ehpb(ccn_type="linear", lr_ccn=0.01, epochs_ccn=100, num_samples_ccn=400):
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create random bipolar keys (factored rank-1)
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
        
    # Superimpose weights
    W_holo_layers = []
    for l in range(L - 1):
        W_holo = torch.zeros(D, D)
        for k in range(K):
            W_holo += V_task_layers[k][l] * frozen_keys[l][k]
        W_holo_layers.append(W_holo)
        
    W_holo_head = torch.zeros(num_classes, D)
    for k in range(K):
        W_holo_head += V_task_head[k] * frozen_keys[L - 1][k]
        
    keys_stacked = {l: torch.stack(frozen_keys[l]) for l in range(L)}
    V_task_stacked = {l: torch.stack([V_task_layers[k][l] for k in range(K)]) for l in range(L - 1)}
    V_task_stacked[L - 1] = torch.stack(V_task_head)
    
    # Train Router
    router = L3_Router_Network()
    optimizer = optim.AdamW(router.parameters(), lr=0.01, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(15):
        optimizer.zero_grad()
        coeffs = router(Psi_cal)
        logits_list = []
        for b in range(Psi_cal.shape[0]):
            h_b = X_cal[b].unsqueeze(0)
            for l in range(L - 1):
                c_b_l = coeffs[l][b]
                U_bl = torch.sum(c_b_l.view(K, 1, 1) * keys_stacked[l], dim=0)
                W_bl = W_base_layers[l] + W_holo_layers[l] * U_bl
                h_b = torch.matmul(h_b, W_bl.t())
                
            c_b_head = coeffs[L - 1][b]
            U_head_b = torch.sum(c_b_head.view(K, 1, 1) * keys_stacked[L - 1], dim=0)
            W_head_b = W_base_head + W_holo_head * U_head_b
            logits_b = torch.matmul(h_b, W_head_b.t())
            logits_list.append(logits_b)
            
        logits_all = torch.cat(logits_list, dim=0)
        loss = criterion(logits_all, Y_class_cal)
        loss.backward()
        optimizer.step()
        
    ccns = {}
    
    if ccn_type != "none":
        with torch.no_grad():
            coeffs_train = router(get_psi(X_train[:num_samples_ccn]))
            
        H_noisy = X_train[:num_samples_ccn].clone()
        H_clean = X_train[:num_samples_ccn].clone()
        
        for l in range(L - 1):
            H_clean_next_list = []
            H_noisy_next_list = []
            
            with torch.no_grad():
                for b in range(num_samples_ccn):
                    c_b_l = coeffs_train[l][b]
                    W_clean_b = W_base_layers[l] + sum(c_b_l[k] * V_task_layers[k][l] for k in range(K))
                    h_clean_next = torch.matmul(H_clean[b].unsqueeze(0), W_clean_b.t())
                    H_clean_next_list.append(h_clean_next)
                    
                    U_bl = torch.sum(c_b_l.view(K, 1, 1) * keys_stacked[l], dim=0)
                    W_bl = W_base_layers[l] + W_holo_layers[l] * U_bl
                    h_noisy_next = torch.matmul(H_noisy[b].unsqueeze(0), W_bl.t())
                    H_noisy_next_list.append(h_noisy_next)
                    
            H_clean_next = torch.cat(H_clean_next_list, dim=0)
            H_noisy_next = torch.cat(H_noisy_next_list, dim=0)
            
            if ccn_type == "linear":
                ccn_layer = nn.Linear(D, D, bias=True)
            elif ccn_type == "nonlinear":
                ccn_layer = NonLinearCCN(D, bottleneck_dim=96)
                
            ccn_optimizer = optim.Adam(ccn_layer.parameters(), lr=lr_ccn)
            ccn_criterion = nn.MSELoss()
            
            for epoch in range(epochs_ccn):
                ccn_optimizer.zero_grad()
                pred = ccn_layer(H_noisy_next)
                loss_ccn = ccn_criterion(pred, H_clean_next)
                loss_ccn.backward()
                ccn_optimizer.step()
                
            ccns[l] = ccn_layer
            
            with torch.no_grad():
                H_noisy = ccn_layer(H_noisy_next)
                H_clean = H_clean_next
                
    # Evaluate
    task_correct = {k: 0 for k in range(K)}
    task_counts = {k: 0 for k in range(K)}
    with torch.no_grad():
        coeffs_test = router(Psi_test)
        for b in range(X_test.shape[0]):
            h_b = X_test[b].unsqueeze(0)
            for l in range(L - 1):
                c_b_l = coeffs_test[l][b]
                U_bl = torch.sum(c_b_l.view(K, 1, 1) * keys_stacked[l], dim=0)
                W_bl = W_base_layers[l] + W_holo_layers[l] * U_bl
                h_b = torch.matmul(h_b, W_bl.t())
                
                if ccn_type != "none":
                    h_b = ccns[l](h_b)
                
            c_b_head = coeffs_test[L - 1][b]
            U_head_b = torch.sum(c_b_head.view(K, 1, 1) * keys_stacked[L - 1], dim=0)
            W_head_b = W_base_head + W_holo_head * U_head_b
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

print("Comparing CCN Architectures (Linear vs. Non-Linear Bottleneck with GeLU):")
print("-" * 80)

accs_base, mean_base = run_ccn_ehpb(ccn_type="none")
print(f"No CCN (Baseline) | Joint Mean: {mean_base:.2f}% | Accs: {[round(a,1) for a in accs_base]}")

accs_lin, mean_lin = run_ccn_ehpb(ccn_type="linear")
print(f"Linear CCN        | Joint Mean: {mean_lin:.2f}% | Accs: {[round(a,1) for a in accs_lin]}")

accs_nonlin, mean_nonlin = run_ccn_ehpb(ccn_type="nonlinear")
print(f"Non-Linear CCN    | Joint Mean: {mean_nonlin:.2f}% | Accs: {[round(a,1) for a in accs_nonlin]}")
