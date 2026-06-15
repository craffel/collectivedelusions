# Empirical Validation of CCN Generalization on In-Distribution vs. Out-of-Distribution Inputs
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

# Normal ID noise scales
id_noise_scales = {0: 0.01, 1: 0.15, 2: 0.30, 3: 0.90}

# OOD noise scales (simulating a high-noise shift)
ood_noise_scales = {0: 0.02, 1: 0.30, 2: 0.60, 3: 1.80}

def generate_samples(num_samples_per_task, noise_dict, prototype_shift=None):
    X_list = []
    Y_task_list = []
    Y_class_list = []
    for k in range(K):
        for _ in range(num_samples_per_task):
            c = np.random.randint(0, num_classes)
            proto = prototypes[k][c]
            if prototype_shift is not None:
                proto = proto + prototype_shift
                proto /= (np.linalg.norm(proto) + 1e-8)
            noise = np.random.normal(0, noise_dict[k], D)
            feat = proto + noise
            X_list.append(feat)
            Y_task_list.append(k)
            Y_class_list.append(c)
    return (torch.tensor(np.array(X_list), dtype=torch.float32), 
            torch.tensor(np.array(Y_task_list), dtype=torch.long), 
            torch.tensor(np.array(Y_class_list), dtype=torch.long))

# Generate dataset splits
X_train, Y_task_train, Y_class_train = generate_samples(1000, id_noise_scales)
X_cal, Y_task_cal, Y_class_cal = generate_samples(16, id_noise_scales)
X_test_id, Y_task_test_id, Y_class_test_id = generate_samples(250, id_noise_scales)

# Generate OOD test sets:
# 1. Noise-Scaled OOD
X_test_ood_noise, Y_task_test_ood_noise, Y_class_test_ood_noise = generate_samples(250, ood_noise_scales)
# 2. Prototype-Shifted OOD (feature drift)
shift_vec = np.random.normal(0, 0.25, D)
X_test_ood_drift, Y_task_test_ood_drift, Y_class_test_ood_drift = generate_samples(250, id_noise_scales, prototype_shift=shift_vec)

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

# Setup L=14 layer backbone simulation
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
Psi_test_id = get_psi(X_test_id)
Psi_test_ood_noise = get_psi(X_test_ood_noise)
Psi_test_ood_drift = get_psi(X_test_ood_drift)

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

def run_evaluation(use_ccn=True):
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
    if use_ccn:
        # Train CCN on 400 train samples
        num_samples_ccn = 400
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
            
            ccn_layer = nn.Linear(D, D, bias=True)
            ccn_optimizer = optim.Adam(ccn_layer.parameters(), lr=0.01)
            ccn_criterion = nn.MSELoss()
            
            for epoch in range(100):
                ccn_optimizer.zero_grad()
                pred = ccn_layer(H_noisy_next)
                loss_ccn = ccn_criterion(pred, H_clean_next)
                loss_ccn.backward()
                ccn_optimizer.step()
                
            ccns[l] = ccn_layer
            with torch.no_grad():
                H_noisy = ccn_layer(H_noisy_next)
                H_clean = H_clean_next

    # Evaluate helper
    def evaluate_set(X, Y_task, Y_class, Psi, coeffs_eval):
        task_correct = {k: 0 for k in range(K)}
        task_counts = {k: 0 for k in range(K)}
        
        # Accumulate pre- and post-cleaned MSEs for intermediate layers
        total_pre_mse = 0.0
        total_post_mse = 0.0
        
        for b in range(X.shape[0]):
            h_b = X[b].unsqueeze(0)
            h_b_clean = X[b].unsqueeze(0)
            
            for l in range(L - 1):
                c_b_l = coeffs_eval[l][b]
                
                # Dynamic targets for clean tracking
                W_clean_b = W_base_layers[l] + sum(c_b_l[k] * V_task_layers[k][l] for k in range(K))
                h_b_clean = torch.matmul(h_b_clean, W_clean_b.t())
                
                # Noisy path
                U_bl = torch.sum(c_b_l.view(K, 1, 1) * keys_stacked[l], dim=0)
                W_bl = W_base_layers[l] + W_holo_layers[l] * U_bl
                h_b = torch.matmul(h_b, W_bl.t())
                
                if use_ccn:
                    # Accumulate pre-CCN MSE
                    total_pre_mse += torch.mean((h_b - h_b_clean)**2).item()
                    h_b = ccns[l](h_b)
                    # Accumulate post-CCN MSE
                    total_post_mse += torch.mean((h_b - h_b_clean)**2).item()
                else:
                    total_pre_mse += torch.mean((h_b - h_b_clean)**2).item()
                    
            c_b_head = coeffs_eval[L - 1][b]
            U_head_b = torch.sum(c_b_head.view(K, 1, 1) * keys_stacked[L - 1], dim=0)
            W_head_b = W_base_head + W_holo_head * U_head_b
            logits_b = torch.matmul(h_b, W_head_b.t())
            pred_b = logits_b.argmax(dim=-1).item()
            
            t_label = Y_task[b].item()
            is_correct = (pred_b == Y_class[b].item())
            if is_correct:
                task_correct[t_label] += 1
            task_counts[t_label] += 1
            
        accs = []
        for k in range(K):
            acc = (task_correct[k] / task_counts[k]) * 100.0
            accs.append(acc)
            
        avg_pre_mse = total_pre_mse / (X.shape[0] * (L - 1))
        avg_post_mse = total_post_mse / (X.shape[0] * (L - 1)) if use_ccn else avg_pre_mse
        return np.mean(accs), avg_pre_mse, avg_post_mse

    with torch.no_grad():
        coeffs_test_id = router(Psi_test_id)
        id_acc, id_pre, id_post = evaluate_set(X_test_id, Y_task_test_id, Y_class_test_id, Psi_test_id, coeffs_test_id)
        
        coeffs_ood_noise = router(Psi_test_ood_noise)
        ood_n_acc, ood_n_pre, ood_n_post = evaluate_set(X_test_ood_noise, Y_task_test_ood_noise, Y_class_test_ood_noise, Psi_test_ood_noise, coeffs_ood_noise)
        
        coeffs_ood_drift = router(Psi_test_ood_drift)
        ood_d_acc, ood_d_pre, ood_d_post = evaluate_set(X_test_ood_drift, Y_task_test_ood_drift, Y_class_test_ood_drift, Psi_test_ood_drift, coeffs_ood_drift)
        
    return {
        "id": {"acc": id_acc, "pre_mse": id_pre, "post_mse": id_post},
        "ood_noise": {"acc": ood_n_acc, "pre_mse": ood_n_pre, "post_mse": ood_n_post},
        "ood_drift": {"acc": ood_d_acc, "pre_mse": ood_d_pre, "post_mse": ood_d_post}
    }

print("Running CCN Generalization Audit...")
print("-" * 80)
results_no_ccn = run_evaluation(use_ccn=False)
results_ccn = run_evaluation(use_ccn=True)

print("\nRESULTS SUMMARY:")
print("-" * 80)
print(f"In-Distribution (ID):")
print(f"  No CCN  | Acc: {results_no_ccn['id']['acc']:.2f}% | Feature MSE: {results_no_ccn['id']['pre_mse']:.6f}")
print(f"  With CCN| Acc: {results_ccn['id']['acc']:.2f}% | Feature MSE: {results_ccn['id']['post_mse']:.6f} (Reduced by {results_ccn['id']['pre_mse']/results_ccn['id']['post_mse']:.1f}x)")

print(f"\nOut-of-Distribution (OOD, Scaled Noise):")
print(f"  No CCN  | Acc: {results_no_ccn['ood_noise']['acc']:.2f}% | Feature MSE: {results_no_ccn['ood_noise']['pre_mse']:.6f}")
print(f"  With CCN| Acc: {results_ccn['ood_noise']['acc']:.2f}% | Feature MSE: {results_ccn['ood_noise']['post_mse']:.6f} (Reduced by {results_ccn['ood_noise']['pre_mse']/results_ccn['ood_noise']['post_mse']:.1f}x)")

print(f"\nOut-of-Distribution (OOD, Prototype Feature Drift):")
print(f"  No CCN  | Acc: {results_no_ccn['ood_drift']['acc']:.2f}% | Feature MSE: {results_no_ccn['ood_drift']['pre_mse']:.6f}")
print(f"  With CCN| Acc: {results_ccn['ood_drift']['acc']:.2f}% | Feature MSE: {results_ccn['ood_drift']['post_mse']:.6f} (Reduced by {results_ccn['ood_drift']['pre_mse']/results_ccn['ood_drift']['post_mse']:.1f}x)")
