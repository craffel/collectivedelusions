# Test Calibration Size Sensitivity of EHPB Router (Vectorized up to B=128)
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("Initializing random seeds...")
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

print("Generating fixed train and test sets...")
X_train, Y_task_train, Y_class_train = generate_samples(1000, noise_scales)
X_test, Y_task_test, Y_class_test = generate_samples(250, noise_scales)

print("Training specialized experts...")
experts = {}
for k in range(K):
    mask = (Y_task_train == k)
    X_k = X_train[mask]
    Y_k = Y_class_train[mask]
    head = nn.Linear(D, num_classes)
    optimizer = optim.Adam(head.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(80):
        optimizer.zero_grad()
        outputs = head(X_k)
        loss = criterion(outputs, Y_k)
        loss.backward()
        optimizer.step()
    experts[k] = head

print("Setting up 14-layer backbone simulation...")
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

print("Generating carrier keys...")
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

# Stack keys for vectorized execution
stacked_keys_layers = []
for l in range(L - 1):
    stacked_keys_layers.append(torch.stack(frozen_keys[l]))  # [K, D, D]
stacked_keys_head = torch.stack(frozen_keys[L - 1])          # [K, num_classes, D]

print("Performing holographic superposition...")
W_holo_layers = []
for l in range(L - 1):
    W_holo = torch.zeros(D, D)
    for k in range(K):
        W_holo += V_task_layers[k][l] * frozen_keys[l][k]
    W_holo_layers.append(W_holo)
    
W_holo_head = torch.zeros(num_classes, D)
for k in range(K):
    W_holo_head += V_task_head[k] * frozen_keys[L - 1][k]

class L3_Router_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(L, K, K) * 0.1)
        self.B = nn.Parameter(torch.zeros(L, K))
    def forward(self, psi):
        coeffs = []
        for l in range(L):
            logits = torch.matmul(psi, self.W[l].t()) + self.B[l]
            coeffs.append(logits.softmax(dim=-1))
        return coeffs

sizes_to_sweep = [16, 32, 64, 128]

print("| Calibration Size (B/task) | Total Samples | Train Cross-Entropy | Test Joint Mean Accuracy | Test Routing Selection Accuracy |")
print("|---------------------------|---------------|---------------------|--------------------------|---------------------------------|")

for B in sizes_to_sweep:
    random.seed(42 + B)
    np.random.seed(42 + B)
    torch.manual_seed(42 + B)
    
    X_cal, Y_task_cal, Y_class_cal = generate_samples(B, noise_scales)
    
    X_cal_centered = X_cal - X_cal.mean(dim=0)
    _, _, V_mat = torch.pca_lowrank(X_cal_centered, q=K)
    P_proj = V_mat[:, :K]
    
    def get_psi_proj(x):
        z = torch.matmul(x, P_proj)
        return z / (torch.norm(z, p=2, dim=-1, keepdim=True) + 1e-8)
        
    Psi_cal = get_psi_proj(X_cal)
    Psi_test = get_psi_proj(X_test)
    
    router = L3_Router_Network()
    optimizer = optim.AdamW(router.parameters(), lr=0.01, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Fully batched, vectorized training loop
    for epoch in range(100):
        optimizer.zero_grad()
        coeffs = router(Psi_cal)
        
        h = X_cal.unsqueeze(1)
        for l in range(L - 1):
            U_l = torch.einsum('bk,kdc->bdc', coeffs[l], stacked_keys_layers[l])
            W_l = torch.eye(D).unsqueeze(0) + W_holo_layers[l].unsqueeze(0) * U_l
            h = torch.bmm(h, W_l.transpose(-1, -2))
            
        U_head = torch.einsum('bk,kcd->bcd', coeffs[L - 1], stacked_keys_head)
        W_head = W_base_head.unsqueeze(0) + W_holo_head.unsqueeze(0) * U_head
        logits = torch.bmm(h, W_head.transpose(-1, -2)).squeeze(1)
        
        loss = criterion(logits, Y_class_cal)
        loss.backward()
        optimizer.step()
        
    final_train_loss = loss.item()
    
    with torch.no_grad():
        coeffs_test = router(Psi_test)
        
        h_test = X_test.unsqueeze(1)
        for l in range(L - 1):
            U_l = torch.einsum('bk,kdc->bdc', coeffs_test[l], stacked_keys_layers[l])
            W_l = torch.eye(D).unsqueeze(0) + W_holo_layers[l].unsqueeze(0) * U_l
            h_test = torch.bmm(h_test, W_l.transpose(-1, -2))
            
        U_head_test = torch.einsum('bk,kcd->bcd', coeffs_test[L - 1], stacked_keys_head)
        W_head_test = W_base_head.unsqueeze(0) + W_holo_head.unsqueeze(0) * U_head_test
        logits_test = torch.bmm(h_test, W_head_test.transpose(-1, -2)).squeeze(1)
        
        preds_test = logits_test.argmax(dim=-1)
        
        total_test = X_test.shape[0]
        task_correct = {k: 0 for k in range(K)}
        task_counts = {k: 0 for k in range(K)}
        for b in range(total_test):
            t_label = Y_task_test[b].item()
            c_label = Y_class_test[b].item()
            pred = preds_test[b].item()
            if pred == c_label:
                task_correct[t_label] += 1
            task_counts[t_label] += 1
            
        accs = []
        for k in range(K):
            acc = (task_correct[k] / task_counts[k]) * 100.0
            accs.append(acc)
        test_joint_mean = np.mean(accs)
        
        stacked_test_coeffs = torch.stack(coeffs_test)
        avg_test_coeffs = stacked_test_coeffs.mean(dim=0)
        routing_preds = avg_test_coeffs.argmax(dim=-1)
        
        routing_corrects = (routing_preds == Y_task_test).sum().item()
        routing_acc = (routing_corrects / total_test) * 100.0
    
    print(f"| {B:25d} | {4*B:13d} | {final_train_loss:19.4f} | {test_joint_mean:23.2f}% | {routing_acc:30.2f}% |")
print("Done!")
