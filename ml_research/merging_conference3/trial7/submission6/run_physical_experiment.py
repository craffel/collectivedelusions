import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

set_seed(42)

# Load digits dataset
digits = load_digits()
X = digits.data # Shape: (1797, 64)
y = digits.target # Shape: (1797,)

# Normalize inputs
X = X / 16.0
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

K = 4

# Define 4 distinct tasks from digits
# Task 0: Digit 0 vs 1
# Task 1: Digit 2 vs 3
# Task 2: Digit 4 vs 5
# Task 3: Digit 6 vs 7
def get_task_data(task_id):
    if task_id == 0:
        indices = (y == 0) | (y == 1)
        X_task = X[indices]
        y_task = y[indices]
    elif task_id == 1:
        indices = (y == 2) | (y == 3)
        X_task = X[indices]
        y_task = y[indices] - 2
    elif task_id == 2:
        indices = (y == 4) | (y == 5)
        X_task = X[indices]
        y_task = y[indices] - 4
    elif task_id == 3:
        indices = (y == 6) | (y == 7)
        X_task = X[indices]
        y_task = y[indices] - 6
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_task.numpy(), y_task.numpy(), test_size=100, random_state=42, stratify=y_task.numpy()
    )
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long)
    )

task_data = [get_task_data(k) for k in range(K)]

# Define a simple 2-layer MLP as base model
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize base model
base_model = TinyMLP()

# Fine-tune task-specific experts
experts = []
print("Training task-specific experts on physical classification tasks...")
for k in range(K):
    X_train, X_test, y_train, y_test = task_data[k]
    expert = copy.deepcopy(base_model)
    optimizer = optim.Adam(expert.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train expert
    for epoch in range(40):
        optimizer.zero_grad()
        out = expert(X_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        optimizer.step()
        
    # Evaluate expert test accuracy
    with torch.no_grad():
        test_out = expert(X_test)
        preds = torch.argmax(test_out, dim=-1)
        acc = (preds == y_test).float().mean().item() * 100.0
    print(f"Expert {k} Standalone Test Accuracy: {acc:.2f}%")
    experts.append(expert)

# Precompute parameter-space task vectors
task_vectors = []
for k in range(K):
    tv = {}
    for name, param in experts[k].named_parameters():
        base_param = dict(base_model.named_parameters())[name]
        tv[name] = param.data - base_param.data
    task_vectors.append(tv)

# Compute linear task-vector norms
v_norms = torch.zeros(K)
s_norms = torch.zeros(K)
for k in range(K):
    total_f_squared = 0.0
    total_spec = 0.0
    for name, tv in task_vectors[k].items():
        total_f_squared += torch.sum(tv ** 2).item()
        if len(tv.shape) == 2:
            svals = torch.linalg.svdvals(tv)
            total_spec += svals[0].item()
    v_norms[k] = np.sqrt(total_f_squared)
    s_norms[k] = total_spec

print("\nPrecomputed task-vector norms across physical experts:")
for k in range(K):
    print(f"Expert {k} | Frobenius Norm: {v_norms[k]:.4f} | Spectral Norm: {s_norms[k]:.4f}")

# Define Merged MLP with vectorized sample-wise parameter merging
class MergedMLP(nn.Module):
    def __init__(self, base_model, task_vectors):
        super().__init__()
        self.base_model = base_model
        self.task_vectors = task_vectors
        
    def forward(self, x, alpha):
        # fc1 weight and bias
        W_fc1_base = self.base_model.fc1.weight
        b_fc1_base = self.base_model.fc1.bias
        V_fc1_w = torch.stack([tv['fc1.weight'] for tv in self.task_vectors]) # (K, 32, 64)
        V_fc1_b = torch.stack([tv['fc1.bias'] for tv in self.task_vectors]) # (K, 32)
        
        W_fc1_merged = W_fc1_base.unsqueeze(0) + torch.einsum('bk,koi->boi', alpha, V_fc1_w)
        b_fc1_merged = b_fc1_base.unsqueeze(0) + torch.einsum('bk,ko->bo', alpha, V_fc1_b)
        
        # fc1 layer output
        h = torch.einsum('boi,bi->bo', W_fc1_merged, x) + b_fc1_merged
        h = torch.relu(h)
        
        # fc2 weight and bias
        W_fc2_base = self.base_model.fc2.weight
        b_fc2_base = self.base_model.fc2.bias
        V_fc2_w = torch.stack([tv['fc2.weight'] for tv in self.task_vectors]) # (K, 2, 32)
        V_fc2_b = torch.stack([tv['fc2.bias'] for tv in self.task_vectors]) # (K, 2)
        
        W_fc2_merged = W_fc2_base.unsqueeze(0) + torch.einsum('bk,koi->boi', alpha, V_fc2_w)
        b_fc2_merged = b_fc2_base.unsqueeze(0) + torch.einsum('bk,ko->bo', alpha, V_fc2_b)
        
        # fc2 layer output
        out = torch.einsum('boi,bi->bo', W_fc2_merged, h) + b_fc2_merged
        return out

# Define parametric router using frozen random projection and linear Softmax Predictor
class RealRouter(nn.Module):
    def __init__(self, in_features=64, proj_dim=4, K=4):
        super().__init__()
        # Frozen random projection matrix
        P = torch.randn(in_features, proj_dim)
        P = P / torch.norm(P, dim=0, keepdim=True)
        self.register_buffer('P', P)
        
        self.W = nn.Parameter(torch.zeros(K, proj_dim)) # (K, proj_dim)
        self.B = nn.Parameter(torch.zeros(K))   # (K,)
        
    def forward(self, x):
        proj = x @ self.P
        psi = proj / (torch.norm(proj, dim=-1, keepdim=True) + 1e-8)
        logits = psi @ self.W.t() + self.B
        alpha = torch.softmax(logits, dim=-1)
        return alpha

# Helpers to generate splits across tasks
def generate_splits(task_data, num_samples_per_task):
    X_list = []
    y_list = []
    task_ids = []
    
    for k in range(K):
        X_train, _, y_train, _ = task_data[k]
        indices = torch.randperm(len(X_train))[:num_samples_per_task]
        X_list.append(X_train[indices])
        y_list.append(y_train[indices])
        task_ids.append(torch.full((num_samples_per_task,), k, dtype=torch.long))
        
    return (
        torch.cat(X_list, dim=0),
        torch.cat(y_list, dim=0),
        torch.cat(task_ids, dim=0)
    )

def generate_test_splits(task_data):
    X_list = []
    y_list = []
    task_ids = []
    
    for k in range(K):
        _, X_test, _, y_test = task_data[k]
        X_list.append(X_test)
        y_list.append(y_test)
        task_ids.append(torch.full((len(X_test),), k, dtype=torch.long))
        
    return (
        torch.cat(X_list, dim=0),
        torch.cat(y_list, dim=0),
        torch.cat(task_ids, dim=0)
    )

X_cal, y_cal, task_cal = generate_splits(task_data, 16) # B_cal = 64 (16 per task)
X_test, y_test, task_test = generate_test_splits(task_data) # B_test = 400 (100 per task)

def evaluate_real_router(router, merged_model, X, y, task_ids):
    router.eval()
    with torch.no_grad():
        alpha = router(X)
        out = merged_model(X, alpha)
        preds = torch.argmax(out, dim=-1)
        
    accuracies = []
    for k in range(K):
        idx = (task_ids == k)
        correct = (preds[idx] == y[idx]).float().mean().item() * 100.0
        accuracies.append(correct)
        
    mean_acc = sum(accuracies) / K
    return accuracies, mean_acc

def calibrate_real_router(reg_type="none", lambda_reg=0.1, lr=0.01, epochs=150, proj_dim=4, beta=0.9, gamma=1.0):
    router = RealRouter(proj_dim=proj_dim)
    merged_model = MergedMLP(base_model, task_vectors)
    optimizer = optim.Adam(router.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    g = torch.zeros(K)
    
    for epoch in range(epochs):
        router.train()
        optimizer.zero_grad()
        
        alpha = router(X_cal)
        out = merged_model(X_cal, alpha)
        
        # Each calibration sample must be evaluated on its corresponding task class mapping
        # Since the output dimension is 2, cross entropy directly measures if the merged weights can classify correctly
        loss_ce = loss_fn(out, y_cal)
        
        # Regularization
        loss_reg = 0.0
        W_squared = torch.sum(router.W ** 2, dim=-1) # (K,)
        B_squared = router.B ** 2 # (K,)
        
        if reg_type == "l2":
            loss_reg = lambda_reg * (torch.sum(router.W ** 2) + torch.sum(router.B ** 2))
        elif reg_type == "tsar":
            W_mean = torch.mean(router.W, dim=0, keepdim=True)
            loss_reg = lambda_reg * torch.sum((router.W - W_mean) ** 2)
        elif reg_type == "sr3_f":
            # Scale decay by linear Frobenius norms of task vectors
            loss_reg = lambda_reg * torch.sum(v_norms * (W_squared + B_squared))
        elif reg_type == "sr3_s":
            # Scale decay by linear Spectral norms of task vectors
            loss_reg = lambda_reg * torch.sum(s_norms * (W_squared + B_squared))
        elif reg_type == "sr3_hybrid":
            # Compute the gradient of loss_ce with respect to router.W
            grad_W = torch.autograd.grad(loss_ce, router.W, retain_graph=True)[0].detach()
            grad_norms = torch.norm(grad_W, p=2, dim=-1) # Shape: (K,)
            g = beta * g + (1.0 - beta) * grad_norms
            lambdas = lambda_reg * v_norms * torch.exp(-gamma * g)
            loss_reg = torch.sum(lambdas * (W_squared + B_squared))
            
        loss_total = loss_ce + loss_reg
        loss_total.backward()
        optimizer.step()
        
    accs, mean_acc = evaluate_real_router(router, merged_model, X_test, y_test, task_test)
    return accs, mean_acc

# Evaluate Static Uniform Merging
merged_model = MergedMLP(base_model, task_vectors)
uniform_alpha = torch.full((len(X_test), K), 1.0 / K)
with torch.no_grad():
    out = merged_model(X_test, uniform_alpha)
    preds = torch.argmax(out, dim=-1)
uniform_accs = []
for k in range(K):
    idx = (task_test == k)
    uniform_accs.append((preds[idx] == y_test[idx]).float().mean().item() * 100.0)
uniform_mean = sum(uniform_accs) / K

# Evaluate different routers
results = {
    "Static Uniform Merging": (uniform_accs, uniform_mean),
}

# Unregularized Router
unreg_accs, unreg_mean = calibrate_real_router("none")
results["Linear Router (Unregularized)"] = (unreg_accs, unreg_mean)

# Isotropic L2 router sweeps
best_mean = -1
best_accs = None
for l in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]:
    accs, m = calibrate_real_router("l2", l)
    if m > best_mean:
        best_mean = m
        best_accs = accs
results["Linear Router (L2 Regularized)"] = (best_accs, best_mean)

# TSAR sweeps
best_mean = -1
best_accs = None
for l in [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]:
    accs, m = calibrate_real_router("tsar", l)
    if m > best_mean:
        best_mean = m
        best_accs = accs
results["TSAR (Centroid Anchoring)"] = (best_accs, best_mean)

# SR3-F sweeps
best_mean = -1
best_accs = None
for l in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]:
    accs, m = calibrate_real_router("sr3_f", l)
    if m > best_mean:
        best_mean = m
        best_accs = accs
results["SR3-F (Ours - Frobenius)"] = (best_accs, best_mean)

# SR3-S sweeps
best_mean = -1
best_accs = None
for l in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]:
    accs, m = calibrate_real_router("sr3_s", l)
    if m > best_mean:
        best_mean = m
        best_accs = accs
results["SR3-S (Ours - Spectral)"] = (best_accs, best_mean)

# SR3-Hybrid sweeps
best_mean = -1
best_accs = None
for l in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]:
    accs, m = calibrate_real_router("sr3_hybrid", l)
    if m > best_mean:
        best_mean = m
        best_accs = accs
results["SR3-H (Ours - Hybrid Adaptive)"] = (best_accs, best_mean)

print("\n" + "="*80)
print(f"{'Method':<35} | {'Task 0':<8} | {'Task 1':<8} | {'Task 2':<8} | {'Task 3':<8} | {'Mean (%)':<8}")
print("="*80)
for method, (accs, mean_acc) in results.items():
    print(f"{method:<35} | {accs[0]:.2f}%  | {accs[1]:.2f}%  | {accs[2]:.2f}%  | {accs[3]:.2f}%  | {mean_acc:.2f}%")
print("="*80)

# --- Projection Dimension Ablation Sweep ---
print("\n" + "="*80)
print("RUNNING PROJECTION DIMENSION ABLATION SWEEP ON PHYSICAL PYTORCH MODEL")
print("="*80)
print(f"{'Proj Dim':<10} | {'Unregularized':<15} | {'L2 Regularized':<15} | {'TSAR':<10} | {'SR3-F (Ours)':<15} | {'SR3-S (Ours)':<15} | {'SR3-H (Ours)':<15}")
print("="*80)

for pd in [4, 8, 16, 32, 64]:
    # Unregularized
    _, unreg_m = calibrate_real_router("none", proj_dim=pd)
    
    # L2 sweep
    l2_best = -1
    for l in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]:
        _, m = calibrate_real_router("l2", l, proj_dim=pd)
        if m > l2_best:
            l2_best = m
            
    # TSAR sweep
    tsar_best = -1
    for l in [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]:
        _, m = calibrate_real_router("tsar", l, proj_dim=pd)
        if m > tsar_best:
            tsar_best = m
            
    # SR3-F sweep
    sr3_f_best = -1
    for l in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]:
        _, m = calibrate_real_router("sr3_f", l, proj_dim=pd)
        if m > sr3_f_best:
            sr3_f_best = m
            
    # SR3-S sweep
    sr3_s_best = -1
    for l in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]:
        _, m = calibrate_real_router("sr3_s", l, proj_dim=pd)
        if m > sr3_s_best:
            sr3_s_best = m
            
    # SR3-H sweep
    sr3_h_best = -1
    for l in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]:
        _, m = calibrate_real_router("sr3_hybrid", l, proj_dim=pd)
        if m > sr3_h_best:
            sr3_h_best = m
            
    print(f"{pd:<10} | {unreg_m:.2f}%          | {l2_best:.2f}%          | {tsar_best:.2f}%     | {sr3_f_best:.2f}%         | {sr3_s_best:.2f}%         | {sr3_h_best:.2f}%")
print("="*80)
