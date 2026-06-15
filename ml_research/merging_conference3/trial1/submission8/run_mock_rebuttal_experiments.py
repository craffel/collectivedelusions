import os
import sys
import math
import copy
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Model Architecture
# ---------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def copy_model_structure(model):
    replica = SimpleMLP()
    replica.load_state_dict(model.state_dict())
    return replica

def robust_svd(A):
    # Convert PyTorch tensor to double-precision NumPy
    device = A.device
    dtype = A.dtype
    A_np = A.detach().cpu().numpy().astype(np.float64)
    try:
        U, S, Vh = np.linalg.svd(A_np, full_matrices=False)
    except Exception:
        A_np += 1e-6 * np.random.randn(*A_np.shape)
        U, S, Vh = np.linalg.svd(A_np, full_matrices=False)
    
    U_t = torch.from_numpy(U).to(device=device, dtype=dtype)
    S_t = torch.from_numpy(S).to(device=device, dtype=dtype)
    Vh_t = torch.from_numpy(Vh).to(device=device, dtype=dtype)
    return U_t, S_t, Vh_t

# ---------------------------------------------------------
# 2. Data Loading & Filtering
# ---------------------------------------------------------
def get_datasets():
    """Load Split-MNIST datasets with fallback."""
    try:
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        return train_dataset, test_dataset, False
    except Exception as e:
        print(f"[Data] Failed to load MNIST: {e}. Falling back to synthetic.")
        
    np.random.seed(42)
    X_train = np.random.randn(6000, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, size=(6000,)).astype(np.int64)
    X_test = np.random.randn(1000, 784).astype(np.float32)
    y_test = np.random.randint(0, 10, size=(1000,)).astype(np.int64)
    
    for i in range(10):
        X_train[y_train == i, i*20:(i+1)*20] += 5.0
        X_test[y_test == i, i*20:(i+1)*20] += 5.0
        
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    return train_dataset, test_dataset, True

def filter_split_dataset(dataset, classes):
    indices = []
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'tensors'):
        targets = dataset.tensors[1]
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]
        
    for idx, target in enumerate(targets):
        if int(target) in classes:
            indices.append(idx)
    return Subset(dataset, indices)

# ---------------------------------------------------------
# 3. Model Training under Different Constraints
# ---------------------------------------------------------
def train_model_standard(model, train_loader, epochs=3, lr=0.01, device="cpu", desc=""):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

def train_model_ortho(model, train_loader, epochs=3, lr=0.01, device="cpu", desc="", ortho_lambda=2.0):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Compute Orthogonal Regularization for all Linear weights
            reg_loss = 0.0
            for name, param in model.named_parameters():
                if len(param.shape) == 2 and "weight" in name:
                    out_d, in_d = param.shape
                    if out_d >= in_d:
                        prod = torch.matmul(param.t(), param)
                        I = torch.eye(in_d, device=param.device)
                    else:
                        prod = torch.matmul(param, param.t())
                        I = torch.eye(out_d, device=param.device)
                    reg_loss += torch.norm(prod - I, p='fro')
                    
            total_loss = loss + ortho_lambda * reg_loss
            total_loss.backward()
            optimizer.step()
    return model

def train_model_hard_ortho(model, train_loader, epochs=3, lr=0.01, device="cpu", desc=""):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Project weights back onto Stiefel/Orthogonal Manifold
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if len(param.shape) == 2 and "weight" in name:
                        U, S, Vh = robust_svd(param)
                        param.copy_(torch.matmul(U, Vh))
    return model

def eval_model(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total if total > 0 else 0.0

# ---------------------------------------------------------
# 4. Geometric Merging Operators & Schur Implementations
# ---------------------------------------------------------
def get_rotation_procrustes(W_k, W_0):
    A = torch.matmul(W_0.t(), W_k)
    U, S, Vh = robust_svd(A)
    R = torch.matmul(U, Vh)
    return R

def cayley_to_skew(R):
    I = torch.eye(R.shape[-1], device=R.device, dtype=R.dtype)
    Q = torch.linalg.solve(R + I, R - I)
    Q = 0.5 * (Q - Q.t())
    return Q

def cayley_from_skew(Q):
    I = torch.eye(Q.shape[-1], device=Q.device, dtype=Q.dtype)
    R = torch.linalg.solve(I - Q, I + Q)
    return R

def merge_cayley_Q_list(Q_list):
    Q_stack = torch.stack(Q_list, dim=0)
    N = Q_stack.shape[0]
    merged_sum = Q_stack.sum(dim=0)
    norms = torch.norm(Q_stack.view(N, -1), p='fro', dim=1)
    sum_of_norms = norms.sum()
    norm_of_sum = torch.norm(merged_sum, p='fro')
    c = sum_of_norms / (norm_of_sum + 1e-8)
    Q_com = (1.0 / N) * c * merged_sum
    Q_com = 0.5 * (Q_com - Q_com.t())
    return Q_com

# SVD spectral balancing & pruning
def rimo_spectral_balancing(Q_com, t=2.0):
    U, S, Vh = robust_svd(Q_com)
    mean_s = S.mean()
    S_balanced = mean_s + (S - mean_s) / math.sqrt(t)
    Q_balanced = torch.matmul(U * S_balanced.unsqueeze(0), Vh)
    Q_balanced = 0.5 * (Q_balanced - Q_balanced.t())
    return Q_balanced

def rimo_spectral_pruning(Q_com, keep_ratio=0.2):
    U, S, Vh = robust_svd(Q_com)
    k = max(2, int(len(S) * keep_ratio))
    if k % 2 != 0:
        k = min(len(S), k + 1)
    S_pruned = torch.zeros_like(S)
    S_pruned[:k] = S[:k]
    Q_pruned = torch.matmul(U * S_pruned.unsqueeze(0), Vh)
    Q_pruned = 0.5 * (Q_pruned - Q_pruned.t())
    return Q_pruned

# Schur-based spectral balancing & pruning
def schur_to_blocks(Q):
    Q_np = Q.detach().cpu().numpy()
    T, Z = scipy.linalg.schur(Q_np, output='real')
    return torch.from_numpy(T).to(Q.device).to(Q.dtype), torch.from_numpy(Z).to(Q.device).to(Q.dtype)

def schur_from_blocks(T, Z):
    return Z @ T @ Z.t()

def rimo_schur_balancing(Q_com, t=2.0):
    T, Z = schur_to_blocks(Q_com)
    d = T.shape[0]
    num_blocks = d // 2
    
    magnitudes = []
    for j in range(num_blocks):
        a_j = T[2 * j, 2 * j + 1]
        magnitudes.append(a_j)
        
    magnitudes = torch.tensor(magnitudes, device=Q_com.device, dtype=Q_com.dtype)
    abs_mags = torch.abs(magnitudes)
    mean_mag = abs_mags.mean()
    
    new_magnitudes = []
    for j in range(num_blocks):
        a_j = magnitudes[j]
        sign_aj = torch.sign(a_j)
        abs_aj = torch.abs(a_j)
        new_abs = mean_mag + (abs_aj - mean_mag) / math.sqrt(t)
        new_aj = sign_aj * new_abs
        new_magnitudes.append(new_aj)
        
    T_balanced = torch.zeros_like(T)
    for j in range(num_blocks):
        new_aj = new_magnitudes[j]
        T_balanced[2 * j, 2 * j + 1] = new_aj
        T_balanced[2 * j + 1, 2 * j] = -new_aj
        
    Q_balanced = schur_from_blocks(T_balanced, Z)
    return Q_balanced

def rimo_schur_pruning(Q_com, keep_ratio=0.2):
    T, Z = schur_to_blocks(Q_com)
    d = T.shape[0]
    num_blocks = d // 2
    
    magnitudes = []
    for j in range(num_blocks):
        a_j = T[2 * j, 2 * j + 1]
        magnitudes.append((torch.abs(a_j).item(), a_j, j))
        
    magnitudes.sort(key=lambda x: x[0], reverse=True)
    k = max(1, int(num_blocks * keep_ratio))
    
    T_pruned = torch.zeros_like(T)
    for idx, (abs_val, a_j, original_j) in enumerate(magnitudes):
        if idx < k:
            T_pruned[2*original_j, 2*original_j+1] = a_j
            T_pruned[2*original_j+1, 2*original_j] = -a_j
            
    Q_pruned = schur_from_blocks(T_pruned, Z)
    return Q_pruned

# Complex Hermitian Eigen-decomposition based balancing & pruning
def rimo_complex_balancing(Q_com, t=2.0):
    iQ = 1j * Q_com
    L, U = torch.linalg.eigh(iQ)
    abs_L = torch.abs(L)
    mean_L = abs_L.mean()
    L_mod = torch.sign(L) * (mean_L + (abs_L - mean_L) / math.sqrt(t))
    L_mod_complex = L_mod.to(U.dtype)
    iQ_mod = U @ torch.diag(L_mod_complex) @ U.conj().T
    Q_mod = (-1j * iQ_mod).real
    Q_mod = 0.5 * (Q_mod - Q_mod.T)
    return Q_mod

def rimo_complex_pruning(Q_com, keep_ratio=0.2):
    iQ = 1j * Q_com
    L, U = torch.linalg.eigh(iQ)
    abs_L = torch.abs(L)
    d = L.shape[0]
    _, indices = torch.sort(abs_L, descending=True)
    k = int(d * keep_ratio)
    if k % 2 != 0:
        k = min(d, k + 1)
    top_indices = indices[:k]
    L_mod = torch.zeros_like(L)
    L_mod[top_indices] = L[top_indices]
    L_mod_complex = L_mod.to(U.dtype)
    iQ_mod = U @ torch.diag(L_mod_complex) @ U.conj().T
    Q_mod = (-1j * iQ_mod).real
    Q_mod = 0.5 * (Q_mod - Q_mod.T)
    return Q_mod

# ---------------------------------------------------------
# 5. Merging Wrapper Functions
# ---------------------------------------------------------
def merge_models_task_arithmetic(models_list, base_model, scaling_factor=0.5):
    merged = copy_model_structure(base_model)
    sd_merged = merged.state_dict()
    sd_base = base_model.state_dict()
    sd_experts = [m.state_dict() for m in models_list]
    for key in sd_base.keys():
        if sd_base[key].dtype.is_floating_point:
            deltas = [sd_exp[key] - sd_base[key] for sd_exp in sd_experts]
            sd_merged[key] = sd_base[key] + scaling_factor * torch.stack(deltas, dim=0).mean(dim=0)
    merged.load_state_dict(sd_merged)
    return merged

def merge_models_orthomerge(models_list, base_model, residual_scale=1.0):
    merged = copy_model_structure(base_model)
    sd_merged = merged.state_dict()
    sd_base = base_model.state_dict()
    sd_experts = [m.state_dict() for m in models_list]
    
    for key in sd_base.keys():
        W_0 = sd_base[key]
        if W_0.dtype.is_floating_point and len(W_0.shape) == 2:
            W_experts = [sd_exp[key] for sd_exp in sd_experts]
            R_list = []
            rho_list = []
            for W_k in W_experts:
                R_k = get_rotation_procrustes(W_k, W_0)
                rho_k = W_k - torch.matmul(W_0, R_k)
                R_list.append(R_k)
                rho_list.append(rho_k)
            Q_list = [cayley_to_skew(R) for R in R_list]
            Q_com = merge_cayley_Q_list(Q_list)
            R_merged = cayley_from_skew(Q_com)
            rho_merged = torch.stack(rho_list, dim=0).mean(dim=0)
            sd_merged[key] = torch.matmul(W_0, R_merged) + residual_scale * rho_merged
        else:
            if W_0.dtype.is_floating_point:
                sd_merged[key] = torch.stack([sd_exp[key] for sd_exp in sd_experts], dim=0).mean(dim=0)
    merged.load_state_dict(sd_merged)
    return merged

def merge_models_saim(models_list, base_model, t=2.0, scaling_factor=0.5):
    merged = copy_model_structure(base_model)
    sd_merged = merged.state_dict()
    sd_base = base_model.state_dict()
    sd_experts = [m.state_dict() for m in models_list]
    for key in sd_base.keys():
        W_0 = sd_base[key]
        if W_0.dtype.is_floating_point and len(W_0.shape) == 2:
            deltas = [sd_exp[key] - W_0 for sd_exp in sd_experts]
            avg_delta = torch.stack(deltas, dim=0).mean(dim=0)
            U, S, Vh = robust_svd(avg_delta)
            mean_s = S.mean()
            S_balanced = mean_s + (S - mean_s) / math.sqrt(t)
            avg_delta_balanced = torch.matmul(U * S_balanced.unsqueeze(0), Vh)
            sd_merged[key] = W_0 + scaling_factor * avg_delta_balanced
        else:
            if W_0.dtype.is_floating_point:
                sd_merged[key] = torch.stack([sd_exp[key] for sd_exp in sd_experts], dim=0).mean(dim=0)
    merged.load_state_dict(sd_merged)
    return merged

def merge_models_rimo(models_list, base_model, t=2.0, residual_scale=1.0, op_type="svd_balanced", keep_ratio=0.2):
    merged = copy_model_structure(base_model)
    sd_merged = merged.state_dict()
    sd_base = base_model.state_dict()
    sd_experts = [m.state_dict() for m in models_list]
    
    for key in sd_base.keys():
        W_0 = sd_base[key]
        if W_0.dtype.is_floating_point and len(W_0.shape) == 2:
            W_experts = [sd_exp[key] for sd_exp in sd_experts]
            R_list = []
            rho_list = []
            for W_k in W_experts:
                R_k = get_rotation_procrustes(W_k, W_0)
                rho_k = W_k - torch.matmul(W_0, R_k)
                R_list.append(R_k)
                rho_list.append(rho_k)
            Q_list = [cayley_to_skew(R) for R in R_list]
            Q_com = merge_cayley_Q_list(Q_list)
            
            if op_type == "svd_balanced":
                Q_mod = rimo_spectral_balancing(Q_com, t=t)
            elif op_type == "svd_pruned":
                Q_mod = rimo_spectral_pruning(Q_com, keep_ratio=keep_ratio)
            elif op_type == "schur_balanced":
                Q_mod = rimo_schur_balancing(Q_com, t=t)
            elif op_type == "schur_pruned":
                Q_mod = rimo_schur_pruning(Q_com, keep_ratio=keep_ratio)
            elif op_type == "complex_balanced":
                Q_mod = rimo_complex_balancing(Q_com, t=t)
            elif op_type == "complex_pruned":
                Q_mod = rimo_complex_pruning(Q_com, keep_ratio=keep_ratio)
            else:
                Q_mod = Q_com
                
            R_merged = cayley_from_skew(Q_mod)
            rho_merged = torch.stack(rho_list, dim=0).mean(dim=0)
            sd_merged[key] = torch.matmul(W_0, R_merged) + residual_scale * rho_merged
        else:
            if W_0.dtype.is_floating_point:
                sd_merged[key] = torch.stack([sd_exp[key] for sd_exp in sd_experts], dim=0).mean(dim=0)
    merged.load_state_dict(sd_merged)
    return merged

def merge_models_adamerging(models_list, base_model, test_loader, device="cpu", epochs=20, lr=5e-2):
    N_tasks = len(models_list)
    alphas = nn.Parameter(torch.ones(N_tasks, device=device) * 0.5)
    
    inputs_list = []
    max_samples = 512
    samples_count = 0
    for inputs, _ in test_loader:
        inputs_list.append(inputs)
        samples_count += inputs.size(0)
        if samples_count >= max_samples:
            break
    inputs_all = torch.cat(inputs_list, dim=0).to(device)
    
    optimizer = optim.Adam([alphas], lr=lr)
    softmax = nn.Softmax(dim=1)
    
    sd_base = base_model.state_dict()
    sd_experts = [m.state_dict() for m in models_list]
    
    class FunctionalMLP(nn.Module):
        def __init__(self, base_sd, expert_sds, num_tasks):
            super().__init__()
            self.base_sd = base_sd
            self.expert_sds = expert_sds
            self.num_tasks = num_tasks
            
        def forward(self, x, alphas):
            x = x.view(x.size(0), -1)
            weights = {}
            for key in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]:
                w_0 = self.base_sd[key]
                deltas = [self.expert_sds[i][key] - w_0 for i in range(self.num_tasks)]
                w_merged = w_0 + sum(alphas[i] * deltas[i] for i in range(self.num_tasks))
                weights[key] = w_merged
                
            x = torch.nn.functional.linear(x, weights["fc1.weight"], weights["fc1.bias"])
            x = torch.relu(x)
            x = torch.nn.functional.linear(x, weights["fc2.weight"], weights["fc2.bias"])
            x = torch.relu(x)
            x = torch.nn.functional.linear(x, weights["fc3.weight"], weights["fc3.bias"])
            return x

    func_model = FunctionalMLP(sd_base, sd_experts, N_tasks).to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = func_model(inputs_all, alphas)
        probs = softmax(outputs)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()
        reg = torch.sum((alphas - 0.5) ** 2) * 0.01
        loss = entropy + reg
        loss.backward()
        optimizer.step()
        
    final_alphas = alphas.detach()
    merged = copy_model_structure(base_model)
    sd_merged = merged.state_dict()
    for key in sd_base.keys():
        if sd_base[key].dtype.is_floating_point:
            deltas = [sd_experts[i][key] - sd_base[key] for i in range(N_tasks)]
            sd_merged[key] = sd_base[key] + sum(final_alphas[i] * deltas[i] for i in range(N_tasks))
        else:
            sd_merged[key] = sd_base[key]
    merged.load_state_dict(sd_merged)
    return merged

# ---------------------------------------------------------
# 6. Run Rebuttal Experiments
# ---------------------------------------------------------
def run_rebuttal():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Rebuttal Experiments on: {device}")
    
    train_dataset, test_dataset, is_synthetic = get_datasets()
    
    classes_t1 = [0, 1, 2, 3, 4]
    classes_t2 = [5, 6, 7, 8, 9]
    
    train_t1 = filter_split_dataset(train_dataset, classes_t1)
    train_t1 = Subset(train_t1, list(range(min(1000, len(train_t1)))))
    train_t2 = filter_split_dataset(train_dataset, classes_t2)
    train_t2 = Subset(train_t2, list(range(min(1000, len(train_t2)))))
    test_t1  = filter_split_dataset(test_dataset, classes_t1)
    test_t2  = filter_split_dataset(test_dataset, classes_t2)
    
    loader_train_t1 = DataLoader(train_t1, batch_size=64, shuffle=True)
    loader_train_t2 = DataLoader(train_t2, batch_size=64, shuffle=True)
    loader_test_t1  = DataLoader(test_t1, batch_size=64, shuffle=False)
    loader_test_t2  = DataLoader(test_t2, batch_size=64, shuffle=False)
    
    # 1. Pretrain base model (mixture)
    mix_train_indices = list(range(0, len(train_dataset), 10))
    mix_train = Subset(train_dataset, mix_train_indices)
    mix_train = Subset(mix_train, list(range(min(500, len(mix_train)))))
    loader_mix = DataLoader(mix_train, batch_size=64, shuffle=True)
    
    all_runs = {}
    
    # Run three training environments
    for env in ["standard", "soft_ortho", "hard_ortho"]:
        print(f"\n--- Training Environment: {env.upper()} ---")
        
        base_model = SimpleMLP()
        expert1 = SimpleMLP()
        expert2 = SimpleMLP()
        
        if env == "standard":
            base_model = train_model_standard(base_model, loader_mix, epochs=2, lr=0.01, device=device)
            expert1.load_state_dict(base_model.state_dict())
            expert1 = train_model_standard(expert1, loader_train_t1, epochs=3, lr=0.005, device=device)
            expert2.load_state_dict(base_model.state_dict())
            expert2 = train_model_standard(expert2, loader_train_t2, epochs=3, lr=0.005, device=device)
        elif env == "soft_ortho":
            base_model = train_model_ortho(base_model, loader_mix, epochs=2, lr=0.01, device=device, ortho_lambda=2.0)
            expert1.load_state_dict(base_model.state_dict())
            expert1 = train_model_ortho(expert1, loader_train_t1, epochs=3, lr=0.005, device=device, ortho_lambda=2.0)
            expert2.load_state_dict(base_model.state_dict())
            expert2 = train_model_ortho(expert2, loader_train_t2, epochs=3, lr=0.005, device=device, ortho_lambda=2.0)
        elif env == "hard_ortho":
            base_model = train_model_hard_ortho(base_model, loader_mix, epochs=2, lr=0.01, device=device)
            expert1.load_state_dict(base_model.state_dict())
            expert1 = train_model_hard_ortho(expert1, loader_train_t1, epochs=3, lr=0.005, device=device)
            expert2.load_state_dict(base_model.state_dict())
            expert2 = train_model_hard_ortho(expert2, loader_train_t2, epochs=3, lr=0.005, device=device)
            
        acc_exp1_t1 = eval_model(expert1, loader_test_t1, device=device)
        acc_exp1_t2 = eval_model(expert1, loader_test_t2, device=device)
        acc_exp2_t1 = eval_model(expert2, loader_test_t1, device=device)
        acc_exp2_t2 = eval_model(expert2, loader_test_t2, device=device)
        print(f"Expert 1 -> T1: {acc_exp1_t1:.4f}, T2: {acc_exp1_t2:.4f}")
        print(f"Expert 2 -> T1: {acc_exp2_t1:.4f}, T2: {acc_exp2_t2:.4f}")
        
        # Test merging strategies
        env_results = {}
        
        # Merging methods dictionary
        merges = {
            "Task Arithmetic (s=0.5)": lambda: merge_models_task_arithmetic([expert1, expert2], base_model, scaling_factor=0.5),
            "AdaMerging (Adaptive)": lambda: merge_models_adamerging([expert1, expert2], base_model, loader_test_t1, device=device),
            "OrthoMerge (res=1.0)": lambda: merge_models_orthomerge([expert1, expert2], base_model, residual_scale=1.0),
            "OrthoMerge (res=0.2)": lambda: merge_models_orthomerge([expert1, expert2], base_model, residual_scale=0.2),
            "SAIM (t=2.0, s=0.5)": lambda: merge_models_saim([expert1, expert2], base_model, t=2.0, scaling_factor=0.5),
            "RIMO SVD-Balanced (t=2.0, res=1.0)": lambda: merge_models_rimo([expert1, expert2], base_model, t=2.0, residual_scale=1.0, op_type="svd_balanced"),
            "RIMO SVD-Balanced (t=2.0, res=0.2)": lambda: merge_models_rimo([expert1, expert2], base_model, t=2.0, residual_scale=0.2, op_type="svd_balanced"),
            "RIMO SVD-Pruned (keep=0.2, res=1.0)": lambda: merge_models_rimo([expert1, expert2], base_model, keep_ratio=0.2, residual_scale=1.0, op_type="svd_pruned"),
            "RIMO SVD-Pruned (keep=0.2, res=0.2)": lambda: merge_models_rimo([expert1, expert2], base_model, keep_ratio=0.2, residual_scale=0.2, op_type="svd_pruned"),
            "RIMO Schur-Balanced (t=2.0, res=1.0)": lambda: merge_models_rimo([expert1, expert2], base_model, t=2.0, residual_scale=1.0, op_type="schur_balanced"),
            "RIMO Schur-Balanced (t=2.0, res=0.2)": lambda: merge_models_rimo([expert1, expert2], base_model, t=2.0, residual_scale=0.2, op_type="schur_balanced"),
            "RIMO Schur-Pruned (keep=0.2, res=1.0)": lambda: merge_models_rimo([expert1, expert2], base_model, keep_ratio=0.2, residual_scale=1.0, op_type="schur_pruned"),
            "RIMO Schur-Pruned (keep=0.2, res=0.2)": lambda: merge_models_rimo([expert1, expert2], base_model, keep_ratio=0.2, residual_scale=0.2, op_type="schur_pruned"),
            "RIMO Complex-Balanced (t=2.0, res=1.0)": lambda: merge_models_rimo([expert1, expert2], base_model, t=2.0, residual_scale=1.0, op_type="complex_balanced"),
            "RIMO Complex-Balanced (t=2.0, res=0.2)": lambda: merge_models_rimo([expert1, expert2], base_model, t=2.0, residual_scale=0.2, op_type="complex_balanced"),
            "RIMO Complex-Pruned (keep=0.2, res=1.0)": lambda: merge_models_rimo([expert1, expert2], base_model, keep_ratio=0.2, residual_scale=1.0, op_type="complex_pruned"),
            "RIMO Complex-Pruned (keep=0.2, res=0.2)": lambda: merge_models_rimo([expert1, expert2], base_model, keep_ratio=0.2, residual_scale=0.2, op_type="complex_pruned"),
        }
        
        for name, fn in merges.items():
            model_merged = fn()
            acc_t1 = eval_model(model_merged, loader_test_t1, device=device)
            acc_t2 = eval_model(model_merged, loader_test_t2, device=device)
            avg = (acc_t1 + acc_t2) / 2.0
            env_results[name] = {"t1": acc_t1, "t2": acc_t2, "avg": avg}
            print(f"  {name:40s} -> T1: {acc_t1:.4f}, T2: {acc_t2:.4f}, Avg: {avg:.4f}")
            
        all_runs[env] = env_results
        
    # Latency comparison
    print("\n--- Latency Comparison: SVD vs Schur (Dimension 256x256) ---")
    Q_bench = torch.randn(256, 256)
    Q_bench = 0.5 * (Q_bench - Q_bench.t()).to(device)
    
    # 1. Measure SVD Sequential Latency
    start_time = time.time()
    for _ in range(50):
        U, S, Vh = robust_svd(Q_bench)
        mean_s = S.mean()
        S_balanced = mean_s + (S - mean_s) / math.sqrt(2.0)
        Q_mod = torch.matmul(U * S_balanced.unsqueeze(0), Vh)
        Q_mod = 0.5 * (Q_mod - Q_mod.t())
    svd_latency = (time.time() - start_time) / 50 * 1000 # ms
    
    # 2. Measure Schur Sequential Latency
    start_time = time.time()
    for _ in range(50):
        T, Z = schur_to_blocks(Q_bench)
        d = T.shape[0]
        num_blocks = d // 2
        
        magnitudes = []
        for j in range(num_blocks):
            a_j = T[2 * j, 2 * j + 1]
            magnitudes.append(a_j)
            
        magnitudes = torch.tensor(magnitudes, device=Q_bench.device, dtype=Q_bench.dtype)
        abs_mags = torch.abs(magnitudes)
        mean_mag = abs_mags.mean()
        
        new_magnitudes = []
        for j in range(num_blocks):
            a_j = magnitudes[j]
            sign_aj = torch.sign(a_j)
            abs_aj = torch.abs(a_j)
            new_abs = mean_mag + (abs_aj - mean_mag) / math.sqrt(2.0)
            new_aj = sign_aj * new_abs
            new_magnitudes.append(new_aj)
            
        T_balanced = torch.zeros_like(T)
        for j in range(num_blocks):
            new_aj = new_magnitudes[j]
            T_balanced[2 * j, 2 * j + 1] = new_aj
            T_balanced[2 * j + 1, 2 * j] = -new_aj
            
        Q_balanced = schur_from_blocks(T_balanced, Z)
    schur_latency = (time.time() - start_time) / 50 * 1000 # ms
    
    # 3. Measure Complex Sequential Latency
    start_time = time.time()
    for _ in range(50):
        Q_mod = rimo_complex_balancing(Q_bench, t=2.0)
    complex_latency = (time.time() - start_time) / 50 * 1000 # ms
    
    print(f"SVD Latency: {svd_latency:.2f} ms")
    print(f"Schur Latency: {schur_latency:.2f} ms")
    print(f"Complex Latency: {complex_latency:.2f} ms")
    
    latency_results = {
        "svd": svd_latency,
        "schur": schur_latency,
        "complex": complex_latency
    }
    
    # Save the output
    final_output = {
        "runs": all_runs,
        "latency": latency_results
    }
    
    with open("results_mock_rebuttal.json", "w") as f:
        json.dump(final_output, f, indent=2)
        
    print("\nRebuttal Experiments complete! Results saved in results_mock_rebuttal.json.")

if __name__ == "__main__":
    run_rebuttal()
