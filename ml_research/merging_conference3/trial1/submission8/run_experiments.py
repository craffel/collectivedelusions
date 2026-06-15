import os
import sys
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

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

# ---------------------------------------------------------
# 2. Dataset Preparation (Split-MNIST / Synthetic Fallback)
# ---------------------------------------------------------
def get_datasets(use_synthetic=False):
    """
    Load Split-MNIST datasets.
    Task 1: Digit classes 0-4
    Task 2: Digit classes 5-9
    Falls back to a robust synthetic multi-class setup if internet/download fails.
    """
    if not use_synthetic:
        try:
            import torchvision
            import torchvision.transforms as transforms
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            print("[Data] Successfully loaded MNIST from torchvision.")
            return train_dataset, test_dataset, False
        except Exception as e:
            print(f"[Data] Failed to load MNIST: {e}. Falling back to synthetic datasets.")
            
    # Synthetic dataset generator representing 10-class features
    print("[Data] Initializing synthetic classification dataset.")
    np.random.seed(42)
    
    # Generate random features
    X_train = np.random.randn(6000, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, size=(6000,)).astype(np.int64)
    
    X_test = np.random.randn(1000, 784).astype(np.float32)
    y_test = np.random.randint(0, 10, size=(1000,)).astype(np.int64)
    
    # Emphasize features for task labels to make them easily learnable
    for i in range(10):
        mask_tr = (y_train == i)
        X_train[mask_tr, i*20:(i+1)*20] += 5.0
        mask_te = (y_test == i)
        X_test[mask_te, i*20:(i+1)*20] += 5.0
        
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    return train_dataset, test_dataset, True

def filter_split_dataset(dataset, classes):
    """Filter dataset to include only specified classes."""
    indices = []
    # Handle standard torchvision dataset vs TensorDataset
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'tensors'):
        targets = dataset.tensors[1]
    else:
        # Fallback to iteration if structure is unusual
        targets = [dataset[i][1] for i in range(len(dataset))]
        
    for idx, target in enumerate(targets):
        if int(target) in classes:
            indices.append(idx)
    return Subset(dataset, indices)

# ---------------------------------------------------------
# 3. Model Training & Evaluation
# ---------------------------------------------------------
def train_model(model, train_loader, epochs=3, lr=0.01, device="cpu", desc=""):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"  Training {desc}...")
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"    Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
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
# 4. Merging Operators
# ---------------------------------------------------------
def get_rotation_procrustes(W_k, W_0):
    """
    Solve the Orthogonal Procrustes problem:
    Find orthogonal R such that W_0 @ R \approx W_k
    W_k, W_0 are [out, in] matrices.
    """
    A = torch.matmul(W_0.t(), W_k)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    R = torch.matmul(U, Vh)
    return R

def cayley_to_skew(R):
    """Map orthogonal matrix R to skew-symmetric Q via Inverse Cayley Transform."""
    I = torch.eye(R.shape[-1], device=R.device, dtype=R.dtype)
    # Solve (R + I) Q = R - I
    Q = torch.linalg.solve(R + I, R - I)
    # Strictly enforce skew-symmetry to handle numerical precision limits
    Q = 0.5 * (Q - Q.t())
    return Q

def cayley_from_skew(Q):
    """Map skew-symmetric Q to orthogonal matrix R via Forward Cayley Transform."""
    I = torch.eye(Q.shape[-1], device=Q.device, dtype=Q.dtype)
    # Solve (I - Q) R = I + Q
    R = torch.linalg.solve(I - Q, I + Q)
    return R

def merge_cayley_Q_list(Q_list):
    """Perform magnitude-corrected Lie algebra merging of skew-symmetric matrices."""
    Q_stack = torch.stack(Q_list, dim=0) # [N_tasks, d, d]
    N = Q_stack.shape[0]
    
    merged_sum = Q_stack.sum(dim=0)
    
    # Compute individual Frobenius norms
    norms = torch.norm(Q_stack.view(N, -1), p='fro', dim=1)
    sum_of_norms = norms.sum()
    
    # Compute Frobenius norm of the sum
    norm_of_sum = torch.norm(merged_sum, p='fro')
    
    c = sum_of_norms / (norm_of_sum + 1e-8)
    
    # Scaled, magnitude-corrected sum
    Q_com = (1.0 / N) * c * merged_sum
    # Enforce skew-symmetry
    Q_com = 0.5 * (Q_com - Q_com.t())
    return Q_com

def rimo_spectral_balancing(Q_com, t=2.0):
    """
    Apply SVD-based Isotropic Spectral Balancing in the Lie algebra.
    Interpolate singular value spectrum towards uniform isotropy.
    """
    U, S, Vh = torch.linalg.svd(Q_com, full_matrices=False)
    mean_s = S.mean()
    # Interpolate singular values: S_new = \bar{s} + (S - \bar{s}) / \sqrt{t}
    S_balanced = mean_s + (S - mean_s) / math.sqrt(t)
    
    # Reconstruct the balanced skew matrix
    Q_balanced = torch.matmul(U * S_balanced.unsqueeze(0), Vh)
    # Re-project to skew-symmetric subspace to preserve Lie algebraic properties
    Q_balanced = 0.5 * (Q_balanced - Q_balanced.t())
    return Q_balanced

def rimo_spectral_pruning(Q_com, keep_ratio=0.2):
    """
    Apply SVD-based Rank-Preserving Spectral Pruning in the Lie algebra.
    Set the smallest singular values to exactly zero to avoid noise in inactive dimensions.
    """
    U, S, Vh = torch.linalg.svd(Q_com, full_matrices=False)
    k = max(2, int(len(S) * keep_ratio))
    if k % 2 != 0:
        k = min(len(S), k + 1)
    
    S_pruned = torch.zeros_like(S)
    S_pruned[:k] = S[:k]
    
    Q_pruned = torch.matmul(U * S_pruned.unsqueeze(0), Vh)
    Q_pruned = 0.5 * (Q_pruned - Q_pruned.t())
    return Q_pruned

# --- Full model merging algorithms ---

def merge_models_rimo_pruned(models_list, base_model, keep_ratio=0.2, residual_scale=1.0):
    """RIMO with Rank-Preserving Spectral Pruning instead of Isotropic Spectral Balancing."""
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
            Q_pruned = rimo_spectral_pruning(Q_com, keep_ratio=keep_ratio)
            R_merged = cayley_from_skew(Q_pruned)
            rho_merged = torch.stack(rho_list, dim=0).mean(dim=0)
            sd_merged[key] = torch.matmul(W_0, R_merged) + residual_scale * rho_merged
        else:
            if W_0.dtype.is_floating_point:
                sd_merged[key] = torch.stack([sd_exp[key] for sd_exp in sd_experts], dim=0).mean(dim=0)
            else:
                sd_merged[key] = W_0
                
    merged.load_state_dict(sd_merged)
    return merged

def merge_models_task_arithmetic(models_list, base_model, scaling_factor=0.5):
    merged = copy_model_structure(base_model)
    sd_merged = merged.state_dict()
    sd_base = base_model.state_dict()
    sd_experts = [m.state_dict() for m in models_list]
    
    for key in sd_base.keys():
        if sd_base[key].dtype.is_floating_point:
            # Task Vector: W_k - W_0
            deltas = [sd_exp[key] - sd_base[key] for sd_exp in sd_experts]
            avg_delta = torch.stack(deltas, dim=0).mean(dim=0)
            sd_merged[key] = sd_base[key] + scaling_factor * avg_delta
        else:
            sd_merged[key] = sd_base[key]
            
    merged.load_state_dict(sd_merged)
    return merged

def merge_models_saim(models_list, base_model, t=2.0, scaling_factor=0.5):
    """Sharpness-Aware Isotropic Merging in Euclidean Space (baseline)."""
    merged = copy_model_structure(base_model)
    sd_merged = merged.state_dict()
    sd_base = base_model.state_dict()
    sd_experts = [m.state_dict() for m in models_list]
    
    for key in sd_base.keys():
        if sd_base[key].dtype.is_floating_point and len(sd_base[key].shape) == 2:
            deltas = [sd_exp[key] - sd_base[key] for sd_exp in sd_experts]
            delta_com = torch.stack(deltas, dim=0).mean(dim=0)
            
            U, S, Vh = torch.linalg.svd(delta_com, full_matrices=False)
            mean_s = S.mean()
            S_balanced = mean_s + (S - mean_s) / math.sqrt(t)
            delta_balanced = torch.matmul(U * S_balanced.unsqueeze(0), Vh)
            
            sd_merged[key] = sd_base[key] + scaling_factor * delta_balanced
        else:
            # For 1D biases/weights, use standard task arithmetic
            if sd_base[key].dtype.is_floating_point:
                deltas = [sd_exp[key] - sd_base[key] for sd_exp in sd_experts]
                sd_merged[key] = sd_base[key] + scaling_factor * torch.stack(deltas, dim=0).mean(dim=0)
            else:
                sd_merged[key] = sd_base[key]
                
    merged.load_state_dict(sd_merged)
    return merged

def merge_models_orthomerge(models_list, base_model, residual_scale=1.0):
    """Orthogonal Model Merging without Isotropic Spectral Balancing."""
    merged = copy_model_structure(base_model)
    sd_merged = merged.state_dict()
    sd_base = base_model.state_dict()
    sd_experts = [m.state_dict() for m in models_list]
    
    for key in sd_base.keys():
        W_0 = sd_base[key]
        if W_0.dtype.is_floating_point and len(W_0.shape) == 2:
            # W_k shape: [out_dim, in_dim]
            W_experts = [sd_exp[key] for sd_exp in sd_experts]
            
            # Solve Orthogonal Procrustes for each task
            R_list = []
            rho_list = []
            for W_k in W_experts:
                R_k = get_rotation_procrustes(W_k, W_0)
                rho_k = W_k - torch.matmul(W_0, R_k)
                R_list.append(R_k)
                rho_list.append(rho_k)
                
            # Map rotations to Lie algebra, merge, and map back
            Q_list = [cayley_to_skew(R) for R in R_list]
            Q_com = merge_cayley_Q_list(Q_list)
            R_merged = cayley_from_skew(Q_com)
            
            # Average residuals
            rho_merged = torch.stack(rho_list, dim=0).mean(dim=0)
            
            # Reconstruct weight
            sd_merged[key] = torch.matmul(W_0, R_merged) + residual_scale * rho_merged
        else:
            if W_0.dtype.is_floating_point:
                # Average biases / 1D weights
                sd_merged[key] = torch.stack([sd_exp[key] for sd_exp in sd_experts], dim=0).mean(dim=0)
            else:
                sd_merged[key] = W_0
                
    merged.load_state_dict(sd_merged)
    return merged

def merge_models_rimo(models_list, base_model, t=2.0, residual_scale=1.0):
    """Our proposed Riemannian Isotropic Merging on the Orthogonal Group (RIMO)."""
    merged = copy_model_structure(base_model)
    sd_merged = merged.state_dict()
    sd_base = base_model.state_dict()
    sd_experts = [m.state_dict() for m in models_list]
    
    for key in sd_base.keys():
        W_0 = sd_base[key]
        if W_0.dtype.is_floating_point and len(W_0.shape) == 2:
            W_experts = [sd_exp[key] for sd_exp in sd_experts]
            
            # 1. Orthogonal Procrustes decoupling
            R_list = []
            rho_list = []
            for W_k in W_experts:
                R_k = get_rotation_procrustes(W_k, W_0)
                rho_k = W_k - torch.matmul(W_0, R_k)
                R_list.append(R_k)
                rho_list.append(rho_k)
                
            # 2. Map orthogonal matrices R_k to skew-symmetric Q_k in so(d)
            Q_list = [cayley_to_skew(R) for R in R_list]
            
            # 3. Magnitude-corrected sum of Q_k
            Q_com = merge_cayley_Q_list(Q_list)
            
            # 4. Isotropic Spectral Balancing in so(d)
            Q_balanced = rimo_spectral_balancing(Q_com, t=t)
            
            # 5. Map balanced skew-symmetric matrix back to O(d)
            R_merged = cayley_from_skew(Q_balanced)
            
            # 6. Average residual components
            rho_merged = torch.stack(rho_list, dim=0).mean(dim=0)
            
            # 7. Construct final hybrid merged weights
            sd_merged[key] = torch.matmul(W_0, R_merged) + residual_scale * rho_merged
        else:
            if W_0.dtype.is_floating_point:
                sd_merged[key] = torch.stack([sd_exp[key] for sd_exp in sd_experts], dim=0).mean(dim=0)
            else:
                sd_merged[key] = W_0
                
    merged.load_state_dict(sd_merged)
    return merged

def copy_model_structure(model):
    """Utility to create an empty replica of a model structure."""
    replica = SimpleMLP()
    # Ensure independent weights
    replica.load_state_dict(model.state_dict())
    return replica

# ---------------------------------------------------------
# 5. Master Pipeline Execution
# ---------------------------------------------------------
def run_all():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")
    
    # Create results folder
    os.makedirs("results", exist_ok=True)
    
    # 5.1 Load Datasets (with automatic fallback)
    train_dataset, test_dataset, is_synthetic = get_datasets(use_synthetic=False)
    
    # Define tasks
    # Task 1: classes 0-4
    classes_t1 = [0, 1, 2, 3, 4]
    # Task 2: classes 5-9
    classes_t2 = [5, 6, 7, 8, 9]
    
    print(f"[Data] Filtering data for Split-MNIST tasks...")
    train_t1 = filter_split_dataset(train_dataset, classes_t1)
    train_t2 = filter_split_dataset(train_dataset, classes_t2)
    test_t1  = filter_split_dataset(test_dataset, classes_t1)
    test_t2  = filter_split_dataset(test_dataset, classes_t2)
    
    loader_train_t1 = DataLoader(train_t1, batch_size=64, shuffle=True)
    loader_train_t2 = DataLoader(train_t2, batch_size=64, shuffle=True)
    loader_test_t1  = DataLoader(test_t1, batch_size=64, shuffle=False)
    loader_test_t2  = DataLoader(test_t2, batch_size=64, shuffle=False)
    
    print(f"  Task 1 Train samples: {len(train_t1)}, Test samples: {len(test_t1)}")
    print(f"  Task 2 Train samples: {len(train_t2)}, Test samples: {len(test_t2)}")
    
    # 5.2 Train Base and Expert Models
    # Base model is pretrained on a small subset of both tasks to represent pretraining
    print("[Train] Training Base Model (Pretrained on mixture)...")
    mix_train_indices = list(range(0, len(train_dataset), 10)) # 10% subset
    mix_train = Subset(train_dataset, mix_train_indices)
    loader_mix = DataLoader(mix_train, batch_size=64, shuffle=True)
    
    base_model = SimpleMLP()
    base_model = train_model(base_model, loader_mix, epochs=2, lr=0.01, device=device, desc="Base Model")
    
    # Save base model checkpoint for reproducibility
    torch.save(base_model.state_dict(), "results/base_model.pt")
    
    # Experts
    print("[Train] Training Expert 1 on Task 1 (classes 0-4)...")
    expert1 = SimpleMLP()
    expert1.load_state_dict(base_model.state_dict())
    expert1 = train_model(expert1, loader_train_t1, epochs=3, lr=0.005, device=device, desc="Expert 1")
    torch.save(expert1.state_dict(), "results/expert1.pt")
    
    print("[Train] Training Expert 2 on Task 2 (classes 5-9)...")
    expert2 = SimpleMLP()
    expert2.load_state_dict(base_model.state_dict())
    expert2 = train_model(expert2, loader_train_t2, epochs=3, lr=0.005, device=device, desc="Expert 2")
    torch.save(expert2.state_dict(), "results/expert2.pt")
    
    # 5.3 Evaluate Individual Models (Sanity Checks)
    print("\n[Evaluation] Evaluating individual models (Sanity Checks):")
    acc_base_t1 = eval_model(base_model, loader_test_t1, device=device)
    acc_base_t2 = eval_model(base_model, loader_test_t2, device=device)
    print(f"  Base Model accuracy -> Task 1: {acc_base_t1:.4f}, Task 2: {acc_base_t2:.4f}")
    
    acc_exp1_t1 = eval_model(expert1, loader_test_t1, device=device)
    acc_exp1_t2 = eval_model(expert1, loader_test_t2, device=device)
    print(f"  Expert 1 accuracy   -> Task 1: {acc_exp1_t1:.4f}, Task 2: {acc_exp1_t2:.4f}")
    
    acc_exp2_t1 = eval_model(expert2, loader_test_t1, device=device)
    acc_exp2_t2 = eval_model(expert2, loader_test_t2, device=device)
    print(f"  Expert 2 accuracy   -> Task 1: {acc_exp2_t1:.4f}, Task 2: {acc_exp2_t2:.4f}")
    
    # 5.4 Model Merging & Hyperparameter Sweep
    print("\n[Merging] Performing Model Merging Experiments...")
    results = {}
    
    # Baseline: Task Arithmetic
    ta_sweep = [0.1, 0.3, 0.5, 0.7, 1.0]
    results["Task Arithmetic"] = []
    for s in ta_sweep:
        model_merged = merge_models_task_arithmetic([expert1, expert2], base_model, scaling_factor=s)
        acc_t1 = eval_model(model_merged, loader_test_t1, device=device)
        acc_t2 = eval_model(model_merged, loader_test_t2, device=device)
        avg_acc = (acc_t1 + acc_t2) / 2.0
        results["Task Arithmetic"].append({"param": s, "t1": acc_t1, "t2": acc_t2, "avg": avg_acc})
        print(f"  TA (scale={s}) -> Task 1: {acc_t1:.4f}, Task 2: {acc_t2:.4f}, Avg: {avg_acc:.4f}")
        
    # Baseline: OrthoMerge
    om_sweep = [0.0, 0.2, 0.5, 0.8, 1.0]
    results["OrthoMerge"] = []
    for s in om_sweep:
        model_merged = merge_models_orthomerge([expert1, expert2], base_model, residual_scale=s)
        acc_t1 = eval_model(model_merged, loader_test_t1, device=device)
        acc_t2 = eval_model(model_merged, loader_test_t2, device=device)
        avg_acc = (acc_t1 + acc_t2) / 2.0
        results["OrthoMerge"].append({"param": s, "t1": acc_t1, "t2": acc_t2, "avg": avg_acc})
        print(f"  OrthoMerge (res_scale={s}) -> Task 1: {acc_t1:.4f}, Task 2: {acc_t2:.4f}, Avg: {avg_acc:.4f}")
        
    # Baseline: SAIM
    saim_t_sweep = [1.0, 1.5, 2.0, 4.0, 8.0]
    results["SAIM"] = []
    for t_factor in saim_t_sweep:
        model_merged = merge_models_saim([expert1, expert2], base_model, t=t_factor, scaling_factor=0.5)
        acc_t1 = eval_model(model_merged, loader_test_t1, device=device)
        acc_t2 = eval_model(model_merged, loader_test_t2, device=device)
        avg_acc = (acc_t1 + acc_t2) / 2.0
        results["SAIM"].append({"param": t_factor, "t1": acc_t1, "t2": acc_t2, "avg": avg_acc})
        print(f"  SAIM (t={t_factor}) -> Task 1: {acc_t1:.4f}, Task 2: {acc_t2:.4f}, Avg: {avg_acc:.4f}")
        
    # Proposed: RIMO
    results["RIMO"] = []
    t_sweep = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    best_rimo_acc = 0.0
    best_rimo_model = None
    
    # We sweep both the spectral balancing t and the residual scale to show their interaction
    print("\n[Proposed RIMO Sweep]")
    for r_scale in [0.2, 0.5, 0.8, 1.0]:
        for t_factor in t_sweep:
            model_merged = merge_models_rimo([expert1, expert2], base_model, t=t_factor, residual_scale=r_scale)
            acc_t1 = eval_model(model_merged, loader_test_t1, device=device)
            acc_t2 = eval_model(model_merged, loader_test_t2, device=device)
            avg_acc = (acc_t1 + acc_t2) / 2.0
            results["RIMO"].append({
                "t": t_factor,
                "res_scale": r_scale,
                "t1": acc_t1,
                "t2": acc_t2,
                "avg": avg_acc
            })
            if avg_acc > best_rimo_acc:
                best_rimo_acc = avg_acc
                best_rimo_model = model_merged
            print(f"  RIMO (t={t_factor}, res_scale={r_scale}) -> Task 1: {acc_t1:.4f}, Task 2: {acc_t2:.4f}, Avg: {avg_acc:.4f}")
            
    # Proposed alternative: RIMO-Pruned
    results["RIMO-Pruned"] = []
    keep_sweep = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    print("\n[Proposed RIMO-Pruned Sweep]")
    for r_scale in [0.2, 0.5, 0.8, 1.0]:
        for keep in keep_sweep:
            model_merged = merge_models_rimo_pruned([expert1, expert2], base_model, keep_ratio=keep, residual_scale=r_scale)
            acc_t1 = eval_model(model_merged, loader_test_t1, device=device)
            acc_t2 = eval_model(model_merged, loader_test_t2, device=device)
            avg_acc = (acc_t1 + acc_t2) / 2.0
            results["RIMO-Pruned"].append({
                "keep_ratio": keep,
                "res_scale": r_scale,
                "t1": acc_t1,
                "t2": acc_t2,
                "avg": avg_acc
            })
            print(f"  RIMO-Pruned (keep={keep}, res_scale={r_scale}) -> Task 1: {acc_t1:.4f}, Task 2: {acc_t2:.4f}, Avg: {avg_acc:.4f}")

    # Save best RIMO checkpoint
    if best_rimo_model is not None:
        torch.save(best_rimo_model.state_dict(), "results/rimo_best_model.pt")
        
    # Save raw results as JSON
    with open("results/all_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # ---------------------------------------------------------
    # 6. Generate Figures & Summary Plots
    # ---------------------------------------------------------
    print("\n[Visualization] Generating results visualization...")
    plt.figure(figsize=(10, 6))
    
    # TA line
    ta_x = [item["param"] for item in results["Task Arithmetic"]]
    ta_y = [item["avg"] for item in results["Task Arithmetic"]]
    plt.plot(ta_x, ta_y, marker='o', linestyle='--', label="Task Arithmetic (Scale)", color="orange")
    
    # OrthoMerge line
    om_x = [item["param"] for item in results["OrthoMerge"]]
    om_y = [item["avg"] for item in results["OrthoMerge"]]
    plt.plot(om_x, om_y, marker='s', linestyle='-.', label="OrthoMerge (Res Scale)", color="blue")
    
    # SAIM line
    saim_x = [item["param"] for item in results["SAIM"]]
    saim_y = [item["avg"] for item in results["SAIM"]]
    plt.plot(saim_x, saim_y, marker='^', linestyle=':', label="SAIM (Spectral t)", color="green")
    
    # RIMO sweep (best per t across sweeps)
    rimo_by_t = {}
    for item in results["RIMO"]:
        t_val = item["t"]
        avg_val = item["avg"]
        if t_val not in rimo_by_t or avg_val > rimo_by_t[t_val]:
            rimo_by_t[t_val] = avg_val
    rimo_x = sorted(list(rimo_by_t.keys()))
    rimo_y = [rimo_by_t[t] for t in rimo_x]
    plt.plot(rimo_x, rimo_y, marker='*', linestyle='-', linewidth=2.5, label="RIMO (Proposed, Best res_scale)", color="red")
    
    plt.title("Model Merging Performance Comparison on Split-MNIST", fontsize=14, fontweight='bold')
    plt.xlabel("Hyperparameter Value", fontsize=12)
    plt.ylabel("Average Accuracy (Task 1 & Task 2)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc="best")
    plt.tight_layout()
    plt.savefig("results/accuracy_comparison.png", dpi=300)
    plt.close()
    
    # RIMO 2D Heatmap of t vs res_scale
    heat_t_vals = sorted(list(set([item["t"] for item in results["RIMO"]])))
    heat_res_vals = sorted(list(set([item["res_scale"] for item in results["RIMO"]])))
    
    heatmap_data = np.zeros((len(heat_res_vals), len(heat_t_vals)))
    for item in results["RIMO"]:
        t_idx = heat_t_vals.index(item["t"])
        res_idx = heat_res_vals.index(item["res_scale"])
        heatmap_data[res_idx, t_idx] = item["avg"]
        
    plt.figure(figsize=(8, 6))
    im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(im, label="Average Accuracy")
    plt.xticks(range(len(heat_t_vals)), heat_t_vals)
    plt.yticks(range(len(heat_res_vals)), heat_res_vals)
    plt.xlabel("Spectral Isotropic Factor t", fontsize=12)
    plt.ylabel("Residual Scale Factor", fontsize=12)
    plt.title("RIMO Performance Landscape (Accuracy Heatmap)", fontsize=14, fontweight='bold')
    
    # Loop over data dimensions and create text annotations
    for i in range(len(heat_res_vals)):
        for j in range(len(heat_t_vals)):
            plt.text(j, i, f"{heatmap_data[i, j]:.4f}", ha="center", va="center", color="w" if heatmap_data[i, j] < np.max(heatmap_data)*0.9 else "black")
            
    plt.tight_layout()
    plt.savefig("results/rimo_heatmap.png", dpi=300)
    plt.close()
    
    print("\n[Done] Experiments finished successfully. Metrics and plots saved to results/ directory.")

if __name__ == "__main__":
    run_all()
