import os
import sys
import copy
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
    def __init__(self, input_dim=3072, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        self.fc3_out = self.fc3(x)
        return self.fc3_out

# ---------------------------------------------------------
# 2. Dataset Preparation (Split-CIFAR-10)
# ---------------------------------------------------------
def get_datasets():
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def filter_split_dataset(dataset, classes, subset_ratio=0.2):
    indices = []
    targets = dataset.targets
    for idx, target in enumerate(targets):
        if int(target) in classes:
            indices.append(idx)
    
    # Take a subset to make training extremely fast on CPU
    set_seed(42)
    np.random.shuffle(indices)
    keep_len = int(len(indices) * subset_ratio)
    indices = indices[:keep_len]
    return Subset(dataset, indices)

# ---------------------------------------------------------
# 3. Model Training with Orthogonal Regularization
# ---------------------------------------------------------
def train_model_ortho(model, train_loader, epochs=3, lr=0.01, device="cpu", desc="", ortho_lambda=0.1):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"  Training {desc} (Ortho Reg={ortho_lambda})...")
    for epoch in range(epochs):
        running_loss = 0.0
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
    A = torch.matmul(W_0.t(), W_k)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
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

def rimo_spectral_balancing(Q_com, t=2.0):
    U, S, Vh = torch.linalg.svd(Q_com, full_matrices=False)
    mean_s = S.mean()
    S_balanced = mean_s + (S - mean_s) / math.sqrt(t)
    Q_balanced = torch.matmul(U * S_balanced.unsqueeze(0), Vh)
    Q_balanced = 0.5 * (Q_balanced - Q_balanced.t())
    return Q_balanced

def rimo_spectral_pruning(Q_com, keep_ratio=0.2):
    U, S, Vh = torch.linalg.svd(Q_com, full_matrices=False)
    k = max(2, int(len(S) * keep_ratio))
    if k % 2 != 0:
        k = min(len(S), k + 1)
    S_pruned = torch.zeros_like(S)
    S_pruned[:k] = S[:k]
    Q_pruned = torch.matmul(U * S_pruned.unsqueeze(0), Vh)
    Q_pruned = 0.5 * (Q_pruned - Q_pruned.t())
    return Q_pruned

def copy_model_structure(model):
    replica = SimpleMLP()
    replica.load_state_dict(model.state_dict())
    return replica

# --- Full model merging algorithms ---
def merge_models_task_arithmetic(models_list, base_model, scaling_factor=0.5):
    merged = copy_model_structure(base_model)
    sd_merged = merged.state_dict()
    sd_base = base_model.state_dict()
    sd_experts = [m.state_dict() for m in models_list]
    for key in sd_base.keys():
        W_0 = sd_base[key]
        if W_0.dtype.is_floating_point:
            W_experts = [sd_exp[key] for sd_exp in sd_experts]
            delta_sum = torch.zeros_like(W_0)
            for W_k in W_experts:
                delta_sum += (W_k - W_0)
            sd_merged[key] = W_0 + scaling_factor * delta_sum
        else:
            sd_merged[key] = W_0
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
            R_merged = torch.stack(R_list, dim=0).mean(dim=0)
            rho_merged = torch.stack(rho_list, dim=0).mean(dim=0)
            sd_merged[key] = torch.matmul(W_0, R_merged) + residual_scale * rho_merged
        else:
            if W_0.dtype.is_floating_point:
                sd_merged[key] = torch.stack([sd_exp[key] for sd_exp in sd_experts], dim=0).mean(dim=0)
            else:
                sd_merged[key] = W_0
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
            W_experts = [sd_exp[key] for sd_exp in sd_experts]
            deltas = [W_k - W_0 for W_k in W_experts]
            delta_com = torch.stack(deltas, dim=0).mean(dim=0)
            U, S, Vh = torch.linalg.svd(delta_com, full_matrices=False)
            mean_s = S.mean()
            S_balanced = mean_s + (S - mean_s) / math.sqrt(t)
            delta_balanced = torch.matmul(U * S_balanced.unsqueeze(0), Vh)
            sd_merged[key] = W_0 + scaling_factor * delta_balanced
        else:
            if W_0.dtype.is_floating_point:
                sd_merged[key] = torch.stack([sd_exp[key] for sd_exp in sd_experts], dim=0).mean(dim=0)
            else:
                sd_merged[key] = W_0
    merged.load_state_dict(sd_merged)
    return merged

def merge_models_rimo(models_list, base_model, t=2.0, residual_scale=1.0):
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
            Q_balanced = rimo_spectral_balancing(Q_com, t=t)
            R_merged = cayley_from_skew(Q_balanced)
            rho_merged = torch.stack(rho_list, dim=0).mean(dim=0)
            sd_merged[key] = torch.matmul(W_0, R_merged) + residual_scale * rho_merged
        else:
            if W_0.dtype.is_floating_point:
                sd_merged[key] = torch.stack([sd_exp[key] for sd_exp in sd_experts], dim=0).mean(dim=0)
            else:
                sd_merged[key] = W_0
    merged.load_state_dict(sd_merged)
    return merged

def merge_models_rimo_pruned(models_list, base_model, keep_ratio=0.2, residual_scale=1.0):
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

# ---------------------------------------------------------
# 5. Master Pipeline Execution
# ---------------------------------------------------------
def run_all(ortho_lambda=2.0):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device} (Ortho Regularization Lambda: {ortho_lambda})")
    
    os.makedirs("results_cifar10", exist_ok=True)
    
    # 5.1 Load Datasets
    train_dataset, test_dataset = get_datasets()
    
    classes_t1 = [0, 1, 2, 3, 4]
    classes_t2 = [5, 6, 7, 8, 9]
    
    # Take 100% of the dataset for fully rigorous, standard training and testing
    train_t1 = filter_split_dataset(train_dataset, classes_t1, subset_ratio=1.0)
    train_t2 = filter_split_dataset(train_dataset, classes_t2, subset_ratio=1.0)
    test_t1  = filter_split_dataset(test_dataset, classes_t1, subset_ratio=1.0)
    test_t2  = filter_split_dataset(test_dataset, classes_t2, subset_ratio=1.0)
    
    loader_train_t1 = DataLoader(train_t1, batch_size=128, shuffle=True)
    loader_train_t2 = DataLoader(train_t2, batch_size=128, shuffle=True)
    loader_test_t1  = DataLoader(test_t1, batch_size=128, shuffle=False)
    loader_test_t2  = DataLoader(test_t2, batch_size=128, shuffle=False)
    
    print(f"Task 1 Train size: {len(train_t1)}, Test size: {len(test_t1)}")
    print(f"Task 2 Train size: {len(train_t2)}, Test size: {len(test_t2)}")
    
    # 5.2 Train Base and Expert Models with Orthogonal Regularization
    print("[Train] Training Base Model with Orthogonal Regularization...")
    # Pretrain base model on 20% subset of both classes
    mix_indices = list(range(0, len(train_dataset), 5)) # 20% subset
    mix_train = Subset(train_dataset, mix_indices)
    loader_mix = DataLoader(mix_train, batch_size=128, shuffle=True)
    
    base_model = SimpleMLP(input_dim=3072, hidden_dim=256, output_dim=10)
    base_model = train_model_ortho(base_model, loader_mix, epochs=2, lr=0.01, device=device, desc="Base Model", ortho_lambda=ortho_lambda)
    torch.save(base_model.state_dict(), "results_cifar10/base_model.pt")
    
    print("[Train] Training Expert 1 on Task 1 with Orthogonal Regularization...")
    expert1 = SimpleMLP(input_dim=3072, hidden_dim=256, output_dim=10)
    expert1.load_state_dict(base_model.state_dict())
    expert1 = train_model_ortho(expert1, loader_train_t1, epochs=3, lr=0.005, device=device, desc="Expert 1", ortho_lambda=ortho_lambda)
    torch.save(expert1.state_dict(), "results_cifar10/expert1.pt")
    
    print("[Train] Training Expert 2 on Task 2 with Orthogonal Regularization...")
    expert2 = SimpleMLP(input_dim=3072, hidden_dim=256, output_dim=10)
    expert2.load_state_dict(base_model.state_dict())
    expert2 = train_model_ortho(expert2, loader_train_t2, epochs=3, lr=0.005, device=device, desc="Expert 2", ortho_lambda=ortho_lambda)
    torch.save(expert2.state_dict(), "results_cifar10/expert2.pt")
    
    # 5.3 Evaluate Individual Models
    print("\n[Evaluation] Evaluating individual models:")
    acc_base_t1 = eval_model(base_model, loader_test_t1, device=device)
    acc_base_t2 = eval_model(base_model, loader_test_t2, device=device)
    print(f"  Base Model accuracy -> Task 1: {acc_base_t1:.4f}, Task 2: {acc_base_t2:.4f}")
    
    acc_exp1_t1 = eval_model(expert1, loader_test_t1, device=device)
    acc_exp1_t2 = eval_model(expert1, loader_test_t2, device=device)
    print(f"  Expert 1 accuracy   -> Task 1: {acc_exp1_t1:.4f}, Task 2: {acc_exp1_t2:.4f}")
    
    acc_exp2_t1 = eval_model(expert2, loader_test_t1, device=device)
    acc_exp2_t2 = eval_model(expert2, loader_test_t2, device=device)
    print(f"  Expert 2 accuracy   -> Task 1: {acc_exp2_t1:.4f}, Task 2: {acc_exp2_t2:.4f}")
    
    # 5.4 Model Merging
    print("\n[Merging] Performing Model Merging Experiments...")
    results = {}
    
    # Task Arithmetic
    ta_sweep = [0.1, 0.3, 0.5, 0.7, 1.0]
    results["Task Arithmetic"] = []
    for s in ta_sweep:
        model_merged = merge_models_task_arithmetic([expert1, expert2], base_model, scaling_factor=s)
        acc_t1 = eval_model(model_merged, loader_test_t1, device=device)
        acc_t2 = eval_model(model_merged, loader_test_t2, device=device)
        avg_acc = (acc_t1 + acc_t2) / 2.0
        results["Task Arithmetic"].append({"param": s, "t1": acc_t1, "t2": acc_t2, "avg": avg_acc})
        print(f"  TA (scale={s}) -> Task 1: {acc_t1:.4f}, Task 2: {acc_t2:.4f}, Avg: {avg_acc:.4f}")
        
    # OrthoMerge
    om_sweep = [0.0, 0.2, 0.5, 0.8, 1.0]
    results["OrthoMerge"] = []
    for s in om_sweep:
        model_merged = merge_models_orthomerge([expert1, expert2], base_model, residual_scale=s)
        acc_t1 = eval_model(model_merged, loader_test_t1, device=device)
        acc_t2 = eval_model(model_merged, loader_test_t2, device=device)
        avg_acc = (acc_t1 + acc_t2) / 2.0
        results["OrthoMerge"].append({"param": s, "t1": acc_t1, "t2": acc_t2, "avg": avg_acc})
        print(f"  OrthoMerge (res_scale={s}) -> Task 1: {acc_t1:.4f}, Task 2: {acc_t2:.4f}, Avg: {avg_acc:.4f}")
        
    # SAIM
    saim_t_sweep = [1.0, 1.5, 2.0, 4.0, 8.0]
    results["SAIM"] = []
    for t_factor in saim_t_sweep:
        model_merged = merge_models_saim([expert1, expert2], base_model, t=t_factor, scaling_factor=0.5)
        acc_t1 = eval_model(model_merged, loader_test_t1, device=device)
        acc_t2 = eval_model(model_merged, loader_test_t2, device=device)
        avg_acc = (acc_t1 + acc_t2) / 2.0
        results["SAIM"].append({"param": t_factor, "t1": acc_t1, "t2": acc_t2, "avg": avg_acc})
        print(f"  SAIM (t={t_factor}) -> Task 1: {acc_t1:.4f}, Task 2: {acc_t2:.4f}, Avg: {avg_acc:.4f}")
        
    # RIMO
    results["RIMO"] = []
    t_sweep = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    best_rimo_acc = 0.0
    best_rimo_model = None
    
    print("\n[Proposed RIMO Sweep with Orthogonalized Parameters]")
    for r_scale in [0.0, 0.2, 0.5, 0.8, 1.0]:
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
    print("\n[Proposed RIMO-Pruned Sweep with Orthogonalized Parameters]")
    for r_scale in [0.0, 0.2, 0.5, 0.8, 1.0]:
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

    if best_rimo_model is not None:
        torch.save(best_rimo_model.state_dict(), "results_cifar10/rimo_best_model.pt")
        
    with open("results_cifar10/all_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # ---------------------------------------------------------
    # 6. Generate Figures & Summary Plots
    # ---------------------------------------------------------
    print("\n[Visualization] Generating results visualization...")
    plt.figure(figsize=(10, 6))
    
    ta_x = [item["param"] for item in results["Task Arithmetic"]]
    ta_y = [item["avg"] for item in results["Task Arithmetic"]]
    plt.plot(ta_x, ta_y, marker='o', linestyle='--', label="Task Arithmetic (Scale)", color="orange")
    
    om_x = [item["param"] for item in results["OrthoMerge"]]
    om_y = [item["avg"] for item in results["OrthoMerge"]]
    plt.plot(om_x, om_y, marker='s', linestyle='-.', label="OrthoMerge (Res Scale)", color="blue")
    
    saim_x = [item["param"] for item in results["SAIM"]]
    saim_y = [item["avg"] for item in results["SAIM"]]
    plt.plot(saim_x, saim_y, marker='^', linestyle=':', label="SAIM (Spectral t)", color="green")
    
    rimo_by_t = {}
    for item in results["RIMO"]:
        t_val = item["t"]
        avg_val = item["avg"]
        if t_val not in rimo_by_t or avg_val > rimo_by_t[t_val]:
            rimo_by_t[t_val] = avg_val
    rimo_x = sorted(list(rimo_by_t.keys()))
    rimo_y = [rimo_by_t[t] for t in rimo_x]
    plt.plot(rimo_x, rimo_y, marker='*', linestyle='-', linewidth=2.5, label="RIMO (Proposed, Best res_scale)", color="red")
    
    plt.title(f"CIFAR-10 Merging Performance Comparison (Ortho Regularized $\lambda={ortho_lambda}$)", fontsize=13, fontweight='bold')
    plt.xlabel("Hyperparameter Value", fontsize=12)
    plt.ylabel("Average Accuracy (Task 1 & Task 2)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc="best")
    plt.tight_layout()
    plt.savefig("results_cifar10/accuracy_comparison.png", dpi=300)
    plt.close()
    
    print("\n[Done] CIFAR-10 experiments finished successfully. Saved to results_cifar10/.")

if __name__ == "__main__":
    run_all(ortho_lambda=2.0)
