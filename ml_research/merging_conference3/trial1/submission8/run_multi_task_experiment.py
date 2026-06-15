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
# 2. Dataset Preparation (5-Split MNIST)
# ---------------------------------------------------------
def get_datasets():
    try:
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        return train_dataset, test_dataset
    except Exception as e:
        print(f"[Data] Failed to load MNIST: {e}. Falling back to synthetic.")
        np.random.seed(42)
        X_train = np.random.randn(6000, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, size=(6000,)).astype(np.int64)
        X_test = np.random.randn(1000, 784).astype(np.float32)
        y_test = np.random.randint(0, 10, size=(1000,)).astype(np.int64)
        for i in range(10):
            mask_tr = (y_train == i)
            X_train[mask_tr, i*20:(i+1)*20] += 5.0
            mask_te = (y_test == i)
            X_test[mask_te, i*20:(i+1)*20] += 5.0
        return TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

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
# 3. Training Function
# ---------------------------------------------------------
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
# 4. Merging Utilities
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

# --- Merging Algorithms ---

def merge_models_task_arithmetic(models_list, base_model, scaling_factor=0.2):
    merged = copy_model_structure(base_model)
    sd_merged = merged.state_dict()
    sd_base = base_model.state_dict()
    sd_experts = [m.state_dict() for m in models_list]
    for key in sd_base.keys():
        if sd_base[key].dtype.is_floating_point:
            deltas = [sd_exp[key] - sd_base[key] for sd_exp in sd_experts]
            avg_delta = torch.stack(deltas, dim=0).mean(dim=0)
            sd_merged[key] = sd_base[key] + scaling_factor * avg_delta
        else:
            sd_merged[key] = sd_base[key]
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
# 5. Main Execution
# ---------------------------------------------------------
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")
    
    # Load and split dataset into 5 tasks
    train_dataset, test_dataset = get_datasets()
    
    tasks = [
        [0, 1],  # Task 1
        [2, 3],  # Task 2
        [4, 5],  # Task 3
        [6, 7],  # Task 4
        [8, 9]   # Task 5
    ]
    
    loader_trains = []
    loader_tests = []
    
    for i, classes in enumerate(tasks):
        train_t = filter_split_dataset(train_dataset, classes)
        test_t  = filter_split_dataset(test_dataset, classes)
        loader_trains.append(DataLoader(train_t, batch_size=64, shuffle=True))
        loader_tests.append(DataLoader(test_t, batch_size=64, shuffle=False))
        print(f"Task {i+1} ({classes}) -> Train size: {len(train_t)}, Test size: {len(test_t)}")
        
    # Pre-train shared base model on 10% mixed MNIST
    print("\n[Train] Training Shared Base Model with Orthogonal Regularization...")
    mix_train_indices = list(range(0, len(train_dataset), 10))
    mix_train = Subset(train_dataset, mix_train_indices)
    loader_mix = DataLoader(mix_train, batch_size=64, shuffle=True)
    
    base_model = SimpleMLP()
    base_model = train_model_ortho(base_model, loader_mix, epochs=2, lr=0.01, device=device, desc="Base Model", ortho_lambda=2.0)
    
    # Train 5 separate experts
    experts = []
    for i, loader_tr in enumerate(loader_trains):
        print(f"[Train] Training Expert {i+1} on Task {i+1} with Orthogonal Regularization...")
        expert = SimpleMLP()
        expert.load_state_dict(base_model.state_dict())
        expert = train_model_ortho(expert, loader_tr, epochs=3, lr=0.005, device=device, desc=f"Expert {i+1}", ortho_lambda=2.0)
        experts.append(expert)
        
    # Evaluate individual experts
    print("\n[Evaluation] Evaluating individual experts:")
    for i, exp in enumerate(experts):
        accs = []
        for j, loader_te in enumerate(loader_tests):
            acc = eval_model(exp, loader_te, device=device)
            accs.append(acc)
            if i == j:
                print(f"  Expert {i+1} on Task {j+1} (Self): {acc*100:.2f}%")
                
    # ---------------------------------------------------------
    # 6. Model Merging and Magnitude Analysis
    # ---------------------------------------------------------
    print("\n[Merging] Performing 5-Expert Model Merging...")
    
    # Merging Methods:
    # 1. Task Arithmetic (sweep scaling factors)
    # 2. OrthoMerge
    # 3. RIMO-Pruned
    
    results = {}
    
    # Utilities to compute matrix norm
    def get_fc1_norm(model):
        return torch.norm(model.fc1.weight, p='fro').item()
    
    base_fc1_norm = get_fc1_norm(base_model)
    print(f"Base Model fc1 Frobenius norm: {base_fc1_norm:.4f}")
    
    # Task Arithmetic Sweeps
    results["Task Arithmetic"] = []
    for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        merged = merge_models_task_arithmetic(experts, base_model, scaling_factor=s)
        accs = [eval_model(merged, l, device=device) for l in loader_tests]
        avg_acc = sum(accs) / len(accs)
        fc1_norm = get_fc1_norm(merged)
        results["Task Arithmetic"].append({"param": s, "avg_acc": avg_acc, "fc1_norm": fc1_norm})
        print(f"  TA (scale={s:.2f}) -> Avg Accuracy: {avg_acc*100:.2f}%, fc1 Frobenius norm: {fc1_norm:.4f} (Change: {(fc1_norm - base_fc1_norm)/base_fc1_norm*100:.2f}%)")
        
    # OrthoMerge
    merged_om = merge_models_orthomerge(experts, base_model, residual_scale=1.0)
    om_accs = [eval_model(merged_om, l, device=device) for l in loader_tests]
    om_avg = sum(om_accs) / len(om_accs)
    om_norm = get_fc1_norm(merged_om)
    results["OrthoMerge"] = {"avg_acc": om_avg, "fc1_norm": om_norm}
    print(f"  OrthoMerge -> Avg Accuracy: {om_avg*100:.2f}%, fc1 Frobenius norm: {om_norm:.4f} (Change: {(om_norm - base_fc1_norm)/base_fc1_norm*100:.2f}%)")
    
    # RIMO-Pruned Sweeps
    results["RIMO-Pruned"] = []
    for keep in [0.1, 0.2, 0.4]:
        merged_rp = merge_models_rimo_pruned(experts, base_model, keep_ratio=keep, residual_scale=1.0)
        rp_accs = [eval_model(merged_rp, l, device=device) for l in loader_tests]
        rp_avg = sum(rp_accs) / len(rp_accs)
        rp_norm = get_fc1_norm(merged_rp)
        results["RIMO-Pruned"].append({"keep_ratio": keep, "avg_acc": rp_avg, "fc1_norm": rp_norm})
        print(f"  RIMO-Pruned (keep={keep:.2f}) -> Avg Accuracy: {rp_avg*100:.2f}%, fc1 Frobenius norm: {rp_norm:.4f} (Change: {(rp_norm - base_fc1_norm)/base_fc1_norm*100:.2f}%)")
        
    # Save results to disk
    os.makedirs("results", exist_ok=True)
    with open("results/multi_task_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Generate Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy Comparison
    ta_s = [x["param"] for x in results["Task Arithmetic"]]
    ta_acc = [x["avg_acc"] * 100 for x in results["Task Arithmetic"]]
    ax1.plot(ta_s, ta_acc, marker='o', linestyle='--', label="Task Arithmetic (Scale)", color="orange")
    
    ax1.axhline(y=om_avg * 100, color="blue", linestyle='-', label="OrthoMerge", linewidth=2)
    
    rp_k = [x["keep_ratio"] for x in results["RIMO-Pruned"]]
    rp_acc = [x["avg_acc"] * 100 for x in results["RIMO-Pruned"]]
    ax1.scatter(rp_k, rp_acc, color="red", marker='*', s=150, label="RIMO-Pruned (keep_ratio)", zorder=5)
    
    ax1.set_title("5-Task Merged Model Accuracy", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Scaling / Keep Factor", fontsize=10)
    ax1.set_ylabel("Average Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    # Plot 2: Frobenius Norm Comparison
    ta_norm = [x["fc1_norm"] for x in results["Task Arithmetic"]]
    ax2.plot(ta_s, ta_norm, marker='o', linestyle='--', label="Task Arithmetic", color="orange")
    ax2.axhline(y=om_norm, color="blue", linestyle='-', label="OrthoMerge", linewidth=2)
    ax2.scatter(rp_k, [x["fc1_norm"] for x in results["RIMO-Pruned"]], color="red", marker='*', s=150, label="RIMO-Pruned", zorder=5)
    ax2.axhline(y=base_fc1_norm, color="grey", linestyle=':', label="Base Model (Ideal Orthogonal Norm)")
    
    ax2.set_title("fc1.weight Frobenius Norm ($N = 5$ Experts)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Scaling / Keep Factor", fontsize=10)
    ax2.set_ylabel("Frobenius Norm", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("results/multi_task_comparison.png", dpi=300)
    plt.close()
    
    print("\n[Done] Multi-task experiment completed successfully! Results saved to results/multi_task_results.json and results/multi_task_comparison.png.")

if __name__ == "__main__":
    main()
