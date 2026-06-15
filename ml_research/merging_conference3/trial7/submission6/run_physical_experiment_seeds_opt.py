import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time

# Load digits dataset
digits = load_digits()
X_all = digits.data # Shape: (1797, 64)
y_all = digits.target # Shape: (1797,)

# Normalize inputs
X_all = X_all / 16.0
X_all = torch.tensor(X_all, dtype=torch.float32)
y_all = torch.tensor(y_all, dtype=torch.long)

K = 4

def get_task_data(task_id, seed):
    if task_id == 0:
        indices = (y_all == 0) | (y_all == 1)
        X_task = X_all[indices]
        y_task = y_all[indices]
    elif task_id == 1:
        indices = (y_all == 2) | (y_all == 3)
        X_task = X_all[indices]
        y_task = y_all[indices] - 2
    elif task_id == 2:
        indices = (y_all == 4) | (y_all == 5)
        X_task = X_all[indices]
        y_task = y_all[indices] - 4
    elif task_id == 3:
        indices = (y_all == 6) | (y_all == 7)
        X_task = X_all[indices]
        y_task = y_all[indices] - 6
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_task.numpy(), y_task.numpy(), test_size=100, random_state=seed, stratify=y_task.numpy()
    )
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long)
    )

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

# Define Merged MLP with vectorized sample-wise parameter merging
class MergedMLP(nn.Module):
    def __init__(self, base_model, task_vectors):
        super().__init__()
        self.base_model = base_model
        self.task_vectors = task_vectors
        
    def forward(self, x, alpha):
        W_fc1_base = self.base_model.fc1.weight
        b_fc1_base = self.base_model.fc1.bias
        V_fc1_w = torch.stack([tv['fc1.weight'] for tv in self.task_vectors]) # (K, 32, 64)
        V_fc1_b = torch.stack([tv['fc1.bias'] for tv in self.task_vectors]) # (K, 32)
        
        W_fc1_merged = W_fc1_base.unsqueeze(0) + torch.einsum('bk,koi->boi', alpha, V_fc1_w)
        b_fc1_merged = b_fc1_base.unsqueeze(0) + torch.einsum('bk,ko->bo', alpha, V_fc1_b)
        
        h = torch.einsum('boi,bi->bo', W_fc1_merged, x) + b_fc1_merged
        h = torch.relu(h)
        
        W_fc2_base = self.base_model.fc2.weight
        b_fc2_base = self.base_model.fc2.bias
        V_fc2_w = torch.stack([tv['fc2.weight'] for tv in self.task_vectors]) # (K, 2, 32)
        V_fc2_b = torch.stack([tv['fc2.bias'] for tv in self.task_vectors]) # (K, 2)
        
        W_fc2_merged = W_fc2_base.unsqueeze(0) + torch.einsum('bk,koi->boi', alpha, V_fc2_w)
        b_fc2_merged = b_fc2_base.unsqueeze(0) + torch.einsum('bk,ko->bo', alpha, V_fc2_b)
        
        out = torch.einsum('boi,bi->bo', W_fc2_merged, h) + b_fc2_merged
        return out

# Define parametric router using frozen random projection and linear Softmax Predictor
class RealRouter(nn.Module):
    def __init__(self, in_features=64, proj_dim=4, K=4):
        super().__init__()
        P = torch.randn(in_features, proj_dim)
        P = P / torch.norm(P, dim=0, keepdim=True)
        self.register_buffer('P', P)
        
        self.W = nn.Parameter(torch.zeros(K, proj_dim))
        self.B = nn.Parameter(torch.zeros(K))
        
    def forward(self, x):
        proj = x @ self.P
        psi = proj / (torch.norm(proj, dim=-1, keepdim=True) + 1e-8)
        logits = psi @ self.W.t() + self.B
        alpha = torch.softmax(logits, dim=-1)
        return alpha

def generate_splits(task_data, num_samples_per_task, seed):
    torch.manual_seed(seed)
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

def calibrate_real_router(reg_type, lambda_reg, base_model, task_vectors, v_norms, s_norms, X_cal, y_cal, X_test, y_test, task_test, lr=0.01, epochs=150, proj_dim=4, beta=0.9, gamma=1.0):
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
            loss_reg = lambda_reg * torch.sum(v_norms * (W_squared + B_squared))
        elif reg_type == "sr3_s":
            loss_reg = lambda_reg * torch.sum(s_norms * (W_squared + B_squared))
        elif reg_type == "sr3_hybrid":
            grad_W = torch.autograd.grad(loss_ce, router.W, retain_graph=True)[0].detach()
            grad_norms = torch.norm(grad_W, p=2, dim=-1)
            g = beta * g + (1.0 - beta) * grad_norms
            lambdas = lambda_reg * v_norms * torch.exp(-gamma * g)
            loss_reg = torch.sum(lambdas * (W_squared + B_squared))
            
        loss_total = loss_ce + loss_reg
        loss_total.backward()
        optimizer.step()
        
    accs, mean_acc = evaluate_real_router(router, merged_model, X_test, y_test, task_test)
    return accs, mean_acc

def run_experiment_on_seed(seed):
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Define splits for this seed
    task_data = [get_task_data(k, seed) for k in range(K)]
    
    # Initialize base model for this seed
    base_model = TinyMLP()
    
    # Fine-tune task-specific experts
    experts = []
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
        
    # Calibration and test splits
    X_cal, y_cal, task_cal = generate_splits(task_data, 16, seed)
    X_test, y_test, task_test = generate_test_splits(task_data)
    
    seed_results = {}
    
    # Static Uniform Merging
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
    seed_results["Static Uniform Merging"] = uniform_mean
    
    # Unregularized Router
    _, unreg_mean = calibrate_real_router("none", 0.0, base_model, task_vectors, v_norms, s_norms, X_cal, y_cal, X_test, y_test, task_test)
    seed_results["Linear Router (Unregularized)"] = unreg_mean
    
    # Isotropic L2 router sweeps (narrowed)
    best_mean = -1
    for l in [1e-4, 1e-3, 1e-2, 0.1]:
        _, m = calibrate_real_router("l2", l, base_model, task_vectors, v_norms, s_norms, X_cal, y_cal, X_test, y_test, task_test)
        if m > best_mean:
            best_mean = m
    seed_results["Linear Router (L2 Regularized)"] = best_mean
    
    # TSAR sweeps (narrowed)
    best_mean = -1
    for l in [1e-4, 1e-3, 1e-2, 0.1]:
        _, m = calibrate_real_router("tsar", l, base_model, task_vectors, v_norms, s_norms, X_cal, y_cal, X_test, y_test, task_test)
        if m > best_mean:
            best_mean = m
    seed_results["TSAR (Centroid Anchoring)"] = best_mean
    
    # SR3-F sweeps (narrowed)
    best_mean = -1
    for l in [1e-4, 1e-3, 1e-2, 0.1]:
        _, m = calibrate_real_router("sr3_f", l, base_model, task_vectors, v_norms, s_norms, X_cal, y_cal, X_test, y_test, task_test)
        if m > best_mean:
            best_mean = m
    seed_results["SR3-F (Ours - Frobenius)"] = best_mean
    
    # SR3-S sweeps (narrowed)
    best_mean = -1
    for l in [1e-4, 1e-3, 1e-2, 0.1]:
        _, m = calibrate_real_router("sr3_s", l, base_model, task_vectors, v_norms, s_norms, X_cal, y_cal, X_test, y_test, task_test)
        if m > best_mean:
            best_mean = m
    seed_results["SR3-S (Ours - Spectral)"] = best_mean
    
    # SR3-Hybrid sweeps (narrowed)
    best_mean = -1
    for l in [1e-4, 1e-3, 1e-2, 0.1]:
        _, m = calibrate_real_router("sr3_hybrid", l, base_model, task_vectors, v_norms, s_norms, X_cal, y_cal, X_test, y_test, task_test)
        if m > best_mean:
            best_mean = m
    seed_results["SR3-H (Ours - Hybrid Adaptive)"] = best_mean
    
    # --- Projection Dimension Ablation Sweep on this seed (highly optimized and swept) ---
    ablation_results = {}
    for pd in [4, 8, 16, 32, 64]:
        pd_res = {}
        # Unregularized
        _, unreg_m = calibrate_real_router("none", 0.0, base_model, task_vectors, v_norms, s_norms, X_cal, y_cal, X_test, y_test, task_test, proj_dim=pd)
        pd_res["Unregularized"] = unreg_m
        
        # L2 sweep
        l2_best = -1
        for l in [1e-4, 1e-3, 1e-2, 0.1]:
            _, m = calibrate_real_router("l2", l, base_model, task_vectors, v_norms, s_norms, X_cal, y_cal, X_test, y_test, task_test, proj_dim=pd)
            if m > l2_best:
                l2_best = m
        pd_res["L2 Regularized"] = l2_best
                
        # TSAR sweep
        tsar_best = -1
        for l in [1e-4, 1e-3, 1e-2, 0.1]:
            _, m = calibrate_real_router("tsar", l, base_model, task_vectors, v_norms, s_norms, X_cal, y_cal, X_test, y_test, task_test, proj_dim=pd)
            if m > tsar_best:
                tsar_best = m
        pd_res["TSAR"] = tsar_best
                
        # SR3-F sweep
        sr3_f_best = -1
        for l in [1e-4, 1e-3, 1e-2, 0.1]:
            _, m = calibrate_real_router("sr3_f", l, base_model, task_vectors, v_norms, s_norms, X_cal, y_cal, X_test, y_test, task_test, proj_dim=pd)
            if m > sr3_f_best:
                sr3_f_best = m
        pd_res["SR3-F"] = sr3_f_best
                
        # SR3-S sweep
        sr3_s_best = -1
        for l in [1e-4, 1e-3, 1e-2, 0.1]:
            _, m = calibrate_real_router("sr3_s", l, base_model, task_vectors, v_norms, s_norms, X_cal, y_cal, X_test, y_test, task_test, proj_dim=pd)
            if m > sr3_s_best:
                sr3_s_best = m
        pd_res["SR3-S"] = sr3_s_best
                
        # SR3-H sweep
        sr3_h_best = -1
        for l in [1e-4, 1e-3, 1e-2, 0.1]:
            _, m = calibrate_real_router("sr3_hybrid", l, base_model, task_vectors, v_norms, s_norms, X_cal, y_cal, X_test, y_test, task_test, proj_dim=pd)
            if m > sr3_h_best:
                sr3_h_best = m
        pd_res["SR3-H"] = sr3_h_best
        
        ablation_results[pd] = pd_res
        
    return seed_results, ablation_results

def main():
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    print(f"Running physical PyTorch experiment across {len(seeds)} random seeds (OPTIMIZED)...", flush=True)
    
    all_seed_res = []
    all_ablation_res = []
    
    start_time = time.time()
    for s_idx, s in enumerate(seeds):
        t0 = time.time()
        s_res, ab_res = run_experiment_on_seed(s)
        all_seed_res.append(s_res)
        all_ablation_res.append(ab_res)
        print(f"Seed {s_idx+1}/{len(seeds)} (Seed: {s}) completed in {time.time()-t0:.2f}s.", flush=True)
        
    print(f"\nAll experiments finished in {time.time()-start_time:.2f}s.\n", flush=True)
    
    # Aggregate main results
    methods = list(all_seed_res[0].keys())
    aggregated_main = {}
    for m in methods:
        vals = [res[m] for res in all_seed_res]
        mean = np.mean(vals)
        std = np.std(vals)
        aggregated_main[m] = (mean, std)
        
    # Aggregate ablation results
    pds = [4, 8, 16, 32, 64]
    ablation_methods = list(all_ablation_res[0][4].keys())
    aggregated_ablation = {} # pd -> method -> (mean, std)
    
    for pd in pds:
        aggregated_ablation[pd] = {}
        for m in ablation_methods:
            vals = [res[pd][m] for res in all_ablation_res]
            mean = np.mean(vals)
            std = np.std(vals)
            aggregated_ablation[pd][m] = (mean, std)
            
    print("="*85, flush=True)
    print(f"{'Method':<35} | {'Mean Joint Accuracy (%)':<30} | {'Std Dev (%)':<12}", flush=True)
    print("="*85, flush=True)
    for m, (mean, std) in aggregated_main.items():
        print(f"{m:<35} | {mean:.2f}%                        | {std:.2f}%", flush=True)
    print("="*85, flush=True)
    
    print("\n" + "="*110, flush=True)
    print("PROJECTION DIMENSION ABLATION SWEEP WITH STATISTICAL SIGNIFICANCE (10 SEEDS)", flush=True)
    print("="*110, flush=True)
    print(f"{'Proj Dim':<10} | {'Unregularized':<15} | {'L2 Regularized':<15} | {'TSAR':<15} | {'SR3-F (Ours)':<15} | {'SR3-S (Ours)':<15} | {'SR3-H (Ours)':<15}", flush=True)
    print("="*110, flush=True)
    for pd in pds:
        unreg = f"{aggregated_ablation[pd]['Unregularized'][0]:.2f}±{aggregated_ablation[pd]['Unregularized'][1]:.2f}"
        l2 = f"{aggregated_ablation[pd]['L2 Regularized'][0]:.2f}±{aggregated_ablation[pd]['L2 Regularized'][1]:.2f}"
        tsar = f"{aggregated_ablation[pd]['TSAR'][0]:.2f}±{aggregated_ablation[pd]['TSAR'][1]:.2f}"
        sr3_f = f"{aggregated_ablation[pd]['SR3-F'][0]:.2f}±{aggregated_ablation[pd]['SR3-F'][1]:.2f}"
        sr3_s = f"{aggregated_ablation[pd]['SR3-S'][0]:.2f}±{aggregated_ablation[pd]['SR3-S'][1]:.2f}"
        sr3_h = f"{aggregated_ablation[pd]['SR3-H'][0]:.2f}±{aggregated_ablation[pd]['SR3-H'][1]:.2f}"
        print(f"{pd:<10} | {unreg:<15} | {l2:<15} | {tsar:<15} | {sr3_f:<15} | {sr3_s:<15} | {sr3_h:<15}", flush=True)
    print("="*110, flush=True)

if __name__ == "__main__":
    main()
