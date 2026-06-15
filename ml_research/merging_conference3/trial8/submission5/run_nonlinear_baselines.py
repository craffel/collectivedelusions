import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Global Hyperparameters
D = 192          # Feature dimension
K = 4            # Number of tasks/experts (MNIST, F-MNIST, CIFAR-10, SVHN)
C = 10           # Classes per task
L = 12           # Number of layers
R = 8            # LoRA rank
LORA_SCALE = 0.1 # Scaling of LoRA adapters to simulate fine-tuning deviation
OOD_THRESHOLD = 0.05

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_subspace_range(k):
    # Overlapping subspaces of size 96
    start = k * 32
    end = start + 96
    return start, end

def generate_sandbox_data(seed, overlap_bleed=0.08):
    setup_seed(seed)
    subspace_size = 96
    
    # Generate class prototypes for each task: (K, C, subspace_size)
    prototypes = torch.randn(K, C, subspace_size)
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
    
    # Noise scales chosen to match the expert ceilings reported in papers:
    # Task 0 (MNIST): ~100%, Task 1 (F-MNIST): ~97%, Task 2 (CIFAR-10): ~84%, Task 3 (SVHN): ~32%
    noise_scales = [0.05, 0.15, 0.40, 1.20]
    
    def generate_split(num_samples_per_task):
        X = []
        Y_task = []
        Y_class = []
        for k in range(K):
            start, end = get_subspace_range(k)
            for _ in range(num_samples_per_task):
                c = torch.randint(0, C, (1,)).item()
                z = torch.zeros(D)
                proto = prototypes[k, c]
                noise = noise_scales[k] * torch.randn(subspace_size)
                z[start:end] = proto + noise
                
                # Critical Flaw 1 Fix: Add representational overlap/bleed across other task subspaces
                bleed = overlap_bleed * torch.randn(D)
                bleed[start:end] = 0.0
                z = z + bleed
                
                X.append(z)
                Y_task.append(k)
                Y_class.append(c)
        return torch.stack(X), torch.tensor(Y_task), torch.tensor(Y_class)
    
    train = generate_split(1000)
    cal = generate_split(16)
    test = generate_split(250)
    
    return prototypes, noise_scales, train, cal, test

class LinearRouterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(D, K)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        return self.linear(x)

def train_linear_router(cal_data):
    X_cal, Y_cal_task, _ = cal_data
    router = LinearRouterModel()
    optimizer = optim.AdamW(router.parameters(), lr=1e-2, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    router.train()
    for epoch in range(100):
        optimizer.zero_grad()
        logits = router(X_cal)
        loss = criterion(logits, Y_cal_task)
        loss.backward()
        optimizer.step()
        
    return router

def run_simulation_seed_nonlinear(seed, use_nonlinear=True):
    prototypes, noise_scales, train_data, cal_data, test_data = generate_sandbox_data(seed)
    X_train, Y_train_task, Y_train_class = train_data
    X_cal, Y_cal_task, Y_cal_class = cal_data
    X_test, Y_test_task, Y_test_class = test_data
    
    # Initialize base model layer transformations
    W_base = torch.zeros(L, D, D)
    for l in range(L):
        W_base[l] = torch.eye(D) + torch.randn(D, D) * 0.015
        
    # LoRA experts initialized densely to prevent hard coordinate masking
    A = torch.randn(K, L, D, R) * LORA_SCALE
    B = torch.randn(K, L, R, D) * LORA_SCALE
    
    # Task localization in LoRA
    for k in range(K):
        start, end = get_subspace_range(k)
        for l in range(L):
            A[k, l, :start, :] *= 0.15
            A[k, l, end:, :] *= 0.15
            B[k, l, :, :start] *= 0.15
            B[k, l, :, end:] *= 0.15
            
    expert_heads_W = torch.randn(K, C, D) * 0.05
    expert_heads_B = torch.zeros(K, C)
    
    for k in range(K):
        start, end = get_subspace_range(k)
        expert_heads_W[k, :, start:end] += prototypes[k]
        
    for k in range(K):
        task_mask = (Y_train_task == k)
        X_task = X_train[task_mask]
        Y_class_task = Y_train_class[task_mask]
        
        z = X_task.clone()
        for l in range(L):
            z = z @ W_base[l]
            delta = z @ B[k, l].t() @ A[k, l].t()
            if use_nonlinear:
                delta = torch.nn.functional.gelu(delta)
            z = z + delta
            
        head_W = nn.Parameter(expert_heads_W[k].clone())
        head_B = nn.Parameter(expert_heads_B[k].clone())
        opt_head = optim.AdamW([head_W, head_B], lr=1e-1)
        
        for _ in range(50):
            opt_head.zero_grad()
            logits = z @ head_W.t() + head_B
            loss = nn.CrossEntropyLoss()(logits, Y_class_task)
            loss.backward()
            opt_head.step()
            
        expert_heads_W[k] = head_W.detach()
        expert_heads_B[k] = head_B.detach()

    # PEAR utilizes Zero-Shot Patch Centroids (ZPC) class-wise at Layer 0
    mu_class = torch.zeros(K, C, D)
    for k in range(K):
        for c in range(C):
            class_mask = (Y_cal_task == k) & (Y_cal_class == c)
            if class_mask.sum() > 0:
                mu_class[k, c] = X_cal[class_mask].mean(dim=0)
            else:
                mu_class[k, c] = X_cal[Y_cal_task == k].mean(dim=0)
        
    # PEAR Dispersion calibration using ZPC
    pear_dispersion = torch.zeros(K)
    for k in range(K):
        task_mask = (Y_cal_task == k)
        X_task_cal = X_cal[task_mask]
        Y_class_cal = Y_cal_class[task_mask]
        sims = []
        for s_idx, s in enumerate(X_task_cal):
            c_label = Y_class_cal[s_idx].item()
            w_c = mu_class[k, c_label]
            cos_sim = torch.dot(s, w_c) / (s.norm() * w_c.norm() + 1e-8)
            sims.append(cos_sim.item())
        pear_dispersion[k] = np.mean(sims)

    trained_router = train_linear_router(cal_data)
    
    # SABLE routes at Layer 10
    X_cal_10 = X_cal.clone()
    for l in range(10):
        X_cal_10 = X_cal_10 @ W_base[l]
            
    mu_sable = torch.zeros(K, C, D)
    for k in range(K):
        for c in range(C):
            class_mask = (Y_cal_task == k) & (Y_cal_class == c)
            if class_mask.sum() > 0:
                mu_sable[k, c] = X_cal_10[class_mask].mean(dim=0)
            else:
                mu_sable[k, c] = X_cal_10[Y_cal_task == k].mean(dim=0)

    def propagate_layers_pear_local(xb, coeffs_batch):
        z = xb.clone()
        for l in range(L):
            base_trans = z @ W_base[l]
            delta_sum = torch.zeros_like(z)
            for k in range(K):
                delta_k = z @ B[k, l].t() @ A[k, l].t()
                if use_nonlinear:
                    delta_k = torch.nn.functional.gelu(delta_k)
                delta_sum += coeffs_batch[:, k:k+1] * delta_k
            z = base_trans + delta_sum
        return z

    def propagate_layers_sable_local(xb, coeffs_batch):
        z = xb.clone()
        for l in range(10):
            z = z @ W_base[l]
        for l in range(10, L):
            base_trans = z @ W_base[l]
            delta_sum = torch.zeros_like(z)
            for k in range(K):
                delta_k = z @ B[k, l].t() @ A[k, l].t()
                if use_nonlinear:
                    delta_k = torch.nn.functional.gelu(delta_k)
                delta_sum += coeffs_batch[:, k:k+1] * delta_k
            z = base_trans + delta_sum
        return z

    def evaluate_heterogeneous_batch_base_local(method, batch_size):
        perm = torch.randperm(len(X_test))
        X_hetero = X_test[perm]
        Y_task_hetero = Y_test_task[perm]
        Y_class_hetero = Y_test_class[perm]
        
        correct_preds = [0]*K
        total_preds = [0]*K
        
        for i in range(0, len(X_hetero), batch_size):
            xb = X_hetero[i : i + batch_size]
            yb_task = Y_task_hetero[i : i + batch_size]
            yb_class = Y_class_hetero[i : i + batch_size]
            actual_B = len(xb)
            
            if method == "expert_ceiling":
                logits = torch.zeros(actual_B, C)
                for b_idx in range(actual_B):
                    k = yb_task[b_idx].item()
                    z = xb[b_idx].clone()
                    for l in range(L):
                        z = z @ W_base[l]
                        delta = z @ B[k, l].t() @ A[k, l].t()
                        if use_nonlinear:
                            delta = torch.nn.functional.gelu(delta)
                        z = z + delta
                    logits[b_idx] = z @ expert_heads_W[k].t() + expert_heads_B[k]
                preds = logits.argmax(dim=-1)
                
            elif method == "uniform":
                coeffs = torch.ones(actual_B, K) * 0.25
                z = propagate_layers_pear_local(xb, coeffs)
                logits = torch.zeros(actual_B, C)
                for j in range(K):
                    logits += 0.25 * (z @ expert_heads_W[j].t() + expert_heads_B[j])
                preds = logits.argmax(dim=-1)
                
            elif method == "linear_router":
                with torch.no_grad():
                    route_logits = trained_router(xb)
                    route_coeffs = torch.softmax(route_logits, dim=-1)
                    mean_coeffs = route_coeffs.mean(dim=0, keepdim=True).repeat(actual_B, 1)
                    
                z = propagate_layers_pear_local(xb, mean_coeffs)
                logits = torch.zeros(actual_B, C)
                for j in range(K):
                    logits += mean_coeffs[:, j:j+1] * (z @ expert_heads_W[j].t() + expert_heads_B[j])
                preds = logits.argmax(dim=-1)
                
            elif method == "pfsr_mbh":
                with torch.no_grad():
                    cos_sims = torch.zeros(actual_B, K)
                    for j in range(K):
                        mu_j = mu_class[j]
                        mu_j_norms = mu_j.norm(dim=-1, keepdim=True) + 1e-8
                        for b_idx in range(actual_B):
                            x_b = xb[b_idx]
                            dot_prods = mu_j @ x_b
                            cos_class = dot_prods / (mu_j_norms.squeeze() * x_b.norm() + 1e-8)
                            cos_sims[b_idx, j] = cos_class.max()
                    route_coeffs = torch.softmax(cos_sims / 0.05, dim=-1)
                    mean_coeffs = route_coeffs.mean(dim=0, keepdim=True).repeat(actual_B, 1)
                    
                z = propagate_layers_pear_local(xb, mean_coeffs)
                logits = torch.zeros(actual_B, C)
                for j in range(K):
                    logits += mean_coeffs[:, j:j+1] * (z @ expert_heads_W[j].t() + expert_heads_B[j])
                preds = logits.argmax(dim=-1)
                
            elif method == "sable":
                with torch.no_grad():
                    z_10 = xb.clone()
                    for l in range(10):
                        z_10 = z_10 @ W_base[l]
                    cos_sims = torch.zeros(actual_B, K)
                    for j in range(K):
                        mu_j = mu_sable[j]
                        mu_j_norms = mu_j.norm(dim=-1, keepdim=True) + 1e-8
                        for b_idx in range(actual_B):
                            x_b = z_10[b_idx]
                            dot_prods = mu_j @ x_b
                            cos_class = dot_prods / (mu_j_norms.squeeze() * x_b.norm() + 1e-8)
                            cos_sims[b_idx, j] = cos_class.max()
                    route_coeffs = torch.softmax(cos_sims / 0.05, dim=-1)
                    
                z = propagate_layers_sable_local(xb, route_coeffs)
                logits = torch.zeros(actual_B, C)
                for j in range(K):
                    logits += route_coeffs[:, j:j+1] * (z @ expert_heads_W[j].t() + expert_heads_B[j])
                preds = logits.argmax(dim=-1)
                
            elif method == "pear":
                with torch.no_grad():
                    cos_sims = torch.zeros(actual_B, K)
                    for j in range(K):
                        mu_j = mu_class[j]
                        mu_j_norms = mu_j.norm(dim=-1, keepdim=True) + 1e-8
                        for b_idx in range(actual_B):
                            x_b = xb[b_idx]
                            dot_prods = mu_j @ x_b
                            cos_class = dot_prods / (mu_j_norms.squeeze() * x_b.norm() + 1e-8)
                            cos_sims[b_idx, j] = cos_class.max()
                    calibrated_sims = cos_sims / (pear_dispersion.unsqueeze(0) + 1e-8)
                    route_coeffs = torch.softmax(calibrated_sims / 0.05, dim=-1)
                    max_sims, _ = cos_sims.max(dim=-1)
                    for b_idx in range(actual_B):
                        if max_sims[b_idx] < OOD_THRESHOLD:
                            route_coeffs[b_idx] = 1.0 / K
                            
                z = propagate_layers_pear_local(xb, route_coeffs)
                logits = torch.zeros(actual_B, C)
                for j in range(K):
                    logits += route_coeffs[:, j:j+1] * (z @ expert_heads_W[j].t() + expert_heads_B[j])
                preds = logits.argmax(dim=-1)
                
            for b_idx in range(actual_B):
                k = yb_task[b_idx].item()
                correct_preds[k] += (preds[b_idx] == yb_class[b_idx]).item()
                total_preds[k] += 1
                
        accs = [correct_preds[k]/total_preds[k]*100 for k in range(K)]
        return accs

    results = {}
    methods = ["expert_ceiling", "uniform", "linear_router", "pfsr_mbh", "sable", "pear"]
    
    results["het_1"] = {}
    for m in methods:
        results["het_1"][m] = evaluate_heterogeneous_batch_base_local(m, 1)
        
    return results

def main():
    seeds = [10, 11, 12, 13, 14]
    all_results = []
    for s in seeds:
        print(f"Running Non-Linear Seed {s}...")
        res = run_simulation_seed_nonlinear(s, use_nonlinear=True)
        all_results.append(res)
        
    methods = ["expert_ceiling", "uniform", "linear_router", "pfsr_mbh", "sable", "pear"]
    
    print("\n================== NON-LINEAR EXPERIMENTAL RESULTS (HET_1) ==================\n")
    for m in methods:
        accs = []
        for s_idx in range(len(seeds)):
            accs.append(all_results[s_idx]["het_1"][m])
        accs = np.array(accs)
        mean_tasks = accs.mean(axis=0)
        std_tasks = accs.std(axis=0)
        
        joint_means = accs.mean(axis=1)
        mean_joint = joint_means.mean()
        std_joint = joint_means.std()
        
        print(f"{m:<18}: "
              f"MNIST: {mean_tasks[0]:.2f} ± {std_tasks[0]:.2f}%, "
              f"F-MNIST: {mean_tasks[1]:.2f} ± {std_tasks[1]:.2f}%, "
              f"CIFAR-10: {mean_tasks[2]:.2f} ± {std_tasks[2]:.2f}%, "
              f"SVHN: {mean_tasks[3]:.2f} ± {std_tasks[3]:.2f}%, "
              f"Joint Mean: {mean_joint:.2f} ± {std_joint:.2f}%")
    print()

if __name__ == "__main__":
    main()
