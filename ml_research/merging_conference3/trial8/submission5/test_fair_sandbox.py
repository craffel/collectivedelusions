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
OOD_THRESHOLD = 0.15

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def generate_sandbox_data(seed, overlap_bleed=0.08):
    setup_seed(seed)
    subspace_dim = 48
    
    # Generate class prototypes for each task: (K, C, subspace_dim)
    prototypes = torch.randn(K, C, subspace_dim)
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
    
    noise_scales = [0.05, 0.15, 0.40, 1.20]
    
    def generate_split(num_samples_per_task):
        X = []
        Y_task = []
        Y_class = []
        for k in range(K):
            for _ in range(num_samples_per_task):
                c = torch.randint(0, C, (1,)).item()
                # Initialize representation as zeros
                z = torch.zeros(D)
                proto = prototypes[k, c]
                noise = noise_scales[k] * torch.randn(subspace_dim)
                z[k*subspace_dim : (k+1)*subspace_dim] = proto + noise
                
                # Critical Flaw 1 Fix: Add representational overlap/bleed across other task subspaces
                # Instead of exactly 0.0 outside the task subspace, we add small random values
                bleed = overlap_bleed * torch.randn(D)
                # Keep task-specific subspace dominant but other dimensions non-zero
                bleed[k*subspace_dim : (k+1)*subspace_dim] = 0.0
                z = z + bleed
                
                X.append(z)
                Y_task.append(k)
                Y_class.append(c)
        return torch.stack(X), torch.tensor(Y_task), torch.tensor(Y_class)
    
    train = generate_split(1000)
    cal = generate_split(16)
    test = generate_split(250)
    
    return prototypes, noise_scales, train, cal, test

def train_linear_router(cal_data):
    X_cal, Y_cal_task, _ = cal_data
    class LinearRouterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(D, K)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        def forward(self, x):
            return self.linear(x)
            
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

def run_simulation_seed(seed, use_base_backbone=True):
    prototypes, noise_scales, train_data, cal_data, test_data = generate_sandbox_data(seed)
    X_train, Y_train_task, Y_train_class = train_data
    X_cal, Y_cal_task, Y_cal_class = cal_data
    X_test, Y_test_task, Y_test_class = test_data
    
    subspace_dim = 48
    
    # Initialize base model transformations (random but frozen orthogonal-like transformations at each layer)
    # This creates a real representational manifold transformation, bridging Layer 0 to Layer 12
    W_base = torch.zeros(L, D, D)
    for l in range(L):
        # Identity with small perturbation to simulate realistic transformer representation shifts
        W_base[l] = torch.eye(D) + torch.randn(D, D) * 0.015
        
    # LoRA experts are initialized densely to prevent hard coordinate masking (Critical Flaw 1)
    A = torch.randn(K, L, D, R) * LORA_SCALE
    B = torch.randn(K, L, R, D) * LORA_SCALE
    
    # Task localization in LoRA: make them dominant in task subspace but non-zero elsewhere
    for k in range(K):
        sub_start = k * subspace_dim
        sub_end = (k + 1) * subspace_dim
        for l in range(L):
            A[k, l, :sub_start, :] *= 0.15
            A[k, l, sub_end:, :] *= 0.15
            B[k, l, :, :sub_start] *= 0.15
            B[k, l, :, sub_end:] *= 0.15
            
    # Expert heads are initialized densely and trained
    expert_heads_W = torch.randn(K, C, D) * 0.05
    expert_heads_B = torch.zeros(K, C)
    for k in range(K):
        expert_heads_W[k, :, k*subspace_dim : (k+1)*subspace_dim] += prototypes[k]
        
    # Train each expert with its respective LoRA adapter and head
    for k in range(K):
        task_mask = (Y_train_task == k)
        X_task = X_train[task_mask]
        Y_class_task = Y_train_class[task_mask]
        
        z = X_task.clone()
        for l in range(L):
            if use_base_backbone:
                z = z @ W_base[l]
            delta = z @ B[k, l].t() @ A[k, l].t()
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

    # PEAR utilizes Zero-Shot Patch Centroids (ZPC) class-wise at Layer 0 (Parameter-Free and space-aligned)
    # Centroids computed at Layer 0 using calibration split for each of the K * C classes
    mu_class = torch.zeros(K, C, D)
    for k in range(K):
        for c in range(C):
            class_mask = (Y_cal_task == k) & (Y_cal_class == c)
            if class_mask.sum() > 0:
                mu_class[k, c] = X_cal[class_mask].mean(dim=0)
            else:
                mu_class[k, c] = X_cal[Y_cal_task == k].mean(dim=0)
        
    # PEAR Dispersion calibration at Layer 0 using ZPC
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
    
    # SABLE routes at Layer 10 using centroids of base-transformed representations (Critical Flaw 2 Fix)
    # Compute base-model representations at Layer 10 for the calibration set
    X_cal_10 = X_cal.clone()
    for l in range(10):
        if use_base_backbone:
            X_cal_10 = X_cal_10 @ W_base[l]
            
    mu_sable = torch.zeros(K, C, D)
    for k in range(K):
        for c in range(C):
            class_mask = (Y_cal_task == k) & (Y_cal_class == c)
            if class_mask.sum() > 0:
                mu_sable[k, c] = X_cal_10[class_mask].mean(dim=0)
            else:
                mu_sable[k, c] = X_cal_10[Y_cal_task == k].mean(dim=0)

    def propagate_layers_pear(xb, coeffs_batch):
        # PEAR propagates through all 12 adapted layers
        z = xb.clone()
        for l in range(L):
            if use_base_backbone:
                base_trans = z @ W_base[l]
            else:
                base_trans = z
            delta_sum = torch.zeros_like(z)
            for k in range(K):
                delta_k = z @ B[k, l].t() @ A[k, l].t()
                delta_sum += coeffs_batch[:, k:k+1] * delta_k
            z = base_trans + delta_sum
        return z

    def propagate_layers_sable(xb, coeffs_batch):
        # SABLE runs unadapted base backbone for layers 0-9
        z = xb.clone()
        for l in range(10):
            if use_base_backbone:
                z = z @ W_base[l]
        # SABLE ensembles only layers 10-12
        for l in range(10, L):
            if use_base_backbone:
                base_trans = z @ W_base[l]
            else:
                base_trans = z
            delta_sum = torch.zeros_like(z)
            for k in range(K):
                delta_k = z @ B[k, l].t() @ A[k, l].t()
                delta_sum += coeffs_batch[:, k:k+1] * delta_k
            z = base_trans + delta_sum
        return z

def main():
    seeds = [10, 11, 12, 13, 14]
    all_results = []
    for s in seeds:
        print(f"Running Seed {s}...")
        # We need to collect results
        # To do so, let's modify run_simulation_seed to return the accuracy dict
        res = run_simulation_seed(s)
        all_results.append(res)
        
    print("\n================== AVERAGED EXPERIMENTAL RESULTS (5 SEEDS) ==================\n")
    baselines = ["ceil", "unif", "lin", "pear", "sable"]
    for b in baselines:
        accs = [res[b] for res in all_results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{b:<10}: Mean Accuracy = {mean_acc:.2f}% ± {std_acc:.2f}%")

def run_simulation_seed(seed, use_base_backbone=True):
    prototypes, noise_scales, train_data, cal_data, test_data = generate_sandbox_data(seed)
    X_train, Y_train_task, Y_train_class = train_data
    X_cal, Y_cal_task, Y_cal_class = cal_data
    X_test, Y_test_task, Y_test_class = test_data
    
    subspace_dim = 48
    
    # Initialize base model transformations (random but frozen orthogonal-like transformations at each layer)
    W_base = torch.zeros(L, D, D)
    for l in range(L):
        W_base[l] = torch.eye(D) + torch.randn(D, D) * 0.015
        
    A = torch.randn(K, L, D, R) * LORA_SCALE
    B = torch.randn(K, L, R, D) * LORA_SCALE
    
    # Task localization in LoRA: make them dominant in task subspace but non-zero elsewhere
    for k in range(K):
        sub_start = k * subspace_dim
        sub_end = (k + 1) * subspace_dim
        for l in range(L):
            A[k, l, :sub_start, :] *= 0.15
            A[k, l, sub_end:, :] *= 0.15
            B[k, l, :, :sub_start] *= 0.15
            B[k, l, :, sub_end:] *= 0.15
            
    expert_heads_W = torch.randn(K, C, D) * 0.05
    expert_heads_B = torch.zeros(K, C)
    for k in range(K):
        expert_heads_W[k, :, k*subspace_dim : (k+1)*subspace_dim] += prototypes[k]
        
    for k in range(K):
        task_mask = (Y_train_task == k)
        X_task = X_train[task_mask]
        Y_class_task = Y_train_class[task_mask]
        
        z = X_task.clone()
        for l in range(L):
            if use_base_backbone:
                z = z @ W_base[l]
            delta = z @ B[k, l].t() @ A[k, l].t()
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

    mu_class = torch.zeros(K, C, D)
    for k in range(K):
        for c in range(C):
            class_mask = (Y_cal_task == k) & (Y_cal_class == c)
            if class_mask.sum() > 0:
                mu_class[k, c] = X_cal[class_mask].mean(dim=0)
            else:
                mu_class[k, c] = X_cal[Y_cal_task == k].mean(dim=0)
        
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
    
    X_cal_10 = X_cal.clone()
    for l in range(10):
        if use_base_backbone:
            X_cal_10 = X_cal_10 @ W_base[l]
            
    mu_sable = torch.zeros(K, C, D)
    for k in range(K):
        for c in range(C):
            class_mask = (Y_cal_task == k) & (Y_cal_class == c)
            if class_mask.sum() > 0:
                mu_sable[k, c] = X_cal_10[class_mask].mean(dim=0)
            else:
                mu_sable[k, c] = X_cal_10[Y_cal_task == k].mean(dim=0)

    def propagate_layers_pear(xb, coeffs_batch):
        z = xb.clone()
        for l in range(L):
            if use_base_backbone:
                base_trans = z @ W_base[l]
            else:
                base_trans = z
            delta_sum = torch.zeros_like(z)
            for k in range(K):
                delta_k = z @ B[k, l].t() @ A[k, l].t()
                delta_sum += coeffs_batch[:, k:k+1] * delta_k
            z = base_trans + delta_sum
        return z

    def propagate_layers_sable(xb, coeffs_batch):
        z = xb.clone()
        for l in range(10):
            if use_base_backbone:
                z = z @ W_base[l]
        for l in range(10, L):
            if use_base_backbone:
                base_trans = z @ W_base[l]
            else:
                base_trans = z
            delta_sum = torch.zeros_like(z)
            for k in range(K):
                delta_k = z @ B[k, l].t() @ A[k, l].t()
                delta_sum += coeffs_batch[:, k:k+1] * delta_k
            z = base_trans + delta_sum
        return z

    def evaluate_test():
        total = len(X_test)
        
        # 1. Expert Ceiling
        z_ceil = torch.zeros(total, D)
        for b_idx in range(total):
            k = Y_test_task[b_idx].item()
            z = X_test[b_idx].clone()
            for l in range(L):
                if use_base_backbone:
                    z = z @ W_base[l]
                delta = z @ B[k, l].t() @ A[k, l].t()
                z = z + delta
            z_ceil[b_idx] = z
            
        logits_ceil = torch.zeros(total, C)
        for b_idx in range(total):
            k = Y_test_task[b_idx].item()
            logits_ceil[b_idx] = z_ceil[b_idx] @ expert_heads_W[k].t() + expert_heads_B[k]
        acc_ceil = (logits_ceil.argmax(dim=-1) == Y_test_class).float().mean().item() * 100

        # 2. Uniform Merging
        uniform_coeffs = torch.ones(total, K) * 0.25
        z_unif = propagate_layers_pear(X_test, uniform_coeffs)
        logits_unif = torch.zeros(total, C)
        for j in range(K):
            logits_unif += 0.25 * (z_unif @ expert_heads_W[j].t() + expert_heads_B[j])
        acc_unif = (logits_unif.argmax(dim=-1) == Y_test_class).float().mean().item() * 100

        # 3. Linear Router
        with torch.no_grad():
            route_logits_lin = trained_router(X_test)
            route_lin = torch.softmax(route_logits_lin, dim=-1)
        z_lin = propagate_layers_pear(X_test, route_lin)
        logits_lin = torch.zeros(total, C)
        for j in range(K):
            logits_lin += route_lin[:, j:j+1] * (z_lin @ expert_heads_W[j].t() + expert_heads_B[j])
        acc_lin = (logits_lin.argmax(dim=-1) == Y_test_class).float().mean().item() * 100

        # 4. PEAR
        cos_sims_pear = torch.zeros(total, K)
        for j in range(K):
            mu_j = mu_class[j]
            mu_j_norms = mu_j.norm(dim=-1, keepdim=True) + 1e-8
            for b_idx in range(total):
                x_b = X_test[b_idx]
                x_b_norm = x_b.norm() + 1e-8
                dot_prods = mu_j @ x_b
                cos_class = dot_prods / (mu_j_norms.squeeze() * x_b_norm)
                cos_sims_pear[b_idx, j] = cos_class.max()
                
        calibrated_pear = cos_sims_pear / (pear_dispersion.unsqueeze(0) + 1e-8)
        route_pear = torch.softmax(calibrated_pear / 0.001, dim=-1)
        
        max_sims, _ = cos_sims_pear.max(dim=-1)
        for b_idx in range(total):
            if max_sims[b_idx] < OOD_THRESHOLD:
                route_pear[b_idx] = 0.0
                
        z_pear = propagate_layers_pear(X_test, route_pear)
        logits_pear = torch.zeros(total, C)
        for j in range(K):
            logits_pear += route_pear[:, j:j+1] * (z_pear @ expert_heads_W[j].t() + expert_heads_B[j])
        acc_pear = (logits_pear.argmax(dim=-1) == Y_test_class).float().mean().item() * 100

        # 5. SABLE
        X_test_10 = X_test.clone()
        for l in range(10):
            if use_base_backbone:
                X_test_10 = X_test_10 @ W_base[l]
                
        cos_sims_sable = torch.zeros(total, K)
        for j in range(K):
            mu_j = mu_sable[j]
            mu_j_norms = mu_j.norm(dim=-1, keepdim=True) + 1e-8
            for b_idx in range(total):
                x_b = X_test_10[b_idx]
                x_b_norm = x_b.norm() + 1e-8
                dot_prods = mu_j @ x_b
                cos_class = dot_prods / (mu_j_norms.squeeze() * x_b_norm)
                cos_sims_sable[b_idx, j] = cos_class.max()
                
        route_sable = torch.softmax(cos_sims_sable / 0.05, dim=-1)
        z_sable = propagate_layers_sable(X_test, route_sable)
        logits_sable = torch.zeros(total, C)
        for j in range(K):
            logits_sable += route_sable[:, j:j+1] * (z_sable @ expert_heads_W[j].t() + expert_heads_B[j])
        acc_sable = (logits_sable.argmax(dim=-1) == Y_test_class).float().mean().item() * 100
        
        return {
            "ceil": acc_ceil,
            "unif": acc_unif,
            "lin": acc_lin,
            "pear": acc_pear,
            "sable": acc_sable
        }

    return evaluate_test()

if __name__ == "__main__":
    main()
