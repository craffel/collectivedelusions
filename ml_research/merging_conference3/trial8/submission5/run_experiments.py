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

def run_simulation_seed(seed, sweep_tau=None, sweep_ood=None):
    prototypes, noise_scales, train_data, cal_data, test_data = generate_sandbox_data(seed)
    X_train, Y_train_task, Y_train_class = train_data
    X_cal, Y_cal_task, Y_cal_class = cal_data
    X_test, Y_test_task, Y_test_class = test_data
    
    # Initialize base model layer transformations (random but frozen orthogonal-like representation shifts)
    W_base = torch.zeros(L, D, D)
    for l in range(L):
        W_base[l] = torch.eye(D) + torch.randn(D, D) * 0.015
        
    # LoRA experts are initialized densely to prevent hard coordinate masking (Critical Flaw 1 Fix)
    A = torch.randn(K, L, D, R) * LORA_SCALE
    B = torch.randn(K, L, R, D) * LORA_SCALE
    
    # Task localization in LoRA: make them dominant in task subspace but non-zero elsewhere
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
            base_trans = z @ W_base[l]
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
            z = z @ W_base[l]
        # SABLE ensembles only layers 10-12
        for l in range(10, L):
            base_trans = z @ W_base[l]
            delta_sum = torch.zeros_like(z)
            for k in range(K):
                delta_k = z @ B[k, l].t() @ A[k, l].t()
                delta_sum += coeffs_batch[:, k:k+1] * delta_k
            z = base_trans + delta_sum
        return z

    def evaluate_heterogeneous_batch(method, batch_size):
        perm = torch.randperm(len(X_test))
        X_hetero = X_test[perm]
        Y_task_hetero = Y_test_task[perm]
        Y_class_hetero = Y_test_class[perm]
        
        correct_preds = [0]*K
        total_preds = [0]*K
        
        tau_val = sweep_tau if sweep_tau is not None else 0.05
        ood_val = sweep_ood if sweep_ood is not None else OOD_THRESHOLD
        
        for i in range(0, len(X_hetero), batch_size):
            xb = X_hetero[i : i + batch_size]
            yb_task = Y_task_hetero[i : i + batch_size]
            yb_class = Y_class_hetero[i : i + batch_size]
            actual_B = len(xb)
            
            if method == "pear":
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
                    route_coeffs = torch.softmax(calibrated_sims / tau_val, dim=-1)
                    
                    # OOD rejection
                    max_sims, _ = cos_sims.max(dim=-1)
                    for b_idx in range(actual_B):
                        if max_sims[b_idx] < ood_val:
                            route_coeffs[b_idx] = 1.0 / K
                            
                z = propagate_layers_pear(xb, route_coeffs)
                logits = torch.zeros(actual_B, C)
                for j in range(K):
                    logits += route_coeffs[:, j:j+1] * (z @ expert_heads_W[j].t() + expert_heads_B[j])
                preds = logits.argmax(dim=-1)
            else:
                raise NotImplementedError()
                
            for b_idx in range(actual_B):
                k = yb_task[b_idx].item()
                correct_preds[k] += (preds[b_idx] == yb_class[b_idx]).item()
                total_preds[k] += 1
                
        accs = [correct_preds[k]/total_preds[k]*100 for k in range(K)]
        return accs

    # Used strictly for sweeps
    if sweep_tau is not None or sweep_ood is not None:
        return evaluate_heterogeneous_batch("pear", 256)

    # Base full evaluations
    results = {}
    methods = ["expert_ceiling", "uniform", "linear_router", "pfsr_mbh", "sable", "pear"]
    
    results["hom_256"] = {}
    for m in methods:
        correct_preds = [0]*K
        total_preds = [0]*K
        
        # Homogeneous evaluation logic
        for k in range(K):
            task_mask = (Y_test_task == k)
            X_task = X_test[task_mask]
            Y_class_task = Y_test_class[task_mask]
            
            B_size = 256
            for i in range(0, len(X_task), B_size):
                xb = X_task[i : i + B_size]
                yb = Y_class_task[i : i + B_size]
                actual_B = len(xb)
                
                if m == "expert_ceiling":
                    z_ceil = torch.zeros(actual_B, D)
                    for b_idx in range(actual_B):
                        z = xb[b_idx].clone()
                        for l in range(L):
                            z = z @ W_base[l]
                            delta = z @ B[k, l].t() @ A[k, l].t()
                            z = z + delta
                        z_ceil[b_idx] = z
                    logits = z_ceil @ expert_heads_W[k].t() + expert_heads_B[k]
                    preds = logits.argmax(dim=-1)
                    
                elif m == "uniform":
                    coeffs = torch.ones(actual_B, K) * 0.25
                    z = propagate_layers_pear(xb, coeffs)
                    logits = torch.zeros(actual_B, C)
                    for j in range(K):
                        logits += 0.25 * (z @ expert_heads_W[j].t() + expert_heads_B[j])
                    preds = logits.argmax(dim=-1)
                    
                elif m == "linear_router":
                    with torch.no_grad():
                        route_logits = trained_router(xb)
                        route_coeffs = torch.softmax(route_logits, dim=-1)
                        mean_coeffs = route_coeffs.mean(dim=0, keepdim=True).repeat(actual_B, 1)
                    z = propagate_layers_pear(xb, mean_coeffs)
                    logits = torch.zeros(actual_B, C)
                    for j in range(K):
                        logits += mean_coeffs[:, j:j+1] * (z @ expert_heads_W[j].t() + expert_heads_B[j])
                    preds = logits.argmax(dim=-1)
                    
                elif m == "pfsr_mbh":
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
                    z = propagate_layers_pear(xb, mean_coeffs)
                    logits = torch.zeros(actual_B, C)
                    for j in range(K):
                        logits += mean_coeffs[:, j:j+1] * (z @ expert_heads_W[j].t() + expert_heads_B[j])
                    preds = logits.argmax(dim=-1)
                    
                elif m == "sable":
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
                    z = propagate_layers_sable(xb, route_coeffs)
                    logits = torch.zeros(actual_B, C)
                    for j in range(K):
                        logits += route_coeffs[:, j:j+1] * (z @ expert_heads_W[j].t() + expert_heads_B[j])
                    preds = logits.argmax(dim=-1)
                    
                elif m == "pear":
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
                    z = propagate_layers_pear(xb, route_coeffs)
                    logits = torch.zeros(actual_B, C)
                    for j in range(K):
                        logits += route_coeffs[:, j:j+1] * (z @ expert_heads_W[j].t() + expert_heads_B[j])
                    preds = logits.argmax(dim=-1)
                    
                correct_preds[k] += (preds == yb).sum().item()
                total_preds[k] += actual_B
                
        results["hom_256"][m] = [correct_preds[k]/total_preds[k]*100 for k in range(K)]

    results["het_256"] = {}
    for m in methods:
        results["het_256"][m] = evaluate_heterogeneous_batch_base(m, 256, X_test, Y_test_task, Y_test_class, expert_heads_W, expert_heads_B, trained_router, pear_dispersion, A, B, W_base, mu_class, mu_sable)

    results["het_1"] = {}
    for m in methods:
        results["het_1"][m] = evaluate_heterogeneous_batch_base(m, 1, X_test, Y_test_task, Y_test_class, expert_heads_W, expert_heads_B, trained_router, pear_dispersion, A, B, W_base, mu_class, mu_sable)

    return results

def evaluate_heterogeneous_batch_base(method, batch_size, X_test, Y_test_task, Y_test_class, expert_heads_W, expert_heads_B, trained_router, pear_dispersion, A, B, W_base, mu_class, mu_sable):
    perm = torch.randperm(len(X_test))
    X_hetero = X_test[perm]
    Y_task_hetero = Y_test_task[perm]
    Y_class_hetero = Y_test_class[perm]
    
    correct_preds = [0]*K
    total_preds = [0]*K
    
    def propagate_layers_pear_local(xb, coeffs_batch):
        z = xb.clone()
        for l in range(L):
            base_trans = z @ W_base[l]
            delta_sum = torch.zeros_like(z)
            for k in range(K):
                delta_k = z @ B[k, l].t() @ A[k, l].t()
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
                delta_sum += coeffs_batch[:, k:k+1] * delta_k
            z = base_trans + delta_sum
        return z

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
                    z = z + z @ B[k, l].t() @ A[k, l].t()
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

def main():
    seeds = [10, 11, 12, 13, 14]
    all_results = []
    for s in seeds:
        print(f"Running Seed {s}...")
        res = run_simulation_seed(s)
        all_results.append(res)
        
    methods = ["expert_ceiling", "uniform", "linear_router", "pfsr_mbh", "sable", "pear"]
    configs = ["hom_256", "het_256", "het_1"]
    
    print("\n================== FINAL EXPERIMENTAL RESULTS ==================\n")
    for cfg in configs:
        print(f"--- Configuration: {cfg.upper()} ---")
        for m in methods:
            accs = []
            for s_idx in range(len(seeds)):
                accs.append(all_results[s_idx][cfg][m])
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
        
    # --- Sweeps (Ablation Studies) on Seed 10 ---
    print("================== ABLATION STUDIES (PEAR on Seed 10) ==================\n")
    
    print("--- Temperature Sensitivity Sweep (tau) ---")
    temperatures = [0.0001, 0.001, 0.01, 0.1, 0.5]
    for t in temperatures:
        accs = run_simulation_seed(10, sweep_tau=t)
        joint_mean = np.mean(accs)
        print(f"Temperature (tau) = {t:<6}: Joint Mean Accuracy = {joint_mean:.2f}% (MNIST: {accs[0]:.2f}%, F-MNIST: {accs[1]:.2f}%, CIFAR-10: {accs[2]:.2f}%, SVHN: {accs[3]:.2f}%)")
    print()
    
    print("--- OOD Rejection Threshold Sweep (gamma_OOD) ---")
    thresholds = [0.0, 0.05, 0.10, 0.15, 0.25, 0.35, 0.45]
    for th in thresholds:
        accs = run_simulation_seed(10, sweep_ood=th)
        joint_mean = np.mean(accs)
        print(f"OOD Threshold = {th:<5}: Joint Mean Accuracy = {joint_mean:.2f}% (MNIST: {accs[0]:.2f}%, F-MNIST: {accs[1]:.2f}%, CIFAR-10: {accs[2]:.2f}%, SVHN: {accs[3]:.2f}%)")
    print()

if __name__ == "__main__":
    main()
