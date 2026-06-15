import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

# Ensure reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Define Controlled Representation Sandbox Feature Generation
def get_datasets(seed, rho=0.33):
    set_seed(seed)
    D = 192
    K = 4
    num_classes = 10
    
    # Generate prototypes with customizable subspace overlap (destructive interference)
    prototypes = torch.zeros(K, num_classes, D)
    overlap_dims = int(rho * 48)
    
    for k in range(K):
        start_dim = k * (48 - overlap_dims)
        q, _ = torch.linalg.qr(torch.randn(48, 48))
        proto_sub = q[:num_classes]
        proto_sub = proto_sub / torch.norm(proto_sub, dim=1, keepdim=True)
        prototypes[k, :, start_dim:start_dim+48] = proto_sub
        
    stds = [0.05, 0.15, 0.40, 1.20]  # MNIST, FashionMNIST, CIFAR-10, SVHN
    
    train_x, train_y = [], []
    for k in range(K):
        x_task, y_task = [], []
        for c in range(num_classes):
            proto = prototypes[k, c]
            noise = torch.randn(100, D) * stds[k]
            samples = proto.unsqueeze(0) + noise
            x_task.append(samples)
            y_task.append(torch.full((100,), c, dtype=torch.long))
        train_x.append(torch.cat(x_task, dim=0))
        train_y.append(torch.cat(y_task, dim=0))
        
    cal_x, cal_y, cal_tasks = [], [], []
    for k in range(K):
        indices = torch.randperm(1000)[:16]
        cal_x.append(train_x[k][indices])
        cal_y.append(train_y[k][indices])
        cal_tasks.append(torch.full((16,), k, dtype=torch.long))
        
    cal_x = torch.cat(cal_x, dim=0)
    cal_y = torch.cat(cal_y, dim=0)
    cal_tasks = torch.cat(cal_tasks, dim=0)
    
    test_x, test_y, test_tasks = [], [], []
    for k in range(K):
        x_task, y_task = [], []
        for c in range(num_classes):
            proto = prototypes[k, c]
            noise = torch.randn(25, D) * stds[k]
            samples = proto.unsqueeze(0) + noise
            x_task.append(samples)
            y_task.append(torch.full((25,), c, dtype=torch.long))
        test_x.append(torch.cat(x_task, dim=0))
        test_y.append(torch.cat(y_task, dim=0))
        test_tasks.append(torch.full((250,), k, dtype=torch.long))
        
    test_x = torch.cat(test_x, dim=0)
    test_y = torch.cat(test_y, dim=0)
    test_tasks = torch.cat(test_tasks, dim=0)
    
    return train_x, train_y, cal_x, cal_y, cal_tasks, test_x, test_y, test_tasks, prototypes

# Expert Pre-Training function
def pretrain_experts(seed, train_x, train_y, test_x, test_y):
    set_seed(seed)
    D = 192
    K = 4
    num_classes = 10
    
    W_base = torch.randn(40, D) * 0.01
    b_base = torch.zeros(40)
    
    expert_weights = []
    expert_biases = []
    
    for k in range(K):
        W_k = nn.Parameter(W_base.clone())
        b_k = nn.Parameter(b_base.clone())
        optimizer = optim.AdamW([W_k, b_k], lr=0.05, weight_decay=1e-4)
        
        tx = train_x[k]
        ty = train_y[k]
        
        for epoch in range(10):
            perm = torch.randperm(1000)
            x_shuffled = tx[perm]
            y_shuffled = ty[perm]
            for i in range(0, 1000, 128):
                bx = x_shuffled[i:i+128]
                by = y_shuffled[i:i+128]
                logits = bx @ W_k.t() + b_k
                logits_active = logits[:, k*10 : (k+1)*10]
                loss = nn.CrossEntropyLoss()(logits_active, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        expert_weights.append(W_k.detach())
        expert_biases.append(b_k.detach())
        
    return W_base, b_base, expert_weights, expert_biases

class L3SoftmaxRouterZero(nn.Module):
    def __init__(self, L, K, d):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(L, K, d))
        self.B = nn.Parameter(torch.zeros(L, K))
    def forward(self, psi):
        out = torch.einsum('bd,lkd->blk', psi, self.W) + self.B
        return torch.softmax(out, dim=2)

def compute_jitter(alpha):
    # alpha is [B, L, K]
    B, L, K = alpha.shape
    if L <= 1:
        return 0.0
    jitter = ((alpha[:, 1:, :] - alpha[:, :-1, :]) ** 2).sum() / (B * (L - 1))
    return jitter.item()

def train_router_with_smoothness(router, cal_x, cal_y, cal_tasks, W_base, b_base, V_weights, V_biases, p_proj, 
                                 epochs=100, lr=1e-3, lambda_wd=1e-2, lambda_smooth=0.0):
    router.train()
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=lambda_wd)
    cal_psi = cal_x @ p_proj
    cal_psi = cal_psi / (torch.norm(cal_psi, dim=1, keepdim=True) + 1e-8)
    
    V_weights_stacked = torch.stack(V_weights, dim=0)
    V_biases_stacked = torch.stack(V_biases, dim=0)
    B = cal_x.shape[0]
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        alpha = router(cal_psi) # [B, L, K]
        alpha_b_k = alpha.mean(dim=1) # [B, K]
        
        W_merged = W_base.unsqueeze(0) + torch.einsum('bk,kod->bod', alpha_b_k, V_weights_stacked)
        b_merged = b_base.unsqueeze(0) + torch.einsum('bk,ko->bo', alpha_b_k, V_biases_stacked)
        logits = torch.einsum('bd,bod->bo', cal_x, W_merged) + b_merged
        targets_40 = cal_tasks * 10 + cal_y
        loss_ce = nn.CrossEntropyLoss()(logits, targets_40)
        
        loss_smooth = 0.0
        if lambda_smooth > 0:
            loss_smooth = ((alpha[:, 1:, :] - alpha[:, :-1, :]) ** 2).sum() / (B * (14 - 1))
            loss_smooth = loss_smooth * lambda_smooth
            
        loss = loss_ce + loss_smooth
        loss.backward()
        optimizer.step()

def evaluate_router_jitter(router, test_x, test_y, test_tasks, W_base, b_base, V_weights, V_biases, p_proj):
    router.eval()
    test_psi = test_x @ p_proj
    test_psi = test_psi / (torch.norm(test_psi, dim=1, keepdim=True) + 1e-8)
    
    V_weights_stacked = torch.stack(V_weights, dim=0)
    V_biases_stacked = torch.stack(V_biases, dim=0)
    targets = test_tasks * 10 + test_y
    
    with torch.no_grad():
        alpha = router(test_psi)
        jitter = compute_jitter(alpha)
        alpha_b_k = alpha.mean(dim=1)
        
        W_merged = W_base.unsqueeze(0) + torch.einsum('bk,kod->bod', alpha_b_k, V_weights_stacked)
        b_merged = b_base.unsqueeze(0) + torch.einsum('bk,ko->bo', alpha_b_k, V_biases_stacked)
        logits = torch.einsum('bd,bod->bo', test_x, W_merged) + b_merged
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean().item()
        
    return acc, jitter

# Decompose task vectors into low-rank representations using SVD
def get_lora_adapters(V_weights, r):
    # V_weights is list of 4 tensors of shape [40, 192]
    A_adapters = []
    B_adapters = []
    for V in V_weights:
        U, S, V_h = torch.linalg.svd(V, full_matrices=False)
        U_r = U[:, :r]
        S_r = S[:r]
        V_r = V_h[:r, :]
        
        # Decompose V \approx B @ A where B is [40, r] and A is [r, 192]
        A_k = torch.diag(S_r) @ V_r  # [r, 192]
        B_k = U_r                    # [40, r]
        
        A_adapters.append(A_k)
        B_adapters.append(B_k)
        
    return torch.stack(A_adapters, dim=0), torch.stack(B_adapters, dim=0)

def train_lora_router(router, cal_x, cal_y, cal_tasks, W_base, b_base, A_stacked, B_stacked, V_biases, p_proj, 
                      epochs=100, lr=1e-3, lambda_wd=1e-2):
    router.train()
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=lambda_wd)
    cal_psi = cal_x @ p_proj
    cal_psi = cal_psi / (torch.norm(cal_psi, dim=1, keepdim=True) + 1e-8)
    
    V_biases_stacked = torch.stack(V_biases, dim=0)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        alpha = router(cal_psi)
        alpha_b_k = alpha.mean(dim=1)
        
        # Dynamic LoRA forward pass
        Y_base = cal_x @ W_base.t() + b_base
        H = torch.einsum('bd,krd->bkr', cal_x, A_stacked)
        H_scaled = H * alpha_b_k.unsqueeze(-1)
        Y_lora = torch.einsum('bkr,kor->bo', H_scaled, B_stacked)
        
        b_merged = b_base.unsqueeze(0) + torch.einsum('bk,ko->bo', alpha_b_k, V_biases_stacked)
        logits = Y_base + Y_lora + (b_merged - b_base.unsqueeze(0))
        
        targets_40 = cal_tasks * 10 + cal_y
        loss_ce = nn.CrossEntropyLoss()(logits, targets_40)
        loss_ce.backward()
        optimizer.step()

def evaluate_lora_router(router, test_x, test_y, test_tasks, W_base, b_base, A_stacked, B_stacked, V_biases, p_proj):
    router.eval()
    test_psi = test_x @ p_proj
    test_psi = test_psi / (torch.norm(test_psi, dim=1, keepdim=True) + 1e-8)
    
    V_biases_stacked = torch.stack(V_biases, dim=0)
    targets = test_tasks * 10 + test_y
    
    with torch.no_grad():
        alpha = router(test_psi)
        alpha_b_k = alpha.mean(dim=1)
        
        # Dynamic LoRA forward pass
        Y_base = test_x @ W_base.t() + b_base
        H = torch.einsum('bd,krd->bkr', test_x, A_stacked)
        H_scaled = H * alpha_b_k.unsqueeze(-1)
        Y_lora = torch.einsum('bkr,kor->bo', H_scaled, B_stacked)
        
        b_merged = b_base.unsqueeze(0) + torch.einsum('bk,ko->bo', alpha_b_k, V_biases_stacked)
        logits = Y_base + Y_lora + (b_merged - b_base.unsqueeze(0))
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean().item()
        
    return acc

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

def run_all_experiments():
    print("Pre-training experts and caching across 10 seeds...")
    cached_data = {}
    for seed in SEEDS:
        train_x, train_y, cal_x, cal_y, cal_tasks, test_x, test_y, test_tasks, prototypes = get_datasets(seed)
        W_base, b_base, expert_weights, expert_biases = pretrain_experts(seed, train_x, train_y, test_x, test_y)
        cached_data[seed] = (train_x, train_y, cal_x, cal_y, cal_tasks, test_x, test_y, test_tasks, W_base, b_base, expert_weights, expert_biases)
        
    # --- Part 1: Sequential Smoothness Regularization Sweep ---
    print("\n--- Running Sequential Smoothness Regularization Sweep ---")
    gamma_smooth_vals = [0.0, 0.01, 0.1, 1.0, 10.0]
    smooth_accs = {g: [] for g in gamma_smooth_vals}
    smooth_jitters = {g: [] for g in gamma_smooth_vals}
    
    for gamma in gamma_smooth_vals:
        for seed in SEEDS:
            train_x, train_y, cal_x, cal_y, cal_tasks, test_x, test_y, test_tasks, W_base, b_base, expert_weights, expert_biases = cached_data[seed]
            V_weights = [expert_weights[k] - W_base for k in range(4)]
            V_biases = [expert_biases[k] - b_base for k in range(4)]
            
            p_proj = torch.randn(192, 4)
            p_proj = p_proj / torch.norm(p_proj, dim=0, keepdim=True)
            
            # Use a slightly perturbed zero-init to allow routing decisions to differentiate,
            # or standard zero-init which behaves identically to L3_Softmax_WellReg
            router = L3SoftmaxRouterZero(14, 4, 4)
            # Inject a tiny amount of noise to break symmetry and evaluate smoothness/jitter
            torch.manual_seed(seed)
            with torch.no_grad():
                router.W.add_(torch.randn_like(router.W) * 0.05)
                router.B.add_(torch.randn_like(router.B) * 0.05)
                
            train_router_with_smoothness(router, cal_x, cal_y, cal_tasks, W_base, b_base, V_weights, V_biases, p_proj, 
                                         epochs=100, lr=1e-3, lambda_wd=1e-2, lambda_smooth=gamma)
            
            acc, jitter = evaluate_router_jitter(router, test_x, test_y, test_tasks, W_base, b_base, V_weights, V_biases, p_proj)
            smooth_accs[gamma].append(acc)
            smooth_jitters[gamma].append(jitter)
            
    print("\nSmoothness Results (Mean +- Std over 10 seeds):")
    for gamma in gamma_smooth_vals:
        mean_acc, std_acc = np.mean(smooth_accs[gamma]) * 100, np.std(smooth_accs[gamma]) * 100
        mean_jit, std_jit = np.mean(smooth_jitters[gamma]), np.std(smooth_jitters[gamma])
        print(f"gamma_smooth = {gamma:<5} | Accuracy: {mean_acc:.2f} +- {std_acc:.2f}% | Jitter (1e-3): {mean_jit*1000:.4f} +- {std_jit*1000:.4f}")
        
    # --- Part 2: Low-Rank Adapter (LoRA) Accuracy Evaluation ---
    print("\n--- Running Dynamic LoRA Router Accuracy Sweep ---")
    ranks = [2, 4, 8, 10, 12]
    lora_accs = {r: [] for r in ranks}
    
    # Baseline Full Parameter Router (Well-Regularized)
    full_accs = []
    for seed in SEEDS:
        train_x, train_y, cal_x, cal_y, cal_tasks, test_x, test_y, test_tasks, W_base, b_base, expert_weights, expert_biases = cached_data[seed]
        V_weights = [expert_weights[k] - W_base for k in range(4)]
        V_biases = [expert_biases[k] - b_base for k in range(4)]
        
        p_proj = torch.randn(192, 4)
        p_proj = p_proj / torch.norm(p_proj, dim=0, keepdim=True)
        
        router = L3SoftmaxRouterZero(14, 4, 4)
        train_router_with_smoothness(router, cal_x, cal_y, cal_tasks, W_base, b_base, V_weights, V_biases, p_proj, 
                                     epochs=100, lr=1e-3, lambda_wd=1e-2, lambda_smooth=0.0)
        acc, _ = evaluate_router_jitter(router, test_x, test_y, test_tasks, W_base, b_base, V_weights, V_biases, p_proj)
        full_accs.append(acc)
        
    print(f"Full-Parameter Router Baseline Accuracy: {np.mean(full_accs)*100:.2f} +- {np.std(full_accs)*100:.2f}%")
    
    for r in ranks:
        for seed in SEEDS:
            train_x, train_y, cal_x, cal_y, cal_tasks, test_x, test_y, test_tasks, W_base, b_base, expert_weights, expert_biases = cached_data[seed]
            V_weights = [expert_weights[k] - W_base for k in range(4)]
            V_biases = [expert_biases[k] - b_base for k in range(4)]
            
            p_proj = torch.randn(192, 4)
            p_proj = p_proj / torch.norm(p_proj, dim=0, keepdim=True)
            
            A_stacked, B_stacked = get_lora_adapters(V_weights, r)
            
            router = L3SoftmaxRouterZero(14, 4, 4)
            train_lora_router(router, cal_x, cal_y, cal_tasks, W_base, b_base, A_stacked, B_stacked, V_biases, p_proj, 
                              epochs=100, lr=1e-3, lambda_wd=1e-2)
            acc = evaluate_lora_router(router, test_x, test_y, test_tasks, W_base, b_base, A_stacked, B_stacked, V_biases, p_proj)
            lora_accs[r].append(acc)
            
    print("\nDynamic LoRA Router Results (Mean +- Std over 10 seeds):")
    for r in ranks:
        mean_acc, std_acc = np.mean(lora_accs[r]) * 100, np.std(lora_accs[r]) * 100
        print(f"LoRA Rank r = {r:<3} | Accuracy: {mean_acc:.2f} +- {std_acc:.2f}%")

if __name__ == "__main__":
    run_all_experiments()
