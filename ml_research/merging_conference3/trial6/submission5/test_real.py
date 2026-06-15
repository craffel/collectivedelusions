import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Ensure reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Define Controlled Representation Sandbox Feature Generation
def get_datasets(seed, rho=0.0):
    set_seed(seed)
    D = 192
    K = 4
    num_classes = 10
    
    # Generate prototypes
    prototypes = torch.zeros(K, num_classes, D)
    
    for k in range(K):
        start_dim = k * 48
        # Generate orthogonal vectors of dimension 48
        q, _ = torch.linalg.qr(torch.randn(48, 48))
        proto_sub = q[:num_classes]  # [10, 48]
        proto_sub = proto_sub / torch.norm(proto_sub, dim=1, keepdim=True)
        
        # Place in disjoint orthogonal subspace
        prototypes[k, :, start_dim:start_dim+48] = proto_sub
        
    # Task noise representing task difficulty
    stds = [0.05, 0.15, 0.40, 1.20]  # MNIST, FashionMNIST, CIFAR-10, SVHN
    
    # Expert training split: 100 samples per class (1000 per task)
    train_x = []
    train_y = []
    for k in range(K):
        x_task = []
        y_task = []
        for c in range(num_classes):
            proto = prototypes[k, c]
            noise = torch.randn(100, D) * stds[k]
            samples = proto.unsqueeze(0) + noise
            x_task.append(samples)
            y_task.append(torch.full((100,), c, dtype=torch.long))
        train_x.append(torch.cat(x_task, dim=0))
        train_y.append(torch.cat(y_task, dim=0))
        
    # Calibration set: exactly 64 samples (16 per task) to train routing parameters
    cal_x = []
    cal_y = []
    cal_tasks = []
    for k in range(K):
        indices = torch.randperm(1000)[:16]
        cal_x.append(train_x[k][indices])
        cal_y.append(train_y[k][indices])
        cal_tasks.append(torch.full((16,), k, dtype=torch.long))
        
    cal_x = torch.cat(cal_x, dim=0)
    cal_y = torch.cat(cal_y, dim=0)
    cal_tasks = torch.cat(cal_tasks, dim=0)
    
    # Test set: exactly 1000 samples (250 per task, 25 per class)
    test_x = []
    test_y = []
    test_tasks = []
    for k in range(K):
        x_task = []
        y_task = []
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
    expert_accuracies = []
    
    for k in range(K):
        W_k = nn.Parameter(W_base.clone())
        b_k = nn.Parameter(b_base.clone())
        optimizer = optim.AdamW([W_k, b_k], lr=0.05, weight_decay=1e-4)
        
        tx = train_x[k]
        ty = train_y[k]
        
        for epoch in range(10):  # increased to 10 for better pre-training
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
                
        with torch.no_grad():
            test_logits = test_x[k*250 : (k+1)*250] @ W_k.t() + b_k
            test_logits_active = test_logits[:, k*10 : (k+1)*10]
            preds = torch.argmax(test_logits_active, dim=1)
            acc = (preds == test_y[k*250 : (k+1)*250]).float().mean().item()
            expert_accuracies.append(acc)
            
        expert_weights.append(W_k.detach())
        expert_biases.append(b_k.detach())
        
    return W_base, b_base, expert_weights, expert_biases, expert_accuracies

class GlobalLinearRouter(nn.Module):
    def __init__(self, d, K):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, K) * 0.01)
    def forward(self, psi):
        return psi @ self.W

class L3LinearRouter(nn.Module):
    def __init__(self, L, K, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(L, K, d) * 0.01)
        self.B = nn.Parameter(torch.zeros(L, K))
    def forward(self, psi):
        return torch.einsum('bd,lkd->blk', psi, self.W) + self.B

class L3SoftmaxRouter(L3LinearRouter):
    def forward(self, psi):
        out = super().forward(psi)
        return torch.softmax(out, dim=2)

class QWSMergeRouter(nn.Module):
    def __init__(self, L, K, d):
        super().__init__()
        self.Phi = nn.Parameter(torch.eye(K).unsqueeze(0).repeat(L, 1, 1) + torch.randn(L, K, d) * 0.1)
        self.R = nn.Parameter(torch.full((L, K), 0.3))
        self.phi = nn.Parameter(torch.full((L, K), -np.pi))
    def forward(self, psi):
        Phi_hat = self.Phi / (torch.norm(self.Phi, dim=2, keepdim=True) + 1e-8)
        inner = torch.einsum('bd,lkd->blk', psi, Phi_hat)
        return self.R * torch.cos(inner + self.phi)

def train_router(router, cal_x, cal_y, cal_tasks, W_base, b_base, V_weights, V_biases, p_proj, 
                 epochs=200, lr=5e-3, lambda_wd=1e-3, lambda_var=0.0, is_qws=False):
    router.train()
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=lambda_wd)
    cal_psi = cal_x @ p_proj
    cal_psi = cal_psi / (torch.norm(cal_psi, dim=1, keepdim=True) + 1e-8)
    
    V_weights_stacked = torch.stack(V_weights, dim=0)
    V_biases_stacked = torch.stack(V_biases, dim=0)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        if isinstance(router, GlobalLinearRouter):
            alpha_b_k = router(cal_psi)
            alpha = alpha_b_k.unsqueeze(1)
        else:
            alpha = router(cal_psi)
            alpha_b_k = alpha.mean(dim=1)
            
        # Sample-wise parameter assembly
        W_merged = W_base.unsqueeze(0) + torch.einsum('bk,kod->bod', alpha_b_k, V_weights_stacked)
        b_merged = b_base.unsqueeze(0) + torch.einsum('bk,ko->bo', alpha_b_k, V_biases_stacked)
        
        logits = torch.einsum('bd,bod->bo', cal_x, W_merged) + b_merged
        targets_40 = cal_tasks * 10 + cal_y
        loss_ce = nn.CrossEntropyLoss()(logits, targets_40)
        
        loss_vr = 0.0
        if lambda_var > 0 and not isinstance(router, GlobalLinearRouter):
            for k in range(4):
                task_mask = (cal_tasks == k)
                if task_mask.sum() > 0:
                    alpha_task = alpha[task_mask]
                    var_task = alpha_task.var(dim=0, unbiased=False)
                    loss_vr = loss_vr + var_task.mean()
            loss_vr = (loss_vr / 4.0) * lambda_var
            
        loss = loss_ce + loss_vr
        loss.backward()
        if is_qws:
            nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()

def evaluate_router(router, test_x, test_y, test_tasks, W_base, b_base, V_weights, V_biases, p_proj, stream_type='hetero', batch_size=256, is_vr_router=False):
    router.eval()
    test_psi = test_x @ p_proj
    test_psi = test_psi / (torch.norm(test_psi, dim=1, keepdim=True) + 1e-8)
    
    V_weights_stacked = torch.stack(V_weights, dim=0)
    V_biases_stacked = torch.stack(V_biases, dim=0)
    B = test_x.shape[0]
    targets = test_tasks * 10 + test_y
    
    with torch.no_grad():
        if isinstance(router, GlobalLinearRouter):
            alpha_b_k = router(test_psi)
        else:
            alpha = router(test_psi)
            alpha_b_k = alpha.mean(dim=1)
            
        if stream_type == 'homo':
            preds = []
            for i in range(0, B, batch_size):
                bx = test_x[i:i+batch_size]
                b_alpha = alpha_b_k[i:i+batch_size]
                b_alpha_avg = b_alpha.mean(dim=0)
                W_merged = W_base + sum(b_alpha_avg[k] * V_weights[k] for k in range(4))
                b_merged = b_base + sum(b_alpha_avg[k] * V_biases[k] for k in range(4))
                b_logits = bx @ W_merged.t() + b_merged
                preds.append(torch.argmax(b_logits, dim=1))
            return (torch.cat(preds, dim=0) == targets).float().mean().item()
            
        elif stream_type == 'hetero':
            if is_vr_router or batch_size == 1:
                W_merged = W_base.unsqueeze(0) + torch.einsum('bk,kod->bod', alpha_b_k, V_weights_stacked)
                b_merged = b_base.unsqueeze(0) + torch.einsum('bk,ko->bo', alpha_b_k, V_biases_stacked)
                logits = torch.einsum('bd,bod->bo', test_x, W_merged) + b_merged
                preds = torch.argmax(logits, dim=1)
                return (preds == targets).float().mean().item()
            else:
                preds = []
                for i in range(0, B, batch_size):
                    bx = test_x[i:i+batch_size]
                    b_alpha = alpha_b_k[i:i+batch_size]
                    b_alpha_avg = b_alpha.mean(dim=0)
                    W_merged = W_base + sum(b_alpha_avg[k] * V_weights[k] for k in range(4))
                    b_merged = b_base + sum(b_alpha_avg[k] * V_biases[k] for k in range(4))
                    b_logits = bx @ W_merged.t() + b_merged
                    preds.append(torch.argmax(b_logits, dim=1))
                return (torch.cat(preds, dim=0) == targets).float().mean().item()

def evaluate_uniform(test_x, test_y, test_tasks, W_base, b_base, V_weights, V_biases):
    alpha_uniform = torch.full((4,), 0.25)
    W_merged = W_base + sum(alpha_uniform[k] * V_weights[k] for k in range(4))
    b_merged = b_base + sum(alpha_uniform[k] * V_biases[k] for k in range(4))
    logits = test_x @ W_merged.t() + b_merged
    preds = torch.argmax(logits, dim=1)
    targets = test_tasks * 10 + test_y
    return (preds == targets).float().mean().item()

# Test run for seed 42
seed = 42
train_x, train_y, cal_x, cal_y, cal_tasks, test_x, test_y, test_tasks, prototypes = get_datasets(seed)
W_base, b_base, expert_weights, expert_biases, expert_accuracies = pretrain_experts(seed, train_x, train_y, test_x, test_y)

print("Expert Accuracies:", expert_accuracies)

V_weights = [expert_weights[k] - W_base for k in range(4)]
V_biases = [expert_biases[k] - b_base for k in range(4)]

# Define projection onto d=4 subspace using random projection
torch.manual_seed(seed)
p_proj = torch.randn(192, 4)
p_proj = p_proj / torch.norm(p_proj, dim=0, keepdim=True)

# Evaluate Uniform
uniform_acc = evaluate_uniform(test_x, test_y, test_tasks, W_base, b_base, V_weights, V_biases)
print("Uniform Accuracy:", uniform_acc)

# Evaluate Linear Router
linear_router = GlobalLinearRouter(4, 4)
train_router(linear_router, cal_x, cal_y, cal_tasks, W_base, b_base, V_weights, V_biases, p_proj, epochs=300, lr=1e-2, lambda_wd=1e-3, lambda_var=0.0)
lr_acc_homo = evaluate_router(linear_router, test_x, test_y, test_tasks, W_base, b_base, V_weights, V_biases, p_proj, 'homo', 256)
lr_acc_hetero = evaluate_router(linear_router, test_x, test_y, test_tasks, W_base, b_base, V_weights, V_biases, p_proj, 'hetero', 256)
print("Linear Router Homo:", lr_acc_homo)
print("Linear Router Hetero (B=256):", lr_acc_hetero)

# Evaluate L3-Linear
l3_linear = L3LinearRouter(14, 4, 4)
train_router(l3_linear, cal_x, cal_y, cal_tasks, W_base, b_base, V_weights, V_biases, p_proj, epochs=300, lr=1e-2, lambda_wd=1e-3, lambda_var=0.0)
l3_acc_homo = evaluate_router(l3_linear, test_x, test_y, test_tasks, W_base, b_base, V_weights, V_biases, p_proj, 'homo', 256)
l3_acc_hetero = evaluate_router(l3_linear, test_x, test_y, test_tasks, W_base, b_base, V_weights, V_biases, p_proj, 'hetero', 256)
print("L3-Linear Homo:", l3_acc_homo)
print("L3-Linear Hetero (B=256):", l3_acc_hetero)

# Evaluate VR-Router
vr_router = L3LinearRouter(14, 4, 4)
train_router(vr_router, cal_x, cal_y, cal_tasks, W_base, b_base, V_weights, V_biases, p_proj, epochs=300, lr=1e-2, lambda_wd=1e-3, lambda_var=2.0)
vr_acc_homo = evaluate_router(vr_router, test_x, test_y, test_tasks, W_base, b_base, V_weights, V_biases, p_proj, 'homo', 256, is_vr_router=True)
vr_acc_hetero = evaluate_router(vr_router, test_x, test_y, test_tasks, W_base, b_base, V_weights, V_biases, p_proj, 'hetero', 256, is_vr_router=True)
print("VR-Router Homo:", vr_acc_homo)
print("VR-Router Hetero (B=256):", vr_acc_hetero)
