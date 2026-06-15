import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import json

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def generate_scale_dataset(K=4, seed=42):
    set_seed(seed)
    D = 192
    C = 10
    
    # 1. Shared subspace for conflicting semantic features
    D_shared = 128
    shared_prototypes = np.random.normal(0, 1, (C, D_shared))
    shared_prototypes = shared_prototypes / np.linalg.norm(shared_prototypes, axis=1, keepdims=True)
    
    # 2. Task-specific domain style cues to allow routing
    D_style = D - D_shared # 64
    style_dim_per_task = D_style // K
    
    # Noise standard deviations scaled between MNIST (0.001) and SVHN (0.8)
    noises = np.linspace(0.001, 0.8, K)
    
    def generate_task_data(k, num_samples, noise_std):
        X = []
        Y = []
        for c in range(C):
            shared_feat = shared_prototypes[c]
            style_feat = np.zeros(D_style)
            style_feat[k * style_dim_per_task : (k + 1) * style_dim_per_task] = 1.5
            
            for _ in range(num_samples):
                feat = np.concatenate([shared_feat, style_feat])
                feat += np.random.normal(0, noise_std, D)
                X.append(feat)
                
                label = (c + k) % C
                Y.append(label)
                
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)
        
    train_data = {}
    test_data = {}
    for k in range(K):
        train_data[k] = generate_task_data(k, 200, noises[k])
        test_data[k] = generate_task_data(k, 100, noises[k])
        
    # Calibration split: 16 samples per task
    calib_X = []
    calib_Y = []
    calib_T = []
    for k in range(K):
        X_t, Y_t = generate_task_data(k, 2, noises[k])
        calib_X.append(X_t[:16])
        calib_Y.append(Y_t[:16])
        calib_T.append(np.full(16, k, dtype=np.int64))
        
    calib_X = np.concatenate(calib_X, axis=0)
    calib_Y = np.concatenate(calib_Y, axis=0)
    calib_T = np.concatenate(calib_T, axis=0)
    
    shuffle_idx = np.random.permutation(len(calib_X))
    calib_X = calib_X[shuffle_idx]
    calib_Y = calib_Y[shuffle_idx]
    calib_T = calib_T[shuffle_idx]
    
    return train_data, test_data, (calib_X, calib_Y, calib_T)

def train_experts(train_data, K):
    experts = {}
    D = 192
    C = 10
    for k in range(K):
        X_tr, Y_tr = train_data[k]
        X_tr_t = torch.tensor(X_tr)
        Y_tr_t = torch.tensor(Y_tr)
        
        model = nn.Linear(D, C)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t), Y_tr_t)
            loss.backward()
            optimizer.step()
        experts[k] = model
    return experts

# Class BWS Router that scales with K
class ScalableBWSRouter(nn.Module):
    def __init__(self, L=12, G=4, d=4, K=4, activation='Sigmoid', lambda_max=0.3):
        super().__init__()
        self.L = L
        self.G = G
        self.d = d
        self.K = K
        self.activation_name = activation
        self.lambda_max = lambda_max
        
        self.W = nn.Parameter(torch.zeros(G, K, d))
        self.B = nn.Parameter(torch.zeros(G, K))
        nn.init.normal_(self.W, std=0.01)
        nn.init.constant_(self.B, 1.0) # bias initialization matching standard
        
    def forward(self, psi):
        logits = torch.einsum("gkd,bd->bgk", self.W, psi) + self.B.unsqueeze(0)
        if self.activation_name == 'Sigmoid':
            alpha_g = self.lambda_max * torch.sigmoid(logits)
        elif self.activation_name == 'Softmax':
            alpha_g = torch.softmax(logits, dim=-1)
        else:
            alpha_g = logits
        M = self.L // self.G
        alpha_l = alpha_g.unsqueeze(2).repeat(1, 1, M, 1).view(-1, self.L, self.K)
        return alpha_l

class PCAPreprojector:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.P = None
        
    def fit(self, X_calib):
        pca = PCA(n_components=self.n_components)
        pca.fit(X_calib)
        self.P = torch.tensor(pca.components_.T, dtype=torch.float32)
        
    def project(self, X):
        proj = torch.matmul(X, self.P)
        psi = proj / (torch.norm(proj, dim=-1, keepdim=True) + 1e-8)
        return psi

def train_router(router, pca_proj, experts, calib_data, K, lr=5e-2, lambda_wd=1e-4):
    router.train()
    X_cal, Y_cal, T_cal = calib_data
    X_cal_t = torch.tensor(X_cal)
    Y_cal_t = torch.tensor(Y_cal)
    
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    W_experts = torch.stack([experts[k].weight.data for k in range(K)], dim=0)
    B_experts = torch.stack([experts[k].bias.data for k in range(K)], dim=0)
    
    for epoch in range(100):
        optimizer.zero_grad()
        psi = pca_proj.project(X_cal_t)
        alpha = router(psi)
        alpha_avg_layers = alpha.mean(dim=1) # [B, K]
        logits_experts = torch.stack([torch.matmul(X_cal_t, W_experts[j].t()) + B_experts[j] for j in range(K)], dim=1) # [B, K, C]
        logits = torch.einsum("bk,bkc->bc", alpha_avg_layers, logits_experts)
        loss = criterion(logits, Y_cal_t)
        
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for name, param in router.named_parameters():
            if 'weight' in name or 'W' in name:
                l2_reg = l2_reg + torch.norm(param, p=2)**2
        total_loss = loss + lambda_wd * l2_reg
        total_loss.backward()
        optimizer.step()

def evaluate_router(router, pca_proj, experts, test_data, K):
    router.eval()
    W_experts = torch.stack([experts[k].weight.data for k in range(K)], dim=0)
    B_experts = torch.stack([experts[k].bias.data for k in range(K)], dim=0)
    
    accuracies = []
    with torch.no_grad():
        for k in range(K):
            X, Y = test_data[k]
            X_t = torch.tensor(X)
            Y_t = torch.tensor(Y)
            psi = pca_proj.project(X_t)
            alpha = router(psi)
            alpha_avg_layers = alpha.mean(dim=1)
            logits_experts = torch.stack([torch.matmul(X_t, W_experts[j].t()) + B_experts[j] for j in range(K)], dim=1)
            logits = torch.einsum("bk,bkc->bc", alpha_avg_layers, logits_experts)
            preds = logits.argmax(dim=-1)
            acc = (preds == Y_t).float().mean().item() * 100
            accuracies.append(acc)
    return np.mean(accuracies)

def evaluate_static_uniform(experts, test_data, K):
    W_experts = torch.stack([experts[k].weight.data for k in range(K)], dim=0)
    B_experts = torch.stack([experts[k].bias.data for k in range(K)], dim=0)
    
    # Uniform routing coefficients summing to 1.2 or similar scale matching baseline scale
    alpha_avg = torch.full((K,), 1.2 / K)
    W_merged = torch.einsum("k,kcd->cd", alpha_avg, W_experts)
    B_merged = torch.einsum("k,kc->c", alpha_avg, B_experts)
    
    accuracies = []
    with torch.no_grad():
        for k in range(K):
            X, Y = test_data[k]
            X_t = torch.tensor(X)
            Y_t = torch.tensor(Y)
            logits = torch.matmul(X_t, W_merged.t()) + B_merged
            preds = logits.argmax(dim=-1)
            acc = (preds == Y_t).float().mean().item() * 100
            accuracies.append(acc)
    return np.mean(accuracies)

SEEDS = [42, 43, 44, 45, 46]
K_values = [4, 6, 8, 10]

results_bws = {k_val: [] for k_val in K_values}
results_static = {k_val: [] for k_val in K_values}

print("Running Scalability Sweep over Expert Count K...")
for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    for k_val in K_values:
        train_data, test_data, calib_data = generate_scale_dataset(K=k_val, seed=seed)
        experts = train_experts(train_data, k_val)
        
        # Scaling the PCA projector dimension sub-linearly: d = log_2(K) rounded up, but at least 2
        d_val = max(2, int(np.ceil(np.log2(k_val))))
        
        pca_proj = PCAPreprojector(n_components=d_val)
        pca_proj.fit(calib_data[0])
        
        # Proposed BWS Router (M=3, G=4)
        router = ScalableBWSRouter(L=12, G=4, d=d_val, K=k_val, activation='Sigmoid')
        train_router(router, pca_proj, experts, calib_data, K=k_val, lr=5e-2, lambda_wd=1e-4)
        
        acc_bws = evaluate_router(router, pca_proj, experts, test_data, K=k_val)
        acc_static = evaluate_static_uniform(experts, test_data, K=k_val)
        
        results_bws[k_val].append(acc_bws)
        results_static[k_val].append(acc_static)
        
        print(f"K = {k_val:2d} (d = {d_val}) | Static Uniform: {acc_static:.2f}% | BWS Router ($M=3$): {acc_bws:.2f}%")

print("\n=== Expert Scaling Sweep Results (Mean ± Std across 5 seeds) ===")
print("| Expert Count (K) | Sub-linear PCA Dim (d) | Static Uniform Acc (%) | BWS-Router ($M=3$) Acc (%) | Improvement (%) |")
print("| :---: | :---: | :---: | :---: | :---: |")
for k_val in K_values:
    d_val = max(2, int(np.ceil(np.log2(k_val))))
    m_bws, s_bws = np.mean(results_bws[k_val]), np.std(results_bws[k_val])
    m_st, s_st = np.mean(results_static[k_val]), np.std(results_static[k_val])
    diff = m_bws - m_st
    print(f"| {k_val:2d} | {d_val} | {m_st:.2f} ± {s_st:.2f}% | {m_bws:.2f} ± {s_bws:.2f}% | **+{diff:.2f}%** |")

# Save results
with open("scale_experts_results.json", "w") as f:
    json.dump({
        "bws": results_bws,
        "static": results_static
    }, f, indent=2)
print("\nSaved scale sweep results to scale_experts_results.json")
