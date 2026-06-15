import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def generate_conflicting_dataset(seed=42):
    set_seed(seed)
    D = 192
    K = 4
    C = 10
    
    # 1. Shared subspace for conflicting semantic features
    D_shared = 128
    shared_prototypes = np.random.normal(0, 1, (C, D_shared))
    shared_prototypes = shared_prototypes / np.linalg.norm(shared_prototypes, axis=1, keepdims=True)
    
    # 2. Task-specific domain style cues
    D_style = D - D_shared # 64
    style_dim_per_task = D_style // K # 16 dimensions per task
    
    noises = [0.001, 0.18, 0.22, 0.8]
    
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
                
                # Full conflict permutation
                label = (c + k) % C
                Y.append(label)
                
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)
        
    train_data = {}
    test_data = {}
    for k in range(K):
        train_data[k] = generate_task_data(k, 200, noises[k])
        test_data[k] = generate_task_data(k, 100, noises[k])
        
    calib_X = []
    calib_Y = []
    for k in range(K):
        X_t, Y_t = generate_task_data(k, 2, noises[k])
        calib_X.append(X_t[:16])
        calib_Y.append(Y_t[:16])
    calib_X = np.concatenate(calib_X, axis=0)
    calib_Y = np.concatenate(calib_Y, axis=0)
    
    shuffle_idx = np.random.permutation(len(calib_X))
    calib_X = calib_X[shuffle_idx]
    calib_Y = calib_Y[shuffle_idx]
    
    return train_data, test_data, (calib_X, calib_Y)

def train_experts(train_data):
    experts = {}
    D = 192
    C = 10
    for k in range(4):
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

def evaluate_static_uniform(experts, test_data):
    K = 4
    W_experts = torch.stack([experts[k].weight.data for k in range(K)], dim=0)
    B_experts = torch.stack([experts[k].bias.data for k in range(K)], dim=0)
    alpha_avg = torch.tensor([0.3, 0.3, 0.3, 0.3])
    W_merged = torch.einsum("k,kcd->cd", alpha_avg, W_experts)
    B_merged = torch.einsum("k,kc->c", alpha_avg, B_experts)
    
    accuracies = []
    for k in range(K):
        X, Y = test_data[k]
        X_t = torch.tensor(X)
        Y_t = torch.tensor(Y)
        logits = torch.matmul(X_t, W_merged.t()) + B_merged
        preds = logits.argmax(dim=-1)
        acc = (preds == Y_t).float().mean().item() * 100
        accuracies.append(acc)
    return np.mean(accuracies)

class BWS_Router(nn.Module):
    def __init__(self, L=12, G=4, d=4, K=4, activation='Sigmoid', init_bias=1.0):
        super().__init__()
        self.L = L
        self.G = G
        self.d = d
        self.K = K
        self.activation_name = activation
        self.W = nn.Parameter(torch.zeros(G, K, d))
        self.B = nn.Parameter(torch.zeros(G, K))
        nn.init.normal_(self.W, std=0.01)
        nn.init.constant_(self.B, init_bias)
        
    def forward(self, psi):
        logits = torch.einsum("gkd,bd->bgk", self.W, psi) + self.B.unsqueeze(0)
        if self.activation_name == 'Sigmoid':
            alpha_g = 0.3 * torch.sigmoid(logits)
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

def train_router(router, pca_proj, experts, calib_data, epochs=100, lr=0.01, lambda_wd=1e-3):
    router.train()
    X_cal, Y_cal = calib_data
    X_cal_t = torch.tensor(X_cal)
    Y_cal_t = torch.tensor(Y_cal)
    
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    K = 4
    W_experts = torch.stack([experts[k].weight.data for k in range(K)], dim=0)
    B_experts = torch.stack([experts[k].bias.data for k in range(K)], dim=0)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        psi = pca_proj.project(X_cal_t)
        alpha = router(psi)
        
        alpha_avg_layers = alpha.mean(dim=1) # [B, K]
        logits_experts = torch.stack([torch.matmul(X_cal_t, W_experts[k].t()) + B_experts[k] for k in range(K)], dim=1) # [B, K, C]
        logits = torch.einsum("bk,bkc->bc", alpha_avg_layers, logits_experts)
        loss = criterion(logits, Y_cal_t)
        
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for name, param in router.named_parameters():
            l2_reg = l2_reg + torch.norm(param, p=2)**2
        total_loss = loss + lambda_wd * l2_reg
        total_loss.backward()
        optimizer.step()

def evaluate_router_acc(router, pca_proj, experts, test_data):
    router.eval()
    K = 4
    W_experts = torch.stack([experts[k].weight.data for k in range(K)], dim=0)
    B_experts = torch.stack([experts[k].bias.data for k in range(K)], dim=0)
    
    accuracies = []
    for k in range(K):
        X, Y = test_data[k]
        X_t = torch.tensor(X)
        Y_t = torch.tensor(Y)
        
        psi = pca_proj.project(X_t)
        alpha = router(psi)
        alpha_avg_layers = alpha.mean(dim=1) # [B, K]
        
        logits_experts = torch.stack([torch.matmul(X_t, W_experts[j].t()) + B_experts[j] for j in range(K)], dim=1) # [B, K, C]
        logits = torch.einsum("bk,bkc->bc", alpha_avg_layers, logits_experts)
        preds = logits.argmax(dim=-1)
        acc = (preds == Y_t).float().mean().item() * 100
        accuracies.append(acc)
    return np.mean(accuracies)

SEEDS = [42, 43, 44] # use subset of seeds for quick sweep

print("=" * 60)
print("RUNNING REVISED HYPERPARAMETER SWEEP (test_tuning.py)")
print("=" * 60)

# Pre-generate and cache all datasets & experts for the seeds
cached_data = {}
for seed in SEEDS:
    train, test, calib = generate_conflicting_dataset(seed)
    experts = train_experts(train)
    static_mean = evaluate_static_uniform(experts, test)
    cached_data[seed] = {
        'train': train,
        'test': test,
        'calib': calib,
        'experts': experts,
        'static_mean': static_mean
    }

print("Pre-training and caching completed!")

# Run a simplified robust sweep
for d_pca in [4, 8]:
    for init_bias in [0.0, 1.0]:
        for lr in [1e-2, 5e-2]:
            for wd in [1e-3, 1e-2]:
                means = []
                static_means = []
                for seed in SEEDS:
                    test = cached_data[seed]['test']
                    calib = cached_data[seed]['calib']
                    experts = cached_data[seed]['experts']
                    static_means.append(cached_data[seed]['static_mean'])
                    
                    pca_proj = PCAPreprojector(n_components=d_pca)
                    pca_proj.fit(calib[0])
                    
                    router = BWS_Router(L=12, G=4, d=d_pca, K=4, activation='Sigmoid', init_bias=init_bias)
                    train_router(router, pca_proj, experts, calib, epochs=120, lr=lr, lambda_wd=wd)
                    means.append(evaluate_router_acc(router, pca_proj, experts, test))
                    
                print(f"d_pca={d_pca}, init_bias={init_bias}, lr={lr:.3f}, wd={wd:.3f} => BWS Joint Mean: {np.mean(means):.3f}%, Static Joint Mean: {np.mean(static_means):.3f}% (Diff: {np.mean(means)-np.mean(static_means):+2.3f}%)")
