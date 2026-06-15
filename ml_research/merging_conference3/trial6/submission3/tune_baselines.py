import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def generate_synthetic_dataset(seed=42):
    set_seed(seed)
    D = 192
    K = 4
    C = 10
    subspace_dim = D // K # 48
    prototypes = {}
    for k in range(K):
        W_proto = np.random.normal(0, 1, (C, subspace_dim))
        W_proto = W_proto / np.linalg.norm(W_proto, axis=1, keepdims=True)
        prototypes[k] = W_proto
    noises = [0.001, 0.18, 0.22, 0.8]
    def generate_task_data(k, num_samples, noise_std):
        X = []
        Y = []
        for c in range(C):
            proto = prototypes[k][c]
            for _ in range(num_samples):
                feat = np.zeros(D)
                feat[k * subspace_dim : (k + 1) * subspace_dim] = proto
                feat += np.random.normal(0, noise_std, D)
                X.append(feat)
                Y.append(c)
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

class BWS_Router(nn.Module):
    def __init__(self, L=12, G=12, d=4, K=4, activation='Linear'):
        super().__init__()
        self.L = L
        self.G = G
        self.d = d
        self.K = K
        self.activation_name = activation
        self.W = nn.Parameter(torch.zeros(G, K, d))
        self.B = nn.Parameter(torch.zeros(G, K))
        nn.init.normal_(self.W, std=0.01)
        nn.init.normal_(self.B, std=0.01)
    def forward(self, psi):
        logits = torch.einsum("gkd,bd->bgk", self.W, psi) + self.B.unsqueeze(0)
        if self.activation_name == 'Sigmoid':
            alpha_g = 0.3 * torch.sigmoid(logits)
        elif self.activation_name == 'Softmax':
            alpha_g = torch.softmax(logits, dim=-1)
        elif self.activation_name == 'Tanh':
            alpha_g = torch.tanh(logits)
        else:
            alpha_g = logits
        M = self.L // self.G
        alpha_l = alpha_g.unsqueeze(2).repeat(1, 1, M, 1).view(-1, self.L, self.K)
        return alpha_l

class Global_Linear_Router(nn.Module):
    def __init__(self, D=192, K=4, L=12):
        super().__init__()
        self.L = L
        self.K = K
        self.fc = nn.Linear(D, K)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.normal_(self.fc.bias, std=0.01)
    def forward(self, z):
        alpha_global = torch.softmax(self.fc(z), dim=-1)
        alpha_l = alpha_global.unsqueeze(1).repeat(1, self.L, 1)
        return alpha_l

class QWS_Merge_Router(nn.Module):
    def __init__(self, L=12, d=4, K=4):
        super().__init__()
        self.L = L
        self.d = d
        self.K = K
        self.Phi = nn.Parameter(torch.zeros(L, K, d))
        with torch.no_grad():
            for l in range(L):
                self.Phi[l].copy_(torch.eye(K, d) + torch.randn(K, d) * 0.1)
        self.R = nn.Parameter(torch.full((L, K), 0.3))
        self.phi = nn.Parameter(torch.full((L, K), -np.pi))
    def forward(self, psi):
        Phi_hat = self.Phi / (torch.norm(self.Phi, dim=-1, keepdim=True) + 1e-8)
        inner = torch.einsum("bd,lkd->blk", psi, Phi_hat)
        alpha = self.R.unsqueeze(0) * torch.cos(np.pi * inner + self.phi.unsqueeze(0))
        return alpha

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
        if isinstance(router, Global_Linear_Router):
            alpha = router(X_cal_t)
        else:
            psi = pca_proj.project(X_cal_t)
            alpha = router(psi)
        alpha_avg = alpha.mean(dim=0).mean(dim=0)
        W_merged = torch.einsum("k,kcd->cd", alpha_avg, W_experts)
        B_merged = torch.einsum("k,kc->c", alpha_avg, B_experts)
        logits = torch.matmul(X_cal_t, W_merged.t()) + B_merged
        loss = criterion(logits, Y_cal_t)
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for param in router.parameters():
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
        if isinstance(router, Global_Linear_Router):
            alpha = router(X_t)
        else:
            psi = pca_proj.project(X_t)
            alpha = router(psi)
        alpha_avg = alpha.mean(dim=0).mean(dim=0)
        W_merged = torch.einsum("k,kcd->cd", alpha_avg, W_experts)
        B_merged = torch.einsum("k,kc->c", alpha_avg, B_experts)
        logits = torch.matmul(X_t, W_merged.t()) + B_merged
        preds = logits.argmax(dim=-1)
        acc = (preds == Y_t).float().mean().item() * 100
        accuracies.append(acc)
    return np.mean(accuracies)

SEEDS = [42, 43, 44, 45, 46]
cached_data = {}
for seed in SEEDS:
    train, test, calib = generate_synthetic_dataset(seed)
    experts = train_experts(train)
    cached_data[seed] = {
        'train': train,
        'test': test,
        'calib': calib,
        'experts': experts
    }

lrs = [1e-3, 5e-3, 1e-2, 5e-2]
wds = [0.0, 1e-4, 1e-3, 1e-2]

print("Tuning Global_Linear_Reg...")
best_global_mean = 0
best_global_cfg = None
for lr in lrs:
    for wd in wds:
        means = []
        for seed in SEEDS:
            test = cached_data[seed]['test']
            calib = cached_data[seed]['calib']
            experts = cached_data[seed]['experts']
            pca_proj = PCAPreprojector(n_components=4)
            pca_proj.fit(calib[0])
            router = Global_Linear_Router(D=192, K=4, L=12)
            train_router(router, pca_proj, experts, calib, lr=lr, lambda_wd=wd)
            means.append(evaluate_router_acc(router, pca_proj, experts, test))
        m = np.mean(means)
        if m > best_global_mean:
            best_global_mean = m
            best_global_cfg = (lr, wd)
print(f"Best Global Linear Reg Joint Mean: {best_global_mean:.3f}% at lr={best_global_cfg[0]}, wd={best_global_cfg[1]}")

print("Tuning QWS_Merge...")
best_qws_mean = 0
best_qws_cfg = None
for lr in lrs:
    for wd in wds:
        means = []
        for seed in SEEDS:
            test = cached_data[seed]['test']
            calib = cached_data[seed]['calib']
            experts = cached_data[seed]['experts']
            pca_proj = PCAPreprojector(n_components=4)
            pca_proj.fit(calib[0])
            router = QWS_Merge_Router(L=12, d=4, K=4)
            train_router(router, pca_proj, experts, calib, lr=lr, lambda_wd=wd)
            means.append(evaluate_router_acc(router, pca_proj, experts, test))
        m = np.mean(means)
        if m > best_qws_mean:
            best_qws_mean = m
            best_qws_cfg = (lr, wd)
print(f"Best QWS Merge Joint Mean: {best_qws_mean:.3f}% at lr={best_qws_cfg[0]}, wd={best_qws_cfg[1]}")

print("Tuning L3_Linear_Reg...")
best_l3_lin_mean = 0
best_l3_lin_cfg = None
for lr in lrs:
    for wd in wds:
        means = []
        for seed in SEEDS:
            test = cached_data[seed]['test']
            calib = cached_data[seed]['calib']
            experts = cached_data[seed]['experts']
            pca_proj = PCAPreprojector(n_components=4)
            pca_proj.fit(calib[0])
            router = BWS_Router(L=12, G=12, d=4, K=4, activation='Linear')
            train_router(router, pca_proj, experts, calib, lr=lr, lambda_wd=wd)
            means.append(evaluate_router_acc(router, pca_proj, experts, test))
        m = np.mean(means)
        if m > best_l3_lin_mean:
            best_l3_lin_mean = m
            best_l3_lin_cfg = (lr, wd)
print(f"Best L3 Linear Reg Joint Mean: {best_l3_lin_mean:.3f}% at lr={best_l3_lin_cfg[0]}, wd={best_l3_lin_cfg[1]}")

print("Tuning L3_Softmax_Reg...")
best_l3_soft_mean = 0
best_l3_soft_cfg = None
for lr in lrs:
    for wd in wds:
        means = []
        for seed in SEEDS:
            test = cached_data[seed]['test']
            calib = cached_data[seed]['calib']
            experts = cached_data[seed]['experts']
            pca_proj = PCAPreprojector(n_components=4)
            pca_proj.fit(calib[0])
            router = BWS_Router(L=12, G=12, d=4, K=4, activation='Softmax')
            train_router(router, pca_proj, experts, calib, lr=lr, lambda_wd=wd)
            means.append(evaluate_router_acc(router, pca_proj, experts, test))
        m = np.mean(means)
        if m > best_l3_soft_mean:
            best_l3_soft_mean = m
            best_l3_soft_cfg = (lr, wd)
print(f"Best L3 Softmax Reg Joint Mean: {best_l3_soft_mean:.3f}% at lr={best_l3_soft_cfg[0]}, wd={best_l3_soft_cfg[1]}")
