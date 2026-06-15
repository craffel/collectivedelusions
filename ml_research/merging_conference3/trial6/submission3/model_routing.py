import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA

# --- Helper function for reproducibility ---
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Dataset Generation ---
def generate_synthetic_dataset(seed=42):
    set_seed(seed)
    D = 192
    K = 4
    C = 10
    
    # 1. Shared subspace for conflicting semantic features
    D_shared = 128
    # Generate 10 shared class prototypes
    shared_prototypes = np.random.normal(0, 1, (C, D_shared))
    shared_prototypes = shared_prototypes / np.linalg.norm(shared_prototypes, axis=1, keepdims=True)
    
    # 2. Task-specific domain style cues to allow routing
    D_style = D - D_shared # 64
    style_dim_per_task = D_style // K # 16 dimensions per task
    
    # Calibrated noise standard deviations (same as original to preserve baseline difficulty)
    noises = [0.001, 0.18, 0.22, 0.8]
    
    def generate_task_data(k, num_samples, noise_std):
        X = []
        Y = []
        for c in range(C):
            # Shared semantic feature
            shared_feat = shared_prototypes[c]
            
            # Task-specific style feature (domain cue)
            style_feat = np.zeros(D_style)
            style_feat[k * style_dim_per_task : (k + 1) * style_dim_per_task] = 1.5
            
            for _ in range(num_samples):
                # Concatenate shared semantic and task style features
                feat = np.concatenate([shared_feat, style_feat])
                # Add Gaussian noise
                feat += np.random.normal(0, noise_std, D)
                X.append(feat)
                
                # Permuted labels to create extreme task conflict in the shared space!
                label = (c + k) % C
                Y.append(label)
                
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)
        
    train_data = {}
    test_data = {}
    
    for k in range(K):
        X_train, Y_train = generate_task_data(k, 200, noises[k])
        X_test, Y_test = generate_task_data(k, 100, noises[k])
        train_data[k] = (X_train, Y_train)
        test_data[k] = (X_test, Y_test)
        
    # Calibration split of 64 samples (16 per task)
    calib_X = []
    calib_Y = []
    calib_T = []
    for k in range(K):
        X_t, Y_t = generate_task_data(k, 2, noises[k]) # 2 per class -> 20 samples per task. Let's slice exactly 16 samples.
        calib_X.append(X_t[:16])
        calib_Y.append(Y_t[:16])
        calib_T.append(np.full(16, k, dtype=np.int64))
        
    calib_X = np.concatenate(calib_X, axis=0)
    calib_Y = np.concatenate(calib_Y, axis=0)
    calib_T = np.concatenate(calib_T, axis=0)
    
    # Shuffle calibration data
    shuffle_idx = np.random.permutation(len(calib_X))
    calib_X = calib_X[shuffle_idx]
    calib_Y = calib_Y[shuffle_idx]
    calib_T = calib_T[shuffle_idx]
    
    return train_data, test_data, (calib_X, calib_Y, calib_T)


# --- Train Specialized Expert Classifiers ---
def train_experts(train_data, test_data):
    experts = {}
    D = 192
    C = 10
    for k in range(4):
        X_tr, Y_tr = train_data[k]
        X_te, Y_te = test_data[k]
        
        X_tr_t = torch.tensor(X_tr)
        Y_tr_t = torch.tensor(Y_tr)
        X_te_t = torch.tensor(X_te)
        Y_te_t = torch.tensor(Y_te)
        
        model = nn.Linear(D, C)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t), Y_tr_t)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            acc = (model(X_te_t).argmax(dim=1) == Y_te_t).float().mean().item() * 100
            
        experts[k] = model
    return experts


# --- Router Architectures ---

class BWS_Router(nn.Module):
    def __init__(self, L=12, G=4, d=4, K=4, activation='Sigmoid', lambda_max=0.3, init_bias=None):
        super().__init__()
        self.L = L
        self.G = G
        self.d = d
        self.K = K
        self.activation_name = activation
        self.lambda_max = lambda_max
        
        # Trainable parameters per group
        self.W = nn.Parameter(torch.zeros(G, K, d))
        self.B = nn.Parameter(torch.zeros(G, K))
        # Initializing around zero matching standard linear routing start
        nn.init.normal_(self.W, std=0.01)
        if init_bias is None:
            if activation == 'Sigmoid':
                init_bias = 1.0
            else:
                init_bias = 0.0
        nn.init.constant_(self.B, init_bias)
        
    def forward(self, psi):
        # psi shape: [B, d]
        logits = torch.einsum("gkd,bd->bgk", self.W, psi) + self.B.unsqueeze(0) # [B, G, K]
        
        if self.activation_name == 'Sigmoid':
            alpha_g = self.lambda_max * torch.sigmoid(logits)
        elif self.activation_name == 'Linear':
            alpha_g = logits
        elif self.activation_name == 'Tanh':
            alpha_g = torch.tanh(logits)
        elif self.activation_name == 'Softmax':
            alpha_g = torch.softmax(logits, dim=-1) # softmax over tasks
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
            
        M = self.L // self.G
        alpha_l = alpha_g.unsqueeze(2).repeat(1, 1, M, 1).view(-1, self.L, self.K)
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


class Global_Linear_Router(nn.Module):
    def __init__(self, D=192, K=4, L=12):
        super().__init__()
        self.L = L
        self.K = K
        self.fc = nn.Linear(D, K)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.normal_(self.fc.bias, std=0.01)
        
    def forward(self, z):
        # z shape: [B, D]
        alpha_global = torch.softmax(self.fc(z), dim=-1) # [B, K]
        alpha_l = alpha_global.unsqueeze(1).repeat(1, self.L, 1) # [B, L, K]
        return alpha_l


# --- Unsupervised PCA Setup ---
class PCAPreprojector:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.P = None
        
    def fit(self, X_calib):
        self.pca.fit(X_calib)
        self.P = torch.tensor(self.pca.components_.T, dtype=torch.float32)
        
    def project(self, X):
        # X shape: [B, D]
        if self.P is None:
            raise ValueError("PCA is not fitted yet.")
        proj = torch.matmul(X, self.P)
        psi = proj / (torch.norm(proj, dim=-1, keepdim=True) + 1e-8)
        return psi


# --- Evaluation and Training Framework ---

def evaluate_router(router, pca_proj, experts, test_data, mode='Homogeneous_B256'):
    # mode can be: 'Homogeneous_B1', 'Homogeneous_B256', 'Heterogeneous_B256'
    router.eval()
    K = 4
    C = 10
    
    # Extract expert weights and biases
    W_experts = []
    B_experts = []
    for k in range(K):
        W_experts.append(experts[k].weight.data) # [C, D]
        B_experts.append(experts[k].bias.data)   # [C]
    W_experts = torch.stack(W_experts, dim=0) # [K, C, D]
    B_experts = torch.stack(B_experts, dim=0) # [K, C]
    
    with torch.no_grad():
        if mode == 'Homogeneous_B1':
            # Sample-wise evaluation
            accuracies = []
            for k in range(K):
                X, Y = test_data[k]
                X_t = torch.tensor(X)
                Y_t = torch.tensor(Y)
                
                # Single sample evaluation
                correct = 0
                for b in range(len(X_t)):
                    xb = X_t[b:b+1]
                    yb = Y_t[b:b+1]
                    
                    if isinstance(router, Global_Linear_Router):
                        alpha = router(xb) # [1, L, K]
                    else:
                        psi = pca_proj.project(xb)
                        alpha = router(psi) # [1, L, K]
                        
                    # Average over layers
                    alpha_avg = alpha.mean(dim=1).squeeze(0) # [K]
                    
                    # Merge classification head
                    W_merged = torch.einsum("k,kcd->cd", alpha_avg, W_experts)
                    B_merged = torch.einsum("k,kc->c", alpha_avg, B_experts)
                    
                    logits = torch.matmul(xb, W_merged.t()) + B_merged
                    pred = logits.argmax(dim=-1)
                    if pred == yb:
                        correct += 1
                accuracies.append(correct / len(X_t) * 100)
            return accuracies, np.mean(accuracies)
            
        elif mode == 'Homogeneous_B256':
            # Task-wise batch evaluation
            accuracies = []
            for k in range(K):
                X, Y = test_data[k]
                X_t = torch.tensor(X)
                Y_t = torch.tensor(Y)
                
                if isinstance(router, Global_Linear_Router):
                    alpha = router(X_t) # [B, L, K]
                else:
                    psi = pca_proj.project(X_t)
                    alpha = router(psi) # [B, L, K]
                    
                # Sample-wise routing logic
                alpha_avg_layers = alpha.mean(dim=1) # [B, K]
                logits_experts = torch.stack([torch.matmul(X_t, W_experts[j].t()) + B_experts[j] for j in range(K)], dim=1) # [B, K, C]
                logits = torch.einsum("bk,bkc->bc", alpha_avg_layers, logits_experts)
                preds = logits.argmax(dim=-1)
                acc = (preds == Y_t).float().mean().item() * 100
                accuracies.append(acc)
            return accuracies, np.mean(accuracies)
            
        elif mode == 'Heterogeneous_B256':
            # Mixed-task batch evaluation (all tasks mixed in a single batch)
            # Create a mixed batch: 64 samples per task, total 256 samples
            mixed_X = []
            mixed_Y = []
            mixed_T = []
            for k in range(K):
                X, Y = test_data[k]
                mixed_X.append(X[:64])
                mixed_Y.append(Y[:64])
                mixed_T.append(np.full(64, k))
                
            mixed_X = torch.tensor(np.concatenate(mixed_X, axis=0))
            mixed_Y = torch.tensor(np.concatenate(mixed_Y, axis=0))
            mixed_T = torch.tensor(np.concatenate(mixed_T, axis=0))
            
            # Shuffle mixed batch
            perm = torch.randperm(len(mixed_X))
            mixed_X = mixed_X[perm]
            mixed_Y = mixed_Y[perm]
            mixed_T = mixed_T[perm]
            
            if isinstance(router, Global_Linear_Router):
                alpha = router(mixed_X) # [B, L, K]
            else:
                psi = pca_proj.project(mixed_X)
                alpha = router(psi) # [B, L, K]
                
            # Sample-wise routing logic instead of batch average coefficients
            alpha_avg_layers = alpha.mean(dim=1) # [B, K]
            logits_experts = torch.stack([torch.matmul(mixed_X, W_experts[j].t()) + B_experts[j] for j in range(K)], dim=1) # [B, K, C]
            logits = torch.einsum("bk,bkc->bc", alpha_avg_layers, logits_experts)
            
            # Compute accuracy per task in the mixed batch
            accuracies = []
            for k in range(K):
                task_mask = (mixed_T == k)
                if task_mask.sum() > 0:
                    task_logits = logits[task_mask]
                    task_Y = mixed_Y[task_mask]
                    preds = task_logits.argmax(dim=-1)
                    acc = (preds == task_Y).float().mean().item() * 100
                    accuracies.append(acc)
                else:
                    accuracies.append(0.0)
            return accuracies, np.mean(accuracies)


def train_router(router, pca_proj, experts, calib_data, epochs=100, lr=0.01, lambda_wd=1e-3):
    router.train()
    X_cal, Y_cal, T_cal = calib_data
    X_cal_t = torch.tensor(X_cal)
    Y_cal_t = torch.tensor(Y_cal)
    T_cal_t = torch.tensor(T_cal)
    
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    K = 4
    # Extract expert weights and biases
    W_experts = []
    B_experts = []
    for k in range(K):
        W_experts.append(experts[k].weight.data)
        B_experts.append(experts[k].bias.data)
    W_experts = torch.stack(W_experts, dim=0) # [K, C, D]
    B_experts = torch.stack(B_experts, dim=0) # [K, C]
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        if isinstance(router, Global_Linear_Router):
            alpha = router(X_cal_t) # [B, L, K]
        else:
            psi = pca_proj.project(X_cal_t)
            alpha = router(psi) # [B, L, K]
            
        # Get sample-wise coefficients averaged over layers
        alpha_avg_layers = alpha.mean(dim=1) # [B, K]
        
        # Compute expert logits for each sample
        logits_experts = torch.stack([torch.matmul(X_cal_t, W_experts[k].t()) + B_experts[k] for k in range(K)], dim=1) # [B, K, C]
        
        # Sample-wise logits
        logits = torch.einsum("bk,bkc->bc", alpha_avg_layers, logits_experts) # [B, C]
        loss = criterion(logits, Y_cal_t)
        
        # Add L2 penalty manually for weights if not handled perfectly by AdamW
        # especially for non-weight parameter structures or custom groupings
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for name, param in router.named_parameters():
            if 'weight' in name or 'W' in name:
                l2_reg = l2_reg + torch.norm(param, p=2)**2
        total_loss = loss + lambda_wd * l2_reg
        
        # Gradient norm clipping for stability (especially for QWS)
        total_loss.backward()
        nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
        optimizer.step()
