import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import json

# --- Helper for Reproducibility ---
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Dataset Generator (Task Conflict Sandbox) ---
def generate_synthetic_dataset(seed=42):
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

# --- Multi-Layer Expert Architecture ---
class MLPExpert(nn.Module):
    def __init__(self, input_dim=192, h1=128, h2=64, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_mlp_experts(train_data, test_data):
    experts = {}
    for k in range(4):
        X_tr, Y_tr = train_data[k]
        X_tr_t = torch.tensor(X_tr)
        Y_tr_t = torch.tensor(Y_tr)
        
        model = MLPExpert()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(120):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t), Y_tr_t)
            loss.backward()
            optimizer.step()
            
        model.eval()
        experts[k] = model
    return experts

# --- PCA Projector for Hidden Activations ---
class PhysicalPCAPreprojector:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.P = None
        
    def fit(self, H):
        if isinstance(H, torch.Tensor):
            H = H.cpu().numpy()
        pca = PCA(n_components=self.n_components)
        pca.fit(H)
        self.P = torch.tensor(pca.components_.T, dtype=torch.float32)
        
    def project(self, H):
        if self.P is None:
            raise ValueError("PCA not fitted yet.")
        proj = torch.matmul(H, self.P.to(H.device))
        psi = proj / (torch.norm(proj, dim=-1, keepdim=True) + 1e-8)
        return psi

# --- Physical BWS-Router ---
class PhysicalBWS_Router(nn.Module):
    def __init__(self, L=3, G=1, d=4, K=4, activation='Sigmoid', lambda_max=0.3, init_bias=-2.0):
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
        nn.init.constant_(self.B, init_bias)
        
    def forward(self, psi_list):
        alpha_list = []
        for l in range(self.L):
            g = l if self.G == self.L else 0
            psi = psi_list[l]
            logits = torch.matmul(psi, self.W[g].t()) + self.B[g].unsqueeze(0)
            if self.activation_name == 'Sigmoid':
                alpha = self.lambda_max * torch.sigmoid(logits)
            elif self.activation_name == 'Softmax':
                alpha = torch.softmax(logits, dim=-1)
            else:
                alpha = logits
            alpha_list.append(alpha)
        return alpha_list

# --- Physical Sequential forward with Residual Routing Link ---
def physical_residual_forward(x, experts, router, pcas, r_factor=0.0, activation_relu=nn.ReLU()):
    K = len(experts)
    
    # Layer 1
    psi1 = pcas[0].project(x)
    alpha_list = router([psi1, psi1, psi1])
    alpha1 = alpha_list[0]
    
    # Apply Residual Interpolation
    if r_factor > 0.0:
        # Interpolate with static uniform coefficient 0.25
        alpha1 = (1.0 - r_factor) * alpha1 + r_factor * 0.25
    
    W1_experts = torch.stack([experts[k].fc1.weight.data for k in range(K)], dim=0)
    B1_experts = torch.stack([experts[k].fc1.bias.data for k in range(K)], dim=0)
    W1_merged = torch.einsum("bk,kod->bod", alpha1, W1_experts)
    B1_merged = torch.einsum("bk,ko->bo", alpha1, B1_experts)
    
    h1 = torch.bmm(x.unsqueeze(1), W1_merged.transpose(1, 2)).squeeze(1) + B1_merged
    h1 = activation_relu(h1)
    
    # Layer 2
    psi2 = pcas[1].project(h1)
    alpha_list2 = router([psi1, psi2, psi2])
    alpha2 = alpha_list2[1]
    
    if r_factor > 0.0:
        alpha2 = (1.0 - r_factor) * alpha2 + r_factor * 0.25
    
    W2_experts = torch.stack([experts[k].fc2.weight.data for k in range(K)], dim=0)
    B2_experts = torch.stack([experts[k].fc2.bias.data for k in range(K)], dim=0)
    W2_merged = torch.einsum("bk,kod->bod", alpha2, W2_experts)
    B2_merged = torch.einsum("bk,ko->bo", alpha2, B2_experts)
    
    h2 = torch.bmm(h1.unsqueeze(1), W2_merged.transpose(1, 2)).squeeze(1) + B2_merged
    h2 = activation_relu(h2)
    
    # Layer 3
    psi3 = pcas[2].project(h2)
    alpha_list3 = router([psi1, psi2, psi3])
    alpha3 = alpha_list3[2]
    
    if r_factor > 0.0:
        alpha3 = (1.0 - r_factor) * alpha3 + r_factor * 0.25
        
    W3_experts = torch.stack([experts[k].fc3.weight.data for k in range(K)], dim=0)
    B3_experts = torch.stack([experts[k].fc3.bias.data for k in range(K)], dim=0)
    W3_merged = torch.einsum("bk,kod->bod", alpha3, W3_experts)
    B3_merged = torch.einsum("bk,ko->bo", alpha3, B3_experts)
    
    logits = torch.bmm(h2.unsqueeze(1), W3_merged.transpose(1, 2)).squeeze(1) + B3_merged
    return logits, [alpha1, alpha2, alpha3]

# --- Fit PCAs sequentially ---
def fit_sequential_pcas(experts, calib_X):
    K = len(experts)
    calib_X_t = torch.tensor(calib_X)
    
    W1 = torch.stack([experts[k].fc1.weight.data for k in range(K)], dim=0).mean(dim=0)
    B1 = torch.stack([experts[k].fc1.bias.data for k in range(K)], dim=0).mean(dim=0)
    W2 = torch.stack([experts[k].fc2.weight.data for k in range(K)], dim=0).mean(dim=0)
    B2 = torch.stack([experts[k].fc2.bias.data for k in range(K)], dim=0).mean(dim=0)
    
    relu = nn.ReLU()
    with torch.no_grad():
        h1_calib = relu(torch.matmul(calib_X_t, W1.t()) + B1)
        h2_calib = relu(torch.matmul(h1_calib, W2.t()) + B2)
        
    pca1 = PhysicalPCAPreprojector()
    pca1.fit(calib_X)
    
    pca2 = PhysicalPCAPreprojector()
    pca2.fit(h1_calib)
    
    pca3 = PhysicalPCAPreprojector()
    pca3.fit(h2_calib)
    
    return [pca1, pca2, pca3]

# --- Router Calibration Training ---
def train_physical_router(router, experts, pcas, calib_data, r_factor=0.0, epochs=120, lr=0.05, lambda_wd=1e-4):
    router.train()
    X_cal, Y_cal, T_cal = calib_data
    X_cal_t = torch.tensor(X_cal)
    Y_cal_t = torch.tensor(Y_cal)
    
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, alphas = physical_residual_forward(X_cal_t, experts, router, pcas, r_factor=r_factor)
        loss = criterion(logits, Y_cal_t)
        
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for name, param in router.named_parameters():
            if 'W' in name:
                l2_reg = l2_reg + torch.norm(param, p=2)**2
        total_loss = loss + lambda_wd * l2_reg
        
        total_loss.backward()
        nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
        optimizer.step()

# --- Evaluate Physical Sequential Routing with Residual routing ---
def evaluate_physical_router(router, experts, pcas, test_data, r_factor=0.0, mode='Heterogeneous_B256'):
    router.eval()
    K = len(experts)
    
    with torch.no_grad():
        if mode == 'Homogeneous_B256':
            accuracies = []
            for k in range(K):
                X, Y = test_data[k]
                X_t = torch.tensor(X)
                Y_t = torch.tensor(Y)
                logits, _ = physical_residual_forward(X_t, experts, router, pcas, r_factor=r_factor)
                preds = logits.argmax(dim=-1)
                acc = (preds == Y_t).float().mean().item() * 100
                accuracies.append(acc)
            return accuracies, np.mean(accuracies)
            
        elif mode == 'Heterogeneous_B256':
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
            
            perm = torch.randperm(len(mixed_X))
            mixed_X = mixed_X[perm]
            mixed_Y = mixed_Y[perm]
            mixed_T = mixed_T[perm]
            
            logits, _ = physical_residual_forward(mixed_X, experts, router, pcas, r_factor=r_factor)
            
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

# --- Main Sweep Routine ---
def main():
    seeds = [42, 43, 44, 45, 46]
    r_factors = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    results = {r: {'homogeneous': [], 'heterogeneous': []} for r in r_factors}
    
    print("=" * 60)
    print("RUNNING RESIDUAL ROUTING SWEEP IN PHYSICAL MERGING")
    print("=" * 60)
    
    for seed in seeds:
        print(f"\n>>> Running seed {seed}...")
        set_seed(seed)
        
        train_data, test_data, calib_data = generate_synthetic_dataset(seed)
        experts = train_mlp_experts(train_data, test_data)
        pcas = fit_sequential_pcas(experts, calib_data[0])
        
        for r in r_factors:
            # Shared Physical BWS M=3
            router = PhysicalBWS_Router(L=3, G=1, d=4, K=4, activation='Sigmoid', init_bias=-2.0)
            # Train router with the corresponding residual factor r
            train_physical_router(router, experts, pcas, calib_data, r_factor=r, lr=0.05, lambda_wd=1e-4)
            
            _, homo_mean = evaluate_physical_router(router, experts, pcas, test_data, r_factor=r, mode='Homogeneous_B256')
            _, hetero_mean = evaluate_physical_router(router, experts, pcas, test_data, r_factor=r, mode='Heterogeneous_B256')
            
            results[r]['homogeneous'].append(homo_mean)
            results[r]['heterogeneous'].append(hetero_mean)
            
            print(f"  r = {r:.1f} | Homo: {homo_mean:.2f}% | Hetero: {hetero_mean:.2f}%")
            
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS ACROSS SEEDS FOR RESIDUAL SWEEP")
    print("=" * 60)
    
    summary_out = {}
    for r in r_factors:
        homo_mean = np.mean(results[r]['homogeneous'])
        homo_std = np.std(results[r]['homogeneous'])
        hetero_mean = np.mean(results[r]['heterogeneous'])
        hetero_std = np.std(results[r]['heterogeneous'])
        
        print(f"r = {r:.1f} | Homogeneous: {homo_mean:.2f} \u00b1 {homo_std:.2f}% | Heterogeneous: {hetero_mean:.2f} \u00b1 {hetero_std:.2f}%")
        
        summary_out[str(r)] = {
            'homogeneous': f"{homo_mean:.2f} \u00b1 {homo_std:.2f}%",
            'heterogeneous': f"{hetero_mean:.2f} \u00b1 {hetero_std:.2f}%"
        }
        
    with open('physical_residual_results.json', 'w') as f:
        json.dump(summary_out, f, indent=4)
    print("\nSaved residual routing sweep results to 'physical_residual_results.json'")

if __name__ == '__main__':
    main()
