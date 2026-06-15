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
    
    # Calibrated noise standard deviations to match original difficulty
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
                
                # Permuted labels to create task conflict
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
        
    # Calibration split of 64 samples
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
        X_te, Y_te = test_data[k]
        
        X_tr_t = torch.tensor(X_tr)
        Y_tr_t = torch.tensor(Y_tr)
        X_te_t = torch.tensor(X_te)
        Y_te_t = torch.tensor(Y_te)
        
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
        with torch.no_grad():
            acc = (model(X_te_t).argmax(dim=1) == Y_te_t).float().mean().item() * 100
        print(f"  Expert {k} Test Accuracy: {acc:.2f}%")
        experts[k] = model
    return experts

# --- PCA Projector for Hidden Activations ---
class PhysicalPCAPreprojector:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.P = None
        
    def fit(self, H):
        # H is [B, D] numpy array or tensor
        if isinstance(H, torch.Tensor):
            H = H.cpu().numpy()
        pca = PCA(n_components=self.n_components)
        pca.fit(H)
        self.P = torch.tensor(pca.components_.T, dtype=torch.float32)
        
    def project(self, H):
        # H is [B, D] tensor
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
        self.G = G # G=1 for shared (M=3), G=3 for unshared (M=1)
        self.d = d
        self.K = K
        self.activation_name = activation
        self.lambda_max = lambda_max
        
        self.W = nn.Parameter(torch.zeros(G, K, d))
        self.B = nn.Parameter(torch.zeros(G, K))
        nn.init.normal_(self.W, std=0.01)
        nn.init.constant_(self.B, init_bias)
        
    def forward(self, psi_list):
        # psi_list is list of L tensors, each [B, d]
        alpha_list = []
        for l in range(self.L):
            g = l if self.G == self.L else 0
            psi = psi_list[l]
            logits = torch.matmul(psi, self.W[g].t()) + self.B[g].unsqueeze(0) # [B, K]
            if self.activation_name == 'Sigmoid':
                alpha = self.lambda_max * torch.sigmoid(logits)
            elif self.activation_name == 'Softmax':
                alpha = torch.softmax(logits, dim=-1)
            else:
                alpha = logits
            alpha_list.append(alpha)
        return alpha_list # list of L tensors of shape [B, K]

# --- Physical Sequential weight-space model-merging forward pass ---
def physical_sequential_forward(x, experts, router, pcas, activation_relu=nn.ReLU()):
    # x is [B, D]
    K = len(experts)
    
    # Layer 1
    psi1 = pcas[0].project(x)
    alpha_list = router([psi1, psi1, psi1]) # Dummy list to allow general forward signature
    alpha1 = alpha_list[0] # [B, K]
    
    # Extract expert layer 1 weights & biases
    W1_experts = torch.stack([experts[k].fc1.weight.data for k in range(K)], dim=0) # [K, 128, 192]
    B1_experts = torch.stack([experts[k].fc1.bias.data for k in range(K)], dim=0)   # [K, 128]
    
    # Vectorized blend
    W1_merged = torch.einsum("bk,kod->bod", alpha1, W1_experts) # [B, 128, 192]
    B1_merged = torch.einsum("bk,ko->bo", alpha1, B1_experts)   # [B, 128]
    
    # Sequential layer 1 pass
    h1 = torch.bmm(x.unsqueeze(1), W1_merged.transpose(1, 2)).squeeze(1) + B1_merged # [B, 128]
    h1 = activation_relu(h1)
    
    # Layer 2
    psi2 = pcas[1].project(h1)
    alpha_list2 = router([psi1, psi2, psi2])
    alpha2 = alpha_list2[1]
    
    W2_experts = torch.stack([experts[k].fc2.weight.data for k in range(K)], dim=0) # [K, 64, 128]
    B2_experts = torch.stack([experts[k].fc2.bias.data for k in range(K)], dim=0)   # [K, 64]
    W2_merged = torch.einsum("bk,kod->bod", alpha2, W2_experts)
    B2_merged = torch.einsum("bk,ko->bo", alpha2, B2_experts)
    
    h2 = torch.bmm(h1.unsqueeze(1), W2_merged.transpose(1, 2)).squeeze(1) + B2_merged # [B, 64]
    h2 = activation_relu(h2)
    
    # Layer 3
    psi3 = pcas[2].project(h2)
    alpha_list3 = router([psi1, psi2, psi3])
    alpha3 = alpha_list3[2]
    
    W3_experts = torch.stack([experts[k].fc3.weight.data for k in range(K)], dim=0) # [K, 10, 64]
    B3_experts = torch.stack([experts[k].fc3.bias.data for k in range(K)], dim=0)   # [K, 10]
    W3_merged = torch.einsum("bk,kod->bod", alpha3, W3_experts)
    B3_merged = torch.einsum("bk,ko->bo", alpha3, B3_experts)
    
    logits = torch.bmm(h2.unsqueeze(1), W3_merged.transpose(1, 2)).squeeze(1) + B3_merged # [B, 10]
    return logits, [alpha1, alpha2, alpha3]

# --- Static Uniform Physical Merging ---
def evaluate_static_uniform_physical(experts, test_data):
    K = 4
    W1 = torch.stack([experts[k].fc1.weight.data for k in range(K)], dim=0).mean(dim=0)
    B1 = torch.stack([experts[k].fc1.bias.data for k in range(K)], dim=0).mean(dim=0)
    W2 = torch.stack([experts[k].fc2.weight.data for k in range(K)], dim=0).mean(dim=0)
    B2 = torch.stack([experts[k].fc2.bias.data for k in range(K)], dim=0).mean(dim=0)
    W3 = torch.stack([experts[k].fc3.weight.data for k in range(K)], dim=0).mean(dim=0)
    B3 = torch.stack([experts[k].fc3.bias.data for k in range(K)], dim=0).mean(dim=0)
    
    relu = nn.ReLU()
    accuracies = []
    with torch.no_grad():
        for k in range(K):
            X, Y = test_data[k]
            X_t = torch.tensor(X)
            Y_t = torch.tensor(Y)
            
            h1 = relu(torch.matmul(X_t, W1.t()) + B1)
            h2 = relu(torch.matmul(h1, W2.t()) + B2)
            logits = torch.matmul(h2, W3.t()) + B3
            preds = logits.argmax(dim=-1)
            acc = (preds == Y_t).float().mean().item() * 100
            accuracies.append(acc)
    return accuracies, np.mean(accuracies)

# --- Fit PCAs sequentially under Static Uniform routing or Experts calibration ---
def fit_sequential_pcas(experts, calib_X):
    K = len(experts)
    calib_X_t = torch.tensor(calib_X)
    
    # PCA 1 (Inputs)
    pca1 = PhysicalPCAPreprojector()
    pca1.fit(calib_X)
    
    # To get realistic hidden activations, propagate with a uniform blending of weights
    # (since we are doing post-hoc ensembling, uniform blending represents the initial/centered regime)
    W1 = torch.stack([experts[k].fc1.weight.data for k in range(K)], dim=0).mean(dim=0)
    B1 = torch.stack([experts[k].fc1.bias.data for k in range(K)], dim=0).mean(dim=0)
    W2 = torch.stack([experts[k].fc2.weight.data for k in range(K)], dim=0).mean(dim=0)
    B2 = torch.stack([experts[k].fc2.bias.data for k in range(K)], dim=0).mean(dim=0)
    
    relu = nn.ReLU()
    with torch.no_grad():
        h1_calib = relu(torch.matmul(calib_X_t, W1.t()) + B1)
        h2_calib = relu(torch.matmul(h1_calib, W2.t()) + B2)
        
    pca2 = PhysicalPCAPreprojector()
    pca2.fit(h1_calib)
    
    pca3 = PhysicalPCAPreprojector()
    pca3.fit(h2_calib)
    
    return [pca1, pca2, pca3]

# --- Router Calibration Training ---
def train_physical_router(router, experts, pcas, calib_data, epochs=120, lr=0.05, lambda_wd=1e-4):
    router.train()
    X_cal, Y_cal, T_cal = calib_data
    X_cal_t = torch.tensor(X_cal)
    Y_cal_t = torch.tensor(Y_cal)
    
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0) # Exclude weight decay from AdamW
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, alphas = physical_sequential_forward(X_cal_t, experts, router, pcas)
        loss = criterion(logits, Y_cal_t)
        
        # Manual L2 weight regularization excluding biases
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for name, param in router.named_parameters():
            if 'W' in name: # Only regularize routing weights W
                l2_reg = l2_reg + torch.norm(param, p=2)**2
        total_loss = loss + lambda_wd * l2_reg
        
        total_loss.backward()
        nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
        optimizer.step()

# --- Evaluate Physical Sequential Routing ---
def evaluate_physical_router(router, experts, pcas, test_data, mode='Homogeneous_B256'):
    router.eval()
    K = len(experts)
    
    with torch.no_grad():
        if mode == 'Homogeneous_B256':
            accuracies = []
            for k in range(K):
                X, Y = test_data[k]
                X_t = torch.tensor(X)
                Y_t = torch.tensor(Y)
                logits, _ = physical_sequential_forward(X_t, experts, router, pcas)
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
            
            # Shuffle mixed batch
            perm = torch.randperm(len(mixed_X))
            mixed_X = mixed_X[perm]
            mixed_Y = mixed_Y[perm]
            mixed_T = mixed_T[perm]
            
            logits, _ = physical_sequential_forward(mixed_X, experts, router, pcas)
            
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
    results = {
        'Static_Uniform': [],
        'Physical_BWS_Unshared_M1': [],
        'Physical_BWS_Shared_M3': [],
        'Hetero_Static_Uniform': [],
        'Hetero_BWS_Unshared_M1': [],
        'Hetero_BWS_Shared_M3': []
    }
    
    print("=" * 60)
    print("RUNNING PHYSICAL SEQUENTIAL WEIGHT-SPACE MODEL MERGING")
    print("=" * 60)
    
    for seed in seeds:
        print(f"\n>>> Running seed {seed}...")
        set_seed(seed)
        
        # 1. Dataset Generation
        train_data, test_data, calib_data = generate_synthetic_dataset(seed)
        
        # 2. Train Multi-Layer Experts
        print("  Training Multi-layer MLP Experts...")
        experts = train_mlp_experts(train_data, test_data)
        
        # 3. Fit Sequential PCAs
        pcas = fit_sequential_pcas(experts, calib_data[0])
        
        # 4. Evaluate Static Uniform Physical Merging
        static_accs, static_mean = evaluate_static_uniform_physical(experts, test_data)
        results['Static_Uniform'].append(static_mean)
        print(f"  Static Uniform Joint Mean: {static_mean:.2f}% (Tasks: {[f'{x:.2f}%' for x in static_accs]})")
        
        static_h_accs, static_h_mean = evaluate_static_uniform_physical(experts, test_data) # Heterogeneous identical for static
        results['Hetero_Static_Uniform'].append(static_h_mean)
        
        # 5. Evaluate Physical BWS Router M=1 (Unshared - 12 parameters per layer, total 36)
        router_m1 = PhysicalBWS_Router(L=3, G=3, d=4, K=4, activation='Sigmoid', init_bias=-2.0)
        train_physical_router(router_m1, experts, pcas, calib_data, lr=0.05, lambda_wd=1e-4)
        m1_accs, m1_mean = evaluate_physical_router(router_m1, experts, pcas, test_data, mode='Homogeneous_B256')
        results['Physical_BWS_Unshared_M1'].append(m1_mean)
        
        m1_h_accs, m1_h_mean = evaluate_physical_router(router_m1, experts, pcas, test_data, mode='Heterogeneous_B256')
        results['Hetero_BWS_Unshared_M1'].append(m1_h_mean)
        print(f"  Physical BWS M=1 Joint Mean: {m1_mean:.2f}% (Tasks: {[f'{x:.2f}%' for x in m1_accs]})")
        
        # 6. Evaluate Physical BWS Router M=3 (Shared - 12 parameters total, 66.7% reduction)
        router_m3 = PhysicalBWS_Router(L=3, G=1, d=4, K=4, activation='Sigmoid', init_bias=-2.0)
        train_physical_router(router_m3, experts, pcas, calib_data, lr=0.05, lambda_wd=1e-4)
        m3_accs, m3_mean = evaluate_physical_router(router_m3, experts, pcas, test_data, mode='Homogeneous_B256')
        results['Physical_BWS_Shared_M3'].append(m3_mean)
        
        m3_h_accs, m3_h_mean = evaluate_physical_router(router_m3, experts, pcas, test_data, mode='Heterogeneous_B256')
        results['Hetero_BWS_Shared_M3'].append(m3_h_mean)
        print(f"  Physical BWS M=3 Joint Mean: {m3_mean:.2f}% (Tasks: {[f'{x:.2f}%' for x in m3_accs]})")
        
    print("\n" + "=" * 60)
    print("FINAL SUMMARY STATISTICS (Across 5 Seeds)")
    print("=" * 60)
    
    for key, val in results.items():
        mean = np.mean(val)
        std = np.std(val)
        print(f"{key:<30} : {mean:.2f} \u00b1 {std:.2f}%")
        
    # Save results to json for paper integration
    summary_out = {}
    for key, val in results.items():
        summary_out[key] = f"{np.mean(val):.2f} \u00b1 {np.std(val):.2f}%"
        
    with open('physical_merging_results.json', 'w') as f:
        json.dump(summary_out, f, indent=4)
    print("\nSaved physical merging results to 'physical_merging_results.json'")

if __name__ == '__main__':
    main()
