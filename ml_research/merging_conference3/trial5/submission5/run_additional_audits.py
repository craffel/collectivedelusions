import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed for basic reproducibility
torch.manual_seed(10)
np.random.seed(10)

K = 4          # Number of tasks (0: MNIST, 1: FashionMNIST, 2: CIFAR-10, 3: SVHN)
D = 192        # Feature dimension
d = 4          # Routing projection dimension
L = 14         # Layer groups

# ==========================================
# HELPER FUNCTIONS FOR DATA GENERATION
# ==========================================
def generate_class_prototypes(rho=0.0):
    # Generates class prototypes with correlation parameter rho
    # rho = 0 means completely orthogonal task subspaces
    # rho > 0 means tasks share some subspace dimension
    orthogonal_prototypes = torch.zeros(K, 10, D)
    for k in range(K):
        for i in range(10):
            orthogonal_prototypes[k, i, k*48 + i*4 : k*48 + (i+1)*4] = 1.0
            
    if rho == 0.0:
        return orthogonal_prototypes / orthogonal_prototypes.norm(dim=2, keepdim=True)
        
    # Shared prototype component to introduce correlation (task overlap)
    shared_prototypes = torch.zeros(10, D)
    for i in range(10):
        # A common set of 10 prototypes shared across all tasks, using indices 0:40
        shared_prototypes[i, i*4 : (i+1)*4] = 1.0
    shared_prototypes = shared_prototypes / shared_prototypes.norm(dim=1, keepdim=True)
    
    correlated_prototypes = torch.zeros(K, 10, D)
    for k in range(K):
        for i in range(10):
            correlated_prototypes[k, i] = (1 - rho) * orthogonal_prototypes[k, i] + rho * shared_prototypes[i]
            
    return correlated_prototypes / correlated_prototypes.norm(dim=2, keepdim=True)

def generate_features(samples_per_task, prototypes, noise_scales=[0.01, 0.10, 0.13, 0.35]):
    z_list = []
    y_list = []
    for k in range(K):
        y_local = torch.randint(0, 10, (samples_per_task,))
        y_global = y_local + k * 10
        
        noise = torch.randn(samples_per_task, D) * noise_scales[k]
        z = prototypes[k, y_local] + noise
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        
        z_list.append(z)
        y_list.append(y_global)
        
    return torch.cat(z_list, dim=0), torch.cat(y_list, dim=0)

def get_data_splits(rho=0.0, seed=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    prototypes = generate_class_prototypes(rho)
    train_z, train_y = generate_features(500, prototypes)
    calib_z, calib_y = generate_features(16, prototypes)
    test_z, test_y = generate_features(250, prototypes)
    
    # PCA Projection
    combined_features = train_z - train_z.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(combined_features, q=d)
    P = V[:, :d]
    
    calib_psi = (calib_z @ P) / ((calib_z @ P).norm(dim=1, keepdim=True) + 1e-8)
    test_psi = (test_z @ P) / ((test_z @ P).norm(dim=1, keepdim=True) + 1e-8)
    
    calib_tasks = calib_y // 10
    calib_local_y = calib_y % 10
    test_tasks = test_y // 10
    test_local_y = test_y % 10
    
    return (train_z, train_y, calib_z, calib_y, test_z, test_y, 
            calib_psi, test_psi, calib_tasks, calib_local_y, test_tasks, test_local_y)

# ==========================================
# ROUTER DEFINITIONS (MATCHING MAIN SUITE)
# ==========================================
class CrippledGlobalLinearRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D, K)
        nn.init.eye_(self.fc.weight[:, :K])
    def forward(self, z, psi=None):
        scores = self.fc(z)
        return torch.softmax(scores, dim=1).unsqueeze(1).repeat(1, L, 1)

class QWSMergeRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.Phi = nn.Parameter(torch.eye(K, d).unsqueeze(0).repeat(L, 1, 1) + torch.randn(L, K, d)*0.1)
        self.R = nn.Parameter(torch.ones(L, K) * 0.3)
        self.phi = nn.Parameter(torch.ones(L, K) * -np.pi)
    def forward(self, z, psi):
        B = psi.size(0)
        alpha = torch.zeros(B, L, K, device=psi.device)
        for l in range(L):
            hat_Phi = self.Phi[l] / (self.Phi[l].norm(dim=1, keepdim=True) + 1e-8)
            for k in range(K):
                overlap = torch.mv(psi, hat_Phi[k])
                alpha[:, l, k] = self.R[l, k] * torch.cos(np.pi * overlap + self.phi[l, k])
        return alpha

class L3Router(nn.Module):
    def __init__(self, mode='linear'):
        super().__init__()
        self.mode = mode
        self.W = nn.Parameter(torch.eye(K, d).unsqueeze(0).repeat(L, 1, 1) * 1.5 + torch.randn(L, K, d)*0.1)
        self.B = nn.Parameter(torch.zeros(L, K))
    def forward(self, z, psi):
        B = psi.size(0)
        alpha = torch.zeros(B, L, K, device=psi.device)
        for l in range(L):
            if self.mode == 'softmax':
                scores = torch.zeros(B, K, device=psi.device)
                for k in range(K):
                    scores[:, k] = torch.mv(psi, self.W[l, k]) + self.B[l, k]
                alpha[:, l] = torch.softmax(scores, dim=1)
            else:
                for k in range(K):
                    score = torch.mv(psi, self.W[l, k]) + self.B[l, k]
                    if self.mode == 'linear':
                        alpha[:, l, k] = score
                    elif self.mode == 'tanh':
                        alpha[:, l, k] = torch.tanh(score)
        return alpha

# ==========================================
# MAIN EXPERT TRAINING & ROUTER OPTIMIZATION
# ==========================================
class ExpertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(D, 10) for _ in range(K)])
    def forward(self, z, task_id):
        return self.experts[task_id](z)

class MergedClassifier(nn.Module):
    def __init__(self, experts, router):
        super().__init__()
        self.experts = experts
        self.router = router
    def forward(self, z, psi, batch_average=True):
        alpha = self.router(z, psi)
        head_alpha = alpha.mean(dim=1)
        if batch_average:
            mean_alpha = head_alpha.mean(dim=0)
            W_merged = torch.zeros(10, D, device=z.device)
            B_merged = torch.zeros(10, device=z.device)
            for k in range(K):
                W_merged += mean_alpha[k] * self.experts.experts[k].weight
                B_merged += mean_alpha[k] * self.experts.experts[k].bias
            logits = z @ W_merged.t() + B_merged
        else:
            B = z.size(0)
            logits = torch.zeros(B, 10, device=z.device)
            for b in range(B):
                W_merged = torch.zeros(10, D, device=z.device)
                B_merged = torch.zeros(10, device=z.device)
                for k in range(K):
                    W_merged += head_alpha[b, k] * self.experts.experts[k].weight
                    B_merged += head_alpha[b, k] * self.experts.experts[k].bias
                logits[b] = z[b] @ W_merged.t() + B_merged
        return logits

def train_experts_and_optimize_router(router, train_z, train_y, calib_z, calib_y, calib_tasks, calib_local_y, calib_psi, wd=0.0):
    # Train experts
    expert_model = ExpertClassifier()
    optimizer_exp = optim.AdamW(expert_model.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for k in range(K):
        task_train_mask = (train_y // 10 == k)
        z_task_train = train_z[task_train_mask]
        y_task_train = train_y[task_train_mask] % 10
        for epoch in range(150):
            expert_model.train()
            optimizer_exp.zero_grad()
            outputs = expert_model(z_task_train, k)
            loss = criterion(outputs, y_task_train)
            loss.backward()
            optimizer_exp.step()
            
    expert_model.eval()
    
    # Optimize router
    merged_classifier = MergedClassifier(expert_model, router)
    optimizer_r = optim.AdamW(router.parameters(), lr=1e-2, weight_decay=wd)
    
    for epoch in range(100):
        router.train()
        optimizer_r.zero_grad()
        loss = 0.0
        for k in range(K):
            task_mask = (calib_tasks == k)
            if task_mask.sum() > 0:
                z_calib = calib_z[task_mask]
                psi_calib = calib_psi[task_mask]
                y_calib = calib_local_y[task_mask]
                # Fully vectorized forward pass task-by-task (no representation leakage!)
                logits = merged_classifier(z_calib, psi_calib, batch_average=True)
                loss += criterion(logits, y_calib)
        loss.backward()
        optimizer_r.step()
        
    router.eval()
    return expert_model, merged_classifier

def evaluate_accuracies_for_router(merged_model, test_z, test_psi, test_tasks, test_local_y):
    merged_model.eval()
    accuracies = []
    with torch.no_grad():
        for k in range(K):
            task_mask = (test_tasks == k)
            z_task = test_z[task_mask]
            psi_task = test_psi[task_mask]
            y_task = test_local_y[task_mask]
            logits = merged_model(z_task, psi_task, batch_average=True)
            preds = logits.argmax(dim=1)
            acc = (preds == y_task).float().mean().item()
            accuracies.append(acc * 100)
    return accuracies

# ==========================================
# AUDIT 1: MULTI-SEED GENERALIZATION EVALUATION
# ==========================================
def run_multi_seed_sweep():
    print("\n" + "="*80)
    print("AUDIT 1: MULTI-SEED GENERALIZATION ACCURACY (5 SEEDS)")
    print("="*80)
    seeds = [10, 11, 12, 13, 14]
    
    results = {
        "Uniform Merging": [],
        "Linear Router": [],
        "QWS-Merge": [],
        "L3-Linear (L2 Reg)": [],
        "L3-Softmax (L2 Reg)": []
    }
    
    for s in seeds:
        data = get_data_splits(rho=0.0, seed=s)
        train_z, train_y, calib_z, calib_y, test_z, test_y, calib_psi, test_psi, calib_tasks, calib_local_y, test_tasks, test_local_y = data
        
        # 1. Uniform
        class UniformRouter(nn.Module):
            def forward(self, z, psi):
                B = psi.size(0)
                return torch.ones(B, L, K, device=psi.device) * 0.25
        # Train temporary experts to evaluate uniform
        expert_model = ExpertClassifier()
        optimizer_exp = optim.AdamW(expert_model.parameters(), lr=1e-2, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        for k in range(K):
            task_train_mask = (train_y // 10 == k)
            z_task_train = train_z[task_train_mask]
            y_task_train = train_y[task_train_mask] % 10
            for epoch in range(150):
                expert_model.train()
                optimizer_exp.zero_grad()
                outputs = expert_model(z_task_train, k)
                loss = criterion(outputs, y_task_train)
                loss.backward()
                optimizer_exp.step()
        expert_model.eval()
        uni_model = MergedClassifier(expert_model, UniformRouter())
        uni_accs = evaluate_accuracies_for_router(uni_model, test_z, test_psi, test_tasks, test_local_y)
        results["Uniform Merging"].append(np.mean(uni_accs))
        
        # 2. Global Linear Router
        lin_router = CrippledGlobalLinearRouter()
        _, opt_lin = train_experts_and_optimize_router(lin_router, train_z, train_y, calib_z, calib_y, calib_tasks, calib_local_y, calib_psi, wd=0.0)
        lin_accs = evaluate_accuracies_for_router(opt_lin, test_z, test_psi, test_tasks, test_local_y)
        results["Linear Router"].append(np.mean(lin_accs))
        
        # 3. QWS-Merge
        qws_router = QWSMergeRouter()
        _, opt_qws = train_experts_and_optimize_router(qws_router, train_z, train_y, calib_z, calib_y, calib_tasks, calib_local_y, calib_psi, wd=0.0)
        qws_accs = evaluate_accuracies_for_router(opt_qws, test_z, test_psi, test_tasks, test_local_y)
        results["QWS-Merge"].append(np.mean(qws_accs))
        
        # 4. L3-Linear (L2 Reg)
        l3_lin = L3Router(mode='linear')
        _, opt_l3_lin = train_experts_and_optimize_router(l3_lin, train_z, train_y, calib_z, calib_y, calib_tasks, calib_local_y, calib_psi, wd=1e-3)
        l3_lin_accs = evaluate_accuracies_for_router(opt_l3_lin, test_z, test_psi, test_tasks, test_local_y)
        results["L3-Linear (L2 Reg)"].append(np.mean(l3_lin_accs))
        
        # 5. L3-Softmax (L2 Reg)
        l3_smax = L3Router(mode='softmax')
        _, opt_l3_smax = train_experts_and_optimize_router(l3_smax, train_z, train_y, calib_z, calib_y, calib_tasks, calib_local_y, calib_psi, wd=1e-3)
        l3_smax_accs = evaluate_accuracies_for_router(opt_l3_smax, test_z, test_psi, test_tasks, test_local_y)
        results["L3-Softmax (L2 Reg)"].append(np.mean(l3_smax_accs))
        
    print(f"{'Method':<25} | {'Accuracies across Seeds':<40} | {'Mean ± Std':<15}")
    print("-"*85)
    for method, accs in results.items():
        formatted_accs = ", ".join([f"{a:.2f}%" for a in accs])
        print(f"{method:<25} | [{formatted_accs}] | {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
    print("="*80 + "\n")

# ==========================================
# AUDIT 2: TASK CORRELATION SWEEP
# ==========================================
def run_task_correlation_sweep():
    print("="*80)
    print("AUDIT 2: TASK CORRELATION (SUB-SPACE OVERLAP) SWEEP")
    print("="*80)
    rhos = [0.0, 0.25, 0.50, 0.75]
    
    results = {
        "Linear Router": [],
        "QWS-Merge": [],
        "L3-Linear (L2 Reg)": [],
        "L3-Softmax (L2 Reg)": []
    }
    
    for rho in rhos:
        data = get_data_splits(rho=rho, seed=10)
        train_z, train_y, calib_z, calib_y, test_z, test_y, calib_psi, test_psi, calib_tasks, calib_local_y, test_tasks, test_local_y = data
        
        # 1. Linear Router
        lin_router = CrippledGlobalLinearRouter()
        _, opt_lin = train_experts_and_optimize_router(lin_router, train_z, train_y, calib_z, calib_y, calib_tasks, calib_local_y, calib_psi, wd=0.0)
        lin_accs = evaluate_accuracies_for_router(opt_lin, test_z, test_psi, test_tasks, test_local_y)
        results["Linear Router"].append(np.mean(lin_accs))
        
        # 2. QWS-Merge
        qws_router = QWSMergeRouter()
        _, opt_qws = train_experts_and_optimize_router(qws_router, train_z, train_y, calib_z, calib_y, calib_tasks, calib_local_y, calib_psi, wd=0.0)
        qws_accs = evaluate_accuracies_for_router(opt_qws, test_z, test_psi, test_tasks, test_local_y)
        results["QWS-Merge"].append(np.mean(qws_accs))
        
        # 3. L3-Linear (L2 Reg)
        l3_lin = L3Router(mode='linear')
        _, opt_l3_lin = train_experts_and_optimize_router(l3_lin, train_z, train_y, calib_z, calib_y, calib_tasks, calib_local_y, calib_psi, wd=1e-3)
        l3_lin_accs = evaluate_accuracies_for_router(opt_l3_lin, test_z, test_psi, test_tasks, test_local_y)
        results["L3-Linear (L2 Reg)"].append(np.mean(l3_lin_accs))
        
        # 4. L3-Softmax (L2 Reg)
        l3_smax = L3Router(mode='softmax')
        _, opt_l3_smax = train_experts_and_optimize_router(l3_smax, train_z, train_y, calib_z, calib_y, calib_tasks, calib_local_y, calib_psi, wd=1e-3)
        l3_smax_accs = evaluate_accuracies_for_router(opt_l3_smax, test_z, test_psi, test_tasks, test_local_y)
        results["L3-Softmax (L2 Reg)"].append(np.mean(l3_smax_accs))
        
    print(f"{'Method':<25} | {'rho = 0.0':<11} | {'rho = 0.25':<11} | {'rho = 0.50':<11} | {'rho = 0.75':<11}")
    print("-"*80)
    for method, accs in results.items():
        print(f"{method:<25} | {accs[0]:.2f}%     | {accs[1]:.2f}%     | {accs[2]:.2f}%     | {accs[3]:.2f}%")
    print("="*80 + "\n")

# ==========================================
# AUDIT 3: TRUE LAYER-BY-LAYER WEIGHT MERGING (NO AVERAGING)
# ==========================================
class DeepExpertLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(out_dim, in_dim) + torch.randn(out_dim, in_dim)*0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

class DeepExpertClassifier(nn.Module):
    def __init__(self, in_dim, mid_dim, num_layers=L):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(DeepExpertLayer(in_dim, mid_dim))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(DeepExpertLayer(mid_dim, mid_dim))
        # Output layer (to class dimension 10)
        self.layers.append(DeepExpertLayer(mid_dim, 10))
        
    def forward(self, x):
        h = x
        for l in range(self.num_layers - 1):
            h = torch.relu(h @ self.layers[l].weight.t() + self.layers[l].bias)
        logits = h @ self.layers[-1].weight.t() + self.layers[-1].bias
        return logits

class TrueLayerWiseMergedClassifier(nn.Module):
    def __init__(self, deep_experts, router):
        super().__init__()
        self.deep_experts = deep_experts # nn.ModuleList of K DeepExpertClassifiers
        self.router = router             # Outputs coefficients [B, L, K]
        
    def forward(self, z, psi, batch_average=True):
        B = z.size(0)
        alpha = self.router(z, psi) # [B, L, K]
        
        if batch_average:
            mean_alpha = alpha.mean(dim=0) # [L, K]
            # Layer-by-layer dynamic weight merging (No averaging across the layer dimension!)
            h = z
            for l in range(L - 1):
                W_merged = torch.zeros_like(self.deep_experts[0].layers[l].weight)
                B_merged = torch.zeros_like(self.deep_experts[0].layers[l].bias)
                for k in range(K):
                    W_merged += mean_alpha[l, k] * self.deep_experts[k].layers[l].weight
                    B_merged += mean_alpha[l, k] * self.deep_experts[k].layers[l].bias
                h = torch.relu(h @ W_merged.t() + B_merged)
                
            W_merged_out = torch.zeros_like(self.deep_experts[0].layers[-1].weight)
            B_merged_out = torch.zeros_like(self.deep_experts[0].layers[-1].bias)
            for k in range(K):
                W_merged_out += mean_alpha[-1, k] * self.deep_experts[k].layers[-1].weight
                B_merged_out += mean_alpha[-1, k] * self.deep_experts[k].layers[-1].bias
            logits = h @ W_merged_out.t() + B_merged_out
        else:
            # Sample-wise merging
            logits = torch.zeros(B, 10, device=z.device)
            for b in range(B):
                h = z[b:b+1]
                for l in range(L - 1):
                    W_merged = torch.zeros_like(self.deep_experts[0].layers[l].weight)
                    B_merged = torch.zeros_like(self.deep_experts[0].layers[l].bias)
                    for k in range(K):
                        W_merged += alpha[b, l, k] * self.deep_experts[k].layers[l].weight
                        B_merged += alpha[b, l, k] * self.deep_experts[k].layers[l].bias
                    h = torch.relu(h @ W_merged.t() + B_merged)
                W_merged_out = torch.zeros_like(self.deep_experts[0].layers[-1].weight)
                B_merged_out = torch.zeros_like(self.deep_experts[0].layers[-1].bias)
                for k in range(K):
                    W_merged_out += alpha[b, -1, k] * self.deep_experts[k].layers[-1].weight
                    B_merged_out += alpha[b, -1, k] * self.deep_experts[k].layers[-1].bias
                logits[b] = h @ W_merged_out.t() + B_merged_out
                
        return logits

def run_layer_wise_no_averaging_audit():
    print("="*80)
    print("AUDIT 3: TRUE LAYER-BY-LAYER WEIGHT MERGING (NO LAYER AVERAGING)")
    print("="*80)
    
    # 1. Set up aligned deep experts in sandbox (initialized from shared base weights, then adapted)
    torch.manual_seed(10)
    np.random.seed(10)
    
    data = get_data_splits(rho=0.0, seed=10)
    train_z, train_y, calib_z, calib_y, test_z, test_y, calib_psi, test_psi, calib_tasks, calib_local_y, test_tasks, test_local_y = data
    
    # Create shared base network
    mid_dim = 16
    base_expert = DeepExpertClassifier(in_dim=D, mid_dim=mid_dim, num_layers=L)
    
    # Construct K deep experts, initialized with minor offset from base to ensure representation alignment
    deep_experts = nn.ModuleList()
    for k in range(K):
        expert = DeepExpertClassifier(in_dim=D, mid_dim=mid_dim, num_layers=L)
        # Copy base weights to keep them aligned
        expert.load_state_dict(base_expert.state_dict())
        deep_experts.append(expert)
        
    # Train deep experts independently on their task splits
    optimizer = optim.AdamW(deep_experts.parameters(), lr=5e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for k in range(K):
        task_train_mask = (train_y // 10 == k)
        z_task_train = train_z[task_train_mask]
        y_task_train = train_y[task_train_mask] % 10
        for epoch in range(150):
            optimizer.zero_grad()
            outputs = deep_experts[k](z_task_train)
            loss = criterion(outputs, y_task_train)
            loss.backward()
            optimizer.step()
            
    deep_experts.eval()
    
    # Define Routers
    lin_router = CrippledGlobalLinearRouter()
    qws_router = QWSMergeRouter()
    l3_lin_reg = L3Router(mode='linear')
    l3_smax_reg = L3Router(mode='softmax')
    
    routers = {
        "Linear Router (Unreg)": (lin_router, 0.0),
        "QWS-Merge (Unreg)": (qws_router, 0.0),
        "L3-Linear (L2 Reg, wd=1e-3)": (l3_lin_reg, 1e-3),
        "L3-Softmax (L2 Reg, wd=1e-3)": (l3_smax_reg, 1e-3)
    }
    
    # Train each router under the True Layer-Wise Merged framework
    for name, (router, wd) in routers.items():
        merged_model = TrueLayerWiseMergedClassifier(deep_experts, router)
        optimizer_r = optim.AdamW(router.parameters(), lr=1e-2, weight_decay=wd)
        
        for epoch in range(100):
            router.train()
            optimizer_r.zero_grad()
            loss = 0.0
            for k in range(K):
                task_mask = (calib_tasks == k)
                if task_mask.sum() > 0:
                    z_calib = calib_z[task_mask]
                    psi_calib = calib_psi[task_mask]
                    y_calib = calib_local_y[task_mask]
                    # Fully vectorized forward pass task-by-task!
                    logits = merged_model(z_calib, psi_calib, batch_average=True)
                    loss += criterion(logits, y_calib)
            loss.backward()
            optimizer_r.step()
            
        router.eval()
        
        # Evaluate true layer-wise dynamic merged model
        accuracies = []
        with torch.no_grad():
            for k in range(K):
                task_mask = (test_tasks == k)
                z_task = test_z[task_mask]
                psi_task = test_psi[task_mask]
                y_task = test_local_y[task_mask]
                logits = merged_model(z_task, psi_task, batch_average=True)
                preds = logits.argmax(dim=1)
                acc = (preds == y_task).float().mean().item()
                accuracies.append(acc * 100)
                
        print(f"{name:<30} | Accs: {[f'{a:.2f}%' for a in accuracies]} | Joint Mean: {np.mean(accuracies):.2f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_multi_seed_sweep()
    run_task_correlation_sweep()
    run_layer_wise_no_averaging_audit()
