import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed for basic reproducibility
torch.manual_seed(10)
np.random.seed(10)

K = 4          # Number of tasks (0: MNIST, 1: FashionMNIST, 2: CIFAR-10, 3: SVHN)
D = 192        # Feature dimension
L = 14         # Layer groups

# ==========================================
# HELPER FUNCTIONS FOR DATA GENERATION
# ==========================================
def generate_class_prototypes():
    orthogonal_prototypes = torch.zeros(K, 10, D)
    for k in range(K):
        for i in range(10):
            orthogonal_prototypes[k, i, k*48 + i*4 : k*48 + (i+1)*4] = 1.0
    return orthogonal_prototypes / orthogonal_prototypes.norm(dim=2, keepdim=True)

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

def get_data_splits(d, seed=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    prototypes = generate_class_prototypes()
    train_z, train_y = generate_features(500, prototypes)
    calib_z, calib_y = generate_features(16, prototypes)
    test_z, test_y = generate_features(250, prototypes)
    
    # PCA Projection with variable dimension d
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
# ROUTER DEFINITIONS
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
    def __init__(self, d_dim):
        super().__init__()
        self.d_dim = d_dim
        self.Phi = nn.Parameter(torch.eye(K, d_dim).unsqueeze(0).repeat(L, 1, 1) + torch.randn(L, K, d_dim)*0.1)
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
    def __init__(self, d_dim, mode='linear'):
        super().__init__()
        self.mode = mode
        self.d_dim = d_dim
        self.W = nn.Parameter(torch.eye(K, d_dim).unsqueeze(0).repeat(L, 1, 1) * 1.5 + torch.randn(L, K, d_dim)*0.1)
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
    def forward(self, z, psi):
        alpha = self.router(z, psi)
        head_alpha = alpha.mean(dim=1)
        mean_alpha = head_alpha.mean(dim=0)
        W_merged = torch.zeros(10, D, device=z.device)
        B_merged = torch.zeros(10, device=z.device)
        for k in range(K):
            W_merged += mean_alpha[k] * self.experts.experts[k].weight
            B_merged += mean_alpha[k] * self.experts.experts[k].bias
        logits = z @ W_merged.t() + B_merged
        return logits

def train_experts_and_optimize_router(router, train_z, train_y, calib_z, calib_y, calib_tasks, calib_local_y, calib_psi, wd=0.0):
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
                logits = merged_classifier(z_calib, psi_calib)
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
            logits = merged_model(z_task, psi_task)
            preds = logits.argmax(dim=1)
            acc = (preds == y_task).float().mean().item()
            accuracies.append(acc * 100)
    return accuracies

def main():
    dimensions = [2, 4, 8]
    print(f"{'d':<5} | {'Linear Router':<15} | {'QWS-Merge':<12} | {'L3-Lin (L2)':<12} | {'L3-Softmax (L2)':<15}")
    print("-" * 70)
    for d_val in dimensions:
        data = get_data_splits(d_val, seed=10)
        train_z, train_y, calib_z, calib_y, test_z, test_y, calib_psi, test_psi, calib_tasks, calib_local_y, test_tasks, test_local_y = data
        
        # Linear Router (Global Unreg)
        lin_router = CrippledGlobalLinearRouter()
        _, opt_lin = train_experts_and_optimize_router(lin_router, train_z, train_y, calib_z, calib_y, calib_tasks, calib_local_y, calib_psi, wd=0.0)
        lin_accs = evaluate_accuracies_for_router(opt_lin, test_z, test_psi, test_tasks, test_local_y)
        lin_mean = np.mean(lin_accs)
        
        # QWS-Merge
        qws_router = QWSMergeRouter(d_val)
        _, opt_qws = train_experts_and_optimize_router(qws_router, train_z, train_y, calib_z, calib_y, calib_tasks, calib_local_y, calib_psi, wd=0.0)
        qws_accs = evaluate_accuracies_for_router(opt_qws, test_z, test_psi, test_tasks, test_local_y)
        qws_mean = np.mean(qws_accs)
        
        # L3-Linear (L2 Reg)
        l3_lin = L3Router(d_val, mode='linear')
        _, opt_l3_lin = train_experts_and_optimize_router(l3_lin, train_z, train_y, calib_z, calib_y, calib_tasks, calib_local_y, calib_psi, wd=1e-3)
        l3_lin_accs = evaluate_accuracies_for_router(opt_l3_lin, test_z, test_psi, test_tasks, test_local_y)
        l3_lin_mean = np.mean(l3_lin_accs)
        
        # L3-Softmax (L2 Reg)
        l3_smax = L3Router(d_val, mode='softmax')
        _, opt_l3_smax = train_experts_and_optimize_router(l3_smax, train_z, train_y, calib_z, calib_y, calib_tasks, calib_local_y, calib_psi, wd=1e-3)
        l3_smax_accs = evaluate_accuracies_for_router(opt_l3_smax, test_z, test_psi, test_tasks, test_local_y)
        l3_smax_mean = np.mean(l3_smax_accs)
        
        print(f"{d_val:<5} | {lin_mean:<15.2f}% | {qws_mean:<12.2f}% | {l3_lin_mean:<12.2f}% | {l3_smax_mean:<15.2f}%")

if __name__ == "__main__":
    main()
