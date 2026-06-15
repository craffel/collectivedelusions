import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed
torch.manual_seed(10)
np.random.seed(10)

K = 4          # Number of tasks
D = 192        # Feature dimension
d = 4          # Routing projection dimension
L = 14         # Number of layers

# Generate synthetic dataset (original run_experiments setup)
train_samples_per_task = 500
calib_samples_per_task = 16
test_samples_per_task = 250

class_prototypes = torch.zeros(K, 10, D)
for k in range(K):
    for i in range(10):
        class_prototypes[k, i, k*48 + i*4 : k*48 + (i+1)*4] = 1.0
class_prototypes = class_prototypes / class_prototypes.norm(dim=2, keepdim=True)

def generate_features(samples_per_task, noise_scales=[0.01, 0.10, 0.13, 0.35]):
    z_list = []
    y_list = []
    for k in range(K):
        y_local = torch.randint(0, 10, (samples_per_task,))
        y_global = y_local + k * 10
        
        noise = torch.randn(samples_per_task, D) * noise_scales[k]
        z = class_prototypes[k, y_local] + noise
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        
        z_list.append(z)
        y_list.append(y_global)
        
    return torch.cat(z_list, dim=0), torch.cat(y_list, dim=0)

train_z, train_y = generate_features(train_samples_per_task)
calib_z, calib_y = generate_features(calib_samples_per_task)
test_z, test_y = generate_features(test_samples_per_task)

# PCA Projection matrix P
combined_features = train_z - train_z.mean(dim=0, keepdim=True)
U, S, V = torch.pca_lowrank(combined_features, q=d)
P = V[:, :d]

def project_features(z):
    projected = z @ P
    return projected / (projected.norm(dim=1, keepdim=True) + 1e-8)

calib_psi = project_features(calib_z)
test_psi = project_features(test_z)

def parse_labels(y):
    y_tasks = torch.div(y, 10, rounding_mode='floor')
    y_local = y % 10
    return y_tasks, y_local

calib_tasks, calib_local_y = parse_labels(calib_y)
test_tasks, test_local_y = parse_labels(test_y)

class ExpertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(D, 10) for _ in range(K)])
        
    def forward(self, z, task_id):
        return self.experts[task_id](z)

expert_model = ExpertClassifier()
optimizer = optim.AdamW(expert_model.parameters(), lr=1e-2, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for k in range(K):
    task_train_mask = (train_y // 10 == k)
    z_task_train = train_z[task_train_mask]
    y_task_train = train_y[task_train_mask] % 10
    
    for epoch in range(150):
        expert_model.train()
        optimizer.zero_grad()
        outputs = expert_model(z_task_train, k)
        loss = criterion(outputs, y_task_train)
        loss.backward()
        optimizer.step()

expert_model.eval()

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

def evaluate_accuracies(router):
    router.eval()
    merged_model = MergedClassifier(expert_model, router)
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

# Sweeping learning rate
for lr in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
    qws_router = QWSMergeRouter()
    merged_classifier = MergedClassifier(expert_model, qws_router)
    optimizer = optim.AdamW(qws_router.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(100):
        qws_router.train()
        optimizer.zero_grad()
        loss = 0.0
        for k in range(K):
            task_mask = (calib_tasks == k)
            if task_mask.sum() > 0:
                z_calib = calib_z[task_mask]
                psi_calib = calib_psi[task_mask]
                y_calib = calib_local_y[task_mask]
                logits = merged_classifier(z_calib, psi_calib, batch_average=False)
                loss += criterion(logits, y_calib)
        loss.backward()
        optimizer.step()
        
    accs = evaluate_accuracies(qws_router)
    print(f"LR: {lr:.1e} | Accs: {[f'{a:.1f}%' for a in accs]} | Mean: {np.mean(accs):.2f}%")
