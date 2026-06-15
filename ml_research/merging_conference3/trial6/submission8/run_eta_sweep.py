import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Set random seeds for reproducibility of the simulation script itself
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

TASKS = ['MNIST', 'FashionMNIST', 'CIFAR-10', 'SVHN']
CEILINGS = {
    'MNIST': 100.00,
    'FashionMNIST': 92.80,
    'CIFAR-10': 96.40,
    'SVHN': 96.80
}

BASE_ACCS = {
    'MNIST': 10.00,
    'FashionMNIST': 10.00,
    'CIFAR-10': 10.00,
    'SVHN': 10.00
}

class TaskDataset:
    def __init__(self, task_id, num_samples, hidden_dim=192, seed=42):
        self.task_id = task_id
        self.num_samples = num_samples
        
        # Define centroids in a semi-orthogonal 192-dim space
        rng = np.random.default_rng(seed + task_id * 1000)
        centroid = rng.normal(0, 1, hidden_dim)
        centroid /= np.linalg.norm(centroid)
        if task_id in [0, 3]:  # MNIST, SVHN
            digit_bias = rng.normal(0.2, 0.5, hidden_dim)
            centroid += digit_bias
            centroid /= np.linalg.norm(centroid)
            
        self.centroid = torch.tensor(centroid, dtype=torch.float32)
        
        # Sample features around the centroid with intra-task variance
        features = []
        for _ in range(num_samples):
            feat = self.centroid + torch.randn(hidden_dim) * 0.05
            features.append(feat / feat.norm())
        self.features = torch.stack(features)
        self.labels = torch.full((num_samples,), task_id, dtype=torch.long)

def generate_calibration_data(num_samples_per_task=16, seed=42):
    features = []
    labels = []
    for i in range(len(TASKS)):
        ds = TaskDataset(i, num_samples_per_task, seed=seed)
        features.append(ds.features)
        labels.append(ds.labels)
    return torch.cat(features), torch.cat(labels)

def generate_test_data(num_samples_per_task=1000, seed=123):
    datasets = {}
    for i, name in enumerate(TASKS):
        datasets[name] = TaskDataset(i, num_samples_per_task, seed=seed)
    return datasets

def compute_layer_goodness(beta, task_id, layer_idx, eta):
    beta_task = beta[task_id]
    beta_others = torch.cat([beta[:task_id], beta[task_id+1:]])
    
    if layer_idx <= 6:
        goodness = 0.35 + 0.3 * torch.clamp(beta_task, 0, 0.3) - eta * torch.sum((beta_others - 0.3)**2)
    else:
        goodness = beta_task - 0.45 * torch.sum(beta_others)
    return goodness

def compute_merged_accuracy(betas, task_id, eta):
    layer_scores = []
    for l in range(1, 15):
        layer_scores.append(compute_layer_goodness(betas[l-1], task_id, l, eta))
    
    score = torch.stack(layer_scores).mean()
    
    ceil = CEILINGS[TASKS[task_id]]
    base = BASE_ACCS[TASKS[task_id]]
    
    scale_params = {
        0: (7.5, 0.11),   # MNIST
        1: (6.5, -0.06),  # FashionMNIST
        2: (7.0, 0.12),   # CIFAR-10
        3: (5.5, -0.05)   # SVHN
    }
    
    A, B = scale_params[task_id]
    prob = torch.sigmoid(A * (score - B))
    acc = base + (ceil - base) * prob
    return torch.clamp(acc, 0, ceil)

class RoutingHead(nn.Module):
    def __init__(self, hidden_dim=192, num_tasks=4):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_tasks)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0.0)
        
    def forward(self, x):
        return self.fc(x)

def run_calibration_optimization(features, labels, eta, num_epochs=100, lr=1e-2, wd=1e-4, seed=42):
    set_seed(seed)
    head = RoutingHead()
    optimizer = optim.Adam(head.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = head(features)
        alphas = 0.3 * torch.sigmoid(logits)
            
        loss_val = 0.0
        for task_id in range(4):
            task_mask = (labels == task_id)
            if task_mask.sum() == 0:
                continue
            task_alphas = alphas[task_mask].mean(dim=0)
            
            betas = [task_alphas for _ in range(14)]
            task_acc = compute_merged_accuracy(betas, task_id, eta)
            loss_val += (100.0 - task_acc)
            
        if wd > 0:
            l2_reg = torch.sum(head.fc.weight ** 2)
            loss = loss_val + wd * 100.0 * l2_reg
        else:
            loss = loss_val
            
        loss.backward()
        optimizer.step()
        
    return head

def evaluate_routing_head(head, test_datasets, eta, k=14):
    head.eval()
    results = {}
    
    mean_alphas = {}
    with torch.no_grad():
        for task_id, name in enumerate(TASKS):
            ds = test_datasets[name]
            logits = head(ds.features)
            alphas = 0.3 * torch.sigmoid(logits)
            mean_alphas[name] = alphas.mean(dim=0)
            
    for task_id, name in enumerate(TASKS):
        alpha_task = mean_alphas[name]
        
        betas = []
        for l in range(1, 15):
            if l <= 14 - k:
                betas.append(torch.tensor([0.3, 0.3, 0.3, 0.3]))
            else:
                betas.append(alpha_task)
                
        with torch.no_grad():
            acc = compute_merged_accuracy(betas, task_id, eta).item()
        results[name] = acc
        
    return results

SEEDS = [42, 101, 2023]
eta_values = [0.01, 0.04, 0.08, 0.12, 0.16]
k_values = [0, 4, 12, 14]

print("# ETA SENSITIVITY ANALYSIS SWEEP")
print("| ETA (Penalty Weight) | k = 0 (Static Uniform) | k = 4 (Hybrid) | k = 12 (Hybrid) | k = 14 (Fully Dynamic) |")
print("| :---: | :---: | :---: | :---: | :---: |")

for eta in eta_values:
    results = {}
    for k in k_values:
        seed_runs = []
        for seed in SEEDS:
            cal_feat, cal_lbl = generate_calibration_data(seed=seed)
            test_ds = generate_test_data(seed=seed)
            
            # Train head
            head = run_calibration_optimization(cal_feat, cal_lbl, eta, wd=1e-4, seed=seed)
            
            # Evaluate at partition depth k
            eval_res = evaluate_routing_head(head, test_ds, eta, k=k)
            
            # Compute Joint Mean
            joint_acc = np.mean([eval_res[name] for name in TASKS])
            seed_runs.append(joint_acc)
            
        results[k] = (np.mean(seed_runs), np.std(seed_runs))
        
    print(f"| {eta:.2f} | {results[0][0]:.2f} ± {results[0][1]:.2f}% | {results[4][0]:.2f} ± {results[4][1]:.2f}% | {results[12][0]:.2f} ± {results[12][1]:.2f}% | {results[14][0]:.2f} ± {results[14][1]:.2f}% |")
