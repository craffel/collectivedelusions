import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy

# --- 1. Constants and Shared Feature Space ---
input_dim = 128
hidden_dim = 64
num_classes = 8

# Generate the shared pre-trained backbone features
torch.manual_seed(10)
W_shared = torch.randn(input_dim, hidden_dim)

def generate_multi_task_data(num_samples=1200, task_id=1, seed=42):
    torch.manual_seed(seed * 1000 + task_id)
    np.random.seed(seed * 1000 + task_id)
    
    X = torch.randn(num_samples, input_dim)
    features = torch.relu(torch.matmul(X, W_shared))
    W_head = torch.randn(hidden_dim, num_classes)
    logits = torch.matmul(features, W_head) + 0.05 * torch.randn(num_samples, num_classes)
    y = torch.argmax(logits, dim=1)
    return X, y

# Pre-train the base model on Task 0 once (using seed 42) to represent a fixed foundation checkpoint
X0, y0 = generate_multi_task_data(2000, task_id=0, seed=42)
pre_loader = DataLoader(TensorDataset(X0, y0), batch_size=32, shuffle=True)

class BaseMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_classes=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

base_model = BaseMLP(128, 64, 8)
optimizer_pre = optim.Adam(base_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
for epoch in range(15):
    base_model.train()
    for inputs, targets in pre_loader:
        optimizer_pre.zero_grad()
        outputs = base_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_pre.step()

pretrained_backbone_weight = base_model.fc1.weight.data.clone()

# --- 2. OFT Layer and Model Definitions ---
class OFTLayer(nn.Module):
    def __init__(self, base_weight, block_size=8):
        super().__init__()
        self.register_buffer("base_weight", base_weight.clone())
        out_features, in_features = base_weight.shape
        self.out_features = out_features
        self.in_features = in_features
        self.block_size = block_size
        
        assert out_features % block_size == 0, f"out_features {out_features} must be divisible by block_size {block_size}"
        self.num_blocks = out_features // block_size
        self.num_params_per_block = block_size * (block_size - 1) // 2
        
        # Initialize q_params to zero (which corresponds to Q=0, R=Identity)
        self.q_params = nn.Parameter(torch.zeros(self.num_blocks, self.num_params_per_block))
        
    def _get_Q(self):
        device = self.q_params.device
        Q_full = torch.zeros(self.out_features, self.out_features, device=device)
        for idx in range(self.num_blocks):
            block_q = torch.zeros(self.block_size, self.block_size, device=device)
            tri_indices = torch.triu_indices(self.block_size, self.block_size, offset=1)
            block_q[tri_indices[0], tri_indices[1]] = self.q_params[idx]
            block_q = block_q - block_q.T
            start_i = idx * self.block_size
            end_i = start_i + self.block_size
            Q_full[start_i:end_i, start_i:end_i] = block_q
        return Q_full
        
    def _get_R(self, Q_full=None):
        if Q_full is None:
            Q_full = self._get_Q()
        device = Q_full.device
        I = torch.eye(self.out_features, device=device)
        R = torch.matmul(I + Q_full, torch.inverse(I - Q_full))
        return R
        
    def forward(self, x):
        R = self._get_R()
        W = torch.matmul(R, self.base_weight)
        return nn.functional.linear(x, W)

class OFTBackboneModel(nn.Module):
    def __init__(self, pretrained_backbone, num_classes=8, block_size=8):
        super().__init__()
        self.fc1 = OFTLayer(pretrained_backbone, block_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes, bias=False)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def evaluate_task(fc1, fc2, val_loader):
    model = nn.Sequential(fc1, nn.ReLU(), fc2)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total

def train_oft(train_loader, val_loader, pretrained_backbone, block_size, lr=0.01, epochs=20):
    model = OFTBackboneModel(pretrained_backbone, num_classes=8, block_size=block_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    final_loss = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        final_loss = epoch_loss / steps
    acc = evaluate_task(model.fc1, model.fc2, val_loader)
    return final_loss, acc

def train_sa_oft(train_loader, val_loader, pretrained_backbone, block_size, lr=0.01, epochs=20, rho=0.1):
    model = OFTBackboneModel(pretrained_backbone, num_classes=8, block_size=block_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    final_loss = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            epoch_loss += loss.item()
            steps += 1
            
            orig_params = {}
            grads = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    orig_params[name] = p.data.clone()
                    grads[name] = p.grad.clone()
            grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads.values()) + 1e-12)
            for name, p in model.named_parameters():
                if name in grads:
                    p.data.add_(rho * grads[name] / grad_norm)
            optimizer.zero_grad()
            outputs_perturbed = model(inputs)
            loss_perturbed = criterion(outputs_perturbed, targets)
            loss_perturbed.backward()
            for name, p in model.named_parameters():
                if name in orig_params:
                    p.data.copy_(orig_params[name])
            optimizer.step()
        final_loss = epoch_loss / steps
    acc = evaluate_task(model.fc1, model.fc2, val_loader)
    return final_loss, acc

# --- 3. Sweep Protocol ---
block_sizes = [2, 4, 8, 16, 32, 64]
seeds = [42, 43, 44, 45, 46]

results_oft_acc = {b: [] for b in block_sizes}
results_oft_loss = {b: [] for b in block_sizes}
results_sa_acc = {b: [] for b in block_sizes}
results_sa_loss = {b: [] for b in block_sizes}

print("Starting Block Size Sweep across 5 seeds...")
for b in block_sizes:
    print(f"\n--- Block Size: {b} ---")
    for s in seeds:
        # Load Task 1 data for seed s
        X1, y1 = generate_multi_task_data(1200, task_id=1, seed=s)
        train_loader = DataLoader(TensorDataset(X1[:1000], y1[:1000]), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X1[1000:], y1[1000:]), batch_size=32, shuffle=False)
        
        # Train OFT
        loss_oft, acc_oft = train_oft(train_loader, val_loader, pretrained_backbone_weight, block_size=b)
        results_oft_acc[b].append(acc_oft)
        results_oft_loss[b].append(loss_oft)
        
        # Train SA-Ortho
        loss_sa, acc_sa = train_sa_oft(train_loader, val_loader, pretrained_backbone_weight, block_size=b)
        results_sa_acc[b].append(acc_sa)
        results_sa_loss[b].append(loss_sa)
        
        print(f"Seed {s} | OFT: Loss={loss_oft:.4f}, Acc={acc_oft*100:.2f}% | SA-Ortho: Loss={loss_sa:.4f}, Acc={acc_sa*100:.2f}%")

# --- 4. Summarize Results ---
print("\n=== SUMMARY OF RESULTS ===")
print("| Block Size (B) | OFT Loss | OFT Val Acc | SA-Ortho Loss | SA-Ortho Val Acc |")
print("|---|---|---|---|---|")
for b in block_sizes:
    oft_l_mean, oft_l_std = np.mean(results_oft_loss[b]), np.std(results_oft_loss[b])
    oft_a_mean, oft_a_std = np.mean(results_oft_acc[b]), np.std(results_oft_acc[b])
    sa_l_mean, sa_l_std = np.mean(results_sa_loss[b]), np.std(results_sa_loss[b])
    sa_a_mean, sa_a_std = np.mean(results_sa_acc[b]), np.std(results_sa_acc[b])
    
    print(f"| {b} | {oft_l_mean:.4f} \u00b1 {oft_l_std:.4f} | {oft_a_mean*100:.2f}% \u00b1 {oft_a_std*100:.2f}% | {sa_l_mean:.4f} \u00b1 {sa_l_std:.4f} | {sa_a_mean*100:.2f}% \u00b1 {sa_a_std*100:.2f}% |")
