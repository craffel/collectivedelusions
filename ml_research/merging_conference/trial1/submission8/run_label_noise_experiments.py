import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
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

def add_label_noise(y, noise_rate=0.20, num_classes=8):
    y_noisy = y.clone()
    num_to_flip = int(noise_rate * len(y))
    flip_idx = np.random.choice(len(y), num_to_flip, replace=False)
    for idx in flip_idx:
        current_class = y[idx].item()
        other_classes = [c for c in range(num_classes) if c != current_class]
        y_noisy[idx] = np.random.choice(other_classes)
    return y_noisy

# Pre-train the base model on Task 0 once
print("--- Pre-training Base Model on Task 0 ---")
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
        self.num_blocks = out_features // block_size
        self.num_params_per_block = block_size * (block_size - 1) // 2
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

def evaluate_task(backbone_layer, head_layer, val_loader):
    model = nn.Sequential(backbone_layer, nn.ReLU(), head_layer)
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

def train_oft(train_loader, val_loader, pretrained_backbone, lr=0.01, epochs=20):
    model = OFTBackboneModel(pretrained_backbone, num_classes=8, block_size=8)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    acc = evaluate_task(model.fc1, model.fc2, val_loader)
    return model.fc1, model.fc2, acc

def train_sa_oft(train_loader, val_loader, pretrained_backbone, lr=0.01, epochs=20, rho=0.1):
    model = OFTBackboneModel(pretrained_backbone, num_classes=8, block_size=8)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
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
    acc = evaluate_task(model.fc1, model.fc2, val_loader)
    return model.fc1, model.fc2, acc

def evaluate_merged_oft(oft1, oft2, fc2_1, fc2_2, pretrained_backbone, alpha, val_loader1, val_loader2):
    merged_oft = OFTLayer(pretrained_backbone, block_size=8)
    merged_oft.q_params.data.copy_(alpha * oft1.q_params.data + (1 - alpha) * oft2.q_params.data)
    acc1 = evaluate_task(merged_oft, fc2_1, val_loader1)
    acc2 = evaluate_task(merged_oft, fc2_2, val_loader2)
    return (acc1 + acc2) / 2

# --- 3. Running Over Multiple Seeds under 20% Label Noise ---
seeds = [42, 43, 44, 45, 46]
alphas = np.linspace(0.0, 1.0, 11)

curves_oft = []
curves_sa = []

peak_oft = []
peak_sa = []

print(f"\n=== Running Label Noise Experiments Over Seeds {seeds} ===")
for seed in seeds:
    print(f">>> Running Seed {seed} (with 20% Train Label Noise) <<<")
    # Generate downstream data
    X1, y1 = generate_multi_task_data(1200, task_id=1, seed=seed)
    X2, y2 = generate_multi_task_data(1200, task_id=2, seed=seed)
    
    # Inject 20% label noise to training sets only
    y1_train_noisy = add_label_noise(y1[:1000], noise_rate=0.20)
    y2_train_noisy = add_label_noise(y2[:1000], noise_rate=0.20)
    
    train_loader1 = DataLoader(TensorDataset(X1[:1000], y1_train_noisy), batch_size=32, shuffle=True)
    val_loader1 = DataLoader(TensorDataset(X1[1000:], y1[1000:]), batch_size=32, shuffle=False) # Clean validation

    train_loader2 = DataLoader(TensorDataset(X2[:1000], y2_train_noisy), batch_size=32, shuffle=True)
    val_loader2 = DataLoader(TensorDataset(X2[1000:], y2[1000:]), batch_size=32, shuffle=False) # Clean validation
    
    # Train OrthoMerge (Standard) and SA-Ortho
    fc1_oft1, fc2_oft1, _ = train_oft(train_loader1, val_loader1, pretrained_backbone_weight)
    fc1_oft2, fc2_oft2, _ = train_oft(train_loader2, val_loader2, pretrained_backbone_weight)
    
    fc1_sa1, fc2_sa1, _ = train_sa_oft(train_loader1, val_loader1, pretrained_backbone_weight, rho=0.1)
    fc1_sa2, fc2_sa2, _ = train_sa_oft(train_loader2, val_loader2, pretrained_backbone_weight, rho=0.1)
    
    # Evaluate merging curves
    seed_oft, seed_sa = [], []
    for alpha in alphas:
        seed_oft.append(evaluate_merged_oft(fc1_oft1, fc1_oft2, fc2_oft1, fc2_oft2, pretrained_backbone_weight, alpha, val_loader1, val_loader2))
        seed_sa.append(evaluate_merged_oft(fc1_sa1, fc1_sa2, fc2_sa1, fc2_sa2, pretrained_backbone_weight, alpha, val_loader1, val_loader2))
        
    curves_oft.append(seed_oft)
    curves_sa.append(seed_sa)
    
    peak_oft.append(max(seed_oft))
    peak_sa.append(max(seed_sa))

# Convert to arrays
curves_oft = np.array(curves_oft)
curves_sa = np.array(curves_sa)

mean_oft, std_oft = curves_oft.mean(axis=0), curves_oft.std(axis=0)
mean_sa, std_sa = curves_sa.mean(axis=0), curves_sa.std(axis=0)

# Print final peak performance stats
print("\n=== Robustness to Label Noise Merging Performance (Mean ± Std) ===")
print(f"OrthoMerge (Riemannian Standard): {np.mean(peak_oft)*100:.2f}% ± {np.std(peak_oft)*100:.2f}%")
print(f"SA-Ortho (Ours, Proposed):       {np.mean(peak_sa)*100:.2f}% ± {np.std(peak_sa)*100:.2f}%")

# Create and save plot
plt.figure(figsize=(8, 5))
plt.plot(alphas, mean_oft, 's-', label='OrthoMerge (Riemannian standard)', color='tab:blue')
plt.fill_between(alphas, mean_oft - std_oft, mean_oft + std_oft, alpha=0.15, color='tab:blue')

plt.plot(alphas, mean_sa, '^--', label='SA-Ortho (Proposed, ours)', color='tab:green', linewidth=2.5)
plt.fill_between(alphas, mean_sa - std_sa, mean_sa + std_sa, alpha=0.18, color='tab:green')

plt.xlabel('Merging Weight (Alpha for Task 1)')
plt.ylabel('Average Multi-task Accuracy')
plt.title('Robustness to 20% Training Label Noise in Model Merging (5 Seeds)')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig('results_label_noise.png', dpi=150)
print("\nLabel noise robustness plot saved to results_label_noise.png")
