import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

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
            
            if len(grads) > 0:
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
            else:
                optimizer.step()
    return model.fc1, model.fc2

def evaluate_merged_oft(oft1, oft2, fc2_1, fc2_2, pretrained_backbone, alpha, val_loader1, val_loader2):
    merged_oft = OFTLayer(pretrained_backbone, block_size=8)
    merged_oft.q_params.data.copy_(alpha * oft1.q_params.data + (1 - alpha) * oft2.q_params.data)
    acc1 = evaluate_task(merged_oft, fc2_1, val_loader1)
    acc2 = evaluate_task(merged_oft, fc2_2, val_loader2)
    return acc1, acc2

# --- 3. Sweep Protocol ---
rhos = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
seeds = [42, 43, 44, 45, 46]
alphas = np.linspace(0.0, 1.0, 11)

results_by_rho = {rho: [] for rho in rhos} # Stores curves for each seed: shape (num_rhos, num_seeds, num_alphas)

print("Starting Perturbation Radius (rho) Sweep across 5 seeds...")
for rho in rhos:
    print(f"\n--- Perturbation Radius (rho): {rho} ---")
    rho_curves = []
    for s in seeds:
        X1, y1 = generate_multi_task_data(1200, task_id=1, seed=s)
        train_loader1 = DataLoader(TensorDataset(X1[:1000], y1[:1000]), batch_size=32, shuffle=True)
        val_loader1 = DataLoader(TensorDataset(X1[1000:], y1[1000:]), batch_size=32, shuffle=False)
        
        X2, y2 = generate_multi_task_data(1200, task_id=2, seed=s)
        train_loader2 = DataLoader(TensorDataset(X2[:1000], y2[:1000]), batch_size=32, shuffle=True)
        val_loader2 = DataLoader(TensorDataset(X2[1000:], y2[1000:]), batch_size=32, shuffle=False)
        
        fc1_1, fc2_1 = train_sa_oft(train_loader1, val_loader1, pretrained_backbone_weight, rho=rho)
        fc1_2, fc2_2 = train_sa_oft(train_loader2, val_loader2, pretrained_backbone_weight, rho=rho)
        
        curve = []
        for alpha in alphas:
            acc1, acc2 = evaluate_merged_oft(fc1_1, fc1_2, fc2_1, fc2_2, pretrained_backbone_weight, alpha, val_loader1, val_loader2)
            curve.append((acc1 + acc2) / 2)
        rho_curves.append(curve)
        print(f"Seed {s} | Max Avg Acc: {max(curve)*100:.2f}%")
    results_by_rho[rho] = np.array(rho_curves)

# --- 4. Plot Results with Confidence/Std Shading ---
plt.figure(figsize=(9, 5.5))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
for idx, rho in enumerate(rhos):
    mean_curve = results_by_rho[rho].mean(axis=0)
    std_curve = results_by_rho[rho].std(axis=0)
    plt.plot(alphas, mean_curve, label=f"rho = {rho}", color=colors[idx])
    plt.fill_between(alphas, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15, color=colors[idx])

plt.xlabel("Merging Weight (Alpha for Task 1)")
plt.ylabel("Average Multi-task Accuracy")
plt.title("Impact of Sharpness Perturbation Magnitude (rho) on Merging (5 Seeds)")
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig("results_sweep_rho.png", dpi=150)
print("\nMulti-seed sweep results plot saved to results_sweep_rho.png")

# --- 5. Summarize Peak Performance Stats ---
print("\n=== SUMMARY OF RESULTS (Peak Merging Performance over 5 seeds) ===")
print("| Perturbation Radius (rho) | Peak Avg Acc (Mean \u00b1 Std) |")
print("|---|---|")
for rho in rhos:
    peaks = results_by_rho[rho].max(axis=1) # Peak accuracy along alpha per seed
    mean_peak = np.mean(peaks)
    std_peak = np.std(peaks)
    print(f"| {rho} | {mean_peak*100:.2f}% \u00b1 {std_peak*100:.2f}% |")
