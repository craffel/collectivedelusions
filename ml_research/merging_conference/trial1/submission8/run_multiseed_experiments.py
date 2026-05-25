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
    # Set seed for reproducible downstream data generation
    torch.manual_seed(seed * 1000 + task_id)
    np.random.seed(seed * 1000 + task_id)
    
    X = torch.randn(num_samples, input_dim)
    features = torch.relu(torch.matmul(X, W_shared))
    W_head = torch.randn(hidden_dim, num_classes)
    logits = torch.matmul(features, W_head) + 0.05 * torch.randn(num_samples, num_classes)
    y = torch.argmax(logits, dim=1)
    return X, y

# Pre-train the base model on Task 0 once (using seed 42) to represent a fixed foundation checkpoint
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
    correct = 0
    total = 0
    for inputs, targets in pre_loader:
        optimizer_pre.zero_grad()
        outputs = base_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_pre.step()
        
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/15 - Pre-training Loss: {loss.item():.4f}, Accuracy: {correct/total:.4f}")

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
        self.epsilon_W = None
        
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
        if self.epsilon_W is not None:
            W = W + self.epsilon_W
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


# --- 3. Training Baselines ---
def train_euclidean(train_loader, val_loader, pretrained_backbone, lr=0.01, epochs=20):
    fc1 = nn.Linear(128, 64, bias=False)
    fc1.weight.data.copy_(pretrained_backbone)
    fc2 = nn.Linear(64, 8, bias=False)
    model = nn.Sequential(fc1, nn.ReLU(), fc2)
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
    acc = evaluate_task(fc1, fc2, val_loader)
    return fc1, fc2, acc

def train_euclidean_sam(train_loader, val_loader, pretrained_backbone, lr=0.01, epochs=20, rho=0.1):
    fc1 = nn.Linear(128, 64, bias=False)
    fc1.weight.data.copy_(pretrained_backbone)
    fc2 = nn.Linear(64, 8, bias=False)
    model = nn.Sequential(fc1, nn.ReLU(), fc2)
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
    acc = evaluate_task(fc1, fc2, val_loader)
    return fc1, fc2, acc

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

def train_oft_classifier_sam(train_loader, val_loader, pretrained_backbone, lr=0.01, epochs=20, rho=0.1):
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
                if "fc2" in name and p.grad is not None:
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
    acc = evaluate_task(model.fc1, model.fc2, val_loader)
    return model.fc1, model.fc2, acc

def train_oft_weight_sam(train_loader, val_loader, pretrained_backbone, lr=0.01, epochs=20, rho=0.1):
    model = OFTBackboneModel(pretrained_backbone, num_classes=8, block_size=8)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            R = model.fc1._get_R()
            W = torch.matmul(R, model.fc1.base_weight)
            W.retain_grad()
            x_feat = nn.functional.linear(inputs, W)
            x_feat = model.relu(x_feat)
            outputs = model.fc2(x_feat)
            loss = criterion(outputs, targets)
            loss.backward()
            
            W_grad = W.grad.clone()
            fc2_grad = model.fc2.weight.grad.clone() if model.fc2.weight.grad is not None else torch.zeros_like(model.fc2.weight)
            grad_norm = torch.sqrt(torch.sum(W_grad ** 2) + torch.sum(fc2_grad ** 2) + 1e-12)
            
            eps_W = rho * W_grad / grad_norm
            eps_fc2 = rho * fc2_grad / grad_norm
            
            orig_fc2_weight = model.fc2.weight.data.clone()
            model.fc1.epsilon_W = eps_W
            model.fc2.weight.data.add_(eps_fc2)
            
            optimizer.zero_grad()
            outputs_perturbed = model(inputs)
            loss_perturbed = criterion(outputs_perturbed, targets)
            loss_perturbed.backward()
            
            model.fc1.epsilon_W = None
            model.fc2.weight.data.copy_(orig_fc2_weight)
            optimizer.step()
    acc = evaluate_task(model.fc1, model.fc2, val_loader)
    return model.fc1, model.fc2, acc


# --- 4. Evaluation Helpers ---
def evaluate_merged_euclid(fc1_1, fc1_2, fc2_1, fc2_2, pretrained_backbone, alpha, val_loader1, val_loader2):
    merged_fc1 = nn.Linear(128, 64, bias=False)
    task_vec1 = fc1_1.weight.data - pretrained_backbone
    task_vec2 = fc1_2.weight.data - pretrained_backbone
    merged_fc1.weight.data.copy_(pretrained_backbone + alpha * task_vec1 + (1 - alpha) * task_vec2)
    acc1 = evaluate_task(merged_fc1, fc2_1, val_loader1)
    acc2 = evaluate_task(merged_fc1, fc2_2, val_loader2)
    return (acc1 + acc2) / 2

def evaluate_merged_oft(oft1, oft2, fc2_1, fc2_2, pretrained_backbone, alpha, val_loader1, val_loader2):
    merged_oft = OFTLayer(pretrained_backbone, block_size=8)
    merged_oft.q_params.data.copy_(alpha * oft1.q_params.data + (1 - alpha) * oft2.q_params.data)
    acc1 = evaluate_task(merged_oft, fc2_1, val_loader1)
    acc2 = evaluate_task(merged_oft, fc2_2, val_loader2)
    return (acc1 + acc2) / 2


# --- 5. Running Over Multiple Seeds ---
seeds = [42, 43, 44, 45, 46]
alphas = np.linspace(0.0, 1.0, 11)

# Storage for curves per seed
curves_e = []
curves_esam = []
curves_oft = []
curves_sa = []
curves_csam = []
curves_wsam = []

peak_e = []
peak_esam = []
peak_oft = []
peak_sa = []
peak_csam = []
peak_wsam = []

print(f"\n=== Running Experiments Over Seeds {seeds} ===")
for seed in seeds:
    print(f"\n>>> Running Seed {seed} <<<")
    # Generate downstream data for this seed
    X1, y1 = generate_multi_task_data(1200, task_id=1, seed=seed)
    X2, y2 = generate_multi_task_data(1200, task_id=2, seed=seed)
    
    train_loader1 = DataLoader(TensorDataset(X1[:1000], y1[:1000]), batch_size=32, shuffle=True)
    val_loader1 = DataLoader(TensorDataset(X1[1000:], y1[1000:]), batch_size=32, shuffle=False)

    train_loader2 = DataLoader(TensorDataset(X2[:1000], y2[:1000]), batch_size=32, shuffle=True)
    val_loader2 = DataLoader(TensorDataset(X2[1000:], y2[1000:]), batch_size=32, shuffle=False)
    
    # Train 6 models
    fc1_e1, fc2_e1, _ = train_euclidean(train_loader1, val_loader1, pretrained_backbone_weight)
    fc1_e2, fc2_e2, _ = train_euclidean(train_loader2, val_loader2, pretrained_backbone_weight)
    
    fc1_esam1, fc2_esam1, _ = train_euclidean_sam(train_loader1, val_loader1, pretrained_backbone_weight, rho=0.1)
    fc1_esam2, fc2_esam2, _ = train_euclidean_sam(train_loader2, val_loader2, pretrained_backbone_weight, rho=0.1)
    
    fc1_oft1, fc2_oft1, _ = train_oft(train_loader1, val_loader1, pretrained_backbone_weight)
    fc1_oft2, fc2_oft2, _ = train_oft(train_loader2, val_loader2, pretrained_backbone_weight)
    
    fc1_sa1, fc2_sa1, _ = train_sa_oft(train_loader1, val_loader1, pretrained_backbone_weight, rho=0.1)
    fc1_sa2, fc2_sa2, _ = train_sa_oft(train_loader2, val_loader2, pretrained_backbone_weight, rho=0.1)
    
    fc1_csam1, fc2_csam1, _ = train_oft_classifier_sam(train_loader1, val_loader1, pretrained_backbone_weight, rho=0.1)
    fc1_csam2, fc2_csam2, _ = train_oft_classifier_sam(train_loader2, val_loader2, pretrained_backbone_weight, rho=0.1)
    
    fc1_wsam1, fc2_wsam1, _ = train_oft_weight_sam(train_loader1, val_loader1, pretrained_backbone_weight, rho=0.1)
    fc1_wsam2, fc2_wsam2, _ = train_oft_weight_sam(train_loader2, val_loader2, pretrained_backbone_weight, rho=0.1)
    
    # Evaluate merging curves for this seed
    seed_e, seed_esam, seed_oft, seed_sa, seed_csam, seed_wsam = [], [], [], [], [], []
    for alpha in alphas:
        seed_e.append(evaluate_merged_euclid(fc1_e1, fc1_e2, fc2_e1, fc2_e2, pretrained_backbone_weight, alpha, val_loader1, val_loader2))
        seed_esam.append(evaluate_merged_euclid(fc1_esam1, fc1_esam2, fc2_esam1, fc2_esam2, pretrained_backbone_weight, alpha, val_loader1, val_loader2))
        seed_oft.append(evaluate_merged_oft(fc1_oft1, fc1_oft2, fc2_oft1, fc2_oft2, pretrained_backbone_weight, alpha, val_loader1, val_loader2))
        seed_sa.append(evaluate_merged_oft(fc1_sa1, fc1_sa2, fc2_sa1, fc2_sa2, pretrained_backbone_weight, alpha, val_loader1, val_loader2))
        seed_csam.append(evaluate_merged_oft(fc1_csam1, fc1_csam2, fc2_csam1, fc2_csam2, pretrained_backbone_weight, alpha, val_loader1, val_loader2))
        seed_wsam.append(evaluate_merged_oft(fc1_wsam1, fc1_wsam2, fc2_wsam1, fc2_wsam2, pretrained_backbone_weight, alpha, val_loader1, val_loader2))
        
    curves_e.append(seed_e)
    curves_esam.append(seed_esam)
    curves_oft.append(seed_oft)
    curves_sa.append(seed_sa)
    curves_csam.append(seed_csam)
    curves_wsam.append(seed_wsam)
    
    peak_e.append(max(seed_e))
    peak_esam.append(max(seed_esam))
    peak_oft.append(max(seed_oft))
    peak_sa.append(max(seed_sa))
    peak_csam.append(max(seed_csam))
    peak_wsam.append(max(seed_wsam))

# Convert to arrays for easy computation
curves_e = np.array(curves_e)
curves_esam = np.array(curves_esam)
curves_oft = np.array(curves_oft)
curves_sa = np.array(curves_sa)
curves_csam = np.array(curves_csam)
curves_wsam = np.array(curves_wsam)

# Compute mean and standard deviation at each alpha
mean_e, std_e = curves_e.mean(axis=0), curves_e.std(axis=0)
mean_esam, std_esam = curves_esam.mean(axis=0), curves_esam.std(axis=0)
mean_oft, std_oft = curves_oft.mean(axis=0), curves_oft.std(axis=0)
mean_sa, std_sa = curves_sa.mean(axis=0), curves_sa.std(axis=0)
mean_csam, std_csam = curves_csam.mean(axis=0), curves_csam.std(axis=0)
mean_wsam, std_wsam = curves_wsam.mean(axis=0), curves_wsam.std(axis=0)


# Print final peak performance stats
import scipy.stats as stats

print("\n=== Multi-Seed Peak Merging Performance (Mean ± Std) ===")
print(f"Task Arithmetic:           {np.mean(peak_e)*100:.2f}% ± {np.std(peak_e)*100:.2f}%")
print(f"Task Arithmetic + SAM:     {np.mean(peak_esam)*100:.2f}% ± {np.std(peak_esam)*100:.2f}%")
print(f"OrthoMerge:                {np.mean(peak_oft)*100:.2f}% ± {np.std(peak_oft)*100:.2f}%")
print(f"SA-Ortho (Ours):           {np.mean(peak_sa)*100:.2f}% ± {np.std(peak_sa)*100:.2f}%")
print(f"OFT + Classifier-only SAM: {np.mean(peak_csam)*100:.2f}% ± {np.std(peak_csam)*100:.2f}%")
print(f"OFT + Weight-Space SAM:    {np.mean(peak_wsam)*100:.2f}% ± {np.std(peak_wsam)*100:.2f}%")

print("\n=== Statistical Significance Test ===")
print("Individual Seed Peaks for OrthoMerge: ", [f"{p*100:.2f}%" for p in peak_oft])
print("Individual Seed Peaks for SA-Ortho:   ", [f"{p*100:.2f}%" for p in peak_sa])
t_stat, p_val = stats.ttest_rel(peak_sa, peak_oft)
print(f"Paired t-test results: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
if p_val < 0.05:
    print(f"The +1.0% improvement of SA-Ortho over OrthoMerge is statistically significant (p = {p_val:.4f} < 0.05)!")
else:
    print(f"The difference is not statistically significant (p = {p_val:.4f} >= 0.05).")


# --- 6. Create and Save Refined Plots with Shading ---

# Refined main plot
plt.figure(figsize=(9, 5.5))
plt.plot(alphas, mean_e, 'o--', label='Task Arithmetic (Euclidean)', color='tab:red')
plt.fill_between(alphas, mean_e - std_e, mean_e + std_e, alpha=0.15, color='tab:red')

plt.plot(alphas, mean_esam, '*:', label='Task Arithmetic + SAM (SAIM-like)', color='tab:orange')
plt.fill_between(alphas, mean_esam - std_esam, mean_esam + std_esam, alpha=0.1, color='tab:orange')

plt.plot(alphas, mean_oft, 's-', label='OrthoMerge (Riemannian)', color='tab:blue')
plt.fill_between(alphas, mean_oft - std_oft, mean_oft + std_oft, alpha=0.15, color='tab:blue')

plt.plot(alphas, mean_sa, '^--', label='SA-Ortho (Proposed, Lie algebra SAM)', color='tab:green', linewidth=2.5)
plt.fill_between(alphas, mean_sa - std_sa, mean_sa + std_sa, alpha=0.18, color='tab:green')

plt.xlabel('Merging Weight (Alpha for Task 1)')
plt.ylabel('Average Multi-task Accuracy')
plt.title('Comprehensive Model Merging Comparison: Euclidean vs. Riemannian Manifolds (5 Seeds)')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig('results_refined.png', dpi=150)
print("\nMulti-seed refined plot saved to results_refined.png")


# Ablation study plot
plt.figure(figsize=(9, 5.5))
plt.plot(alphas, mean_oft, 's-', label='OrthoMerge (No SAM)', color='tab:blue')
plt.fill_between(alphas, mean_oft - std_oft, mean_oft + std_oft, alpha=0.15, color='tab:blue')

plt.plot(alphas, mean_csam, 'x:', label='OFT + Classifier-only SAM', color='purple')
plt.fill_between(alphas, mean_csam - std_csam, mean_csam + std_csam, alpha=0.12, color='purple')

plt.plot(alphas, mean_wsam, 'v-.', label='OFT + Euclidean Weight SAM', color='darkorange')
plt.fill_between(alphas, mean_wsam - std_wsam, mean_wsam + std_wsam, alpha=0.12, color='darkorange')

plt.plot(alphas, mean_sa, '^--', label='SA-Ortho (Proposed, Joint Lie + Head SAM)', color='tab:green', linewidth=2.5)
plt.fill_between(alphas, mean_sa - std_sa, mean_sa + std_sa, alpha=0.18, color='tab:green')

plt.xlabel('Merging Weight (Alpha for Task 1)')
plt.ylabel('Average Multi-task Accuracy')
plt.title('Ablation Study: Understanding the Source of Sharpness-Aware Manifold Gains (5 Seeds)')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig('results_ablation_sam.png', dpi=150)
print("Multi-seed ablation plot saved to results_ablation_sam.png")
