import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Synthetic Dataset Generation ---
def generate_synthetic_data(num_samples=1200, input_dim=128, num_classes=8, task_id=1):
    X = torch.randn(num_samples, input_dim)
    torch.manual_seed(100 + task_id)
    W_task = torch.randn(input_dim, num_classes)
    logits = torch.matmul(X, W_task) + 0.05 * torch.randn(num_samples, num_classes)
    y = torch.argmax(logits, dim=1)
    return X, y

# Generate datasets: Task 0 (Pre-training), Task 1, and Task 2
X0, y0 = generate_synthetic_data(2000, 128, 8, task_id=0)
X1, y1 = generate_synthetic_data(1200, 128, 8, task_id=1)
X2, y2 = generate_synthetic_data(1200, 128, 8, task_id=2)

pre_loader = DataLoader(TensorDataset(X0, y0), batch_size=32, shuffle=True)

train_loader1 = DataLoader(TensorDataset(X1[:1000], y1[:1000]), batch_size=32, shuffle=True)
val_loader1 = DataLoader(TensorDataset(X1[1000:], y1[1000:]), batch_size=32, shuffle=False)

train_loader2 = DataLoader(TensorDataset(X2[:1000], y2[:1000]), batch_size=32, shuffle=True)
val_loader2 = DataLoader(TensorDataset(X2[1000:], y2[1000:]), batch_size=32, shuffle=False)

# --- 2. Base Model Definition ---
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

# Instantiate and pre-train the base model
base_model = BaseMLP(128, 64, 8)
print("--- Pre-training Base Model on Task 0 ---")
optimizer_pre = optim.Adam(base_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
for epoch in range(30):
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
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/30 - Pre-training Loss: {loss.item():.4f}, Accuracy: {correct/total:.4f}")

# --- 3. Orthogonal Fine-Tuning (OFT) Layer ---
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

# OFT MLP Model
class OFTMLP(nn.Module):
    def __init__(self, base_mlp, block_size=8):
        super().__init__()
        self.fc1 = OFTLayer(base_mlp.fc1.weight, block_size)
        self.relu = base_mlp.relu
        self.fc2 = OFTLayer(base_mlp.fc2.weight, block_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- 4. Evaluation Helper ---
def evaluate(model, val_loader):
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

# --- 5. Fine-Tuning ---

# 5a. Euclidean Fine-Tuning
def train_euclidean(train_loader, val_loader, base_model, lr=0.005, epochs=15):
    model = BaseMLP(128, 64, 8)
    model.load_state_dict(base_model.state_dict())
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
            
    acc = evaluate(model, val_loader)
    return model, acc

print("\n--- Training Euclidean Fine-Tuning ---")
euclid_model1, acc1_e = train_euclidean(train_loader1, val_loader1, base_model)
print(f"Task 1 Euclidean Acc: {acc1_e:.4f}")
euclid_model2, acc2_e = train_euclidean(train_loader2, val_loader2, base_model)
print(f"Task 2 Euclidean Acc: {acc2_e:.4f}")


# 5b. Orthogonal Fine-Tuning (OFT)
def train_oft(train_loader, val_loader, base_model, lr=0.01, epochs=15):
    model = OFTMLP(base_model, block_size=8)
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
            
    acc = evaluate(model, val_loader)
    return model, acc

print("\n--- Training Orthogonal Fine-Tuning (OFT) ---")
oft_model1, acc1_oft = train_oft(train_loader1, val_loader1, base_model)
print(f"Task 1 OFT Acc: {acc1_oft:.4f}")
oft_model2, acc2_oft = train_oft(train_loader2, val_loader2, base_model)
print(f"Task 2 OFT Acc: {acc2_oft:.4f}")


# 5c. Sharpness-Aware Orthogonal (SA-Ortho) Fine-Tuning (Our proposed method)
def train_sa_oft(train_loader, val_loader, base_model, lr=0.01, epochs=15, rho=0.1):
    model = OFTMLP(base_model, block_size=8)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            # Step 1: standard forward/backward to get gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Save parameters and gradients
            orig_params = {}
            grads = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    orig_params[name] = p.data.clone()
                    grads[name] = p.grad.clone()
            
            # Compute gradient norm
            grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads.values()) + 1e-12)
            
            # Step 2: Perturb parameters in the direction of the gradient
            for name, p in model.named_parameters():
                if name in grads:
                    eps = rho * grads[name] / grad_norm
                    p.data.add_(eps)
                    
            # Step 3: Compute gradients at perturbed point
            optimizer.zero_grad()
            outputs_perturbed = model(inputs)
            loss_perturbed = criterion(outputs_perturbed, targets)
            loss_perturbed.backward()
            
            # Step 4: Restore original parameters and step with the perturbed gradients
            for name, p in model.named_parameters():
                if name in orig_params:
                    p.data.copy_(orig_params[name])
                    
            optimizer.step()
            
    acc = evaluate(model, val_loader)
    return model, acc

print("\n--- Training Sharpness-Aware Orthogonal Fine-Tuning (SA-Ortho) ---")
sa_model1, acc1_sa = train_sa_oft(train_loader1, val_loader1, base_model, rho=0.1)
print(f"Task 1 SA-Ortho Acc: {acc1_sa:.4f}")
sa_model2, acc2_sa = train_sa_oft(train_loader2, val_loader2, base_model, rho=0.1)
print(f"Task 2 SA-Ortho Acc: {acc2_sa:.4f}")


# --- 6. Model Merging and Evaluation ---

def evaluate_merged_euclid(model1, model2, base_model, alpha, val_loader1, val_loader2):
    merged = BaseMLP(128, 64, 8)
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    base_state = base_model.state_dict()
    merged_state = {}
    
    for k in base_state.keys():
        task_vec1 = state1[k] - base_state[k]
        task_vec2 = state2[k] - base_state[k]
        merged_state[k] = base_state[k] + alpha * task_vec1 + (1 - alpha) * task_vec2
        
    merged.load_state_dict(merged_state)
    return evaluate(merged, val_loader1), evaluate(merged, val_loader2)

def evaluate_merged_oft(model1, model2, base_model, alpha, val_loader1, val_loader2):
    merged = OFTMLP(base_model, block_size=8)
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    merged_state = {}
    
    for k in state1.keys():
        if "q_params" in k:
            merged_state[k] = alpha * state1[k] + (1 - alpha) * state2[k]
        else:
            merged_state[k] = state1[k]
            
    merged.load_state_dict(merged_state)
    return evaluate(merged, val_loader1), evaluate(merged, val_loader2)

# Grid search across alpha values
alphas = np.linspace(0.0, 1.0, 11)

results_e = []
results_oft = []
results_sa = []

for alpha in alphas:
    # Task Arithmetic
    acc1, acc2 = evaluate_merged_euclid(euclid_model1, euclid_model2, base_model, alpha, val_loader1, val_loader2)
    results_e.append((acc1, acc2, (acc1 + acc2) / 2))
    
    # OrthoMerge
    acc1_o, acc2_o = evaluate_merged_oft(oft_model1, oft_model2, base_model, alpha, val_loader1, val_loader2)
    results_oft.append((acc1_o, acc2_o, (acc1_o + acc2_o) / 2))
    
    # SA-Ortho
    acc1_sa, acc2_sa = evaluate_merged_oft(sa_model1, sa_model2, base_model, alpha, val_loader1, val_loader2)
    results_sa.append((acc1_sa, acc2_sa, (acc1_sa + acc2_sa) / 2))

avg_e = [r[2] for r in results_e]
avg_oft = [r[2] for r in results_oft]
avg_sa = [r[2] for r in results_sa]

print("\n--- Model Merging Results (Average Multi-task Accuracy) ---")
print(f"Alpha:       " + "  ".join([f"{a:.1f}" for a in alphas]))
print(f"Arithmetic:  " + "  ".join([f"{a:.4f}" for a in avg_e]))
print(f"OrthoMerge:  " + "  ".join([f"{a:.4f}" for a in avg_oft]))
print(f"SA-Ortho:    " + "  ".join([f"{a:.4f}" for a in avg_sa]))

# Find peak average accuracies
print(f"\nPeak Average Accuracy:")
print(f"Task Arithmetic: {max(avg_e):.4f} at alpha={alphas[np.argmax(avg_e)]:.1f}")
print(f"OrthoMerge:      {max(avg_oft):.4f} at alpha={alphas[np.argmax(avg_oft)]:.1f}")
print(f"SA-Ortho (Ours): {max(avg_sa):.4f} at alpha={alphas[np.argmax(avg_sa)]:.1f}")

# --- 7. Save Results and Create Plot ---
plt.figure(figsize=(8, 5))
plt.plot(alphas, avg_e, 'o--', label='Task Arithmetic (Euclidean)', color='tab:red')
plt.plot(alphas, avg_oft, 's-', label='OrthoMerge (Riemannian)', color='tab:blue')
plt.plot(alphas, avg_sa, '^--', label='SA-Ortho (Proposed, Flat Manifold)', color='tab:green', linewidth=2.5)
plt.xlabel('Merging Weight (Alpha for Task 1)')
plt.ylabel('Average Multi-task Accuracy')
plt.title('Model Merging Performance: Euclidean vs. Riemannian Manifolds')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig('results_refined.png', dpi=150)
print("\nRefined plot saved successfully to results_refined.png")
