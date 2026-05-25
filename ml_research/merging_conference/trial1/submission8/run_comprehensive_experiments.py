import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import copy

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Synthetic Dataset Generation ---
input_dim = 128
hidden_dim = 64
num_classes = 8

# Generate a shared underlying feature structure (representation space)
torch.manual_seed(10)
W_shared = torch.randn(input_dim, hidden_dim)

def generate_multi_task_data(num_samples=1200, task_id=1):
    X = torch.randn(num_samples, input_dim)
    # The actual features mapped to representations
    features = torch.relu(torch.matmul(X, W_shared))
    # Task-specific label projection
    torch.manual_seed(100 + task_id)
    W_head = torch.randn(hidden_dim, num_classes)
    logits = torch.matmul(features, W_head) + 0.05 * torch.randn(num_samples, num_classes)
    y = torch.argmax(logits, dim=1)
    return X, y

# Generate datasets: Task 0 (Pre-training), Task 1, and Task 2
X0, y0 = generate_multi_task_data(2000, task_id=0)
X1, y1 = generate_multi_task_data(1200, task_id=1)
X2, y2 = generate_multi_task_data(1200, task_id=2)

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

# Pre-train the base model on Task 0 to learn the shared backbone representation
base_model = BaseMLP(128, 64, 8)
print("--- Pre-training Base Model on Task 0 ---")
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

# Save pre-trained backbone weight
pretrained_backbone_weight = base_model.fc1.weight.data.clone()

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
        self.epsilon_W = None  # Used for Euclidean Weight-Space SAM
        
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

# Model with OFT Backbone and Task-Specific Classification Head
class OFTBackboneModel(nn.Module):
    def __init__(self, pretrained_backbone, num_classes=8, block_size=8):
        super().__init__()
        self.fc1 = OFTLayer(pretrained_backbone, block_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes, bias=False)  # Task-specific classification head
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Helper to evaluate
def evaluate_task(backbone_layer, head_layer, val_loader):
    model = nn.Sequential(
        backbone_layer,
        nn.ReLU(),
        head_layer
    )
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

# --- 4. Fine-Tuning Baselines ---

# 4a. Euclidean Fine-Tuning
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

# 4b. Euclidean Fine-Tuning with SAM (SAIM-like)
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
            # First pass
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
            
            # Apply perturbation
            for name, p in model.named_parameters():
                if name in grads:
                    eps = rho * grads[name] / grad_norm
                    p.data.add_(eps)
                    
            # Second pass
            optimizer.zero_grad()
            outputs_perturbed = model(inputs)
            loss_perturbed = criterion(outputs_perturbed, targets)
            loss_perturbed.backward()
            
            # Restore original parameters and step
            for name, p in model.named_parameters():
                if name in orig_params:
                    p.data.copy_(orig_params[name])
                    
            optimizer.step()
            
    acc = evaluate_task(fc1, fc2, val_loader)
    return fc1, fc2, acc

# 4c. Orthogonal Fine-Tuning (OFT) (OrthoMerge Baseline)
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

# 4d. Sharpness-Aware Orthogonal Fine-Tuning (SA-Ortho) (Proposed Method)
def train_sa_oft(train_loader, val_loader, pretrained_backbone, lr=0.01, epochs=20, rho=0.1):
    model = OFTBackboneModel(pretrained_backbone, num_classes=8, block_size=8)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            # First pass: standard gradient computation
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
            
            # Apply perturbation (Lie algebra SAM)
            for name, p in model.named_parameters():
                if name in grads:
                    eps = rho * grads[name] / grad_norm
                    p.data.add_(eps)
                    
            # Second pass: compute gradients at perturbed point
            optimizer.zero_grad()
            outputs_perturbed = model(inputs)
            loss_perturbed = criterion(outputs_perturbed, targets)
            loss_perturbed.backward()
            
            # Restore original parameters and step with perturbed gradients
            for name, p in model.named_parameters():
                if name in orig_params:
                    p.data.copy_(orig_params[name])
                    
            optimizer.step()
            
    acc = evaluate_task(model.fc1, model.fc2, val_loader)
    return model.fc1, model.fc2, acc

# 4e. OFT with Classifier-only SAM (Ablation)
def train_oft_classifier_sam(train_loader, val_loader, pretrained_backbone, lr=0.01, epochs=20, rho=0.1):
    model = OFTBackboneModel(pretrained_backbone, num_classes=8, block_size=8)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            # First pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Save classifier head parameters and gradients
            orig_params = {}
            grads = {}
            for name, p in model.named_parameters():
                if "fc2" in name and p.grad is not None:
                    orig_params[name] = p.data.clone()
                    grads[name] = p.grad.clone()
            
            if len(grads) > 0:
                grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads.values()) + 1e-12)
                # Apply perturbation to classifier head ONLY
                for name, p in model.named_parameters():
                    if name in grads:
                        eps = rho * grads[name] / grad_norm
                        p.data.add_(eps)
                        
            # Second pass
            optimizer.zero_grad()
            outputs_perturbed = model(inputs)
            loss_perturbed = criterion(outputs_perturbed, targets)
            loss_perturbed.backward()
            
            # Restore head parameters and step
            for name, p in model.named_parameters():
                if name in orig_params:
                    p.data.copy_(orig_params[name])
                    
            optimizer.step()
            
    acc = evaluate_task(model.fc1, model.fc2, val_loader)
    return model.fc1, model.fc2, acc

# 4f. OFT with Euclidean Weight-Space SAM (Ablation/Alternative)
def train_oft_weight_sam(train_loader, val_loader, pretrained_backbone, lr=0.01, epochs=20, rho=0.1):
    model = OFTBackboneModel(pretrained_backbone, num_classes=8, block_size=8)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            # First pass: forward and retain gradients on intermediate weight
            optimizer.zero_grad()
            
            # We manually trace the forward pass to obtain intermediate W = R * W_0
            R = model.fc1._get_R()
            W = torch.matmul(R, model.fc1.base_weight)
            W.retain_grad()
            
            # Forward through linear layer using intermediate W
            x_feat = nn.functional.linear(inputs, W)
            x_feat = model.relu(x_feat)
            outputs = model.fc2(x_feat)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Now we have grad on W and on fc2.weight
            W_grad = W.grad.clone()
            fc2_grad = model.fc2.weight.grad.clone() if model.fc2.weight.grad is not None else torch.zeros_like(model.fc2.weight)
            
            grad_norm = torch.sqrt(torch.sum(W_grad ** 2) + torch.sum(fc2_grad ** 2) + 1e-12)
            
            # Compute Euclidean perturbations
            eps_W = rho * W_grad / grad_norm
            eps_fc2 = rho * fc2_grad / grad_norm
            
            # Save original head and set weight perturbation
            orig_fc2_weight = model.fc2.weight.data.clone()
            model.fc1.epsilon_W = eps_W
            model.fc2.weight.data.add_(eps_fc2)
            
            # Second pass with perturbed weights
            optimizer.zero_grad()
            outputs_perturbed = model(inputs)
            loss_perturbed = criterion(outputs_perturbed, targets)
            loss_perturbed.backward()
            
            # Restore original states
            model.fc1.epsilon_W = None
            model.fc2.weight.data.copy_(orig_fc2_weight)
            
            # Optimizer step using perturbed gradients
            optimizer.step()
            
    acc = evaluate_task(model.fc1, model.fc2, val_loader)
    return model.fc1, model.fc2, acc


# --- 5. Run All Training ---

print("\n=== Training All Methods ===")
# 5a. Euclidean Task Arithmetic
print("\n[1/6] Training Task Arithmetic (Euclidean)...")
fc1_e1, fc2_e1, acc1_e = train_euclidean(train_loader1, val_loader1, pretrained_backbone_weight)
fc1_e2, fc2_e2, acc2_e = train_euclidean(train_loader2, val_loader2, pretrained_backbone_weight)
print(f"Task 1 Acc: {acc1_e:.4f} | Task 2 Acc: {acc2_e:.4f}")

# 5b. Euclidean SAM (SAIM-like)
print("\n[2/6] Training Task Arithmetic + SAM (Euclidean)...")
fc1_esam1, fc2_esam1, acc1_esam = train_euclidean_sam(train_loader1, val_loader1, pretrained_backbone_weight, rho=0.1)
fc1_esam2, fc2_esam2, acc2_esam = train_euclidean_sam(train_loader2, val_loader2, pretrained_backbone_weight, rho=0.1)
print(f"Task 1 Acc: {acc1_esam:.4f} | Task 2 Acc: {acc2_esam:.4f}")

# 5c. Standard OrthoMerge
print("\n[3/6] Training OrthoMerge (Riemannian standard)...")
fc1_oft1, fc2_oft1, acc1_oft = train_oft(train_loader1, val_loader1, pretrained_backbone_weight)
fc1_oft2, fc2_oft2, acc2_oft = train_oft(train_loader2, val_loader2, pretrained_backbone_weight)
print(f"Task 1 Acc: {acc1_oft:.4f} | Task 2 Acc: {acc2_oft:.4f}")

# 5d. Proposed SA-Ortho (Lie algebra SAM)
print("\n[4/6] Training SA-Ortho (Proposed)...")
fc1_sa1, fc2_sa1, acc1_sa = train_sa_oft(train_loader1, val_loader1, pretrained_backbone_weight, rho=0.1)
fc1_sa2, fc2_sa2, acc2_sa = train_sa_oft(train_loader2, val_loader2, pretrained_backbone_weight, rho=0.1)
print(f"Task 1 Acc: {acc1_sa:.4f} | Task 2 Acc: {acc2_sa:.4f}")

# 5e. OFT with Classifier-only SAM
print("\n[5/6] Training OFT + Classifier-only SAM...")
fc1_csam1, fc2_csam1, acc1_csam = train_oft_classifier_sam(train_loader1, val_loader1, pretrained_backbone_weight, rho=0.1)
fc1_csam2, fc2_csam2, acc2_csam = train_oft_classifier_sam(train_loader2, val_loader2, pretrained_backbone_weight, rho=0.1)
print(f"Task 1 Acc: {acc1_csam:.4f} | Task 2 Acc: {acc2_csam:.4f}")

# 5f. OFT with Euclidean Weight-Space SAM
print("\n[6/6] Training OFT + Euclidean Weight-Space SAM...")
fc1_wsam1, fc2_wsam1, acc1_wsam = train_oft_weight_sam(train_loader1, val_loader1, pretrained_backbone_weight, rho=0.1)
fc1_wsam2, fc2_wsam2, acc2_wsam = train_oft_weight_sam(train_loader2, val_loader2, pretrained_backbone_weight, rho=0.1)
print(f"Task 1 Acc: {acc1_wsam:.4f} | Task 2 Acc: {acc2_wsam:.4f}")


# --- 6. Model Merging and Evaluation ---

def evaluate_merged_euclid(fc1_1, fc1_2, fc2_1, fc2_2, pretrained_backbone, alpha, val_loader1, val_loader2):
    merged_fc1 = nn.Linear(128, 64, bias=False)
    task_vec1 = fc1_1.weight.data - pretrained_backbone
    task_vec2 = fc1_2.weight.data - pretrained_backbone
    merged_fc1.weight.data.copy_(pretrained_backbone + alpha * task_vec1 + (1 - alpha) * task_vec2)
    acc1 = evaluate_task(merged_fc1, fc2_1, val_loader1)
    acc2 = evaluate_task(merged_fc1, fc2_2, val_loader2)
    return acc1, acc2

def evaluate_merged_oft(oft1, oft2, fc2_1, fc2_2, pretrained_backbone, alpha, val_loader1, val_loader2):
    merged_oft = OFTLayer(pretrained_backbone, block_size=8)
    merged_oft.q_params.data.copy_(alpha * oft1.q_params.data + (1 - alpha) * oft2.q_params.data)
    acc1 = evaluate_task(merged_oft, fc2_1, val_loader1)
    acc2 = evaluate_task(merged_oft, fc2_2, val_loader2)
    return acc1, acc2

alphas = np.linspace(0.0, 1.0, 11)

results_e = []
results_esam = []
results_oft = []
results_sa = []
results_csam = []
results_wsam = []

for alpha in alphas:
    # 1. Task Arithmetic
    a1, a2 = evaluate_merged_euclid(fc1_e1, fc1_e2, fc2_e1, fc2_e2, pretrained_backbone_weight, alpha, val_loader1, val_loader2)
    results_e.append((a1 + a2) / 2)
    
    # 2. Task Arithmetic + SAM
    a1_es, a2_es = evaluate_merged_euclid(fc1_esam1, fc1_esam2, fc2_esam1, fc2_esam2, pretrained_backbone_weight, alpha, val_loader1, val_loader2)
    results_esam.append((a1_es + a2_es) / 2)
    
    # 3. OrthoMerge
    a1_o, a2_o = evaluate_merged_oft(fc1_oft1, fc1_oft2, fc2_oft1, fc2_oft2, pretrained_backbone_weight, alpha, val_loader1, val_loader2)
    results_oft.append((a1_o + a2_o) / 2)
    
    # 4. SA-Ortho
    a1_s, a2_s = evaluate_merged_oft(fc1_sa1, fc1_sa2, fc2_sa1, fc2_sa2, pretrained_backbone_weight, alpha, val_loader1, val_loader2)
    results_sa.append((a1_s + a2_s) / 2)
    
    # 5. OFT + Classifier-only SAM
    a1_c, a2_c = evaluate_merged_oft(fc1_csam1, fc1_csam2, fc2_csam1, fc2_csam2, pretrained_backbone_weight, alpha, val_loader1, val_loader2)
    results_csam.append((a1_c + a2_c) / 2)
    
    # 6. OFT + Euclidean Weight SAM
    a1_w, a2_w = evaluate_merged_oft(fc1_wsam1, fc1_wsam2, fc2_wsam1, fc2_wsam2, pretrained_backbone_weight, alpha, val_loader1, val_loader2)
    results_wsam.append((a1_w + a2_w) / 2)

print("\n=== Model Merging Results (Average Multi-task Accuracy) ===")
print("Alpha:      " + "  ".join([f"{a:.1f}" for a in alphas]))
print("Arithmetic: " + "  ".join([f"{a:.4f}" for a in results_e]))
print("Arith+SAM:  " + "  ".join([f"{a:.4f}" for a in results_esam]))
print("OrthoMerge: " + "  ".join([f"{a:.4f}" for a in results_oft]))
print("SA-Ortho:   " + "  ".join([f"{a:.4f}" for a in results_sa]))
print("OFT+ClsSAM: " + "  ".join([f"{a:.4f}" for a in results_csam]))
print("OFT+WtSAM:  " + "  ".join([f"{a:.4f}" for a in results_wsam]))

# Find peak average accuracies
print(f"\nPeak Average Accuracies:")
print(f"Task Arithmetic:           {max(results_e):.4f} at alpha={alphas[np.argmax(results_e)]:.1f}")
print(f"Task Arithmetic + SAM:     {max(results_esam):.4f} at alpha={alphas[np.argmax(results_esam)]:.1f}")
print(f"OrthoMerge:                {max(results_oft):.4f} at alpha={alphas[np.argmax(results_oft)]:.1f}")
print(f"SA-Ortho (Ours):           {max(results_sa):.4f} at alpha={alphas[np.argmax(results_sa)]:.1f}")
print(f"OFT + Classifier-only SAM: {max(results_csam):.4f} at alpha={alphas[np.argmax(results_csam)]:.1f}")
print(f"OFT + Weight-Space SAM:    {max(results_wsam):.4f} at alpha={alphas[np.argmax(results_wsam)]:.1f}")

# --- 7. Create and Save Plots ---

# Main Plot: Main Methods and New Baselines
plt.figure(figsize=(9, 5.5))
plt.plot(alphas, results_e, 'o--', label='Task Arithmetic (Euclidean)', color='tab:red')
plt.plot(alphas, results_esam, '*:', label='Task Arithmetic + SAM (SAIM-like)', color='tab:orange')
plt.plot(alphas, results_oft, 's-', label='OrthoMerge (Riemannian)', color='tab:blue')
plt.plot(alphas, results_sa, '^--', label='SA-Ortho (Proposed, Lie algebra SAM)', color='tab:green', linewidth=2.5)
plt.xlabel('Merging Weight (Alpha for Task 1)')
plt.ylabel('Average Multi-task Accuracy')
plt.title('Comprehensive Model Merging Comparison: Euclidean vs. Riemannian Manifolds')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig('results_refined.png', dpi=150)
print("\nRefined plot saved to results_refined.png")

# Ablation Plot: Understanding Where SAM Matters in OFT
plt.figure(figsize=(9, 5.5))
plt.plot(alphas, results_oft, 's-', label='OrthoMerge (No SAM)', color='tab:blue')
plt.plot(alphas, results_csam, 'x:', label='OFT + Classifier-only SAM', color='purple')
plt.plot(alphas, results_wsam, 'v-.', label='OFT + Euclidean Weight SAM', color='darkorange')
plt.plot(alphas, results_sa, '^--', label='SA-Ortho (Proposed, Joint Lie + Head SAM)', color='tab:green', linewidth=2.5)
plt.xlabel('Merging Weight (Alpha for Task 1)')
plt.ylabel('Average Multi-task Accuracy')
plt.title('Ablation Study: Understanding the Source of Sharpness-Aware Manifold Gains')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig('results_ablation_sam.png', dpi=150)
print("Ablation plot saved to results_ablation_sam.png")
