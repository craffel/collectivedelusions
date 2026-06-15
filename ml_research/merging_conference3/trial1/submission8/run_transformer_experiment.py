import os
import sys
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ---------------------------------------------------------
# 1. Custom Vision Transformer (ViT) Architecture
# ---------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_chans=1, embed_dim=32):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 1, 28, 28] -> [B, embed_dim, 4, 4] -> [B, embed_dim, 16] -> [B, 16, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=32, num_heads=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)

class CustomTransformerLayer(nn.Module):
    def __init__(self, embed_dim=32, num_heads=2, mlp_dim=32):
        super().__init__()
        self.attn = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Linear(embed_dim, mlp_dim, bias=False)
        self.fc2 = nn.Linear(mlp_dim, embed_dim, bias=False)
        
    def forward(self, x):
        # Pre-LN
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(torch.relu(self.fc1(self.norm2(x))))
        return x

class ViTClassifier(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_chans=1, num_classes=10, embed_dim=32, depth=1, num_heads=2, mlp_dim=32):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        
        self.layers = nn.ModuleList([
            CustomTransformerLayer(embed_dim, num_heads, mlp_dim)
            for _ in range(depth)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        cls_rep = x[:, 0]
        return self.mlp_head(cls_rep)

# ---------------------------------------------------------
# 2. Dataset Preparation
# ---------------------------------------------------------
def get_datasets():
    try:
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        return train_dataset, test_dataset
    except Exception as e:
        print(f"Failed to load MNIST from torchvision: {e}. Generating synthetic datasets.")
        np.random.seed(42)
        X_train = np.random.randn(6000, 1, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, size=(6000,)).astype(np.int64)
        X_test = np.random.randn(1000, 1, 28, 28).astype(np.float32)
        y_test = np.random.randint(0, 10, size=(1000,)).astype(np.int64)
        for i in range(10):
            mask_tr = (y_train == i)
            # Add simple patterns
            X_train[mask_tr, 0, (i*2)%20:(i*2)%20+7, (i*2)%20:(i*2)%20+7] += 5.0
            mask_te = (y_test == i)
            X_test[mask_te, 0, (i*2)%20:(i*2)%20+7, (i*2)%20:(i*2)%20+7] += 5.0
        return TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

def filter_split_dataset(dataset, classes):
    indices = []
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'tensors'):
        targets = dataset.tensors[1]
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]
    for idx, target in enumerate(targets):
        if int(target) in classes:
            indices.append(idx)
    return Subset(dataset, indices)

# ---------------------------------------------------------
# 3. Model Training & Evaluation
# ---------------------------------------------------------
def train_model_ortho(model, train_loader, epochs=3, lr=0.01, device="cpu", desc="", ortho_lambda=0.1):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"  Training {desc} (Ortho Regularization Lambda = {ortho_lambda})...")
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Orthogonal Regularization for linear weights
            reg_loss = 0.0
            for name, param in model.named_parameters():
                if len(param.shape) == 2 and "weight" in name and "patch_embed" not in name:
                    out_d, in_d = param.shape
                    if out_d >= in_d:
                        prod = torch.matmul(param.t(), param)
                        I = torch.eye(in_d, device=param.device)
                    else:
                        prod = torch.matmul(param, param.t())
                        I = torch.eye(out_d, device=param.device)
                    reg_loss += torch.norm(prod - I, p='fro')
                    
            total_loss = loss + ortho_lambda * reg_loss
            total_loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"    Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    return model

def eval_model(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total if total > 0 else 0.0

# ---------------------------------------------------------
# 4. Merging Operators
# ---------------------------------------------------------
def get_rotation_procrustes(W_k, W_0):
    A = torch.matmul(W_0.t(), W_k)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    R = torch.matmul(U, Vh)
    return R

def cayley_to_skew(R):
    I = torch.eye(R.shape[-1], device=R.device, dtype=R.dtype)
    Q = torch.linalg.solve(R + I, R - I)
    Q = 0.5 * (Q - Q.t())
    return Q

def skew_to_cayley(Q):
    I = torch.eye(Q.shape[-1], device=Q.device, dtype=Q.dtype)
    R = torch.matmul(I + Q, torch.linalg.inv(I - Q))
    return R

def rimo_spectral_balancing(Q_com, t=2.0):
    U, S, Vh = torch.linalg.svd(Q_com, full_matrices=False)
    d = Q_com.shape[-1]
    mean_sigma = torch.sum(S) / d
    # SAIM-style spectral balancing inside Lie algebra
    S_new = mean_sigma + (S - mean_sigma) / math.sqrt(t)
    # Reconstruct skew-symmetric matrix
    Q_new = torch.matmul(U, torch.matmul(torch.diag(S_new), Vh))
    return 0.5 * (Q_new - Q_new.t())

def rimo_spectral_pruning(Q_com, keep_ratio=0.2):
    U, S, Vh = torch.linalg.svd(Q_com, full_matrices=False)
    k = int(len(S) * keep_ratio)
    k = max(1, min(k, len(S)))
    # Prune elements of spectrum
    S_new = S.clone()
    S_new[k:] = 0.0
    Q_new = torch.matmul(U, torch.matmul(torch.diag(S_new), Vh))
    return 0.5 * (Q_new - Q_new.t())

# ---------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------
if __name__ == '__main__':
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running RIMO on Vision Transformer (ViT) ===\nDevice: {device}")
    
    train_dataset, test_dataset = get_datasets()
    
    # Task 1 (classes 0-4), Task 2 (classes 5-9)
    train_loader_t1 = DataLoader(filter_split_dataset(train_dataset, range(5)), batch_size=128, shuffle=True)
    train_loader_t2 = DataLoader(filter_split_dataset(train_dataset, range(5, 10)), batch_size=128, shuffle=True)
    train_loader_all = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    test_loader_t1 = DataLoader(filter_split_dataset(test_dataset, range(5)), batch_size=128, shuffle=False)
    test_loader_t2 = DataLoader(filter_split_dataset(test_dataset, range(5, 10)), batch_size=128, shuffle=False)
    test_loader_all = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 5.1 Train base model on all classes
    print("\n[Step 1] Training ViT Base Model...")
    base_model = ViTClassifier()
    base_model = train_model_ortho(base_model, train_loader_all, epochs=4, lr=0.005, device=device, desc="Base Model", ortho_lambda=2.0)
    
    acc_t1_base = eval_model(base_model, test_loader_t1, device)
    acc_t2_base = eval_model(base_model, test_loader_t2, device)
    acc_all_base = eval_model(base_model, test_loader_all, device)
    print(f"Base Model Accuracies -> Task 1: {acc_t1_base:.4f}, Task 2: {acc_t2_base:.4f}, Overall: {acc_all_base:.4f}")
    
    # 5.2 Train Expert 1 on Task 1 and Expert 2 on Task 2 starting from Base Model
    print("\n[Step 2] Fine-tuning ViT Experts with Orthogonal Regularization...")
    expert1 = ViTClassifier()
    expert1.load_state_dict(base_model.state_dict())
    expert1 = train_model_ortho(expert1, train_loader_t1, epochs=3, lr=0.005, device=device, desc="Expert 1 (Task 1)", ortho_lambda=2.0)
    
    expert2 = ViTClassifier()
    expert2.load_state_dict(base_model.state_dict())
    expert2 = train_model_ortho(expert2, train_loader_t2, epochs=3, lr=0.005, device=device, desc="Expert 2 (Task 2)", ortho_lambda=2.0)
    
    # Evaluate individual expert accuracy
    print(f"\n[Expert 1 Accuracies] Task 1: {eval_model(expert1, test_loader_t1, device):.4f}, Task 2: {eval_model(expert1, test_loader_t2, device):.4f}, Overall: {eval_model(expert1, test_loader_all, device):.4f}")
    print(f"[Expert 2 Accuracies] Task 1: {eval_model(expert2, test_loader_t1, device):.4f}, Task 2: {eval_model(expert2, test_loader_t2, device):.4f}, Overall: {eval_model(expert2, test_loader_all, device):.4f}")
    
    # 5.3 Merge models
    print("\n[Step 3] Merging Experts via different strategies...")
    
    # Strategy A: Euclidean Task Arithmetic (Baseline)
    merged_ta = ViTClassifier()
    for name, param in merged_ta.named_parameters():
        p0 = dict(base_model.named_parameters())[name].data
        p1 = dict(expert1.named_parameters())[name].data
        p2 = dict(expert2.named_parameters())[name].data
        # TA: W_0 + 0.5*(W1 - W0) + 0.5*(W2 - W0)
        param.data.copy_(p0 + 0.5 * (p1 - p0) + 0.5 * (p2 - p0))
        
    acc_t1_ta = eval_model(merged_ta, test_loader_t1, device)
    acc_t2_ta = eval_model(merged_ta, test_loader_t2, device)
    acc_all_ta = eval_model(merged_ta, test_loader_all, device)
    print(f"Task Arithmetic (TA) -> Task 1: {acc_t1_ta:.4f}, Task 2: {acc_t2_ta:.4f}, Overall: {acc_all_ta:.4f}")
    
    # Helper for Riemannian Merging
    def rimo_merge(balancing_fn=None):
        merged = ViTClassifier()
        # Initialize
        merged.load_state_dict(base_model.state_dict())
        with torch.no_grad():
            for name, param in merged.named_parameters():
                if len(param.shape) == 2 and "weight" in name and "patch_embed" not in name:
                    # Retrieve base and experts
                    W_0 = dict(base_model.named_parameters())[name].data
                    W_1 = dict(expert1.named_parameters())[name].data
                    W_2 = dict(expert2.named_parameters())[name].data
                    
                    # Procrustes decomposition
                    R1 = get_rotation_procrustes(W_1, W_0)
                    R2 = get_rotation_procrustes(W_2, W_0)
                    
                    # Residuals
                    rho1 = W_1 - torch.matmul(W_0, R1)
                    rho2 = W_2 - torch.matmul(W_0, R2)
                    
                    # Inverse Cayley
                    Q1 = cayley_to_skew(R1)
                    Q2 = cayley_to_skew(R2)
                    
                    # Average generators in Lie Algebra
                    Q_merged = 0.5 * (Q1 + Q2)
                    if balancing_fn is not None:
                        Q_merged = balancing_fn(Q_merged)
                        
                    # Forward Cayley
                    R_merged = skew_to_cayley(Q_merged)
                    
                    # Average residuals
                    rho_merged = 0.5 * (rho1 + rho2)
                    
                    # Reconstruct merged weights
                    W_merged = torch.matmul(W_0, R_merged) + rho_merged
                    param.copy_(W_merged)
                elif "patch_embed" in name or len(param.shape) != 2:
                    # Linear task arithmetic for other parameters (e.g. bias, positional embed, patch embed)
                    p0 = dict(base_model.named_parameters())[name].data
                    p1 = dict(expert1.named_parameters())[name].data
                    p2 = dict(expert2.named_parameters())[name].data
                    param.copy_(p0 + 0.5 * (p1 - p0) + 0.5 * (p2 - p0))
        return merged

    # Strategy B: RIMO (Euclidean Residual + Riemannian Lie Average) t=1.0 (No Balancing/Pruning)
    print("\nPerforming RIMO (t=1.0)...")
    merged_rimo_t1 = rimo_merge(balancing_fn=None)
    acc_t1_rimo = eval_model(merged_rimo_t1, test_loader_t1, device)
    acc_t2_rimo = eval_model(merged_rimo_t1, test_loader_t2, device)
    acc_all_rimo = eval_model(merged_rimo_t1, test_loader_all, device)
    print(f"RIMO (t=1.0) -> Task 1: {acc_t1_rimo:.4f}, Task 2: {acc_t2_rimo:.4f}, Overall: {acc_all_rimo:.4f}")
    
    # Strategy C: RIMO with Spectral Balancing (t=2.0) - Spectral Balancing Pitfall
    print("\nPerforming RIMO with Spectral Balancing (t=2.0) to observe pitfall...")
    merged_rimo_bal = rimo_merge(balancing_fn=lambda Q: rimo_spectral_balancing(Q, t=2.0))
    acc_t1_bal = eval_model(merged_rimo_bal, test_loader_t1, device)
    acc_t2_bal = eval_model(merged_rimo_bal, test_loader_t2, device)
    acc_all_bal = eval_model(merged_rimo_bal, test_loader_all, device)
    print(f"RIMO (Spectral Balancing t=2.0) -> Task 1: {acc_t1_bal:.4f}, Task 2: {acc_t2_bal:.4f}, Overall: {acc_all_bal:.4f}")
    
    # Strategy D: RIMO-Pruned with Spectral Pruning (keep_ratio=0.2)
    print("\nPerforming RIMO-Pruned (Spectral Pruning, keep_ratio=0.2) to bypass pitfall...")
    merged_rimo_pruned = rimo_merge(balancing_fn=lambda Q: rimo_spectral_pruning(Q, keep_ratio=0.2))
    acc_t1_pruned = eval_model(merged_rimo_pruned, test_loader_t1, device)
    acc_t2_pruned = eval_model(merged_rimo_pruned, test_loader_t2, device)
    acc_all_pruned = eval_model(merged_rimo_pruned, test_loader_all, device)
    print(f"RIMO-Pruned (keep_ratio=0.2) -> Task 1: {acc_t1_pruned:.4f}, Task 2: {acc_t2_pruned:.4f}, Overall: {acc_all_pruned:.4f}")
    
    results = {
        "Base Model": {"Task 1": acc_t1_base, "Task 2": acc_t2_base, "Overall": acc_all_base},
        "Task Arithmetic": {"Task 1": acc_t1_ta, "Task 2": acc_t2_ta, "Overall": acc_all_ta},
        "RIMO (t=1.0)": {"Task 1": acc_t1_rimo, "Task 2": acc_t2_rimo, "Overall": acc_all_rimo},
        "RIMO (Spectral Balancing t=2.0)": {"Task 1": acc_t1_bal, "Task 2": acc_t2_bal, "Overall": acc_all_bal},
        "RIMO-Pruned": {"Task 1": acc_t1_pruned, "Task 2": acc_t2_pruned, "Overall": acc_all_pruned}
    }
    
    with open("transformer_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nResults successfully saved to 'transformer_results.json'.")
