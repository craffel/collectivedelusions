import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set parameters
D = 192
K = 4
L = 14
d = 48
sigmas = [0.05, 0.15, 0.40, 1.20]
biases = [0.0, 0.0, -0.90, -2.30]
gamma_val = 0.05

def get_signatures(rho=0.0):
    v_orth = np.zeros((K, D))
    for k in range(K):
        v_orth[k, k*d:(k+1)*d] = 1.0 / np.sqrt(d)
        
    if rho > 0.0:
        Sigma = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                Sigma[i, j] = rho ** abs(i - j)
        U, S, Vt = np.linalg.svd(Sigma)
        Sigma_half = U @ np.diag(np.sqrt(S)) @ Vt
        
        v = np.zeros((K, D))
        for k in range(K):
            v[k] = Sigma_half @ v_orth[k]
            v[k] /= np.linalg.norm(v[k])
    else:
        v = v_orth.copy()
    return v

def generate_dataset(v, num_samples_per_task=250, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    samples = []
    labels = []
    for k in range(K):
        for _ in range(num_samples_per_task):
            eps = np.random.normal(0, sigmas[k], D)
            samples.append(v[k] + eps)
            labels.append(k)
            
    return torch.tensor(np.array(samples), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

class SandboxModule(nn.Module):
    def __init__(self, v, biases, gating='softmax', zero_init=True):
        super().__init__()
        self.gating = gating
        self.register_buffer('v', torch.tensor(v, dtype=torch.float32))
        self.register_buffer('biases', torch.tensor(biases, dtype=torch.float32))
        
        self.weight = nn.Parameter(torch.empty(D, K))
        self.bias = nn.Parameter(torch.empty(K))
        if zero_init:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)
        else:
            nn.init.normal_(self.weight, std=0.01)
            nn.init.zeros_(self.bias)
            
    def forward(self, h0):
        logits_g = h0 @ self.weight + self.bias
        if self.gating == 'softmax':
            alpha = torch.softmax(logits_g, dim=-1)
        else:
            alpha = torch.sigmoid(logits_g)
            
        h = h0
        for l in range(4, L + 1):
            update = torch.zeros_like(h)
            for k in range(K):
                update += alpha[:, k:k+1] * gamma_val * (self.v[k] - h)
            h = h + update
            
        h_expanded = h.unsqueeze(1)
        v_expanded = self.v.unsqueeze(0)
        dists = torch.sum((h_expanded - v_expanded)**2, dim=-1)
        logits = -dists + self.biases
        return logits, alpha

class SandboxModuleLayerWise(nn.Module):
    def __init__(self, v, biases, gating='softmax', zero_init=True):
        super().__init__()
        self.gating = gating
        self.register_buffer('v', torch.tensor(v, dtype=torch.float32))
        self.register_buffer('biases', torch.tensor(biases, dtype=torch.float32))
        
        self.num_layers = L - 4 + 1
        self.weight = nn.Parameter(torch.empty(self.num_layers, D, K))
        self.bias = nn.Parameter(torch.empty(self.num_layers, K))
        if zero_init:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)
        else:
            nn.init.normal_(self.weight, std=0.01)
            nn.init.zeros_(self.bias)
            
    def forward(self, h0):
        h = h0
        alphas = []
        for l_idx in range(self.num_layers):
            logits_g = h @ self.weight[l_idx] + self.bias[l_idx]
            if self.gating == 'softmax':
                alpha = torch.softmax(logits_g, dim=-1)
            else:
                alpha = torch.sigmoid(logits_g)
            alphas.append(alpha)
            
            update = torch.zeros_like(h)
            for k in range(K):
                update += alpha[:, k:k+1] * gamma_val * (self.v[k] - h)
            h = h + update
            
        h_expanded = h.unsqueeze(1)
        v_expanded = self.v.unsqueeze(0)
        dists = torch.sum((h_expanded - v_expanded)**2, dim=-1)
        logits = -dists + self.biases
        return logits, alphas

# Test run for rho = 0.3
rho = 0.3
v = get_signatures(rho)
train_small, train_labels_small = generate_dataset(v, num_samples_per_task=16, seed=42)
train_large, train_labels_large = generate_dataset(v, num_samples_per_task=1000, seed=42)
test_samples, test_labels = generate_dataset(v, num_samples_per_task=250, seed=42)

def train_eval(model_class, train_data, train_lbls, gating, zero_init, wd, epochs=120):
    model = model_class(v, biases, gating=gating, zero_init=zero_init)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits_train, _ = model(train_data)
        loss = criterion(logits_train, train_lbls)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        logits_test, alphas = model(test_samples)
        acc = (torch.argmax(logits_test, dim=-1) == test_labels).float().mean().item()
        
    # Calculate jitter
    if isinstance(alphas, list):
        jitter_sum = 0.0
        for l in range(1, len(alphas)):
            diff = torch.norm(alphas[l] - alphas[l-1], p=2, dim=-1)
            jitter_sum += diff.mean().item()
        jitter = jitter_sum / (L - 4)
    else:
        jitter = 0.0
        
    return acc, jitter

print("Evaluating Small-Sample Regime (N=64):")
acc_inv, j_inv = train_eval(SandboxModule, train_small, train_labels_small, 'softmax', True, 1e-2, epochs=100)
print(f"Layer-Invariant Softmax (WD=1e-2): Acc={acc_inv*100:.2f}%, Jitter={j_inv:.4f}")

acc_lw, j_lw = train_eval(SandboxModuleLayerWise, train_small, train_labels_small, 'softmax', True, 1e-2, epochs=100)
print(f"Layer-Wise Softmax (WD=1e-2): Acc={acc_lw*100:.2f}%, Jitter={j_lw:.4f}")

acc_lw_unreg, j_lw_unreg = train_eval(SandboxModuleLayerWise, train_small, train_labels_small, 'softmax', False, 0.0, epochs=100)
print(f"Layer-Wise Unreg Softmax: Acc={acc_lw_unreg*100:.2f}%, Jitter={j_lw_unreg:.4f}")

print("\nEvaluating Large-Sample Regime (N=4000):")
acc_inv_lg, j_inv_lg = train_eval(SandboxModule, train_large, train_labels_large, 'softmax', True, 1e-2, epochs=120)
print(f"Layer-Invariant Softmax (WD=1e-2): Acc={acc_inv_lg*100:.2f}%, Jitter={j_inv_lg:.4f}")

acc_lw_lg, j_lw_lg = train_eval(SandboxModuleLayerWise, train_large, train_labels_large, 'softmax', True, 1e-2, epochs=120)
print(f"Layer-Wise Softmax (WD=1e-2): Acc={acc_lw_lg*100:.2f}%, Jitter={j_lw_lg:.4f}")

acc_lw_lg_unreg, j_lw_lg_unreg = train_eval(SandboxModuleLayerWise, train_large, train_labels_large, 'softmax', False, 0.0, epochs=120)
print(f"Layer-Wise Unreg Softmax: Acc={acc_lw_lg_unreg*100:.2f}%, Jitter={j_lw_lg_unreg:.4f}")
