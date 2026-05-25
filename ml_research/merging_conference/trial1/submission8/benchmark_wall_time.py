import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)

# --- 1. Synthetic Dataset Generation ---
def generate_synthetic_data(num_samples=1200, input_dim=128, num_classes=8, task_id=1):
    X = torch.randn(num_samples, input_dim)
    torch.manual_seed(100 + task_id)
    W_task = torch.randn(input_dim, num_classes)
    logits = torch.matmul(X, W_task) + 0.05 * torch.randn(num_samples, num_classes)
    y = torch.argmax(logits, dim=1)
    return X, y

# Generate datasets: Task 1 and Task 2
X1, y1 = generate_synthetic_data(1200, 128, 8, task_id=1)
train_loader1 = DataLoader(TensorDataset(X1[:1000], y1[:1000]), batch_size=32, shuffle=True)

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

base_model = BaseMLP(128, 64, 8)

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

def measure_euclidean(train_loader, base_model, epochs=5):
    model = BaseMLP(128, 64, 8)
    model.load_state_dict(base_model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    times = []
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times)

def measure_oft(train_loader, base_model, epochs=5):
    model = OFTMLP(base_model, block_size=8)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    times = []
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times)

def measure_sa_oft(train_loader, base_model, epochs=5, rho=0.1):
    model = OFTMLP(base_model, block_size=8)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    times = []
    for epoch in range(epochs):
        start_time = time.time()
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
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times)

print("Starting benchmark...")
t_eucl = measure_euclidean(train_loader1, base_model, epochs=10)
print(f"Euclidean Fine-Tuning: {t_eucl * 1000:.2f} ms per epoch")

t_oft = measure_oft(train_loader1, base_model, epochs=10)
print(f"Standard OFT: {t_oft * 1000:.2f} ms per epoch")

t_sa = measure_sa_oft(train_loader1, base_model, epochs=10)
print(f"SA-Ortho: {t_sa * 1000:.2f} ms per epoch")

print(f"Relative Overhead of SA-Ortho vs Standard OFT: {t_sa / t_oft:.2f}x")
