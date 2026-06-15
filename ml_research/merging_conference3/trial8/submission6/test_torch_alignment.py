import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Dimensions
D_in = 64
D = 192
r = 8
K = 3 # number of tasks
num_classes = 5

# Shared backbone (frozen random projection)
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D_in, D)
        # Freeze
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        return torch.relu(self.proj(x))

backbone = Backbone()

# Generate task prototype vectors in input space (orthogonal to each other)
prototypes = []
for k in range(K):
    v = torch.zeros(D_in)
    v[k * 10 : (k + 1) * 10] = 1.0 # orthogonal features
    v /= torch.norm(v)
    prototypes.append(v)

# Data generator for task k
def generate_task_data(k, num_samples):
    proto = prototypes[k]
    # Add random noise around the prototype
    X = proto.unsqueeze(0).repeat(num_samples, 1) + torch.randn(num_samples, D_in) * 0.05
    # Classification targets based on task-specific random projections
    W_target = torch.randn(D_in, num_classes)
    logits = X @ W_target
    y = torch.argmax(logits, dim=1)
    return X, y

# Class representing the Model with shared backbone and task-specific adapters/heads
class TaskModel(nn.Module):
    def __init__(self, backbone, k):
        super().__init__()
        self.backbone = backbone
        self.A = nn.Parameter(torch.randn(D, r) * 0.01)
        self.B = nn.Parameter(torch.randn(r, D) * 0.01)
        self.head = nn.Linear(D, num_classes)
        
    def forward(self, x):
        h = self.backbone(x)
        # Adapter path
        h_adapt = h @ self.A @ self.B
        out = h + h_adapt
        logits = self.head(out)
        return logits, h

# Train adapters for each task
models = []
for k in range(K):
    model = TaskModel(backbone, k)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Train data
    X_train, y_train = generate_task_data(k, 512)
    
    # Simple training loop
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
    models.append(model)
    print(f"Task {k} trained with final loss {loss.item():.4f}")

# Verify Weight-Activation Alignment Assumption
# We extract Q_k from QR of A_k
Qs = []
for k in range(K):
    A_k = models[k].A.detach()
    Q_k, R_k = torch.linalg.qr(A_k)
    Qs.append(Q_k)

# Compute alignment of activations with each Q_k
for test_k in range(K):
    X_test, _ = generate_task_data(test_k, 100)
    with torch.no_grad():
        h = backbone(X_test) # activations
        h_norm = torch.norm(h, dim=1, keepdim=True)
        
        # Measure alignment: || h Q_k ||_2 / || h ||_2
        alignments = []
        for k in range(K):
            proj = h @ Qs[k]
            proj_norm = torch.norm(proj, dim=1, keepdim=True)
            align = (proj_norm / (h_norm + 1e-8)).mean().item()
            alignments.append(align)
            
        print(f"Test Task {test_k} activations alignment with: Q0: {alignments[0]:.4f}, Q1: {alignments[1]:.4f}, Q2: {alignments[2]:.4f}")
