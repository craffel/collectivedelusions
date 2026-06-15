import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import os
import copy

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("--- Real-World Validation: Merging MLP Experts on MNIST and FashionMNIST ---")

# Define MLP Architecture
class MLPBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return x

class MultiTaskModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head1 = nn.Linear(64, 10) # MNIST Head
        self.head2 = nn.Linear(64, 10) # FashionMNIST Head
        
    def forward(self, x, task_id):
        features = self.backbone(x)
        if task_id == 0:
            return self.head1(features)
        else:
            return self.head2(features)

# 1. Load Datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

print("Loading MNIST and FashionMNIST...")
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

fashion_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
fashion_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Use subset for fast training on CPU
mnist_train_sub = Subset(mnist_train, list(range(10000)))
fashion_train_sub = Subset(fashion_train, list(range(10000)))

train_loader_mnist = DataLoader(mnist_train_sub, batch_size=128, shuffle=True)
train_loader_fashion = DataLoader(fashion_train_sub, batch_size=128, shuffle=True)

test_loader_mnist = DataLoader(mnist_test, batch_size=256, shuffle=False)
test_loader_fashion = DataLoader(fashion_test, batch_size=256, shuffle=False)

# 2. Train Experts
def train_expert(model, loader, task_id, epochs=2):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x, task_id)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

backbone_base = MLPBackbone()
# Deep copy state dict at initialization to keep base weights clean
base_backbone_state = copy.deepcopy(backbone_base.state_dict())

print("Training Expert 1 on MNIST...")
model_mnist = MultiTaskModel(copy.deepcopy(backbone_base))
train_expert(model_mnist, train_loader_mnist, task_id=0, epochs=2)

print("Training Expert 2 on FashionMNIST...")
model_fashion = MultiTaskModel(copy.deepcopy(backbone_base))
train_expert(model_fashion, train_loader_fashion, task_id=1, epochs=2)

# Extract learned state dicts from independent trained copies
mnist_backbone_state = model_mnist.backbone.state_dict()
fashion_backbone_state = model_fashion.backbone.state_dict()

# Calculate Task Vectors
task_vector_mnist = {k: mnist_backbone_state[k] - base_backbone_state[k] for k in base_backbone_state}
task_vector_fashion = {k: fashion_backbone_state[k] - base_backbone_state[k] for k in base_backbone_state}

# Differentiable weight reconstruction and forward pass
def functional_forward(x, lambdas):
    # lambdas is of shape [2, 2] (2 tasks, 2 layer groups)
    # fc1 layer group
    fc1_weight = base_backbone_state["fc1.weight"] + \
                 lambdas[0, 0] * task_vector_mnist["fc1.weight"] + \
                 lambdas[1, 0] * task_vector_fashion["fc1.weight"]
    fc1_bias = base_backbone_state["fc1.bias"] + \
               lambdas[0, 0] * task_vector_mnist["fc1.bias"] + \
               lambdas[1, 0] * task_vector_fashion["fc1.bias"]
               
    # fc2 layer group
    fc2_weight = base_backbone_state["fc2.weight"] + \
                 lambdas[0, 1] * task_vector_mnist["fc2.weight"] + \
                 lambdas[1, 1] * task_vector_fashion["fc2.weight"]
    fc2_bias = base_backbone_state["fc2.bias"] + \
               lambdas[0, 1] * task_vector_mnist["fc2.bias"] + \
               lambdas[1, 1] * task_vector_fashion["fc2.bias"]
               
    # Perform forward pass functionally
    x = x.view(-1, 28*28)
    x = F.linear(x, fc1_weight, fc1_bias)
    x = F.relu(x)
    x = F.linear(x, fc2_weight, fc2_bias)
    x = F.relu(x)
    return x

# Evaluate merged model functionally
def evaluate_merged(lambdas, noise_scale=0.0):
    model_mnist.eval()
    model_fashion.eval()
    
    # MNIST accuracy
    correct_mnist = 0
    total_mnist = 0
    with torch.no_grad():
        for x, y in test_loader_mnist:
            if noise_scale > 0:
                x = x + torch.randn_like(x) * noise_scale
            features = functional_forward(x, lambdas)
            out = model_mnist.head1(features)
            preds = torch.argmax(out, dim=-1)
            correct_mnist += (preds == y).sum().item()
            total_mnist += len(y)
    acc_mnist = correct_mnist / total_mnist
    
    # FashionMNIST accuracy
    correct_fashion = 0
    total_fashion = 0
    with torch.no_grad():
        for x, y in test_loader_fashion:
            if noise_scale > 0:
                x = x + torch.randn_like(x) * noise_scale
            features = functional_forward(x, lambdas)
            out = model_fashion.head2(features)
            preds = torch.argmax(out, dim=-1)
            correct_fashion += (preds == y).sum().item()
            total_fashion += len(y)
    acc_fashion = correct_fashion / total_fashion
    
    return acc_mnist * 100.0, acc_fashion * 100.0

# 3. Test-Time Adaptation Setup
def get_tta_batch(batch_size=128, noise_scale=0.0):
    # Retrieve half from MNIST and half from FashionMNIST
    x_m, _ = next(iter(DataLoader(mnist_test, batch_size=batch_size//2, shuffle=True)))
    x_f, _ = next(iter(DataLoader(fashion_test, batch_size=batch_size//2, shuffle=True)))
    x = torch.cat([x_m, x_f], dim=0)
    if noise_scale > 0:
        x = x + torch.randn_like(x) * noise_scale
    return x

def entropy_loss(logits):
    probs = torch.softmax(logits, dim=-1)
    return -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))

# Run First-Order AdaMerging (using Autograd)
def run_adamerging_tta(x_batch, steps=100, lr=0.1):
    lambdas = torch.full((2, 2), 0.3, requires_grad=True)
    optimizer = optim.Adam([lambdas], lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()
        features = functional_forward(x_batch, lambdas)
        logits1 = model_mnist.head1(features[:len(x_batch)//2])
        logits2 = model_fashion.head2(features[len(x_batch)//2:])
        
        loss = entropy_loss(logits1) + entropy_loss(logits2)
        loss.backward()
        if step % 20 == 0:
            print(f"    [AdaMerging Step {step}] Loss: {loss.item():.6f}, Grad: {lambdas.grad.detach().numpy() if lambdas.grad is not None else 'None'}")
        optimizer.step()
        
    return lambdas.detach().numpy()

# Run ZO-FlatMerge (using Zeroth-Order randomized smoothing)
def run_zo_flatmerge_tta(x_batch, steps=100, lr=0.1, sigma=0.05, num_samples=10):
    lambdas = torch.full((2, 2), 0.3)
    
    for step in range(steps):
        grad_estimate = torch.zeros_like(lambdas)
        for _ in range(num_samples):
            # Sample direction and normalize to unit vector
            E = torch.randn_like(lambdas)
            U = E / (torch.norm(E) + 1e-12)
            
            # Positive loss
            features_pos = functional_forward(x_batch, lambdas + sigma * U)
            loss_pos = entropy_loss(model_mnist.head1(features_pos[:len(x_batch)//2])) + \
                       entropy_loss(model_fashion.head2(features_pos[len(x_batch)//2:]))
            
            # Negative loss
            features_neg = functional_forward(x_batch, lambdas - sigma * U)
            loss_neg = entropy_loss(model_mnist.head1(features_neg[:len(x_batch)//2])) + \
                       entropy_loss(model_fashion.head2(features_neg[len(x_batch)//2:]))
            
            grad_estimate += ((loss_pos - loss_neg) / (2.0 * sigma)) * U
            
        grad_estimate /= num_samples
        if step % 20 == 0:
            print(f"    [ZO-FlatMerge Step {step}] Loss Grad Estimate: {grad_estimate.detach().numpy()}")
        # Update coefficients
        lambdas -= lr * grad_estimate
        
    return lambdas.detach().numpy()

# 4. Sweep Noise Scales
noise_scales = [0.0, 1.0, 2.0, 3.0]
results = []

for ns in noise_scales:
    print(f"\nEvaluating Noise Scale (Gamma) = {ns:.1f}")
    
    # Task Arithmetic (Uniform blending of 0.3)
    ta_lambdas = torch.full((2, 2), 0.3)
    ta_mnist, ta_fashion = evaluate_merged(ta_lambdas, noise_scale=ns)
    ta_joint = (ta_mnist + ta_fashion) / 2.0
    print(f"  Task Arithmetic: MNIST={ta_mnist:.2f}%, Fashion={ta_fashion:.2f}%, Joint={ta_joint:.2f}%")
    
    # Get TTA Batch (large batch for stable adaptation)
    x_batch = get_tta_batch(batch_size=128, noise_scale=ns)
    
    # AdaMerging TTA
    ada_lambdas = run_adamerging_tta(x_batch, steps=100, lr=0.05)
    ada_mnist, ada_fashion = evaluate_merged(torch.tensor(ada_lambdas), noise_scale=ns)
    ada_joint = (ada_mnist + ada_fashion) / 2.0
    print(f"  AdaMerging (FO): MNIST={ada_mnist:.2f}%, Fashion={ada_fashion:.2f}%, Joint={ada_joint:.2f}%")
    print(f"    Optimized Lambdas: \n{ada_lambdas}")
    
    # ZO-FlatMerge TTA (Ours)
    zo_lambdas = run_zo_flatmerge_tta(x_batch, steps=100, lr=0.05, sigma=0.05, num_samples=10)
    zo_mnist, zo_fashion = evaluate_merged(torch.tensor(zo_lambdas), noise_scale=ns)
    zo_joint = (zo_mnist + zo_fashion) / 2.0
    print(f"  ZO-FlatMerge (Ours): MNIST={zo_mnist:.2f}%, Fashion={zo_fashion:.2f}%, Joint={zo_joint:.2f}%")
    print(f"    Optimized Lambdas: \n{zo_lambdas}")
    
    results.append({
        "noise_scale": ns,
        "ta": [ta_mnist, ta_fashion, ta_joint],
        "ada": [ada_mnist, ada_fashion, ada_joint],
        "zo": [zo_mnist, zo_fashion, zo_joint]
    })

# Write results to file
import json
with open("results/real_mnist_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved real-world validation metrics to results/real_mnist_metrics.json")
