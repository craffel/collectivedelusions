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
import json
import gc

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=== Starting Real-World Vision Model Merging: 5-Layer CNN on MNIST, FashionMNIST, and KMNIST ===")

# Define 5-Layer CNN Architecture
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 32)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class MultiTaskCNN(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head1 = nn.Linear(32, 10) # MNIST Head
        self.head2 = nn.Linear(32, 10) # FashionMNIST Head
        self.head3 = nn.Linear(32, 10) # KMNIST Head
        
    def forward(self, x, task_id):
        features = self.backbone(x)
        if isinstance(task_id, int):
            if task_id == 0:
                return self.head1(features)
            elif task_id == 1:
                return self.head2(features)
            else:
                return self.head3(features)
        else:
            out1 = self.head1(features)
            out2 = self.head2(features)
            out3 = self.head3(features)
            
            out = torch.zeros_like(out1)
            mask0 = (task_id == 0).unsqueeze(-1)
            mask1 = (task_id == 1).unsqueeze(-1)
            mask2 = (task_id == 2).unsqueeze(-1)
            
            out = out + mask0 * out1 + mask1 * out2 + mask2 * out3
            return out

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load Datasets
print("Loading MNIST, FashionMNIST, and KMNIST datasets...")
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
fashion_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
kmnist_train = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)

mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
fashion_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

# 1. Pre-training a shared base model on joint mixture
print("Pre-training base model on joint mixture...")
base_backbone = CNNBackbone()
base_model = MultiTaskCNN(base_backbone)
base_model.train()

sub_m = Subset(mnist_train, list(range(2000)))
sub_f = Subset(fashion_train, list(range(2000)))
sub_k = Subset(kmnist_train, list(range(2000)))

class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, ds_list):
        self.ds_list = ds_list
        self.lens = [len(ds) for ds in ds_list]
        self.total_len = sum(self.lens)
        
    def __len__(self):
        return self.total_len
        
    def __getitem__(self, idx):
        if idx < self.lens[0]:
            img, label = self.ds_list[0][idx]
            return img, label, 0
        elif idx < self.lens[0] + self.lens[1]:
            img, label = self.ds_list[1][idx - self.lens[0]]
            return img, label, 1
        else:
            img, label = self.ds_list[2][idx - self.lens[0] - self.lens[1]]
            return img, label, 2

pretrain_ds = MultiTaskDataset([sub_m, sub_f, sub_k])
pretrain_loader = DataLoader(pretrain_ds, batch_size=128, shuffle=True)

optimizer = optim.Adam(base_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(2):
    for x, y, tid in pretrain_loader:
        optimizer.zero_grad()
        out = base_model(x, tid)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

base_backbone_state = copy.deepcopy(base_backbone.state_dict())

# 2. Fine-tune expert models sequentially from base model
print("Fine-tuning Expert 1 on MNIST...")
model_mnist = MultiTaskCNN(copy.deepcopy(base_backbone))
model_mnist.load_state_dict(base_model.state_dict())
model_mnist.train()
optimizer = optim.Adam(model_mnist.parameters(), lr=2e-4)
loader_m = DataLoader(Subset(mnist_train, list(range(3000))), batch_size=128, shuffle=True)
for x, y in loader_m:
    optimizer.zero_grad()
    loss = criterion(model_mnist(x, 0), y)
    loss.backward()
    optimizer.step()

mnist_backbone_state = copy.deepcopy(model_mnist.backbone.state_dict())
mnist_head_state = copy.deepcopy(model_mnist.head1.state_dict())

print("Fine-tuning Expert 2 on FashionMNIST...")
model_fashion = MultiTaskCNN(copy.deepcopy(base_backbone))
model_fashion.load_state_dict(base_model.state_dict())
model_fashion.train()
optimizer = optim.Adam(model_fashion.parameters(), lr=2e-4)
loader_f = DataLoader(Subset(fashion_train, list(range(3000))), batch_size=128, shuffle=True)
for x, y in loader_f:
    optimizer.zero_grad()
    loss = criterion(model_fashion(x, 1), y)
    loss.backward()
    optimizer.step()

fashion_backbone_state = copy.deepcopy(model_fashion.backbone.state_dict())
fashion_head_state = copy.deepcopy(model_fashion.head2.state_dict())

print("Fine-tuning Expert 3 on KMNIST...")
model_kmnist = MultiTaskCNN(copy.deepcopy(base_backbone))
model_kmnist.load_state_dict(base_model.state_dict())
model_kmnist.train()
optimizer = optim.Adam(model_kmnist.parameters(), lr=2e-4)
loader_k = DataLoader(Subset(kmnist_train, list(range(3000))), batch_size=128, shuffle=True)
for x, y in loader_k:
    optimizer.zero_grad()
    loss = criterion(model_kmnist(x, 2), y)
    loss.backward()
    optimizer.step()

kmnist_backbone_state = copy.deepcopy(model_kmnist.backbone.state_dict())
kmnist_head_state = copy.deepcopy(model_kmnist.head3.state_dict())

# Extract Task Vectors
task_vector_mnist = {k: mnist_backbone_state[k] - base_backbone_state[k] for k in base_backbone_state}
task_vector_fashion = {k: fashion_backbone_state[k] - base_backbone_state[k] for k in base_backbone_state}
task_vector_kmnist = {k: kmnist_backbone_state[k] - base_backbone_state[k] for k in base_backbone_state}

# Instantiate test-time heads
head_mnist = nn.Linear(32, 10)
head_mnist.load_state_dict(mnist_head_state)
head_mnist.eval()

head_fashion = nn.Linear(32, 10)
head_fashion.load_state_dict(fashion_head_state)
head_fashion.eval()

head_kmnist = nn.Linear(32, 10)
head_kmnist.load_state_dict(kmnist_head_state)
head_kmnist.eval()

# Load subsets of test sets (1000 images each) for fast evaluation
print("Loading subsets of test sets for evaluation...")
mnist_test_sub = Subset(mnist_test, list(range(1000)))
fashion_test_sub = Subset(fashion_test, list(range(1000)))
kmnist_test_sub = Subset(kmnist_test, list(range(1000)))

test_loader_mnist = DataLoader(mnist_test_sub, batch_size=256, shuffle=False)
test_loader_fashion = DataLoader(fashion_test_sub, batch_size=256, shuffle=False)
test_loader_kmnist = DataLoader(kmnist_test_sub, batch_size=256, shuffle=False)

# Layer groups matching the 5 parameter layer blocks:
# conv1, conv2, conv3, fc1, fc2
layer_groups = [
    ["conv1.weight", "conv1.bias"],
    ["conv2.weight", "conv2.bias"],
    ["conv3.weight", "conv3.bias"],
    ["fc1.weight", "fc1.bias"],
    ["fc2.weight", "fc2.bias"]
]

# Differentiable weight reconstruction and forward pass
def functional_forward(x, lambdas):
    reconstructed_state = {}
    
    for g_idx, keys in enumerate(layer_groups):
        for k in keys:
            val = base_backbone_state[k].clone()
            val += lambdas[0, g_idx] * task_vector_mnist[k]
            val += lambdas[1, g_idx] * task_vector_fashion[k]
            val += lambdas[2, g_idx] * task_vector_kmnist[k]
            reconstructed_state[k] = val
            
    # Perform forward pass functionally
    out = F.conv2d(x, reconstructed_state["conv1.weight"], reconstructed_state["conv1.bias"], padding=1)
    out = F.relu(out)
    out = F.max_pool2d(out, 2, 2)
    
    out = F.conv2d(out, reconstructed_state["conv2.weight"], reconstructed_state["conv2.bias"], padding=1)
    out = F.relu(out)
    out = F.max_pool2d(out, 2, 2)
    
    out = F.conv2d(out, reconstructed_state["conv3.weight"], reconstructed_state["conv3.bias"], padding=1)
    out = F.relu(out)
    out = F.max_pool2d(out, 2, 2)
    
    out = out.view(-1, 64 * 3 * 3)
    
    out = F.linear(out, reconstructed_state["fc1.weight"], reconstructed_state["fc1.bias"])
    out = F.relu(out)
    
    out = F.linear(out, reconstructed_state["fc2.weight"], reconstructed_state["fc2.bias"])
    out = F.relu(out)
    
    return out

# Evaluate merged model functionally
def evaluate_merged_cnn(lambdas, noise_scale=0.0):
    # MNIST accuracy
    correct_mnist = 0
    total_mnist = 0
    with torch.no_grad():
        for x, y in test_loader_mnist:
            if noise_scale > 0:
                x = x + torch.randn_like(x) * noise_scale
            features = functional_forward(x, lambdas)
            out = head_mnist(features)
            preds = torch.argmax(out, dim=-1)
            correct_mnist += (preds == y).sum().item()
            total_mnist += len(y)
    acc_mnist = (correct_mnist / total_mnist) * 100.0
    
    # FashionMNIST accuracy
    correct_fashion = 0
    total_fashion = 0
    with torch.no_grad():
        for x, y in test_loader_fashion:
            if noise_scale > 0:
                x = x + torch.randn_like(x) * noise_scale
            features = functional_forward(x, lambdas)
            out = head_fashion(features)
            preds = torch.argmax(out, dim=-1)
            correct_fashion += (preds == y).sum().item()
            total_fashion += len(y)
    acc_fashion = (correct_fashion / total_fashion) * 100.0
    
    # KMNIST accuracy
    correct_kmnist = 0
    total_kmnist = 0
    with torch.no_grad():
        for x, y in test_loader_kmnist:
            if noise_scale > 0:
                x = x + torch.randn_like(x) * noise_scale
            features = functional_forward(x, lambdas)
            out = head_kmnist(features)
            preds = torch.argmax(out, dim=-1)
            correct_kmnist += (preds == y).sum().item()
            total_kmnist += len(y)
    acc_kmnist = (correct_kmnist / total_kmnist) * 100.0
    
    return acc_mnist, acc_fashion, acc_kmnist

# Test-Time Adaptation Setup
def get_tta_batch_cnn(batch_size=126, noise_scale=0.0):
    x_m, _ = next(iter(DataLoader(mnist_test_sub, batch_size=batch_size//3, shuffle=True)))
    x_f, _ = next(iter(DataLoader(fashion_test_sub, batch_size=batch_size//3, shuffle=True)))
    x_k, _ = next(iter(DataLoader(kmnist_test_sub, batch_size=batch_size//3, shuffle=True)))
    x = torch.cat([x_m, x_f, x_k], dim=0)
    if noise_scale > 0:
        x = x + torch.randn_like(x) * noise_scale
    return x

def entropy_loss(logits):
    probs = torch.softmax(logits, dim=-1)
    return -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))

# Unconstrained AdaMerging (FO)
def run_adamerging_fo(x_batch, steps=100, lr=0.05):
    lambdas = torch.full((3, 5), 0.3, requires_grad=True)
    optimizer = optim.Adam([lambdas], lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()
        features = functional_forward(x_batch, lambdas)
        
        split = len(x_batch) // 3
        logits1 = head_mnist(features[:split])
        logits2 = head_fashion(features[split:2*split])
        logits3 = head_kmnist(features[2*split:])
        
        loss = entropy_loss(logits1) + entropy_loss(logits2) + entropy_loss(logits3)
        loss.backward()
        optimizer.step()
        
    return lambdas.detach().numpy()

# PolyMerge (FO with quadratic polynomial subspace)
def poly_reconstruct(w):
    lambdas = torch.zeros((3, 5))
    for l in range(5):
        norm_l = l / 4.0
        lambdas[:, l] = w[:, 0] + w[:, 1] * norm_l + w[:, 2] * (norm_l ** 2)
    return lambdas

def run_poly_fo(x_batch, steps=100, lr=0.05):
    w = torch.full((3, 3), 0.3, requires_grad=True)
    optimizer = optim.Adam([w], lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()
        lambdas = poly_reconstruct(w)
        features = functional_forward(x_batch, lambdas)
        
        split = len(x_batch) // 3
        logits1 = head_mnist(features[:split])
        logits2 = head_fashion(features[split:2*split])
        logits3 = head_kmnist(features[2*split:])
        
        loss = entropy_loss(logits1) + entropy_loss(logits2) + entropy_loss(logits3)
        loss.backward()
        optimizer.step()
        
    return poly_reconstruct(w).detach().numpy()

# ZO-FlatMerge (Ours: Zeroth-Order randomized smoothing in polynomial subspace)
def run_zo_flatmerge(x_batch, steps=100, lr=0.05, sigma=0.05, num_samples=10):
    w = torch.full((3, 3), 0.3)
    
    for step in range(steps):
        grad_estimate = torch.zeros_like(w)
        with torch.no_grad():
            for _ in range(num_samples):
                # Sample direction and normalize to unit vector
                E = torch.randn_like(w)
                U = E / (torch.norm(E) + 1e-12)
                
                # Positive loss
                lambdas_pos = poly_reconstruct(w + sigma * U)
                features_pos = functional_forward(x_batch, lambdas_pos)
                split = len(x_batch) // 3
                loss_pos = entropy_loss(head_mnist(features_pos[:split])) + \
                           entropy_loss(head_fashion(features_pos[split:2*split])) + \
                           entropy_loss(head_kmnist(features_pos[2*split:]))
                
                # Negative loss
                lambdas_neg = poly_reconstruct(w - sigma * U)
                features_neg = functional_forward(x_batch, lambdas_neg)
                loss_neg = entropy_loss(head_mnist(features_neg[:split])) + \
                           entropy_loss(head_fashion(features_neg[split:2*split])) + \
                           entropy_loss(head_kmnist(features_neg[2*split:]))
                
                grad_estimate += ((loss_pos - loss_neg) / (2.0 * sigma)) * U
                
            grad_estimate /= num_samples
        w -= lr * grad_estimate
        
    return poly_reconstruct(w).detach().numpy()

# 4. Sweep Noise Scales
noise_scales = [0.0, 1.0, 2.0, 3.0]
results = []

for ns in noise_scales:
    print(f"\nEvaluating Noise Scale (Gamma) = {ns:.1f}")
    
    # Task Arithmetic (Uniform blending of 0.3)
    ta_lambdas = torch.full((3, 5), 0.3)
    ta_mnist, ta_fashion, ta_kmnist = evaluate_merged_cnn(ta_lambdas, noise_scale=ns)
    ta_joint = (ta_mnist + ta_fashion + ta_kmnist) / 3.0
    print(f"  Task Arithmetic: MNIST={ta_mnist:.2f}%, Fashion={ta_fashion:.2f}%, KMNIST={ta_kmnist:.2f}%, Joint={ta_joint:.2f}%")
    
    # Get TTA Batch
    x_batch = get_tta_batch_cnn(batch_size=126, noise_scale=ns)
    
    # AdaMerging (FO)
    ada_lambdas = run_adamerging_fo(x_batch, steps=100, lr=0.05)
    ada_mnist, ada_fashion, ada_kmnist = evaluate_merged_cnn(torch.tensor(ada_lambdas), noise_scale=ns)
    ada_joint = (ada_mnist + ada_fashion + ada_kmnist) / 3.0
    print(f"  AdaMerging (FO): MNIST={ada_mnist:.2f}%, Fashion={ada_fashion:.2f}%, KMNIST={ada_kmnist:.2f}%, Joint={ada_joint:.2f}%")
    
    # PolyMerge (FO d=2)
    poly_lambdas = run_poly_fo(x_batch, steps=100, lr=0.05)
    poly_mnist, poly_fashion, poly_kmnist = evaluate_merged_cnn(torch.tensor(poly_lambdas), noise_scale=ns)
    poly_joint = (poly_mnist + poly_fashion + poly_kmnist) / 3.0
    print(f"  PolyMerge (FO d=2): MNIST={poly_mnist:.2f}%, Fashion={poly_fashion:.2f}%, KMNIST={poly_kmnist:.2f}%, Joint={poly_joint:.2f}%")
    
    # ZO-FlatMerge (Ours)
    zo_lambdas = run_zo_flatmerge(x_batch, steps=100, lr=0.05, sigma=0.05, num_samples=10)
    zo_mnist, zo_fashion, zo_kmnist = evaluate_merged_cnn(torch.tensor(zo_lambdas), noise_scale=ns)
    zo_joint = (zo_mnist + zo_fashion + zo_kmnist) / 3.0
    print(f"  ZO-FlatMerge (Ours): MNIST={zo_mnist:.2f}%, Fashion={zo_fashion:.2f}%, KMNIST={zo_kmnist:.2f}%, Joint={zo_joint:.2f}%")
    
    results.append({
        "noise_scale": ns,
        "ta": [ta_mnist, ta_fashion, ta_kmnist, ta_joint],
        "ada": [ada_mnist, ada_fashion, ada_kmnist, ada_joint],
        "poly": [poly_mnist, poly_fashion, poly_kmnist, poly_joint],
        "zo": [zo_mnist, zo_fashion, zo_kmnist, zo_joint]
    })

# Write results to results/real_cnn_metrics.json
os.makedirs("results", exist_ok=True)
with open("results/real_cnn_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved 5-Layer CNN real-world validation metrics to results/real_cnn_metrics.json")
print("=== Real-World Vision Model Merging Experiment Finished Successfully! ===")
