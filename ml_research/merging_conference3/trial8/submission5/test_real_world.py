import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import timm
import time
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Initializing Real-World ViT Routing Evaluation...")

# 1. Load Pre-trained ViT-Tiny
device = torch.device("cpu")
vit = timm.create_model('vit_tiny_patch16_224', pretrained=True)
vit.eval()

# Helper function to extract Layer 0, Layer 1, and Layer 2 features
def extract_vit_features(model, x, layer=0):
    with torch.no_grad():
        # Layer 0: Patch Embedding
        x_embed = model.patch_embed(x)  # (B, N, D)
        if layer == 0:
            # Spatial average pooling
            return x_embed.mean(dim=1)
        
        # Positional Embedding and prefix tokens
        x_pos = model._pos_embed(x_embed)
        x_drop = model.patch_drop(x_pos)
        x_norm = model.norm_pre(x_drop)
        
        # Layer 1: Block 0
        x1 = model.blocks[0](x_norm)
        if layer == 1:
            # Exclude prefix tokens (cls_token) for pure spatial mean
            return x1[:, 1:, :].mean(dim=1)
            
        # Layer 2: Block 1
        x2 = model.blocks[1](x1)
        if layer == 2:
            return x2[:, 1:, :].mean(dim=1)
            
    return None

# 2. Setup Data Loading for 4 Tasks
# MNIST, FashionMNIST, CIFAR10, SVHN
transform_gray = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# For Lightweight CNN Router, we'll use 32x32 images
transform_gray_32 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_rgb_32 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load datasets
datasets_dict = {
    "MNIST": (datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray),
              datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray_32)),
    "Fashion-MNIST": (datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray),
                     datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray_32)),
    "CIFAR-10": (datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb),
                 datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb_32)),
    "SVHN": (datasets.SVHN(root='./data', split='test', download=True, transform=transform_rgb),
             datasets.SVHN(root='./data', split='test', download=True, transform=transform_rgb_32))
}

# Create subsets: 64 for calibration, 64 for testing
N_cal = 64
N_test = 64

task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10", "SVHN"]
data_splits = {}

for name in task_names:
    ds_224, ds_32 = datasets_dict[name]
    
    # Select indices
    indices = np.random.choice(len(ds_224), N_cal + N_test, replace=False)
    cal_idx = indices[:N_cal]
    test_idx = indices[N_cal:]
    
    cal_subset_224 = Subset(ds_224, cal_idx)
    test_subset_224 = Subset(ds_224, test_idx)
    
    cal_subset_32 = Subset(ds_32, cal_idx)
    test_subset_32 = Subset(ds_32, test_idx)
    
    data_splits[name] = {
        "cal_224": cal_subset_224,
        "test_224": test_subset_224,
        "cal_32": cal_subset_32,
        "test_32": test_subset_32
    }

print("Loaded subsets for 4 tasks. Starting feature extraction for PEAR calibration...")

# Extract Calibration Features & Compute Centroids
centroids = {layer: {} for layer in [0, 1, 2]}
dispersion = {layer: {} for layer in [0, 1, 2]}

for layer in [0, 1, 2]:
    print(f"--- Calibrating PEAR on Layer {layer} ---")
    for k, name in enumerate(task_names):
        subset = data_splits[name]["cal_224"]
        dl = DataLoader(subset, batch_size=N_cal, shuffle=False)
        x_batch, _ = next(iter(dl))
        
        # Extract features
        feats = extract_vit_features(vit, x_batch, layer=layer)  # (N_cal, D)
        
        # Compute Zero-Shot Centroid (ZPC)
        centroid = feats.mean(dim=0)  # (D,)
        centroids[layer][name] = centroid
        
        # Compute Intra-Task Dispersion Calibration (IDC)
        # Cosine similarity to centroid
        denom_feat = feats.norm(p=2, dim=1)
        denom_cent = centroid.norm(p=2)
        sims = (feats @ centroid) / (denom_feat * denom_cent + 1e-8)
        disp = sims.mean().item()
        dispersion[layer][name] = disp
        
        print(f"Task: {name:15} Centroid Norm: {centroid.norm().item():.4f} | Dispersion Scale: {disp:.4f}")

# Setup Lightweight Pre-Backbone Classifier (3-Layer CNN)
class TinyCNNRouter(nn.Module):
    def __init__(self, num_classes=4):
        super(TinyCNNRouter, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Prepare Training Data for TinyCNNRouter (from 64 calibration samples per task)
train_x_list = []
train_y_list = []
for k, name in enumerate(task_names):
    dl = DataLoader(data_splits[name]["cal_32"], batch_size=N_cal, shuffle=False)
    x_b, _ = next(iter(dl))
    train_x_list.append(x_b)
    train_y_list.append(torch.full((N_cal,), k, dtype=torch.long))

train_x = torch.cat(train_x_list, dim=0)
train_y = torch.cat(train_y_list, dim=0)

cnn_router = TinyCNNRouter(num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_router.parameters(), lr=0.005)

print("\nTraining Lightweight Pre-Backbone CNN Router...")
cnn_router.train()
for epoch in range(30):
    optimizer.zero_grad()
    outputs = cnn_router(train_x)
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        acc = (outputs.argmax(dim=1) == train_y).float().mean().item() * 100
        print(f"Epoch {epoch+1:2d}/30 | Loss: {loss.item():.4f} | Training Acc: {acc:.2f}%")
cnn_router.eval()

# 3. Evaluation: Test PEAR and CNN Router on 4 Tasks
print("\nEvaluating Routing Accuracy on Test Sets...")

pear_accuracies = {layer: {name: 0.0 for name in task_names} for layer in [0, 1, 2]}
cnn_accuracies = {name: 0.0 for name in task_names}

for k, name in enumerate(task_names):
    # Test PEAR on 224x224
    dl_224 = DataLoader(data_splits[name]["test_224"], batch_size=N_test, shuffle=False)
    x_test_224, _ = next(iter(dl_224))
    
    # Test CNN on 32x32
    dl_32 = DataLoader(data_splits[name]["test_32"], batch_size=N_test, shuffle=False)
    x_test_32, _ = next(iter(dl_32))
    
    # Evaluate PEAR Layer 0, 1, 2
    for layer in [0, 1, 2]:
        feats = extract_vit_features(vit, x_test_224, layer=layer)  # (N_test, D)
        
        # Compute calibrated similarities for all tasks
        all_cal_sims = []
        for target_name in task_names:
            centroid = centroids[layer][target_name]
            disp = dispersion[layer][target_name]
            
            denom_feat = feats.norm(p=2, dim=1)
            denom_cent = centroid.norm(p=2)
            raw_sims = (feats @ centroid) / (denom_feat * denom_cent + 1e-8)
            
            cal_sims = raw_sims / disp
            all_cal_sims.append(cal_sims.unsqueeze(1))
            
        all_cal_sims = torch.cat(all_cal_sims, dim=1)  # (N_test, K=4)
        predictions = all_cal_sims.argmax(dim=1)
        
        correct = (predictions == k).sum().item()
        pear_accuracies[layer][name] = (correct / N_test) * 100
        
    # Evaluate CNN Router
    with torch.no_grad():
        cnn_out = cnn_router(x_test_32)
        cnn_pred = cnn_out.argmax(dim=1)
        cnn_correct = (cnn_pred == k).sum().item()
        cnn_accuracies[name] = (cnn_correct / N_test) * 100

# Format results
print("\n" + "="*50)
print(f"{'Routing Router Accuracy Summary (%)':^50}")
print("="*50)
print(f"{'Task':15} | {'PEAR L0':10} | {'PEAR L1':10} | {'PEAR L2':10} | {'Tiny CNN':10}")
print("-"*50)
for name in task_names:
    print(f"{name:15} | {pear_accuracies[0][name]:9.2f}% | {pear_accuracies[1][name]:9.2f}% | {pear_accuracies[2][name]:9.2f}% | {cnn_accuracies[name]:9.2f}%")
print("-"*50)

# Compute Joint Mean Accuracies
mean_pear_l0 = np.mean([pear_accuracies[0][n] for n in task_names])
mean_pear_l1 = np.mean([pear_accuracies[1][n] for n in task_names])
mean_pear_l2 = np.mean([pear_accuracies[2][n] for n in task_names])
mean_cnn = np.mean([cnn_accuracies[n] for n in task_names])
print(f"{'Joint Mean':15} | {mean_pear_l0:9.2f}% | {mean_pear_l1:9.2f}% | {mean_pear_l2:9.2f}% | {mean_cnn:9.2f}%")
print("="*50)

# 4. Latency Measurements (Single-sample forward, run 100 times to get stable mean)
print("\nMeasuring Single-Sample Processing Latency...")

single_x_224 = torch.randn(1, 3, 224, 224)
single_x_32 = torch.randn(1, 3, 32, 32)

# PEAR Layer 0
t_start = time.perf_counter()
for _ in range(100):
    _ = extract_vit_features(vit, single_x_224, layer=0)
t_end = time.perf_counter()
latency_l0 = ((t_end - t_start) / 100) * 1000  # ms

# PEAR Layer 1
t_start = time.perf_counter()
for _ in range(100):
    _ = extract_vit_features(vit, single_x_224, layer=1)
t_end = time.perf_counter()
latency_l1 = ((t_end - t_start) / 100) * 1000  # ms

# PEAR Layer 2
t_start = time.perf_counter()
for _ in range(100):
    _ = extract_vit_features(vit, single_x_224, layer=2)
t_end = time.perf_counter()
latency_l2 = ((t_end - t_start) / 100) * 1000  # ms

# Tiny CNN Router
t_start = time.perf_counter()
with torch.no_grad():
    for _ in range(100):
        _ = cnn_router(single_x_32)
t_end = time.perf_counter()
latency_cnn = ((t_end - t_start) / 100) * 1000  # ms

# Base ViT Full Pass
t_start = time.perf_counter()
with torch.no_grad():
    for _ in range(100):
        _ = vit(single_x_224)
t_end = time.perf_counter()
latency_vit_full = ((t_end - t_start) / 100) * 1000  # ms

print(f"PEAR L0 (Patch Embedding) Latency: {latency_l0:.4f} ms")
print(f"PEAR L1 (Block 0 Output) Latency:  {latency_l1:.4f} ms")
print(f"PEAR L2 (Block 1 Output) Latency:  {latency_l2:.4f} ms")
print(f"Tiny CNN Router (32x32 Input) Latency: {latency_cnn:.4f} ms")
print(f"Base ViT-Tiny Full Pass Latency:    {latency_vit_full:.4f} ms")

print("\nRouting Compute Overhead relative to Base ViT Full Pass:")
print(f"PEAR L0 Overhead: {latency_l0 / latency_vit_full * 100:.2f}%")
print(f"PEAR L1 Overhead: {latency_l1 / latency_vit_full * 100:.2f}%")
print(f"PEAR L2 Overhead: {latency_l2 / latency_vit_full * 100:.2f}%")
print(f"Tiny CNN Overhead: {latency_cnn / latency_vit_full * 100:.2f}%")

# Save results for LaTeX inclusion
with open("real_world_results.txt", "w") as f:
    f.write(f"PEAR L0 Joint Mean: {mean_pear_l0:.2f}%\n")
    f.write(f"PEAR L1 Joint Mean: {mean_pear_l1:.2f}%\n")
    f.write(f"PEAR L2 Joint Mean: {mean_pear_l2:.2f}%\n")
    f.write(f"Tiny CNN Joint Mean: {mean_cnn:.2f}%\n")
    for name in task_names:
        f.write(f"{name} PEAR L0: {pear_accuracies[0][name]:.2f}%\n")
        f.write(f"{name} PEAR L1: {pear_accuracies[1][name]:.2f}%\n")
        f.write(f"{name} PEAR L2: {pear_accuracies[2][name]:.2f}%\n")
        f.write(f"{name} Tiny CNN: {cnn_accuracies[name]:.2f}%\n")
    f.write(f"Latency PEAR L0: {latency_l0:.4f} ms\n")
    f.write(f"Latency PEAR L1: {latency_l1:.4f} ms\n")
    f.write(f"Latency PEAR L2: {latency_l2:.4f} ms\n")
    f.write(f"Latency Tiny CNN: {latency_cnn:.4f} ms\n")
    f.write(f"Latency ViT Full: {latency_vit_full:.4f} ms\n")

print("\nSuccessfully finished evaluation and saved results to 'real_world_results.txt'")
