import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import timm
import time
import numpy as np
import math

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("="*60)
print("REAL-WORLD LORA ADAPTER ENSEMBLING WITH PEAR")
print("="*60)

# 1. Load Pre-trained ViT-Tiny and Freeze Backbone
device = torch.device("cpu")
vit = timm.create_model('vit_tiny_patch16_224', pretrained=True)
for param in vit.parameters():
    param.requires_grad = False

# Helper function to extract features for routing
def extract_vit_features(model, x, layer=2):
    with torch.no_grad():
        # Layer 0: Patch Embedding
        x_embed = model.patch_embed(x)  # (B, N, D)
        if layer == 0:
            return x_embed.mean(dim=1)
        
        # Positional Embedding and prefix tokens
        x_pos = model._pos_embed(x_embed)
        x_drop = model.patch_drop(x_pos)
        x_norm = model.norm_pre(x_drop)
        
        # Layer 1: Block 0
        x1 = model.blocks[0](x_norm)
        if layer == 1:
            return x1[:, 1:, :].mean(dim=1)
            
        # Layer 2: Block 1
        x2 = model.blocks[1](x1)
        if layer == 2:
            return x2[:, 1:, :].mean(dim=1)
    return None

# 2. Setup Data Loading for MNIST and CIFAR-10
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

# Load datasets
ds_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray)
ds_cifar = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb)

# Sample 64 calibration and 64 test samples
N_cal = 64
N_test = 64

idx_mnist = np.random.choice(len(ds_mnist), N_cal + N_test, replace=False)
idx_cifar = np.random.choice(len(ds_cifar), N_cal + N_test, replace=False)

mnist_cal = Subset(ds_mnist, idx_mnist[:N_cal])
mnist_test = Subset(ds_mnist, idx_mnist[N_cal:])

cifar_cal = Subset(ds_cifar, idx_cifar[:N_cal])
cifar_test = Subset(ds_cifar, idx_cifar[N_cal:])

# 3. Formulate manual LoRA patch for Block 11 Attn QKV
class LoRAWrappedQKV(nn.Module):
    def __init__(self, original_qkv, r=8):
        super().__init__()
        self.original_qkv = original_qkv
        self.r = r
        
        # MNIST adapter (rank r)
        self.lora_A_mnist = nn.Parameter(torch.zeros(r, 192))
        self.lora_B_mnist = nn.Parameter(torch.zeros(576, r))
        
        # CIFAR-10 adapter (rank r)
        self.lora_A_cifar = nn.Parameter(torch.zeros(r, 192))
        self.lora_B_cifar = nn.Parameter(torch.zeros(576, r))
        
        # Initialize (A: Kaiming, B: Zero)
        nn.init.kaiming_uniform_(self.lora_A_mnist, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_cifar, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_mnist)
        nn.init.zeros_(self.lora_B_cifar)
        
        # Blending coefficients (dynamic per-sample)
        # Shape: (B,)
        self.alpha_mnist = None
        self.alpha_cifar = None
        
    def forward(self, x):
        # Base forward pass
        out = self.original_qkv(x)
        B, N, _ = x.shape
        
        # Compute LoRA updates
        if self.alpha_mnist is not None:
            # Broadcast alpha to (B, 1, 1)
            a_m = self.alpha_mnist.view(B, 1, 1)
            mnist_up = (x @ self.lora_A_mnist.t()) @ self.lora_B_mnist.t()
            out = out + a_m * mnist_up
            
        if self.alpha_cifar is not None:
            a_c = self.alpha_cifar.view(B, 1, 1)
            cifar_up = (x @ self.lora_A_cifar.t()) @ self.lora_B_cifar.t()
            out = out + a_c * cifar_up
            
        return out

# Replace the original QKV in Block 11 with our wrapped QKV
original_qkv = vit.blocks[11].attn.qkv
wrapped_qkv = LoRAWrappedQKV(original_qkv, r=8)
vit.blocks[11].attn.qkv = wrapped_qkv

# Create task classification heads
head_mnist = nn.Linear(192, 10)
head_cifar = nn.Linear(192, 10)

# 4. Calibrate PEAR Centroids and Dispersion on Layer 2 (Early-Layer Routing Compromise)
print("\nExtracting calibration features for PEAR on Layer 2...")
dl_mnist_cal = DataLoader(mnist_cal, batch_size=N_cal, shuffle=False)
dl_cifar_cal = DataLoader(cifar_cal, batch_size=N_cal, shuffle=False)

x_mnist_cal, _ = next(iter(dl_mnist_cal))
x_cifar_cal, _ = next(iter(dl_cifar_cal))

feats_mnist_cal = extract_vit_features(vit, x_mnist_cal, layer=2)
feats_cifar_cal = extract_vit_features(vit, x_cifar_cal, layer=2)

centroid_mnist = feats_mnist_cal.mean(dim=0)
centroid_cifar = feats_cifar_cal.mean(dim=0)

# Compute dispersion scales (IDC)
sims_mnist = (feats_mnist_cal @ centroid_mnist) / (feats_mnist_cal.norm(dim=1) * centroid_mnist.norm() + 1e-8)
disp_mnist = sims_mnist.mean().item()

sims_cifar = (feats_cifar_cal @ centroid_cifar) / (feats_cifar_cal.norm(dim=1) * centroid_cifar.norm() + 1e-8)
disp_cifar = sims_cifar.mean().item()

print(f"MNIST Centroid Norm: {centroid_mnist.norm().item():.4f} | Dispersion: {disp_mnist:.4f}")
print(f"CIFAR Centroid Norm: {centroid_cifar.norm().item():.4f} | Dispersion: {disp_cifar:.4f}")

# Helper to run PEAR routing on Layer 2 features
def compute_pear_weights(feats, tau=0.05):
    # MNIST similarity
    norm_feats = feats.norm(p=2, dim=1)
    
    sim_mnist = (feats @ centroid_mnist) / (norm_feats * centroid_mnist.norm() + 1e-8)
    sim_mnist_cal = sim_mnist / disp_mnist
    
    # CIFAR similarity
    sim_cifar = (feats @ centroid_cifar) / (norm_feats * centroid_cifar.norm() + 1e-8)
    sim_cifar_cal = sim_cifar / disp_cifar
    
    # Softmax
    scores = torch.stack([sim_mnist_cal, sim_cifar_cal], dim=1)  # (B, 2)
    weights = torch.softmax(scores / tau, dim=1)
    
    return weights[:, 0], weights[:, 1]  # alpha_mnist, alpha_cifar

# 5. Train Task-Specific LoRA Adapters and Heads
print("\nTraining MNIST Task-Specific Adapter and Head...")
optimizer_mnist = optim.Adam([
    {'params': [wrapped_qkv.lora_A_mnist, wrapped_qkv.lora_B_mnist]},
    {'params': head_mnist.parameters()}
], lr=0.01)

criterion = nn.CrossEntropyLoss()

dl_mnist_train = DataLoader(mnist_cal, batch_size=16, shuffle=True)
for epoch in range(15):
    mnist_loss = 0.0
    correct = 0
    total = 0
    for x_b, y_b in dl_mnist_train:
        # Set QKV routing to pure MNIST
        wrapped_qkv.alpha_mnist = torch.ones(x_b.shape[0])
        wrapped_qkv.alpha_cifar = torch.zeros(x_b.shape[0])
        
        optimizer_mnist.zero_grad()
        features = vit.forward_features(x_b)
        z = vit.forward_head(features, pre_logits=True)
        logits = head_mnist(z)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer_mnist.step()
        
        mnist_loss += loss.item() * x_b.shape[0]
        correct += (logits.argmax(dim=1) == y_b).sum().item()
        total += x_b.shape[0]
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/15 | Loss: {mnist_loss/total:.4f} | Accuracy: {correct/total*100:.2f}%")

print("\nTraining CIFAR-10 Task-Specific Adapter and Head...")
optimizer_cifar = optim.Adam([
    {'params': [wrapped_qkv.lora_A_cifar, wrapped_qkv.lora_B_cifar]},
    {'params': head_cifar.parameters()}
], lr=0.01)

dl_cifar_train = DataLoader(cifar_cal, batch_size=16, shuffle=True)
for epoch in range(15):
    cifar_loss = 0.0
    correct = 0
    total = 0
    for x_b, y_b in dl_cifar_train:
        # Set QKV routing to pure CIFAR
        wrapped_qkv.alpha_mnist = torch.zeros(x_b.shape[0])
        wrapped_qkv.alpha_cifar = torch.ones(x_b.shape[0])
        
        optimizer_cifar.zero_grad()
        features = vit.forward_features(x_b)
        z = vit.forward_head(features, pre_logits=True)
        logits = head_cifar(z)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer_cifar.step()
        
        cifar_loss += loss.item() * x_b.shape[0]
        correct += (logits.argmax(dim=1) == y_b).sum().item()
        total += x_b.shape[0]
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/15 | Loss: {cifar_loss/total:.4f} | Accuracy: {correct/total*100:.2f}%")

# 6. Evaluation on Heterogeneous Test Set
print("\nPreparing Heterogeneous Test Set (64 MNIST + 64 CIFAR-10)...")
dl_mnist_test = DataLoader(mnist_test, batch_size=N_test, shuffle=False)
dl_cifar_test = DataLoader(cifar_test, batch_size=N_test, shuffle=False)

x_mnist_test, y_mnist_test = next(iter(dl_mnist_test))
x_cifar_test, y_cifar_test = next(iter(dl_cifar_test))

# Create joint evaluation batch
x_test = torch.cat([x_mnist_test, x_cifar_test], dim=0)
y_test = torch.cat([y_mnist_test, y_cifar_test], dim=0)
task_labels = torch.cat([torch.zeros(N_test), torch.ones(N_test)], dim=0) # 0: MNIST, 1: CIFAR-10

print("Evaluating 4 ensembling methods...")

# Pre-extract Layer 2 features for PEAR
feats_test_l2 = extract_vit_features(vit, x_test, layer=2)
alpha_mnist_pear, alpha_cifar_pear = compute_pear_weights(feats_test_l2, tau=0.05)

results = {}

# --- Method A: Static Uniform Merging ---
with torch.no_grad():
    wrapped_qkv.alpha_mnist = torch.full((x_test.shape[0],), 0.5)
    wrapped_qkv.alpha_cifar = torch.full((x_test.shape[0],), 0.5)
    
    features = vit.forward_features(x_test)
    z = vit.forward_head(features, pre_logits=True)
    
    logits_m = head_mnist(z)
    logits_c = head_cifar(z)
    final_logits = 0.5 * logits_m + 0.5 * logits_c
    
    # Calculate classification accuracy
    # For MNIST samples (0 to N_test-1), check if argmax(logits_m) is correct
    # For CIFAR samples (N_test to end), check if argmax(logits_c) is correct
    pred_m = logits_m.argmax(dim=1)
    pred_c = logits_c.argmax(dim=1)
    
    correct_mnist = (pred_m[:N_test] == y_test[:N_test]).sum().item()
    correct_cifar = (pred_c[N_test:] == y_test[N_test:]).sum().item()
    
    results["Static Uniform"] = {
        "MNIST": correct_mnist / N_test * 100,
        "CIFAR-10": correct_cifar / N_test * 100,
        "Joint Mean": (correct_mnist + correct_cifar) / (2 * N_test) * 100
    }

# --- Method B: SABLE SOTA (Late Adaptation) ---
# Leaves Block 11 unadapted (alpha_mnist=0, alpha_cifar=0) and ensembles head predictions
with torch.no_grad():
    wrapped_qkv.alpha_mnist = torch.zeros(x_test.shape[0])
    wrapped_qkv.alpha_cifar = torch.zeros(x_test.shape[0])
    
    features = vit.forward_features(x_test)
    z = vit.forward_head(features, pre_logits=True)
    
    logits_m = head_mnist(z)
    logits_c = head_cifar(z)
    
    pred_m = logits_m.argmax(dim=1)
    pred_c = logits_c.argmax(dim=1)
    
    correct_mnist = (pred_m[:N_test] == y_test[:N_test]).sum().item()
    correct_cifar = (pred_c[N_test:] == y_test[N_test:]).sum().item()
    
    results["SABLE SOTA (Late Adapt)"] = {
        "MNIST": correct_mnist / N_test * 100,
        "CIFAR-10": correct_cifar / N_test * 100,
        "Joint Mean": (correct_mnist + correct_cifar) / (2 * N_test) * 100
    }

# --- Method C: PEAR (Ours) ---
with torch.no_grad():
    wrapped_qkv.alpha_mnist = alpha_mnist_pear
    wrapped_qkv.alpha_cifar = alpha_cifar_pear
    
    features = vit.forward_features(x_test)
    z = vit.forward_head(features, pre_logits=True)
    
    logits_m = head_mnist(z)
    logits_c = head_cifar(z)
    
    pred_m = logits_m.argmax(dim=1)
    pred_c = logits_c.argmax(dim=1)
    
    correct_mnist = (pred_m[:N_test] == y_test[:N_test]).sum().item()
    correct_cifar = (pred_c[N_test:] == y_test[N_test:]).sum().item()
    
    results["PEAR (Ours)"] = {
        "MNIST": correct_mnist / N_test * 100,
        "CIFAR-10": correct_cifar / N_test * 100,
        "Joint Mean": (correct_mnist + correct_cifar) / (2 * N_test) * 100
    }

# --- Method D: Expert Ceiling (Perfect Routing) ---
with torch.no_grad():
    # MNIST samples: pure MNIST LoRA
    wrapped_qkv.alpha_mnist = torch.cat([torch.ones(N_test), torch.zeros(N_test)], dim=0)
    wrapped_qkv.alpha_cifar = torch.cat([torch.zeros(N_test), torch.ones(N_test)], dim=0)
    
    features = vit.forward_features(x_test)
    z = vit.forward_head(features, pre_logits=True)
    
    logits_m = head_mnist(z)
    logits_c = head_cifar(z)
    
    pred_m = logits_m.argmax(dim=1)
    pred_c = logits_c.argmax(dim=1)
    
    correct_mnist = (pred_m[:N_test] == y_test[:N_test]).sum().item()
    correct_cifar = (pred_c[N_test:] == y_test[N_test:]).sum().item()
    
    results["Expert Ceiling"] = {
        "MNIST": correct_mnist / N_test * 100,
        "CIFAR-10": correct_cifar / N_test * 100,
        "Joint Mean": (correct_mnist + correct_cifar) / (2 * N_test) * 100
    }

# Format and display table
print("\n" + "="*60)
print(f"{'REAL-WORLD MULTI-TASK LORA CLASSIFICATION RESULTS':^60}")
print("="*60)
print(f"{'Method':28} | {'MNIST Acc (%)':13} | {'CIFAR Acc (%)':13} | {'Joint Mean (%)':13}")
print("-"*60)
for method in ["Static Uniform", "SABLE SOTA (Late Adapt)", "PEAR (Ours)", "Expert Ceiling"]:
    res = results[method]
    print(f"{method:28} | {res['MNIST']:11.2f}% | {res['CIFAR-10']:11.2f}% | {res['Joint Mean']:11.2f}%")
print("="*60)

# Save results for use in the paper's LaTeX
with open("real_world_lora_results.txt", "w") as f:
    f.write(f"Uniform Joint: {results['Static Uniform']['Joint Mean']:.2f}%\n")
    f.write(f"Uniform MNIST: {results['Static Uniform']['MNIST']:.2f}%\n")
    f.write(f"Uniform CIFAR: {results['Static Uniform']['CIFAR-10']:.2f}%\n")
    f.write(f"SABLE Joint: {results['SABLE SOTA (Late Adapt)']['Joint Mean']:.2f}%\n")
    f.write(f"SABLE MNIST: {results['SABLE SOTA (Late Adapt)']['MNIST']:.2f}%\n")
    f.write(f"SABLE CIFAR: {results['SABLE SOTA (Late Adapt)']['CIFAR-10']:.2f}%\n")
    f.write(f"PEAR Joint: {results['PEAR (Ours)']['Joint Mean']:.2f}%\n")
    f.write(f"PEAR MNIST: {results['PEAR (Ours)']['MNIST']:.2f}%\n")
    f.write(f"PEAR CIFAR: {results['PEAR (Ours)']['CIFAR-10']:.2f}%\n")
    f.write(f"Ceiling Joint: {results['Expert Ceiling']['Joint Mean']:.2f}%\n")
    f.write(f"Ceiling MNIST: {results['Expert Ceiling']['MNIST']:.2f}%\n")
    f.write(f"Ceiling CIFAR: {results['Expert Ceiling']['CIFAR-10']:.2f}%\n")

print("\nSuccessfully saved real-world LoRA results to 'real_world_lora_results.txt'")
