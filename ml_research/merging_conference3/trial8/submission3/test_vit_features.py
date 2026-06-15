import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import timm
import numpy as np

# Set seed
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")
print("Loading pre-trained ViT-Tiny...")
model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
model.eval()

# Helper to resize and replicate gray to 3-channel
transform_gray = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_color = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading datasets (test splits)...")
# Loader subsets for speed
subset_size = 100
mnist = datasets.MNIST(root='./data', train=False, download=False, transform=transform_gray)
fmnist = datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_gray)
cifar = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_color)
svhn = datasets.SVHN(root='./data', split='test', download=False, transform=transform_color)

mnist_loader = DataLoader(Subset(mnist, list(range(subset_size))), batch_size=subset_size)
fmnist_loader = DataLoader(Subset(fmnist, list(range(subset_size))), batch_size=subset_size)
cifar_loader = DataLoader(Subset(cifar, list(range(subset_size))), batch_size=subset_size)
svhn_loader = DataLoader(Subset(svhn, list(range(subset_size))), batch_size=subset_size)

# Extract Layer 3 features
# In timm ViT-Tiny: patch_embed -> blocks (0 to 11) -> norm -> head
# Block 3 output is blocks[2] output
def extract_layer3_features(x):
    # Pass through patch_embed and pos_embed
    x = model.patch_embed(x)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    x = model.norm_pre(x)
    # Pass through first 3 blocks (idx 0, 1, 2)
    for i in range(3):
        x = model.blocks[i](x)
    # Pool class token representation (index 0)
    return x[:, 0] # Class token (B, D)

print("Extracting features...")
features = {}
mnist_feats = extract_layer3_features(next(iter(mnist_loader))[0])
fmnist_feats = extract_layer3_features(next(iter(fmnist_loader))[0])
cifar_feats = extract_layer3_features(next(iter(cifar_loader))[0])
svhn_feats = extract_layer3_features(next(iter(svhn_loader))[0])

print(f"MNIST feature shape: {mnist_feats.shape}")
print(f"Fashion-MNIST feature shape: {fmnist_feats.shape}")
print(f"CIFAR-10 feature shape: {cifar_feats.shape}")
print(f"SVHN feature shape: {svhn_feats.shape}")

# Compute Centroids (using first 50 samples as calibration, rest as test)
calib_size = 50
centroids = {
    0: mnist_feats[:calib_size].mean(dim=0),
    1: fmnist_feats[:calib_size].mean(dim=0),
    2: cifar_feats[:calib_size].mean(dim=0),
    3: svhn_feats[:calib_size].mean(dim=0),
}

# Test dynamic routing (Q-ZCA)
test_feats = [
    (mnist_feats[calib_size:], 0),
    (fmnist_feats[calib_size:], 1),
    (cifar_feats[calib_size:], 2),
    (svhn_feats[calib_size:], 3)
]

correct = 0
total = 0
routing_by_task = {0: 0, 1: 0, 2: 0, 3: 0}
total_by_task = {0: 0, 1: 0, 2: 0, 3: 0}

for feats, task_id in test_feats:
    for f in feats:
        # Compute cosine similarity against centroids
        similarities = []
        for k in range(4):
            c = centroids[k]
            cos_sim = torch.dot(f, c) / (torch.norm(f) * torch.norm(c))
            similarities.append(cos_sim.item())
        
        pred_task = np.argmax(similarities)
        if pred_task == task_id:
            correct += 1
            routing_by_task[task_id] += 1
        total_by_task[task_id] += 1
        total += 1

print(f"\nReal-World ViT-Tiny Early-Stage (Layer 3) Routing Accuracy: {correct/total*100.0:.2f}%")
for k, name in [(0, "MNIST"), (1, "Fashion-MNIST"), (2, "CIFAR-10"), (3, "SVHN")]:
    acc = routing_by_task[k]/total_by_task[k]*100.0
    print(f"  {name} Routing Specificity (Accuracy): {acc:.2f}%")
