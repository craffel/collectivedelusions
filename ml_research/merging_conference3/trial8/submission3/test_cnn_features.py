import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
import numpy as np

# Set seed
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")
print("Loading pre-trained ResNet-18...")
# Load pre-trained ResNet-18
model = models.resnet18(pretrained=True)
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

# Extract Stage 1 / Layer 1 features from ResNet-18
# In ResNet-18: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
# Layer 1 output is (B, 64, 56, 56). We can globally average pool it to get a (B, 64) representation.
def extract_resnet_layer1_features(x):
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        # Global average pool over spatial dimensions (H, W)
        x = torch.mean(x, dim=[2, 3]) # (B, 64)
    return x

print("Extracting features from ResNet-18 Layer 1...")
features = {}
mnist_feats = extract_resnet_layer1_features(next(iter(mnist_loader))[0])
fmnist_feats = extract_resnet_layer1_features(next(iter(fmnist_loader))[0])
cifar_feats = extract_resnet_layer1_features(next(iter(cifar_loader))[0])
svhn_feats = extract_resnet_layer1_features(next(iter(svhn_loader))[0])

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

print(f"\nReal-World ResNet-18 Layer 1 Routing Accuracy: {correct/total*100.0:.2f}%")
for k, name in [(0, "MNIST"), (1, "Fashion-MNIST"), (2, "CIFAR-10"), (3, "SVHN")]:
    acc = routing_by_task[k]/total_by_task[k]*100.0
    print(f"  {name} Routing Specificity (Accuracy): {acc:.2f}%")
