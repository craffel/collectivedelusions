import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_base_model():
    model = resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 10)
    return model

# Load models
expert_mnist = get_base_model().to(device)
expert_mnist.load_state_dict(torch.load("./checkpoints/expert_mnist.pth", map_location=device))
expert_mnist.eval()

expert_kmnist = get_base_model().to(device)
expert_kmnist.load_state_dict(torch.load("./checkpoints/expert_kmnist.pth", map_location=device))
expert_kmnist.eval()

# Load data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
kmnist_dataset = torchvision.datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
fmnist_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# Extract features
def get_features(model, x):
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)

# Compute prototypes & means
def compute_protos(model, dataset):
    loader = DataLoader(Subset(dataset, list(range(500))), batch_size=64, shuffle=False)
    feats_list = []
    labels_list = []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        feats_list.append(get_features(model, inputs).cpu())
        labels_list.append(labels)
    feats = torch.cat(feats_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    mean = feats.mean(dim=0).to(device)
    protos = {}
    for c in range(10):
        protos[c] = feats[labels == c].mean(dim=0).to(device)
    return mean, protos

mean_mnist, protos_mnist = compute_protos(expert_mnist, mnist_dataset)
mean_kmnist, protos_kmnist = compute_protos(expert_kmnist, kmnist_dataset)

# Test cohesion on a batch
mnist_batch = next(iter(DataLoader(mnist_subset := Subset(mnist_dataset, list(range(64))), batch_size=64)))[0].to(device)
kmnist_batch = next(iter(DataLoader(kmnist_subset := Subset(kmnist_dataset, list(range(64))), batch_size=64)))[0].to(device)
fmnist_batch = next(iter(DataLoader(fmnist_subset := Subset(fmnist_dataset, list(range(64))), batch_size=64)))[0].to(device)

def check_cohesion(feats, mean_d, protos_d, center_protos=False):
    # Centered features
    feats_centered = feats - mean_d
    
    cohesion_list = []
    for feat in feats_centered:
        sims = []
        for c in range(10):
            proto = protos_d[c]
            if center_protos:
                proto = proto - mean_d
            sims.append(F.cosine_similarity(feat, proto, dim=0).item())
        cohesion_list.append(max(sims))
    return np.mean(cohesion_list)

# Extract feats from expert models
feats_mnist_ex = get_features(expert_mnist, mnist_batch)
feats_kmnist_ex = get_features(expert_kmnist, kmnist_batch)
feats_fmnist_ex = get_features(expert_mnist, fmnist_batch) # Use MNIST expert for novel domain features

print("Uncentered Prototypes Cohesion:")
print("  MNIST feats vs MNIST protos:", check_cohesion(feats_mnist_ex, mean_mnist, protos_mnist, center_protos=False))
print("  MNIST feats vs KMNIST protos:", check_cohesion(feats_mnist_ex, mean_kmnist, protos_kmnist, center_protos=False))
print("  KMNIST feats vs KMNIST protos:", check_cohesion(feats_kmnist_ex, mean_kmnist, protos_kmnist, center_protos=False))
print("  FashionMNIST feats vs MNIST protos:", check_cohesion(feats_fmnist_ex, mean_mnist, protos_mnist, center_protos=False))

print("\nCentered Prototypes Cohesion:")
print("  MNIST feats vs MNIST protos (Centered):", check_cohesion(feats_mnist_ex, mean_mnist, protos_mnist, center_protos=True))
print("  MNIST feats vs KMNIST protos (Centered):", check_cohesion(feats_mnist_ex, mean_kmnist, protos_kmnist, center_protos=True))
print("  KMNIST feats vs KMNIST protos (Centered):", check_cohesion(feats_kmnist_ex, mean_kmnist, protos_kmnist, center_protos=True))
print("  FashionMNIST feats vs MNIST protos (Centered):", check_cohesion(feats_fmnist_ex, mean_mnist, protos_mnist, center_protos=True))
