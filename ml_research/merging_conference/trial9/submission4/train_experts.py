import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import numpy as np
from models import SimpleCNN

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Preprocessing: normalize images to mean=0.5, std=0.5 (range [-1.0, 1.0])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Load datasets
print("Downloading and loading datasets...")
mnist_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
fashion_train = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
kmnist_test = datasets.KMNIST(root="data", train=False, download=True, transform=transform)

# Extract subsets of 10,000 samples for pre-training each expert
subset_mnist = Subset(mnist_train, list(range(10000)))
subset_fashion = Subset(fashion_train, list(range(10000)))

loader_mnist = DataLoader(subset_mnist, batch_size=64, shuffle=True)
loader_fashion = DataLoader(subset_fashion, batch_size=64, shuffle=True)

def train_model(model, dataloader, epochs=2, is_cosface=False):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if is_cosface:
                outputs = model(x, label=y)
            else:
                outputs = model(x)
                
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
    return model

def compute_prototypes(model, dataset, is_cosface=False, num_samples=256):
    model.eval()
    model.to(device)
    
    # We take 256 samples from the dataset to construct prototypes
    loader = DataLoader(Subset(dataset, list(range(num_samples))), batch_size=num_samples, shuffle=False)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            features = model.get_features(x) # [B, 128]
            
        # Compute L2 (Euclidean) prototypes
        l2_prototypes = {}
        for c in range(10):
            mask = (y == c)
            if mask.sum() > 0:
                l2_prototypes[c] = features[mask].mean(0).cpu().numpy()
            else:
                l2_prototypes[c] = np.zeros(128)
                
        # Compute spherical (cosine) prototypes
        sph_prototypes = {}
        # Normalize features first
        features_norm = features / (features.norm(p=2, dim=1, keepdim=True) + 1e-8)
        for c in range(10):
            mask = (y == c)
            if mask.sum() > 0:
                avg_f = features_norm[mask].mean(0)
                avg_f_norm = avg_f / (avg_f.norm(p=2) + 1e-8)
                sph_prototypes[c] = avg_f_norm.cpu().numpy()
            else:
                sph_prototypes[c] = np.zeros(128)
                
    return l2_prototypes, sph_prototypes

# 1. Train Standard MNIST Expert
print("\n--- Training Standard MNIST Expert ---")
std_mnist = SimpleCNN(is_cosface=False)
std_mnist = train_model(std_mnist, loader_mnist, is_cosface=False)
std_mnist_l2_proto, std_mnist_sph_proto = compute_prototypes(std_mnist, mnist_train, is_cosface=False)

# 2. Train Standard FashionMNIST Expert
print("\n--- Training Standard FashionMNIST Expert ---")
std_fashion = SimpleCNN(is_cosface=False)
std_fashion = train_model(std_fashion, loader_fashion, is_cosface=False)
std_fashion_l2_proto, std_fashion_sph_proto = compute_prototypes(std_fashion, fashion_train, is_cosface=False)

# 3. Train CosFace MNIST Expert
print("\n--- Training CosFace MNIST Expert ---")
cos_mnist = SimpleCNN(is_cosface=True)
cos_mnist = train_model(cos_mnist, loader_mnist, is_cosface=True)
cos_mnist_l2_proto, cos_mnist_sph_proto = compute_prototypes(cos_mnist, mnist_train, is_cosface=True)

# 4. Train CosFace FashionMNIST Expert
print("\n--- Training CosFace FashionMNIST Expert ---")
cos_fashion = SimpleCNN(is_cosface=True)
cos_fashion = train_model(cos_fashion, loader_fashion, is_cosface=True)
cos_fashion_l2_proto, cos_fashion_sph_proto = compute_prototypes(cos_fashion, fashion_train, is_cosface=True)

# Save everything
print("\nSaving checkpoints and prototypes...")
torch.save({
    'state_dict': std_mnist.state_dict(),
    'l2_proto': std_mnist_l2_proto,
    'sph_proto': std_mnist_sph_proto
}, 'checkpoints/standard_mnist.pt')

torch.save({
    'state_dict': std_fashion.state_dict(),
    'l2_proto': std_fashion_l2_proto,
    'sph_proto': std_fashion_sph_proto
}, 'checkpoints/standard_fashion.pt')

torch.save({
    'state_dict': cos_mnist.state_dict(),
    'l2_proto': cos_mnist_l2_proto,
    'sph_proto': cos_mnist_sph_proto
}, 'checkpoints/cosface_mnist.pt')

torch.save({
    'state_dict': cos_fashion.state_dict(),
    'l2_proto': cos_fashion_l2_proto,
    'sph_proto': cos_fashion_sph_proto
}, 'checkpoints/cosface_fashion.pt')

print("Expert training and prototype precomputation successfully complete!")
