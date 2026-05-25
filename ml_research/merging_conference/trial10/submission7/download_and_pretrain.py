import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        feats = self.dropout(feats)
        out = self.fc2(feats)
        return out

def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1.0, 1.0]
    ])
    
    # Download datasets
    print("Downloading MNIST...")
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    print("Downloading FashionMNIST...")
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    print("Downloading KMNIST...")
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    return mnist_train, mnist_test, fmnist_train, fmnist_test, kmnist_test

def train_expert(dataset_train, dataset_test, name):
    print(f"\n--- Training Expert: {name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Select subset of 10,000 samples
    indices = list(range(10000))
    subset_train = Subset(dataset_train, indices)
    
    train_loader = DataLoader(subset_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False)
    
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(2):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            _, predicted = out.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        train_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/2: Loss={total_loss/total:.4f}, Train Acc={train_acc:.2f}%")
        
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, predicted = out.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    test_acc = 100.0 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return model

def compute_prototypes(model, dataset, num_samples=10000):
    print("Computing prototypes...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    indices = list(range(num_samples))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=128, shuffle=False)
    
    # Accumulate features for each class
    class_features = {c: [] for c in range(10)}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = model.forward_features(x) # shape: (B, 128)
            for i in range(x.size(0)):
                label = y[i].item()
                class_features[label].append(feats[i].cpu())
                
    # Compute mean and normalize for each class
    prototypes = torch.zeros(10, 128)
    for c in range(10):
        feats_c = torch.stack(class_features[c])
        mean_feat = feats_c.mean(dim=0)
        norm = mean_feat.norm(p=2)
        if norm > 0:
            mean_feat = mean_feat / norm
        prototypes[c] = mean_feat
        
    return prototypes

if __name__ == "__main__":
    mnist_train, mnist_test, fmnist_train, fmnist_test, kmnist_test = get_datasets()
    
    expert_mnist = train_expert(mnist_train, mnist_test, "MNIST")
    expert_fashion = train_expert(fmnist_train, fmnist_test, "FashionMNIST")
    
    # Save expert models
    torch.save(expert_mnist.state_dict(), "expert_mnist.pth")
    torch.save(expert_fashion.state_dict(), "expert_fashion.pth")
    print("Saved experts to expert_mnist.pth and expert_fashion.pth.")
    
    # Compute and save prototypes
    proto_mnist = compute_prototypes(expert_mnist, mnist_train)
    proto_fashion = compute_prototypes(expert_fashion, fmnist_train)
    
    torch.save({
        "mnist": proto_mnist,
        "fashion": proto_fashion
    }, "prototypes.pth")
    print("Saved prototypes to prototypes.pth.")
