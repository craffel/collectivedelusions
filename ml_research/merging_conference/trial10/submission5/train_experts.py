import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 1. Define SimpleCNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x, return_features=True):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        features = self.fc1(x)
        features_relu = F.relu(features)
        x = self.dropout(features_relu)
        logits = self.classifier(x)
        if return_features:
            return logits, features_relu
        return logits

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, return_features=False)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, return_features=False)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 2. Setup transforms and download datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_train_full = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    
    fmnist_train_full = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    
    # Paper uses a subset of 10,000 samples for fine-tuning
    # Let's subset 10,000 samples for joint and specialized training
    mnist_train_subset = Subset(mnist_train_full, list(range(10000)))
    fmnist_train_subset = Subset(fmnist_train_full, list(range(10000)))
    
    mnist_loader = DataLoader(mnist_train_subset, batch_size=64, shuffle=True)
    fmnist_loader = DataLoader(fmnist_train_subset, batch_size=64, shuffle=True)
    
    mnist_test_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)
    fmnist_test_loader = DataLoader(fmnist_test, batch_size=256, shuffle=False)
    
    # 3. Create Joint Dataset for Base Pre-training
    # We will alternate/mix samples or combine them
    joint_train_data = []
    # MNIST subset
    for idx in range(10000):
        img, label = mnist_train_full[idx]
        joint_train_data.append((img, label))
    # FashionMNIST subset
    for idx in range(10000):
        img, label = fmnist_train_full[idx]
        joint_train_data.append((img, label))
        
    joint_loader = DataLoader(joint_train_data, batch_size=64, shuffle=True)
    
    print("\n--- Pre-training Base Model Jointly (1 Epoch) ---")
    base_model = SimpleCNN().to(device)
    optimizer = optim.Adam(base_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    loss = train_epoch(base_model, joint_loader, optimizer, criterion, device)
    mnist_acc = evaluate(base_model, mnist_test_loader, device)
    fmnist_acc = evaluate(base_model, fmnist_test_loader, device)
    print(f"Joint Base Model -> Loss: {loss:.4f} | MNIST Test Acc: {mnist_acc*100:.2f}% | FashionMNIST Test Acc: {fmnist_acc*100:.2f}%")
    
    # Save base model
    torch.save(base_model.state_dict(), "models/joint_base.pt")
    
    print("\n--- Fine-tuning Expert 0 on MNIST (1 Epoch) ---")
    expert_0 = SimpleCNN().to(device)
    expert_0.load_state_dict(torch.load("models/joint_base.pt"))
    optimizer_0 = optim.Adam(expert_0.parameters(), lr=2e-4, weight_decay=1e-5)
    
    loss_0 = train_epoch(expert_0, mnist_loader, optimizer_0, criterion, device)
    mnist_acc_0 = evaluate(expert_0, mnist_test_loader, device)
    print(f"Expert 0 (MNIST) -> Loss: {loss_0:.4f} | MNIST Test Acc: {mnist_acc_0*100:.2f}%")
    torch.save(expert_0.state_dict(), "models/expert_0.pt")
    
    print("\n--- Fine-tuning Expert 1 on FashionMNIST (1 Epoch) ---")
    expert_1 = SimpleCNN().to(device)
    expert_1.load_state_dict(torch.load("models/joint_base.pt"))
    optimizer_1 = optim.Adam(expert_1.parameters(), lr=2e-4, weight_decay=1e-5)
    
    loss_1 = train_epoch(expert_1, fmnist_loader, optimizer_1, criterion, device)
    fmnist_acc_1 = evaluate(expert_1, fmnist_test_loader, device)
    print(f"Expert 1 (Fashion) -> Loss: {loss_1:.4f} | FashionMNIST Test Acc: {fmnist_acc_1*100:.2f}%")
    torch.save(expert_1.state_dict(), "models/expert_1.pt")
    
    print("\nCheckpoints saved successfully in models/")

if __name__ == "__main__":
    main()
