import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from models import SimpleCNN

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard MNIST and FashionMNIST loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load full datasets
    mnist_train_full = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    fashion_train_full = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

    # Subsets of 10,000 training samples
    mnist_train_subset = Subset(mnist_train_full, list(range(10000)))
    fashion_train_subset = Subset(fashion_train_full, list(range(10000)))

    # DataLoaders
    mnist_train_loader = DataLoader(mnist_train_subset, batch_size=64, shuffle=True)
    mnist_test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)
    fashion_train_loader = DataLoader(fashion_train_subset, batch_size=64, shuffle=True)
    fashion_test_loader = DataLoader(fashion_test, batch_size=128, shuffle=False)

    # Shared base model initialization
    base_model = SimpleCNN().to(device)
    torch.save(base_model.state_dict(), "base_model.pth")
    print("Shared base model initialized and saved as base_model.pth")

    # Define helper training function
    def train_expert(name, train_loader, test_loader):
        print(f"\n--- Training {name} expert ---")
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load("base_model.pth", weights_only=True))
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(2):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct / total
            print(f"Epoch {epoch+1}/2 - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        test_acc = correct / total
        print(f"{name} Test Accuracy: {test_acc:.4f}")
        torch.save(model.state_dict(), f"expert_{name}.pth")
        print(f"Saved expert_{name}.pth")
        return model

    train_expert("mnist", mnist_train_loader, mnist_test_loader)
    train_expert("fashion", fashion_train_loader, fashion_test_loader)

if __name__ == "__main__":
    main()
