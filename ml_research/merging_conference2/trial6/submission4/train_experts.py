import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # Bypass cuDNN initialization bugs on this cluster

def get_datasets():
    # Transforms
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download datasets
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_gray)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_color)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_color)
    
    return (mnist_train, mnist_test), (fmnist_train, fmnist_test), (cifar_train, cifar_test)

def train_expert(name, train_dataset, test_dataset, device):
    print(f"\n--- Training Expert: {name} ---")
    
    # Deterministic 5,000-sample subset
    indices = list(range(len(train_dataset)))
    random.seed(42)
    random.shuffle(indices)
    subset_indices = indices[:5000]
    train_subset = Subset(train_dataset, subset_indices)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize ResNet-18 with ImageNet weights and dropout
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(model.fc.in_features, 10)
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_subset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    test_acc = 100.0 * correct / total
    print(f"Test Accuracy for {name}: {test_acc:.2f}%")
    
    # Save checkpoint
    torch.save(model.state_dict(), f"expert_{name.lower()}.pt")
    print(f"Saved expert_{name.lower()}.pt")
    return test_acc

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    (mnist_tr, mnist_te), (fmnist_tr, fmnist_te), (cifar_tr, cifar_te) = get_datasets()
    
    accs = {}
    accs['MNIST'] = train_expert('MNIST', mnist_tr, mnist_te, device)
    accs['FashionMNIST'] = train_expert('FashionMNIST', fmnist_tr, fmnist_te, device)
    accs['CIFAR10'] = train_expert('CIFAR10', cifar_tr, cifar_te, device)
    
    print("\n--- Training Completed ---")
    for name, acc in accs.items():
        print(f"{name} Expert Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
