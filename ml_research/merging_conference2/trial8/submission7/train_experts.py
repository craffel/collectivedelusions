import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.enabled = False
print(f"Using device: {device}")

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Datasets and loaders
def get_loaders():
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])

    fmnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
    ])

    cifar10_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load datasets
    train_mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=mnist_transform)
    test_mnist = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=mnist_transform)

    train_fmnist = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=fmnist_transform)
    test_fmnist = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=fmnist_transform)

    train_cifar = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=cifar10_transform)
    test_cifar = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=cifar10_transform)

    # Dataloaders
    loaders = {
        "mnist": {
            "train": DataLoader(train_mnist, batch_size=128, shuffle=True, num_workers=4, pin_memory=True),
            "test": DataLoader(test_mnist, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        },
        "fmnist": {
            "train": DataLoader(train_fmnist, batch_size=128, shuffle=True, num_workers=4, pin_memory=True),
            "test": DataLoader(test_fmnist, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        },
        "cifar10": {
            "train": DataLoader(train_cifar, batch_size=128, shuffle=True, num_workers=4, pin_memory=True),
            "test": DataLoader(test_cifar, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        }
    }
    return loaders

def get_base_model():
    # Load ImageNet pretrained ResNet-18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return model

def train_expert(task, train_loader, test_loader):
    print(f"\n=== Training Expert for {task.upper()} ===")
    model = get_base_model()
    
    # Replace classification head with 10-class linear head
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
        
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            
    val_acc = 100.0 * val_correct / val_total
    print(f"Validation Accuracy for {task.upper()}: {val_acc:.2f}%")
    
    # Save the full model (backbone + fc head)
    torch.save(model.state_dict(), f"models/{task}_expert.pt")
    print(f"Saved models/{task}_expert.pt")

def main():
    loaders = get_loaders()
    
    # Save progenitor backbone weights
    progenitor = get_base_model()
    # Extract backbone state_dict (all layers except fc)
    backbone_sd = {k: v for k, v in progenitor.state_dict().items() if not k.startswith("fc.")}
    torch.save(backbone_sd, "models/progenitor_backbone.pt")
    print("Saved models/progenitor_backbone.pt (imageNet pretrained backbone)")
    
    # Train experts
    for task in ["mnist", "fmnist", "cifar10"]:
        train_expert(task, loaders[task]["train"], loaders[task]["test"])

if __name__ == "__main__":
    main()
