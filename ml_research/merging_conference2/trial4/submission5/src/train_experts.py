import os
import argparse
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import copy

def get_datasets(data_dir):
    # Standard transformations: resize to 32x32, normalize to (0.5,), (0.5,) for gray, (0.5, 0.5, 0.5) for color
    # MNIST and FashionMNIST will be duplicated to 3 channels to match ResNet input
    
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Custom transform to duplicate channels for grayscale datasets
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_mnist = datasets.MNIST(data_dir, train=True, download=True, transform=transform_mnist)
    test_mnist = datasets.MNIST(data_dir, train=False, download=True, transform=transform_mnist)
    
    train_fashion = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform_mnist)
    test_fashion = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform_mnist)
    
    train_cifar = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_cifar)
    test_cifar = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_cifar)
    
    return {
        'mnist': (train_mnist, test_mnist),
        'fashion': (train_fashion, test_fashion),
        'cifar': (train_cifar, test_cifar)
    }

def train_one_expert(name, train_dataset, test_dataset, save_path, epochs, lr, wd, batch_size, device):
    print(f"--- Training {name.upper()} Expert ---")
    
    # Limit training set to 3,000 deterministic samples
    # We use a fixed generator seed for reproducibility
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset), generator=g)[:3000].tolist()
    train_subset = Subset(train_dataset, indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Initialize ResNet-18 with ImageNet weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace the linear head (fc)
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
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
            
        scheduler.step()
        epoch_loss = running_loss / len(train_subset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
        
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100.0 * test_correct / test_total
    print(f"{name.upper()} Expert Test Accuracy: {test_acc:.2f}%\n")
    
    # Save checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 Expert Models")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for datasets")
    parser.add_argument("--save_dir", type=str, default="./experts", help="Directory to save expert models")
    parser.add_argument("--epochs", type=type(5), default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    all_datasets = get_datasets(args.data_dir)
    
    results = {}
    for name, (train_ds, test_ds) in all_datasets.items():
        save_path = os.path.join(args.save_dir, f"{name}_expert.pt")
        test_acc = train_one_expert(
            name=name,
            train_dataset=train_ds,
            test_dataset=test_ds,
            save_path=save_path,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
            batch_size=args.batch_size,
            device=device
        )
        results[name] = test_acc
        
    print("--- Expert Training Completed ---")
    for name, acc in results.items():
        print(f"{name.upper()} Accuracy: {acc:.2f}%")
