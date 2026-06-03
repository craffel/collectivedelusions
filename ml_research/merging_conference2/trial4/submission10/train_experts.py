import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import copy

def get_transforms(dataset_name):
    if dataset_name in ["mnist", "fmnist"]:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Convert to 3 channels
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else: # cifar10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transform

def get_dataset(name, train=True):
    transform = get_transforms(name)
    if name == "mnist":
        dataset = datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    elif name == "fmnist":
        dataset = datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform)
    elif name == "cifar10":
        dataset = datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {name}")
    return dataset

def train_expert(dataset_name, device):
    print(f"--- Training expert for {dataset_name.upper()} ---")
    train_dataset = get_dataset(dataset_name, train=True)
    
    # Select exactly 3,000 images for training
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset), generator=generator)[:3000].tolist()
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
    
    # Instantiate pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace classification head
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    # Define optimizer, scheduler and loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):
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
        print(f"Epoch {epoch+1}/5 | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    # Evaluate on full test set to verify performance
    test_dataset = get_dataset(dataset_name, train=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
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
    print(f"Test Accuracy for {dataset_name.upper()}: {test_acc:.2f}%")
    
    # Save the expert model
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), f"weights/expert_{dataset_name}.pth")
    print(f"Saved expert weights to weights/expert_{dataset_name}.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for ds in ["mnist", "fmnist", "cifar10"]:
        train_expert(ds, device)
