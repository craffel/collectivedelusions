import os
import torch
import torch.nn as nn

# Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED errors on the cluster
torch.backends.cudnn.enabled = False
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Settings
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(task):
    if task == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
        ])
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    elif task == 'fashion':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
        ])
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    elif task == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    else:
        raise ValueError(f"Unknown task: {task}")
        
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, test_loader

def train_expert(task):
    print(f"\n--- Training Expert for {task.upper()} ---")
    train_loader, test_loader = get_dataloaders(task)
    
    # Initialize from pre-trained ResNet18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Modify fc layer to output 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
        
    # Final evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100.0 * test_correct / test_total
    print(f"Final Test Accuracy for {task.upper()}: {test_acc:.2f}%")
    
    # Save checkpoint
    save_path = f"{task}_expert.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved {task} expert model to {save_path}")

if __name__ == "__main__":
    tasks = ['mnist', 'fashion', 'cifar10']
    for t in tasks:
        train_expert(t)
    print("\nAll expert training completed successfully!")
