import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import argparse

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED error
torch.backends.cudnn.enabled = False

def get_dataloader(dataset_name, batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset_name == 'cifar10':
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'svhn':
        train_set = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_set = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    elif dataset_name == 'fashionmnist':
        train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

def train(dataset_name, epochs=5, lr=1e-4):
    print(f"Training on {dataset_name}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace the FC layer to have 10 output classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    
    train_loader, test_loader = get_dataloader(dataset_name)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100.0 * test_correct / test_total
    print(f"Test Accuracy on {dataset_name}: {test_acc:.2f}%")
    
    # Save checkpoint
    checkpoint_path = f"checkpoint_{dataset_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}\n")
    return test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'svhn', 'fashionmnist'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=type(1e-4), default=1e-4)
    args = parser.parse_args()
    
    train(args.dataset, epochs=args.epochs, lr=args.lr)
