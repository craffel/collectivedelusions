import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False

def get_dataset(dataset_name, download=True):
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Replicate to 3 channels
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
        ])
        train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)
        
    elif dataset_name == "fmnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Replicate to 3 channels
            transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
        ])
        train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=download, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=download, transform=transform)
        
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=download, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    return train_dataset, test_dataset

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 Expert")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "fmnist", "cifar10"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load ImageNet pretrained ResNet-18
    print("Loading pretrained ResNet-18 progenitor...")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Modify the classification head to output 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    train_dataset, test_dataset = get_dataset(args.dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Optimizer (AdamW as specified in paper)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting training on {args.dataset} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {running_loss/(batch_idx+1):.4f} | Acc: {100.0 * correct / total:.2f}%")
                
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1} Complete | Test Accuracy: {test_acc:.2f}%")
        
    print(f"Saving final model checkpoint to {args.save_path}...")
    torch.save(model.state_dict(), args.save_path)
    print("Training finished successfully.")

if __name__ == "__main__":
    main()
