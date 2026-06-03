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
    # Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on cluster GPUs
    torch.backends.cudnn.enabled = False

def get_dataloader(dataset_name, batch_size, data_dir="./data"):
    # ResNet-18 expects 3-channel images, so grayscale datasets are repeated to 3 channels
    transform_gray = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_color = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if dataset_name == "MNIST":
        train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=False, transform=transform_gray)
        test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=transform_gray)
    elif dataset_name == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=transform_gray)
        test_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=transform_gray)
    elif dataset_name == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_color)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_color)
    elif dataset_name == "SVHN":
        train_dataset = torchvision.datasets.SVHN(root=data_dir, split='train', download=False, transform=transform_color)
        test_dataset = torchvision.datasets.SVHN(root=data_dir, split='test', download=False, transform=transform_color)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 on 4 vision tasks")
    parser.add_argument("--dataset", type=str, required=True, choices=["MNIST", "FashionMNIST", "CIFAR10", "SVHN"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for training on {args.dataset}")
    
    # Load pre-trained ResNet-18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Re-initialize the classifier head to 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    
    train_loader, test_loader = get_dataloader(args.dataset, args.batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Starting training on {args.dataset} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
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
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
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
    print(f"Finished Training! Test Accuracy on {args.dataset}: {test_acc:.2f}%")
    
    # Save checkpoint
    checkpoint_path = os.path.join(args.save_dir, f"{args.dataset}_seed{args.seed}.pt")
    torch.save({
        'state_dict': model.state_dict(),
        'test_acc': test_acc
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
