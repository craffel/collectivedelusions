import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import os
import copy

torch.backends.cudnn.enabled = False

def get_resnet18_expert():
    # Load pre-trained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace final classification head (model.fc) with linear layer mapping 512 -> 10
    model.fc = nn.Linear(512, 10)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms
    gray_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Replicate to 3 channels
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    color_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download datasets
    print("Loading datasets...")
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=gray_transform)
    fmnist_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=gray_transform)
    cifar_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=color_transform)
    
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=gray_transform)
    fmnist_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=gray_transform)
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=color_transform)
    
    # Use first 3,000 samples for training
    subset_indices = list(range(3000))
    mnist_train_sub = Subset(mnist_train, subset_indices)
    fmnist_train_sub = Subset(fmnist_train, subset_indices)
    cifar_train_sub = Subset(cifar_train, subset_indices)
    
    tasks = {
        "mnist": (mnist_train_sub, mnist_test),
        "fmnist": (fmnist_train_sub, fmnist_test),
        "cifar10": (cifar_train_sub, cifar_test)
    }
    
    # Save the base model checkpoint first (ImageNet pre-trained base)
    base_model = get_resnet18_expert()
    torch.save(base_model.state_dict(), "base_model.pt")
    print("Saved pre-trained base model weights to base_model.pt")
    
    for name, (train_data, test_data) in tasks.items():
        print(f"\n--- Training {name.upper()} Expert ---")
        model = get_resnet18_expert().to(device)
        
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2)
        
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(5):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            scheduler.step()
            train_acc = 100.0 * correct / total
            print(f"Epoch {epoch+1}/5 - Loss: {total_loss/total:.4f}, Train Acc: {train_acc:.2f}%")
            
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_acc = 100.0 * correct / total
        print(f"Finished {name.upper()} - Test Acc: {test_acc:.2f}%")
        
        # Save expert model state dict
        torch.save(model.state_dict(), f"{name}_expert.pt")
        print(f"Saved {name} expert weights to {name}_expert.pt")

if __name__ == '__main__':
    main()
