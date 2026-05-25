import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import time

def get_resnet18_1channel():
    # Load pretrained resnet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Modify conv1 to accept 1 channel by summing weights
    old_weight = model.conv1.weight.data  # [64, 3, 7, 7]
    new_weight = old_weight.sum(dim=1, keepdim=True)  # [64, 1, 7, 7]
    
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    new_conv.weight.data = new_weight
    model.conv1 = new_conv
    
    # Modify fc layer for 10 classes
    model.fc = nn.Linear(512, 10)
    return model

def train_expert(dataset_name, save_path, epochs=3, device="cuda"):
    print(f"--- Training expert on {dataset_name} ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name == "MNIST":
        train_set = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=".", train=False, download=True, transform=transform)
    elif dataset_name == "KMNIST":
        train_set = torchvision.datasets.KMNIST(root=".", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.KMNIST(root=".", train=False, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        train_set = torchvision.datasets.FashionMNIST(root=".", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root=".", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    model = get_resnet18_1channel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()
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
            
        train_acc = 100.0 * correct / total
        train_loss = total_loss / total
        
        # Eval
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Time: {time.time()-t0:.2f}s")
        
    torch.save(model.state_dict(), save_path)
    print(f"Saved expert model to {save_path}\n")
    return val_acc

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on this cluster
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        print("Disabled cuDNN for stability")
    
    mnist_acc = train_expert("MNIST", "expert_mnist.pth", epochs=3, device=device)
    kmnist_acc = train_expert("KMNIST", "expert_kmnist.pth", epochs=3, device=device)
    fashion_acc = train_expert("FashionMNIST", "expert_fashionmnist.pth", epochs=3, device=device)
    
    print("--- Training Summary ---")
    print(f"MNIST Expert Val Acc: {mnist_acc:.2f}%")
    print(f"KMNIST Expert Val Acc: {kmnist_acc:.2f}%")
    print(f"FashionMNIST Expert Val Acc: {fashion_acc:.2f}%")
