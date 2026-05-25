import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import os

def modify_resnet18_for_grayscale(model):
    # Sum the weights of the first conv layer along the input channel dimension (3 -> 1)
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.conv1 = new_conv
    
    # Modify final fully connected layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def train_model(model, train_loader, epochs=2, device='cuda'):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
    return model

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    acc = 100. * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 1. Train MNIST Expert
    print("\n--- Training MNIST Expert ---")
    train_mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_mnist, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_mnist, batch_size=256, shuffle=False, num_workers=2)
    
    # Load ResNet-18
    resnet18_mnist = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet18_mnist = modify_resnet18_for_grayscale(resnet18_mnist)
    resnet18_mnist = train_model(resnet18_mnist, train_loader, epochs=2, device=device)
    evaluate_model(resnet18_mnist, test_loader, device=device)
    torch.save(resnet18_mnist.state_dict(), "checkpoints/expert_mnist.pth")
    
    # 2. Train KMNIST Expert
    print("\n--- Training KMNIST Expert ---")
    train_kmnist = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
    test_kmnist = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_kmnist, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_kmnist, batch_size=256, shuffle=False, num_workers=2)
    
    resnet18_kmnist = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet18_kmnist = modify_resnet18_for_grayscale(resnet18_kmnist)
    resnet18_kmnist = train_model(resnet18_kmnist, train_loader, epochs=2, device=device)
    evaluate_model(resnet18_kmnist, test_loader, device=device)
    torch.save(resnet18_kmnist.state_dict(), "checkpoints/expert_kmnist.pth")
    
    # 3. Train FashionMNIST Expert
    print("\n--- Training FashionMNIST Expert ---")
    train_fmnist = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_fmnist = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_fmnist, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_fmnist, batch_size=256, shuffle=False, num_workers=2)
    
    resnet18_fmnist = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet18_fmnist = modify_resnet18_for_grayscale(resnet18_fmnist)
    resnet18_fmnist = train_model(resnet18_fmnist, train_loader, epochs=2, device=device)
    evaluate_model(resnet18_fmnist, test_loader, device=device)
    torch.save(resnet18_fmnist.state_dict(), "checkpoints/expert_fashionmnist.pth")
    
    print("\nAll experts trained and saved.")

if __name__ == "__main__":
    main()
