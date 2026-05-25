import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on the GPU nodes
torch.backends.cudnn.enabled = False


def get_resnet18_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Modify conv1: 1 input channel, kernel size 3x3, stride 1, padding 1, no bias
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Modify fc: 10 output classes
    model.fc = nn.Linear(512, 10)
    return model

def train_expert(dataset_name, save_path):
    print(f"=== Training Expert on {dataset_name} ===")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load dataset
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "KMNIST":
        train_dataset = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet18_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training Loop
    for epoch in range(3):
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
        print(f"Epoch {epoch+1}/3 - Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        
    # Test Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    test_acc = 100.0 * correct / total
    print(f"Test Accuracy on {dataset_name}: {test_acc:.2f}%\n")
    
    # Save model weights
    torch.save(model.state_dict(), save_path)
    print(f"Saved {dataset_name} expert weights to {save_path}\n")
    return test_acc

if __name__ == "__main__":
    os.makedirs("./checkpoints", exist_ok=True)
    train_expert("MNIST", "./checkpoints/expert_mnist.pth")
    train_expert("KMNIST", "./checkpoints/expert_kmnist.pth")
    train_expert("FashionMNIST", "./checkpoints/expert_fashionmnist.pth")
