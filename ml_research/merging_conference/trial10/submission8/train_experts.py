import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import os
import random
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x)
        return logits

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
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
    return running_loss / total, correct / total

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load Full Datasets
    print("Loading datasets...")
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # 1. Joint Pre-training on MNIST + FashionMNIST
    # Create combined dataset
    combined_dataset = ConcatDataset([mnist_train, fmnist_train])
    combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

    print("Step 1: Joint Pre-training Base Model on MNIST + FashionMNIST...")
    base_model = SimpleCNN().to(device)
    optimizer = optim.Adam(base_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Train for 1 epoch
    loss, acc = train_one_epoch(base_model, combined_loader, optimizer, criterion, device)
    print(f"Base Model Joint Pre-training -> Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")

    # Save base model checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(base_model.state_dict(), 'checkpoints/base_model.pth')

    # Create 10,000 subsets for fine-tuning
    indices = list(range(len(mnist_train)))
    random.shuffle(indices)
    mnist_subset = Subset(mnist_train, indices[:10000])

    findices = list(range(len(fmnist_train)))
    random.shuffle(findices)
    fmnist_subset = Subset(fmnist_train, findices[:10000])

    # Data loaders
    mnist_sub_loader = DataLoader(mnist_subset, batch_size=64, shuffle=True)
    fmnist_sub_loader = DataLoader(fmnist_subset, batch_size=64, shuffle=True)
    
    mnist_test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)
    fmnist_test_loader = DataLoader(fmnist_test, batch_size=128, shuffle=False)

    # 2. Fine-tune Expert 0 on MNIST subset
    print("Step 2: Fine-tuning Expert 0 on MNIST...")
    expert0 = SimpleCNN().to(device)
    expert0.load_state_dict(torch.load('checkpoints/base_model.pth'))
    # Use small learning rate for tight alignment
    optimizer0 = optim.Adam(expert0.parameters(), lr=2e-4, weight_decay=1e-5)
    
    for epoch in range(2):
        loss, acc = train_one_epoch(expert0, mnist_sub_loader, optimizer0, criterion, device)
        test_acc = evaluate(expert0, mnist_test_loader, device)
        print(f"Expert 0 (MNIST) Epoch {epoch+1} -> Loss: {loss:.4f}, Train Acc: {acc*100:.2f}%, Test Acc: {test_acc*100:.2f}%")
    torch.save(expert0.state_dict(), 'checkpoints/expert0.pth')

    # 3. Fine-tune Expert 1 on FashionMNIST subset
    print("Step 3: Fine-tuning Expert 1 on FashionMNIST...")
    expert1 = SimpleCNN().to(device)
    expert1.load_state_dict(torch.load('checkpoints/base_model.pth'))
    # Use small learning rate for tight alignment
    optimizer1 = optim.Adam(expert1.parameters(), lr=2e-4, weight_decay=1e-5)
    
    for epoch in range(2):
        loss, acc = train_one_epoch(expert1, fmnist_sub_loader, optimizer1, criterion, device)
        test_acc = evaluate(expert1, fmnist_test_loader, device)
        print(f"Expert 1 (Fashion) Epoch {epoch+1} -> Loss: {loss:.4f}, Train Acc: {acc*100:.2f}%, Test Acc: {test_acc*100:.2f}%")
    torch.save(expert1.state_dict(), 'checkpoints/expert1.pth')

    print("Expert training completed and saved!")

if __name__ == "__main__":
    main()
