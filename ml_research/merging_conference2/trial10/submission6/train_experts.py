import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# 1. Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN for compatibility")

# 2. Data Transforms
mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
])

fmnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860, 0.2860, 0.2860], std=[0.3530, 0.3530, 0.3530])
])

cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# 3. Load Datasets
print("Loading datasets...")
train_sets = {
    'mnist': datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform),
    'fmnist': datasets.FashionMNIST(root='./data', train=True, download=True, transform=fmnist_transform),
    'cifar10': datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
}

test_sets = {
    'mnist': datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform),
    'fmnist': datasets.FashionMNIST(root='./data', train=False, download=True, transform=fmnist_transform),
    'cifar10': datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
}

loaders = {
    name: {
        'train': DataLoader(train_sets[name], batch_size=256, shuffle=True, num_workers=4),
        'test': DataLoader(test_sets[name], batch_size=256, shuffle=False, num_workers=4)
    }
    for name in train_sets
}

# 4. Progenitor Model Definition
def get_progenitor():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model

# 5. Expert Training Helper
def train_expert(task_name):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    backbone = get_progenitor()
    # Replace classification head with linear layer (10 classes for all tasks)
    model = nn.Sequential(
        backbone,
        nn.Linear(512, 10)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in loaders[task_name]['train']:
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
        epoch_acc = correct / total * 100.0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    # Evaluate Expert
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in loaders[task_name]['test']:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    test_acc = test_correct / test_total * 100.0
    print(f"Expert {task_name.upper()} Test Accuracy: {test_acc:.2f}%")
    
    # Save checkpoint (save state_dict)
    save_path = f"checkpoint_{task_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint to {save_path}")

# Save Progenitor Backbone
print("\nSaving progenitor backbone...")
progenitor = get_progenitor()
torch.save(progenitor.state_dict(), "checkpoint_progenitor.pth")
print("Saved progenitor checkpoint.")

# Train all 3 experts
for task in ['mnist', 'fmnist', 'cifar10']:
    train_expert(task)

print("\nAll experts trained successfully!")
