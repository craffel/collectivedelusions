import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import os
import random
import numpy as np

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  # Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms
image_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# For grayscale datasets, we replicate channels
grayscale_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Helper to get deterministic 5000-sample training subset
def get_dataset_subset(dataset, subset_size=5000):
    indices = list(range(len(dataset)))
    # Use deterministic shuffle with seed 42 to match papers
    rng = random.Random(42)
    rng.shuffle(indices)
    subset_indices = indices[:subset_size]
    return Subset(dataset, subset_indices)

# Load datasets
print("Loading MNIST...")
train_mnist_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=grayscale_transforms)
train_mnist = get_dataset_subset(train_mnist_full)
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=grayscale_transforms)

print("Loading Fashion-MNIST...")
train_fmnist_full = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=grayscale_transforms)
train_fmnist = get_dataset_subset(train_fmnist_full)
test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=grayscale_transforms)

print("Loading CIFAR-10...")
train_cifar_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=image_transforms)
train_cifar = get_dataset_subset(train_cifar_full)
test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=image_transforms)

# Dataloaders
train_loaders = {
    "mnist": DataLoader(train_mnist, batch_size=64, shuffle=True, num_workers=2),
    "fmnist": DataLoader(train_fmnist, batch_size=64, shuffle=True, num_workers=2),
    "cifar10": DataLoader(train_cifar, batch_size=64, shuffle=True, num_workers=2)
}

test_loaders = {
    "mnist": DataLoader(test_mnist, batch_size=128, shuffle=False, num_workers=2),
    "fmnist": DataLoader(test_fmnist, batch_size=128, shuffle=False, num_workers=2),
    "cifar10": DataLoader(test_cifar, batch_size=128, shuffle=False, num_workers=2)
}

def train_expert(task_name, train_loader, test_loader):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    set_seed(42)  # Enforce deterministic initialization of each expert
    
    # Load ImageNet pre-trained ResNet-18
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    
    # Add Dropout (0.1) before linear classification head to prevent overfitting
    model.fc = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(512, 10)
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
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
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
        
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            test_total += y.size(0)
            test_correct += predicted.eq(y).sum().item()
            
    test_acc = 100. * test_correct / test_total
    print(f"Final Test Accuracy for {task_name.upper()}: {test_acc:.2f}%")
    
    # Save checkpoint
    checkpoint_path = f"checkpoints/expert_{task_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    return test_acc

# Save baseline/pre-trained base ResNet-18 model for Task Arithmetic task-vector computation
print("\nSaving baseline pre-trained ResNet-18 base model...")
base_model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
base_model.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 10))
torch.save(base_model.state_dict(), "checkpoints/base_model.pt")

accuracies = {}
for task_name in ["mnist", "fmnist", "cifar10"]:
    accuracies[task_name] = train_expert(task_name, train_loaders[task_name], test_loaders[task_name])

print("\n--- Training Complete! ---")
for task_name, acc in accuracies.items():
    print(f"{task_name.upper()} Expert Test Acc: {acc:.2f}%")
