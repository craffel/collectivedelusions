import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import os
import copy

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on cluster
torch.backends.cudnn.enabled = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Datasets & Transforms
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Replicate to 3 channels
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_color = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading datasets...")
train_sets = {
    "mnist": MNIST(root="./data", train=True, download=True, transform=transform_gray),
    "fmnist": FashionMNIST(root="./data", train=True, download=True, transform=transform_gray),
    "cifar10": CIFAR10(root="./data", train=True, download=True, transform=transform_color)
}

test_sets = {
    "mnist": MNIST(root="./data", train=False, download=True, transform=transform_gray),
    "fmnist": FashionMNIST(root="./data", train=False, download=True, transform=transform_gray),
    "cifar10": CIFAR10(root="./data", train=False, download=True, transform=transform_color)
}

# Create loaders
train_loaders = {name: DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True) for name, ds in train_sets.items()}
test_loaders = {name: DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True) for name, ds in test_sets.items()}

# Function to train a model
def train_expert(task_name, train_loader, test_loader):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    
    # Initialize backbone from pretrained resnet18
    # Use weights parameter instead of pretrained to avoid deprecation warnings
    backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace classification head
    backbone.fc = nn.Linear(512, 10)
    backbone = backbone.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(backbone.parameters(), lr=1e-4, weight_decay=1e-2)
    
    epochs = 5
    for epoch in range(epochs):
        backbone.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = backbone(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    # Evaluate
    backbone.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = backbone(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    test_acc = 100.0 * correct / total
    print(f"Expert {task_name.upper()} Test Accuracy: {test_acc:.2f}%")
    
    # Save checkpoint
    checkpoint_path = f"checkpoints/expert_{task_name}.pth"
    # Save state_dict of the full model (backbone + head)
    torch.save(backbone.state_dict(), checkpoint_path)
    print(f"Saved expert to {checkpoint_path}")
    return test_acc

# Save progenitor backbone weights for model merging reference
print("\nSaving progenitor backbone weights...")
progenitor = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
# Create a dummy head just to have a complete model structure matching experts
progenitor.fc = nn.Linear(512, 10)
torch.save(progenitor.state_dict(), "checkpoints/progenitor.pth")
print("Saved progenitor to checkpoints/progenitor.pth")

# Train all three experts
results = {}
for task in ["mnist", "fmnist", "cifar10"]:
    results[task] = train_expert(task, train_loaders[task], test_loaders[task])

print("\nAll experts trained successfully!")
for task, acc in results.items():
    print(f" - {task.upper()}: {acc:.2f}%")
