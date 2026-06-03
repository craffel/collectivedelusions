import os
import torch
# Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED errors on some cluster nodes
torch.backends.cudnn.enabled = False

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# Create output directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transforms
# Resizing to 32x32 and replicating to 3 channels
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_color = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_datasets = {
    "mnist": datasets.MNIST(root="data", train=True, download=True, transform=transform_gray),
    "fmnist": datasets.FashionMNIST(root="data", train=True, download=True, transform=transform_gray),
    "cifar10": datasets.CIFAR10(root="data", train=True, download=True, transform=transform_color)
}

test_datasets = {
    "mnist": datasets.MNIST(root="data", train=False, download=True, transform=transform_gray),
    "fmnist": datasets.FashionMNIST(root="data", train=False, download=True, transform=transform_gray),
    "cifar10": datasets.CIFAR10(root="data", train=False, download=True, transform=transform_color)
}

train_loaders = {name: DataLoader(ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True) for name, ds in train_datasets.items()}
test_loaders = {name: DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True) for name, ds in test_datasets.items()}

# Function to evaluate a model
def evaluate(model, test_loader, head=None):
    model.eval()
    if head is not None:
        head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            features = model(inputs)
            if head is not None:
                outputs = head(features)
            else:
                outputs = features
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

# Progenitor initialization
print("Loading pre-trained ResNet-18 progenitor...")
progenitor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# Modify fc layer to map to 512-dim features
# In standard ResNet-18, progenitor has fc = Linear(512, 1000).
# We can treat the backbone as everything except the fc layer.
# So, we save the progenitor backbone state dict, and for each expert we train backbone + task-specific fc head.
# Let's verify that fc will be the task head.
# In evaluating, we pass features = backbone(inputs) and then head(features).
# To make it easier, we can replace progenitor.fc with identity or keep it.
# Let's replace progenitor.fc with nn.Identity() to treat everything else as the backbone.
progenitor.fc = nn.Identity()
progenitor = progenitor.to(device)

# Save progenitor backbone
torch.save(progenitor.state_dict(), "checkpoints/progenitor_backbone.pt")

# Fine-tune experts
for task in ["mnist", "fmnist", "cifar10"]:
    print(f"\n--- Training Expert for {task.upper()} ---")
    
    # Reload progenitor backbone
    backbone = resnet18()
    backbone.fc = nn.Identity()
    backbone.load_state_dict(torch.load("checkpoints/progenitor_backbone.pt"))
    backbone = backbone.to(device)
    
    # Task head
    head = nn.Linear(512, 10).to(device)
    
    # Optimizer & Criterion
    optimizer = optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    
    # Train loop for 5 epochs
    for epoch in range(5):
        backbone.train()
        head.train()
        running_loss = 0.0
        for inputs, targets in train_loaders[task]:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            features = backbone(inputs)
            outputs = head(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_datasets[task])
        val_acc = evaluate(backbone, test_loaders[task], head)
        print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss:.4f} - Test Acc: {val_acc:.2f}%")
        
    # Save expert checkpoints
    torch.save(backbone.state_dict(), f"checkpoints/{task}_backbone.pt")
    torch.save(head.state_dict(), f"checkpoints/{task}_head.pt")
    print(f"Saved {task} expert checkpoints.")
