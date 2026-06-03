import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# Disable cuDNN to bypass driver compatibility issues
torch.backends.cudnn.enabled = False

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Preprocessing transforms as specified in Section 4.1:
# "All inputs are resized to 32x32, converted to 3-channel RGB format, and normalized according to dataset-specific running statistics."
# Note: For MNIST and FMNIST, we convert Grayscale to RGB by duplicating channels, then resize, then normalize.

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

cifar10_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# Load Datasets
print("Loading datasets...")
train_datasets = {
    'mnist': datasets.MNIST(root='data', train=True, download=True, transform=mnist_transform),
    'fmnist': datasets.FashionMNIST(root='data', train=True, download=True, transform=fmnist_transform),
    'cifar10': datasets.CIFAR10(root='data', train=True, download=True, transform=cifar10_transform)
}

test_datasets = {
    'mnist': datasets.MNIST(root='data', train=False, download=True, transform=mnist_transform),
    'fmnist': datasets.FashionMNIST(root='data', train=False, download=True, transform=fmnist_transform),
    'cifar10': datasets.CIFAR10(root='data', train=False, download=True, transform=cifar10_transform)
}

train_loaders = {
    name: DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    for name, ds in train_datasets.items()
}

test_loaders = {
    name: DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    for name, ds in test_datasets.items()
}

def get_progenitor():
    # Load standard ResNet-18 with ImageNet weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity() # Replace FC layer with identity
    return model

# Save progenitor backbone
print("Saving progenitor backbone...")
progenitor = get_progenitor()
torch.save(progenitor.state_dict(), "checkpoints/progenitor.pt")

def train_expert(task_name):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    backbone = get_progenitor()
    head = nn.Linear(512, 10)
    
    # Put model on device
    backbone = backbone.to(device)
    head = head.to(device)
    
    # Define optimizer as specified in Section 4.1:
    # "AdamW with a learning rate of 1e-4 and weight decay of 1e-4."
    optimizer = optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    
    loader = train_loaders[task_name]
    backbone.train()
    head.train()
    
    for epoch in range(5):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            features = backbone(x)
            outputs = head(features)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/5 | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    # Evaluate
    backbone.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loaders[task_name]:
            x, y = x.to(device), y.to(device)
            features = backbone(x)
            outputs = head(features)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    test_acc = 100.0 * correct / total
    print(f"Test Accuracy for {task_name.upper()}: {test_acc:.2f}%")
    
    # Save expert checkpoints
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'head_state_dict': head.state_dict(),
        'test_acc': test_acc
    }, f"checkpoints/expert_{task_name}.pt")

# Train all 3 experts
for task in ['mnist', 'fmnist', 'cifar10']:
    train_expert(task)

print("\nAll experts trained and saved successfully!")
