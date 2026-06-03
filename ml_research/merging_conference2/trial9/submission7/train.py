import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import os
import copy

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define directories
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Datasets & Transforms
mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

fmnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Load Datasets
print("Loading datasets...")
mnist_train = datasets.MNIST(root='data', train=True, download=True, transform=mnist_transform)
mnist_test = datasets.MNIST(root='data', train=False, download=True, transform=mnist_transform)

fmnist_train = datasets.FashionMNIST(root='data', train=True, download=True, transform=fmnist_transform)
fmnist_test = datasets.FashionMNIST(root='data', train=False, download=True, transform=fmnist_transform)

cifar_train = datasets.CIFAR10(root='data', train=True, download=True, transform=cifar_transform)
cifar_test = datasets.CIFAR10(root='data', train=False, download=True, transform=cifar_transform)

# DataLoaders
batch_size = 256
mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2)

fmnist_train_loader = DataLoader(fmnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
fmnist_test_loader = DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, num_workers=2)

cifar_train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=2)
cifar_test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=2)

loaders = {
    'mnist': (mnist_train_loader, mnist_test_loader),
    'fmnist': (fmnist_train_loader, fmnist_test_loader),
    'cifar10': (cifar_train_loader, cifar_test_loader)
}

# Model definitions
def get_resnet_progenitor():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()
    return model

class MLPBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

class JointModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.fc = head
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

def train_expert(backbone, name, loader, epochs=5, lr=1e-3, weight_decay=1e-4):
    backbone = copy.deepcopy(backbone).to(device)
    # Head maps from features to 10 classes (ResNet has 512 features, MLP has 512 features)
    head = nn.Linear(512, 10).to(device)
    model = JointModel(backbone, head)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- Training Expert for {name} ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total * 100.0
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    return model.backbone.cpu(), model.fc.cpu()

def evaluate(backbone, head, loader):
    backbone = copy.deepcopy(backbone).to(device)
    head = copy.deepcopy(head).to(device)
    model = JointModel(backbone, head)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    return correct / total * 100.0

# 1. ResNet-18 Experts Training
print("\n=== Fine-Tuning ResNet-18 Experts ===")
resnet_progenitor = get_resnet_progenitor()
torch.save(resnet_progenitor.state_dict(), 'checkpoints/resnet_progenitor.pt')

resnet_experts = {}
resnet_heads = {}

for task_name in ['mnist', 'fmnist', 'cifar10']:
    train_loader, test_loader = loaders[task_name]
    # Train
    expert_backbone, expert_fc = train_expert(resnet_progenitor, f"ResNet18-{task_name}", train_loader)
    # Evaluate
    test_acc = evaluate(expert_backbone, expert_fc, test_loader)
    print(f"ResNet18-{task_name} Expert Test Accuracy: {test_acc:.2f}%")
    
    # Save
    torch.save(expert_backbone.state_dict(), f'checkpoints/resnet_{task_name}_backbone.pt')
    torch.save(expert_fc.state_dict(), f'checkpoints/resnet_{task_name}_fc.pt')
    
    resnet_experts[task_name] = expert_backbone
    resnet_heads[task_name] = expert_fc

# 2. MLP Experts Training
print("\n=== Fine-Tuning MLP Experts ===")
mlp_progenitor = MLPBackbone()
torch.save(mlp_progenitor.state_dict(), 'checkpoints/mlp_progenitor.pt')

mlp_experts = {}
mlp_heads = {}

for task_name in ['mnist', 'fmnist', 'cifar10']:
    train_loader, test_loader = loaders[task_name]
    # Train
    expert_backbone, expert_fc = train_expert(mlp_progenitor, f"MLP-{task_name}", train_loader)
    # Evaluate
    test_acc = evaluate(expert_backbone, expert_fc, test_loader)
    print(f"MLP-{task_name} Expert Test Accuracy: {test_acc:.2f}%")
    
    # Save
    torch.save(expert_backbone.state_dict(), f'checkpoints/mlp_{task_name}_backbone.pt')
    torch.save(expert_fc.state_dict(), f'checkpoints/mlp_{task_name}_fc.pt')
    
    mlp_experts[task_name] = expert_backbone
    mlp_heads[task_name] = expert_fc

print("\nAll experts successfully trained and checkpoints saved in checkpoints/")
