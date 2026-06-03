import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED error.")

# Create directories
os.makedirs('checkpoints/resnet18', exist_ok=True)
os.makedirs('checkpoints/mlp', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Define ResNet18 Expert
class ResNet18Expert(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        try:
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            print("Loaded pretrained ResNet-18 weights.")
        except Exception as e:
            print(f"Could not load pretrained ResNet-18 weights: {e}. Using random init.")
            self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# Define MLP Expert
class MLPBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 512)
        self.relu3 = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        return x

class MLPExpert(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = MLPBackbone()
        self.head = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# Data loaders
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_dataloader(name, train=True, batch_size=128):
    if name == 'mnist':
        dataset = datasets.MNIST('data', train=train, download=True, transform=transform)
    elif name == 'fashion':
        dataset = datasets.FashionMNIST('data', train=train, download=True, transform=transform)
    elif name == 'cifar10':
        dataset = datasets.CIFAR10('data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2 if torch.cuda.is_available() else 0)

# Training loop
def train_expert(model, dataloader, epochs=5, lr=5e-4, wd=1e-2):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in dataloader:
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
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
def evaluate_expert(model, dataloader):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return 100.0 * correct / total

if __name__ == '__main__':
    # 1. Create and save Progenitors
    print("=== Creating Progenitor Models ===")
    resnet_progenitor = ResNet18Expert()
    mlp_progenitor = MLPExpert()
    
    torch.save(resnet_progenitor.state_dict(), 'checkpoints/resnet18/progenitor.pt')
    torch.save(mlp_progenitor.state_dict(), 'checkpoints/mlp/progenitor.pt')
    
    tasks = ['mnist', 'fashion', 'cifar10']
    
    # 2. Train ResNet-18 Experts
    print("\n=== Training ResNet-18 Experts ===")
    for task in tasks:
        print(f"\nTraining ResNet-18 Expert on {task.upper()}...")
        # Start from shared progenitor
        model = ResNet18Expert()
        model.load_state_dict(torch.load('checkpoints/resnet18/progenitor.pt'))
        
        train_loader = get_dataloader(task, train=True)
        test_loader = get_dataloader(task, train=False)
        
        train_expert(model, train_loader, epochs=5)
        test_acc = evaluate_expert(model, test_loader)
        print(f"ResNet-18 {task.upper()} Test Accuracy: {test_acc:.2f}%")
        
        # Save model
        torch.save(model.state_dict(), f'checkpoints/resnet18/{task}_expert.pt')
        
    # 3. Train MLP Experts
    print("\n=== Training MLP Experts ===")
    for task in tasks:
        print(f"\nTraining MLP Expert on {task.upper()}...")
        # Start from shared progenitor
        model = MLPExpert()
        model.load_state_dict(torch.load('checkpoints/mlp/progenitor.pt'))
        
        train_loader = get_dataloader(task, train=True)
        test_loader = get_dataloader(task, train=False)
        
        train_expert(model, train_loader, epochs=5)
        test_acc = evaluate_expert(model, test_loader)
        print(f"MLP {task.upper()} Test Accuracy: {test_acc:.2f}%")
        
        # Save model
        torch.save(model.state_dict(), f'checkpoints/mlp/{task}_expert.pt')
        
    print("\nTraining completed successfully! All checkpoints saved.")
