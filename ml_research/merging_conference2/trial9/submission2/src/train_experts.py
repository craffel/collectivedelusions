import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED error
torch.backends.cudnn.enabled = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Helper to load dataset
def get_dataset(task_name, is_train=True):
    os.makedirs('data', exist_ok=True)
    
    # Pre-processing transforms
    if task_name in ['mnist', 'fmnist']:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # duplicate channels
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else: # cifar10
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
    if task_name == 'mnist':
        dataset = torchvision.datasets.MNIST('data', train=is_train, download=False, transform=transform)
    elif task_name == 'fmnist':
        dataset = torchvision.datasets.FashionMNIST('data', train=is_train, download=False, transform=transform)
    elif task_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10('data', train=is_train, download=False, transform=transform)
    else:
        raise ValueError(f"Unknown task: {task_name}")
        
    return dataset

# Expert Model Definition
class ExpertModel(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super(ExpertModel, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

def train_expert(task_name):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    
    # Load dataset
    train_dataset = get_dataset(task_name, is_train=True)
    test_dataset = get_dataset(task_name, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    # Instantiate backbone from Imagenet pretrained resnet18
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity() # replace fc with identity
    
    model = ExpertModel(backbone).to(device)
    
    # Optimizer & Criterion
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
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
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        
        # Simple validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100. * val_correct / val_total
        print(f"Epoch [{epoch}/{epochs}] - Test Acc: {val_acc:.2f}%")
        
    # Save the expert model
    os.makedirs('checkpoints', exist_ok=True)
    save_path = f"checkpoints/expert_{task_name}.pt"
    torch.save({
        'backbone_state_dict': model.backbone.state_dict(),
        'fc_state_dict': model.fc.state_dict(),
        'test_acc': val_acc
    }, save_path)
    print(f"Saved {task_name} expert to {save_path}")

if __name__ == "__main__":
    # Save base progenitor model first
    os.makedirs('checkpoints', exist_ok=True)
    base_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_backbone.fc = nn.Identity()
    torch.save(base_backbone.state_dict(), "checkpoints/progenitor.pt")
    print("Saved base progenitor model to checkpoints/progenitor.pt")
    
    # Train each expert
    for task in ['mnist', 'fmnist', 'cifar10']:
        train_expert(task)
