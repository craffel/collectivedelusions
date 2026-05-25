import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import copy
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled = False

# Directory setup
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("prototypes", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define base ResNet-18 and wrapper
class ResNetBackbone(nn.Module):
    def __init__(self, original_resnet):
        super().__init__()
        # Extract all layers except the last fully connected layer
        self.resnet_layers = nn.Sequential(*list(original_resnet.children())[:-1])
    def forward(self, x):
        x = self.resnet_layers(x)
        x = torch.flatten(x, 1)
        return x

def get_base_model():
    original_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone = ResNetBackbone(original_resnet)
    return backbone

# Transformations: Resize 28x28 to 32x32, convert to RGB by duplicating, and normalize
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Grayscale to 3 channels
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Helper to get dataset subsets
def get_task_loader(name, train=True, subset_size=10000, batch_size=128):
    if name == "mnist":
        dataset = torchvision.datasets.MNIST(root="data", train=train, transform=transform, download=False)
    elif name == "fashionmnist":
        dataset = torchvision.datasets.FashionMNIST(root="data", train=train, transform=transform, download=False)
    elif name == "kmnist":
        dataset = torchvision.datasets.KMNIST(root="data", train=train, transform=transform, download=False)
    else:
        raise ValueError("Unknown dataset")
    
    if train and subset_size < len(dataset):
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        dataset = Subset(dataset, indices)
    elif not train and subset_size is not None and subset_size < len(dataset):
        # Allow small subset for quick evaluation or prototype calibration
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)
    return loader, dataset

# Training function for one expert
def train_expert(task_name):
    print(f"\n--- Training Expert for Task: {task_name.upper()} ---")
    backbone = get_base_model().to(device)
    head = nn.Linear(512, 10).to(device)
    
    train_loader, _ = get_task_loader(task_name, train=True, subset_size=10000, batch_size=128)
    test_loader, _ = get_task_loader(task_name, train=False, subset_size=5000, batch_size=128)
    
    optimizer = optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(4):
        backbone.train()
        head.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            features = backbone(images)
            outputs = head(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/4 - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
        # Test evaluation
        backbone.eval()
        head.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                features = backbone(images)
                outputs = head(features)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        test_acc = 100. * test_correct / test_total
        print(f"Test Acc: {test_acc:.2f}%")
        
    # Save checkpoints
    torch.save(backbone.state_dict(), f"checkpoints/{task_name}_backbone.pt")
    torch.save(head.state_dict(), f"checkpoints/{task_name}_head.pt")
    print(f"Saved {task_name} checkpoints!")
    
    # Compute and save class prototypes
    print(f"Computing class prototypes for {task_name}...")
    backbone.eval()
    prototypes = torch.zeros(10, 512).to(device)
    counts = torch.zeros(10).to(device)
    
    # Use a clean subset from training set to compute prototypes
    proto_loader, _ = get_task_loader(task_name, train=True, subset_size=1000, batch_size=128)
    with torch.no_grad():
        for images, labels in proto_loader:
            images, labels = images.to(device), labels.to(device)
            features = backbone(images)
            for c in range(10):
                mask = (labels == c)
                if mask.any():
                    prototypes[c] += features[mask].sum(dim=0)
                    counts[c] += mask.sum()
                    
    # Divide to get the mean
    for c in range(10):
        if counts[c] > 0:
            prototypes[c] /= counts[c]
        else:
            # Fallback to mean of all features if class not represented
            prototypes[c] = features.mean(dim=0)
            
    # Normalize prototypes for stable cosine similarity
    prototypes = nn.functional.normalize(prototypes, p=2, dim=1)
    torch.save(prototypes.cpu(), f"prototypes/{task_name}_prototypes.pt")
    print(f"Prototypes for {task_name} saved!")

if __name__ == "__main__":
    # Also save the original base pre-trained backbone
    base_backbone = get_base_model()
    torch.save(base_backbone.state_dict(), "checkpoints/base_backbone.pt")
    print("Saved original pre-trained base backbone to checkpoints/base_backbone.pt")
    
    for task in ["mnist", "fashionmnist", "kmnist"]:
        train_expert(task)
    print("\nAll experts trained and prototypes computed!")
