import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable TF32 for faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Define datasets and transforms
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_gray = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

os.makedirs("./data", exist_ok=True)
os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# Datasets loading helper
def load_task_datasets():
    datasets = {}
    
    # 1. MNIST
    print("Loading MNIST...")
    datasets["MNIST"] = {
        "train": torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_gray),
        "test": torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_gray)
    }
    
    # 2. FashionMNIST
    print("Loading FashionMNIST...")
    datasets["FashionMNIST"] = {
        "train": torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_gray),
        "test": torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_gray)
    }
    
    # 3. CIFAR-10
    print("Loading CIFAR10...")
    datasets["CIFAR10"] = {
        "train": torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_rgb),
        "test": torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_rgb)
    }
    
    # 4. SVHN
    print("Loading SVHN...")
    datasets["SVHN"] = {
        "train": torchvision.datasets.SVHN(root="./data", split="train", download=True, transform=transform_rgb),
        "test": torchvision.datasets.SVHN(root="./data", split="test", download=True, transform=transform_rgb)
    }
    
    return datasets

# Helper to prepare dataloaders
def prepare_dataloaders(datasets):
    train_loaders = {}
    test_loaders = {}
    
    for task_name, task_ds in datasets.items():
        train_loaders[task_name] = DataLoader(
            task_ds["train"], batch_size=256, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loaders[task_name] = DataLoader(
            task_ds["test"], batch_size=256, shuffle=False, num_workers=4, pin_memory=True
        )
        
    return train_loaders, test_loaders

# Define Joint MTL Model
class JointMTLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.heads = nn.ModuleDict({
            "MNIST": nn.Linear(in_features, 10),
            "FashionMNIST": nn.Linear(in_features, 10),
            "CIFAR10": nn.Linear(in_features, 10),
            "SVHN": nn.Linear(in_features, 10)
        })
        
    def forward(self, x, task_name):
        features = self.backbone(x)
        return self.heads[task_name](features)

def main():
    datasets = load_task_datasets()
    train_loaders, test_loaders = prepare_dataloaders(datasets)
    
    print("\nInitializing Joint MTL Model...")
    model = JointMTLModel().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    print("Starting Joint Multi-Task Learning Training (3 Epochs)...")
    
    for epoch in range(3):
        model.train()
        running_loss = 0.0
        total_steps = 0
        
        # Zip the loaders together (will iterate for the length of the shortest loader)
        zipped_loaders = zip(
            train_loaders["MNIST"],
            train_loaders["FashionMNIST"],
            train_loaders["CIFAR10"],
            train_loaders["SVHN"]
        )
        
        for step, (batch_mnist, batch_fmnist, batch_cifar, batch_svhn) in enumerate(zipped_loaders):
            optimizer.zero_grad()
            
            # 1. MNIST Forward & Backward
            imgs, labels = batch_mnist[0].to(device), batch_mnist[1].to(device)
            with torch.cuda.amp.autocast():
                out = model(imgs, "MNIST")
                loss_mnist = criterion(out, labels)
            
            # 2. FashionMNIST Forward & Backward
            imgs, labels = batch_fmnist[0].to(device), batch_fmnist[1].to(device)
            with torch.cuda.amp.autocast():
                out = model(imgs, "FashionMNIST")
                loss_fmnist = criterion(out, labels)
                
            # 3. CIFAR10 Forward & Backward
            imgs, labels = batch_cifar[0].to(device), batch_cifar[1].to(device)
            with torch.cuda.amp.autocast():
                out = model(imgs, "CIFAR10")
                loss_cifar = criterion(out, labels)
                
            # 4. SVHN Forward & Backward
            imgs, labels = batch_svhn[0].to(device), batch_svhn[1].to(device)
            with torch.cuda.amp.autocast():
                out = model(imgs, "SVHN")
                loss_svhn = criterion(out, labels)
                
            # Combine losses
            total_loss = loss_mnist + loss_fmnist + loss_cifar + loss_svhn
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            total_steps += 1
            
            if (step + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/3 - Step {step+1} - Combined Loss: {total_loss.item():.4f}")
                
        print(f"Epoch {epoch+1}/3 Complete. Avg Step Combined Loss: {running_loss/total_steps:.4f}")
        
    # Evaluate Joint MTL model on each task
    print("\n--- Evaluating Joint MTL Model on Test Sets ---")
    model.eval()
    test_accuracies = {}
    
    with torch.no_grad():
        for task_name, loader in test_loaders.items():
            correct = 0
            total = 0
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs, task_name)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            acc = 100.0 * correct / total
            test_accuracies[task_name] = acc
            print(f"Joint MTL Test Accuracy for {task_name}: {acc:.2f}%")
            
    joint_mean = np.mean(list(test_accuracies.values()))
    test_accuracies["Joint Mean"] = joint_mean
    print(f"\nJoint MTL Mean Accuracy: {joint_mean:.2f}%")
    
    # Save checkpoint
    torch.save(model.state_dict(), "./checkpoints/MTL_joint.pt")
    print("Saved Joint MTL checkpoint to ./checkpoints/MTL_joint.pt")
    
    # Save metrics to json
    with open("./results/mtl_metrics.json", "w") as f:
        json.dump(test_accuracies, f, indent=4)
    print("Saved MTL metrics to ./results/mtl_metrics.json")

if __name__ == "__main__":
    main()
