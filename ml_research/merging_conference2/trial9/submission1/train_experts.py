import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)

class ExpertModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

def get_dataloader(dataset_name, batch_size=256, is_train=True):
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.MNIST(root="./data", train=is_train, download=True, transform=transform)
    elif dataset_name == "fmnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.FashionMNIST(root="./data", train=is_train, download=True, transform=transform)
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        dataset = datasets.CIFAR10(root="./data", train=is_train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    # Standard stable test subset of 1000 samples for evaluation
    if not is_train:
        dataset = Subset(dataset, list(range(1000)))
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4, pin_memory=True)
    return loader

def train_expert(dataset_name):
    print(f"\n--- Training Expert for {dataset_name.upper()} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model from progenitor
    model = ExpertModel().to(device)
    
    # Save a clean progenitor model at the start (just once)
    if not os.path.exists("checkpoints/progenitor.pt"):
        # Save progenitor state dict
        torch.save(model.state_dict(), "checkpoints/progenitor.pt")
        print("Saved progenitor progenitor.pt")
        
    train_loader = get_dataloader(dataset_name, batch_size=256, is_train=True)
    test_loader = get_dataloader(dataset_name, batch_size=256, is_train=False)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
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
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
        
        # Eval at epoch end
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        test_acc = 100. * test_correct / test_total
        print(f"Epoch {epoch}/{epochs} | Test Acc: {test_acc:.2f}%")
        
    # Save the expert model
    torch.save(model.state_dict(), f"checkpoints/{dataset_name}_expert.pt")
    print(f"Saved checkpoints/{dataset_name}_expert.pt")

if __name__ == "__main__":
    # Disable cuDNN to bypass driver issues as per submission5 instructions
    torch.backends.cudnn.enabled = False
    
    train_expert("mnist")
    train_expert("fmnist")
    train_expert("cifar10")
    print("\nAll experts trained and checkpoints saved successfully!")
