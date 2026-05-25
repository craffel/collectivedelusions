import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Datasets path
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Transforms
transform_rgb = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
def get_dataloaders(task, batch_size=128):
    if task == "cifar10":
        train_set = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_rgb)
        test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_rgb)
    elif task == "svhn":
        train_set = datasets.SVHN(root=DATA_DIR, split="train", download=True, transform=transform_rgb)
        test_set = datasets.SVHN(root=DATA_DIR, split="test", download=True, transform=transform_rgb)
    elif task == "mnist":
        train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform_gray)
        test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform_gray)
    else:
        raise ValueError(f"Unknown task: {task}")
        
    # Take a subset to make CPU/GPU training extremely fast (2000 train, 500 test)
    train_indices = list(range(min(2000, len(train_set))))
    test_indices = list(range(min(500, len(test_set))))
    train_subset = Subset(train_set, train_indices)
    test_subset = Subset(test_set, test_indices)
        
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# Define Model Creator
def get_resnet18(num_classes=10):
    # Use standard weights
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Train loop
def train_task(task, epochs=5, lr=1e-4):
    print(f"\n--- Training Expert for Task: {task.upper()} ---")
    train_loader, test_loader = get_dataloaders(task)
    
    model = get_resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
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
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
        
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100. * test_correct / test_total
    print(f"Expert model for {task} evaluated on test set: {test_acc:.2f}%")
    
    # Save model
    save_path = os.path.join(MODEL_DIR, f"resnet18_{task}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved expert weights to {save_path}")

if __name__ == "__main__":
    # Save the base pre-trained resnet18 model first
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_save_path = os.path.join(MODEL_DIR, "resnet18_base.pt")
    torch.save(base_model.state_dict(), base_save_path)
    print(f"Saved base pre-trained model weights to {base_save_path}")
    
    # Train each task
    train_task("cifar10", epochs=5, lr=1e-4)
    train_task("svhn", epochs=5, lr=1e-4)
    train_task("mnist", epochs=5, lr=1e-4)
