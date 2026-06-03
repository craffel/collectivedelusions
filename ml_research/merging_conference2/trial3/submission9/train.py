import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on cluster nodes
torch.backends.cudnn.enabled = False

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)

def get_transforms():
    # Common transforms: resize to 32x32, convert to RGB, and normalize with ImageNet stats
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Repeat 1-channel grayscale to 3-channel RGB
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform_rgb, transform_gray

def get_dataset(task, is_train=True):
    transform_rgb, transform_gray = get_transforms()
    if task == "mnist":
        return datasets.MNIST(root="data", train=is_train, download=True, transform=transform_gray)
    elif task == "fashion":
        return datasets.FashionMNIST(root="data", train=is_train, download=True, transform=transform_gray)
    elif task == "cifar10":
        return datasets.CIFAR10(root="data", train=is_train, download=True, transform=transform_rgb)
    else:
        raise ValueError(f"Unknown task: {task}")

def initialize_base_model():
    # Load pre-trained ResNet-18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Modify the classification head to output 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    # Initialize head weights properly
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    return model

def train_expert(task, epochs=5, batch_size=128, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Load base model checkpoint or initialize if it doesn't exist
    base_path = "checkpoints/base.pt"
    if not os.path.exists(base_path):
        print("Base model checkpoint not found. Creating and saving base model...")
        model = initialize_base_model()
        torch.save(model.state_dict(), base_path)
    else:
        print("Loading base model checkpoint...")
        model = initialize_base_model()
        model.load_state_dict(torch.load(base_path, map_location="cpu"))
        
    model = model.to(device)
    
    train_dataset = get_dataset(task, is_train=True)
    test_dataset = get_dataset(task, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
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
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = (correct / total) * 100.0
        
        # Validation
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
        val_acc = (val_correct / val_total) * 100.0
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"checkpoints/{task}.pt")
            print(f"Saved new best checkpoint with Val Acc: {best_acc:.2f}%")
            
    print(f"Finished training for {task}. Best Val Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["mnist", "fashion", "cifar10", "base"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    if args.task == "base":
        print("Initializing and saving base model checkpoint...")
        model = initialize_base_model()
        torch.save(model.state_dict(), "checkpoints/base.pt")
        print("Base model saved to checkpoints/base.pt")
    else:
        train_expert(args.task, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
