import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# Disable cuDNN due to environment cuDNN mismatch
torch.backends.cudnn.enabled = False

# Create models directory
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def get_transforms(task):
    if task in ["mnist", "fashionmnist"]:
        # Grayscale to RGB via repeating channels, resize to 32x32
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
    elif task == "cifar10":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        raise ValueError(f"Unknown task {task}")

def get_dataset(task, train=True):
    transform = get_transforms(task)
    if task == "mnist":
        return datasets.MNIST("data", train=train, download=True, transform=transform)
    elif task == "fashionmnist":
        return datasets.FashionMNIST("data", train=train, download=True, transform=transform)
    elif task == "cifar10":
        return datasets.CIFAR10("data", train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown task {task}")

def train_task(task, epochs=5, lr=1e-3, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {task} on {device}...")

    # Load shared progenitor ResNet-18 (pre-trained)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Save base model if it does not exist yet
    base_path = "models/resnet18_base.pt"
    if not os.path.exists(base_path):
        torch.save(model.state_dict(), base_path)
        print(f"Saved base pre-trained progenitor model to {base_path}")

    # Replace head for 10 classes
    model.fc = nn.Linear(512, 10)
    model = model.to(device)

    # Get data
    train_dataset = get_dataset(task, train=True)
    val_dataset = get_dataset(task, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
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

        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}% | Val Loss: {val_loss/val_total:.4f} Val Acc: {val_acc*100:.2f}%")

    # Save model weights
    save_path = f"models/resnet18_{task}.pt"
    # Save state dict
    torch.save(model.state_dict(), save_path)
    print(f"Saved expert model for {task} to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["mnist", "fashionmnist", "cifar10", "all"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    if args.task == "all":
        for t in ["mnist", "fashionmnist", "cifar10"]:
            train_task(t, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    else:
        train_task(args.task, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
