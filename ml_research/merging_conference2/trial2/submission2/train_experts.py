import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import os
import random
import numpy as np

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False  # Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on Slurm nodes

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms for 3-channel 32x32 inputs
transform_cifar_svhn = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_fmnist = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_expert_data(dataset_name):
    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_cifar_svhn)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_cifar_svhn)
    elif dataset_name == "svhn":
        train_dataset = datasets.SVHN(root="./data", split="train", download=True, transform=transform_cifar_svhn)
        test_dataset = datasets.SVHN(root="./data", split="test", download=True, transform=transform_cifar_svhn)
    elif dataset_name == "fmnist":
        train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_fmnist)
        test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_fmnist)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create a random subset of 2000 samples for training
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    subset_train = Subset(train_dataset, indices[:2000])

    train_loader = DataLoader(subset_train, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    return train_loader, test_loader

def train_expert(dataset_name, epochs=10):
    print(f"\n--- Training Expert on {dataset_name.upper()} ---")
    train_loader, test_loader = get_expert_data(dataset_name)

    # Load pre-trained ResNet-18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Replace the FC layer for 10 classes
    model.fc = nn.Linear(512, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / total
    test_acc = 100.0 * correct / total
    print(f"Test Accuracy on {dataset_name.upper()}: {test_acc:.2f}%")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/expert_{dataset_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", choices=["cifar10", "svhn", "fmnist", "all"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    if args.dataset == "all":
        for dataset in ["cifar10", "svhn", "fmnist"]:
            train_expert(dataset, epochs=args.epochs)
    else:
        train_expert(args.dataset, epochs=args.epochs)
