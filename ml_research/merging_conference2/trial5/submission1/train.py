import os
import argparse
import torch
# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on this cluster setup
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Define Dataset transform helper
def get_transforms(dataset_name):
    if dataset_name in ["mnist", "fashion"]:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_name == "cifar10":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# Get Dataset classes
def get_dataset(dataset_name, train=True):
    transform = get_transforms(dataset_name)
    if dataset_name == "mnist":
        return torchvision.datasets.MNIST(root="./data", train=train, download=False, transform=transform)
    elif dataset_name == "fashion":
        return torchvision.datasets.FashionMNIST(root="./data", train=train, download=False, transform=transform)
    elif dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10(root="./data", train=train, download=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def train_expert(dataset_name, device_idx=0, epochs=5):
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    print(f"Training {dataset_name} expert on device: {device}...")

    # Load Full Dataset
    full_train_dataset = get_dataset(dataset_name, train=True)
    full_test_dataset = get_dataset(dataset_name, train=False)

    # Subset the datasets
    # Fine-Tuning Set (first 5,000 samples)
    train_subset = Subset(full_train_dataset, list(range(5000)))
    # Calibration Set (next 128 samples: indices 5000 to 5127)
    cal_subset = Subset(full_train_dataset, list(range(5000, 5128)))
    # Test Set (full 10,000 samples)
    test_subset = full_test_dataset

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=2)

    # Instantiate ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace the classification head
    model.fc = nn.Linear(512, 10)
    model = model.to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
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
        print(f"Epoch [{epoch}/{epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")

    # Evaluation
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100. * correct / total
    print(f"Finished training {dataset_name} expert! Test Acc: {test_acc:.2f}%")

    # Ensure output folder exists
    os.makedirs("./checkpoints", exist_ok=True)
    save_path = f"./checkpoints/{dataset_name}_expert.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved {dataset_name} expert weights to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "fashion", "cifar10", "all"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    if args.dataset == "all":
        # Train all datasets sequentially (or we can launch multiple python commands if we want parallel)
        for ds in ["mnist", "fashion", "cifar10"]:
            train_expert(ds, device_idx=args.device, epochs=args.epochs)
    else:
        train_expert(args.dataset, device_idx=args.device, epochs=args.epochs)
