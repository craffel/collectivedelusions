import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

def get_dataloader(dataset_name, batch_size, is_train=True):
    # Grayscale transforms (MNIST, Fashion-MNIST)
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Replicate grayscale to 3 channels
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
        ])
        dataset = datasets.MNIST(root="./data", train=is_train, download=True, transform=transform)
    elif dataset_name == "fmnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
        ])
        dataset = datasets.FashionMNIST(root="./data", train=is_train, download=True, transform=transform)
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = datasets.CIFAR10(root="./data", train=is_train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0, pin_memory=True)
    return loader

def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 Expert on Vision Suite")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "fmnist", "cifar10"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.enabled = False

    print(f"Training parameters: Dataset={args.dataset}, Epochs={args.epochs}, WD={args.weight_decay}, LR={args.lr}, Seed={args.seed}")

    # Output directory
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/{args.dataset}_epochs{args.epochs}_wd{args.weight_decay}_seed{args.seed}.pth"

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint already exists at {checkpoint_path}. Skipping training.")
        return

    # Load progenitor (ImageNet pre-trained ResNet-18)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    
    # Modify classification head for 10 classes
    model.fc = nn.Linear(512, 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model loaded and transferred to device.")

    # Dataloaders
    train_loader = get_dataloader(args.dataset, args.batch_size, is_train=True)
    test_loader = get_dataloader(args.dataset, args.batch_size, is_train=False)
    print("Dataloaders initialized.")

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    print("Optimizer and criterion initialized.")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        print(f"Starting epoch {epoch}...")
        batch_idx = 0
        for inputs, targets in train_loader:
            if batch_idx == 0:
                print("  First training batch loaded. Transferring to device...")
            inputs, targets = inputs.to(device), targets.to(device)
            if batch_idx == 0:
                print("  First training batch transferred. Running forward pass...")
            optimizer.zero_grad()
            outputs = model(inputs)
            if batch_idx == 0:
                print("  Forward pass complete. Running backward pass...")
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if batch_idx == 0:
                print("  First backward pass and step complete.")

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            batch_idx += 1
            if batch_idx % 50 == 0:
                print(f"  Processed training batch {batch_idx}/{len(train_loader)}...")

        epoch_loss = train_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch}/{args.epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")

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

    test_loss /= total
    test_acc = 100.0 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Save model checkpoint
    torch.save({
        "state_dict": model.state_dict(),
        "test_acc": test_acc,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "dataset": args.dataset
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}\n")

if __name__ == "__main__":
    main()
