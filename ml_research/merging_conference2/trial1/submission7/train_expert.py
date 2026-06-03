import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm

def get_args():
    parser = argparse.ArgumentParser(description="Train a task-specific expert model")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "svhn"], help="Dataset to train on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {args.dataset} expert with seed {args.seed} on {device}")

    # Transforms
    if args.dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_test)
    else: # svhn
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        train_dataset = torchvision.datasets.SVHN(root="./data", split="train", download=False, transform=transform_train)
        test_dataset = torchvision.datasets.SVHN(root="./data", split="test", download=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model definition
    # Backbone is pre-trained ResNet-18
    backbone = timm.create_model("resnet18", pretrained=True, num_classes=0)
    # Head is task-specific (10 output classes)
    head = nn.Linear(512, 10)
    
    model = nn.Sequential(
        backbone,
        head
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    for epoch in range(args.epochs):
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
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.2f}%")

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
            
    val_loss = test_loss / total
    val_acc = 100.0 * correct / total
    print(f"Final Test - Loss: {val_loss:.4f} - Acc: {val_acc:.2f}%")

    # Save checkpoints
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_path = f"./checkpoints/{args.dataset}_seed{args.seed}.pth"
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'head_state_dict': head.state_dict(),
        'val_acc': val_acc
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
