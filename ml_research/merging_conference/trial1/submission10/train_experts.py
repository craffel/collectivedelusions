import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

def get_transforms(task):
    if task == 'fashion_mnist':
        # FashionMNIST is 1-channel, so we convert to 3-channel and normalize
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # CIFAR-10 and SVHN are 3-channel
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataset(task, train=True):
    transform = get_transforms(task)
    if task == 'cifar10':
        return datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif task == 'svhn':
        split = 'train' if train else 'test'
        return datasets.SVHN(root='./data', split=split, download=True, transform=transform)
    elif task == 'fashion_mnist':
        return datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown task: {task}")

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
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
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 on vision datasets")
    parser.add_argument("--task", type=str, required=True, choices=["cifar10", "svhn", "fashion_mnist"], help="Task name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test on CPU")
    args = parser.parse_args()

    print(f"Starting fine-tuning for {args.task} on {device}...")
    os.makedirs("checkpoints", exist_ok=True)

    # Initialize model
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Replace classification head
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # Get datasets
    train_set = get_dataset(args.task, train=True)
    test_set = get_dataset(args.task, train=False)

    if args.smoke_test:
        print("Running smoke test (subset of dataset)...")
        train_set = Subset(train_set, range(128))
        test_set = Subset(test_set, range(128))
        args.epochs = 1
        args.batch_size = 32

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

    if not args.smoke_test:
        save_path = f"checkpoints/expert_{args.task}.pt"
        torch.save({
            'state_dict': model.state_dict(),
            'task': args.task,
            'val_acc': val_acc
        }, save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
