import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
from models import ResNet18CIFAR, MLPCIFAR
from data_utils import get_dataloader

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
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

def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    test_loss = running_loss / total
    test_acc = correct / total
    return test_loss, test_acc

def main():
    parser = argparse.ArgumentParser(description="Train model merging experts")
    parser.add_argument("--model_type", type=str, required=True, choices=["resnet18", "mlp"], help="Model architecture")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "fmnist", "cifar10"], help="Dataset to fine-tune on")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_progenitor", action="store_true", help="Initialize and save the progenitor base model")
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
        torch.backends.cudnn.enabled = False  # Disable cuDNN to avoid initialization errors
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    os.makedirs("checkpoints", exist_ok=True)
    
    # Define model
    if args.model_type == "resnet18":
        model = ResNet18CIFAR(num_classes=10)
        progenitor_path = "checkpoints/progenitor_resnet18.pt"
    else:
        model = MLPCIFAR(num_classes=10)
        progenitor_path = "checkpoints/progenitor_mlp.pt"

    # If save_progenitor is requested, we just save the initial weights and exit
    if args.save_progenitor:
        torch.save(model.state_dict(), progenitor_path)
        print(f"Saved un-trained progenitor model to {progenitor_path}")
        return

    # Otherwise, we load the progenitor weights first to ensure aligned starting point
    if os.path.exists(progenitor_path):
        model.load_state_dict(torch.load(progenitor_path, map_location="cpu"))
        print(f"Loaded progenitor weights from {progenitor_path}")
    else:
        print(f"Progenitor path {progenitor_path} not found! Initializing fresh base model.")
        torch.save(model.state_dict(), progenitor_path)
        print(f"Saved newly initialized progenitor base to {progenitor_path}")

    model = model.to(device)

    # Dataloaders
    train_loader = get_dataloader(args.dataset, batch_size=args.batch_size, train=True)
    test_loader = get_dataloader(args.dataset, batch_size=args.batch_size, train=False)

    # Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting training on {args.dataset} for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | Train Loss: {loss:.4f} Acc: {acc*100:.2f}% | Test Loss: {test_loss:.4f} Acc: {test_acc*100:.2f}%")

    expert_path = f"checkpoints/expert_{args.dataset}_{args.model_type}.pt"
    torch.save(model.state_dict(), expert_path)
    print(f"Saved expert weights to {expert_path}")

if __name__ == "__main__":
    main()
