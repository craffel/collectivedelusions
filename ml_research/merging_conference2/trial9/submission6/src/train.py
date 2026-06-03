import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataset
from src.models import get_model

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
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
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

def main():
    parser = argparse.ArgumentParser(description="Train Expert Models")
    parser.add_argument("--arch", type=str, required=True, choices=["resnet18", "mlp"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # Disable cuDNN due to driver compatibility issues on this cluster
    torch.backends.cudnn.enabled = False

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Using device: {device}")
    
    # 1. Create and save the Progenitor Model
    print(f"Creating progenitor model for {args.arch}...")
    progenitor = get_model(args.arch, num_classes=10, pretrained=True)
    progenitor.to(device)
    
    progenitor_path = os.path.join(args.out_dir, f"{args.arch}_progenitor.pt")
    torch.save(progenitor.state_dict(), progenitor_path)
    print(f"Progenitor saved to {progenitor_path}")

    # 2. Train experts for each task
    tasks = ["mnist", "fmnist", "cifar10"]
    
    for task in tasks:
        print(f"\n--- Training Expert for {task.upper()} on {args.arch} ---")
        
        # Reload progenitor weights to start from the shared progenitor
        model = get_model(args.arch, num_classes=10, pretrained=False)
        model.load_state_dict(torch.load(progenitor_path, map_location="cpu"))
        model.to(device)
        
        train_loader = get_dataset(task, train=True, batch_size=args.batch_size)
        test_loader = get_dataset(task, train=False, batch_size=args.batch_size)
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
            
        expert_path = os.path.join(args.out_dir, f"{args.arch}_{task}_expert.pt")
        torch.save(model.state_dict(), expert_path)
        print(f"Expert model for {task} saved to {expert_path}")

if __name__ == "__main__":
    main()
