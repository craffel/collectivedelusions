import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_datasets, get_dataloaders
from models import create_expert_model, save_checkpoint

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on some cluster nodes
torch.backends.cudnn.enabled = False

def train_expert(task, epochs=5, batch_size=128, lr=5e-4, weight_decay=1e-4, device='cuda'):
    print(f"=== Training Expert for Task: {task} ===")
    
    # Load dataset
    splits = get_datasets()
    loaders = get_dataloaders(splits, batch_size=batch_size)
    task_loaders = loaders[task]
    
    # Create model
    model = create_expert_model(num_classes=10)
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in task_loaders['train']:
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
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    # Evaluate on Test Set
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in task_loaders['test']:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100. * test_correct / test_total
    print(f"Final Test Accuracy on {task}: {test_acc:.2f}%")
    
    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    save_path = f"checkpoints/{task}_expert.pt"
    save_checkpoint(model, save_path)
    print(f"Saved expert checkpoint to {save_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Expert Models')
    parser.add_argument('--task', type=str, required=True, choices=['mnist', 'fmnist', 'cifar10'], help='Task to train')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    train_expert(task=args.task, epochs=args.epochs, lr=args.lr, device=args.device)
