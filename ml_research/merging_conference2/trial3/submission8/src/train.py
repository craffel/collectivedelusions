import os
import argparse
import torch
# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED on the GPU partition
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from models import MultiTaskResNet18

def train_expert(task, epochs=5, lr=5e-4, weight_decay=1e-4, batch_size=128, subset_size=5000, seed=42, device='cuda'):
    print(f"\n=========================================")
    print(f"Training Expert for Task: {task.upper()}")
    print(f"=========================================")
    
    # Get dataloaders
    train_loaders, test_loaders = get_dataloaders(batch_size=batch_size, subset_size=subset_size, seed=seed)
    train_loader = train_loaders[task]
    test_loader = test_loaders[task]
    
    # Initialize multi-task model
    model = MultiTaskResNet18(pretrained=True).to(device)
    
    # Define optimizer: AdamW with lr and weight decay
    # We only optimize parameters in backbone and the specific task head
    optimizer_params = list(model.backbone.parameters()) + list(model.heads[task].parameters())
    optimizer = optim.AdamW(optimizer_params, lr=lr, weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, task)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = total_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
        
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, task)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    test_loss /= total
    test_acc = 100.0 * correct / total
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f'checkpoints/expert_{task}.pt'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved expert checkpoint to {checkpoint_path}")
    
    return test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ResNet-18 expert models.")
    parser.add_argument('--task', type=str, required=True, choices=['mnist', 'fashion', 'cifar', 'all'],
                        help="Task to train ('mnist', 'fashion', 'cifar', or 'all' to train all sequentially).")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--subset_size', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Save initial pre-trained model for task arithmetic baseline reference
    pretrained_path = 'checkpoints/pretrained.pt'
    if not os.path.exists(pretrained_path):
        os.makedirs('checkpoints', exist_ok=True)
        init_model = MultiTaskResNet18(pretrained=True)
        torch.save(init_model.state_dict(), pretrained_path)
        print(f"Saved baseline pretrained model to {pretrained_path}")
        
    if args.task == 'all':
        for t in ['mnist', 'fashion', 'cifar']:
            train_expert(t, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                         batch_size=args.batch_size, subset_size=args.subset_size, seed=args.seed, device=device)
    else:
        train_expert(args.task, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                     batch_size=args.batch_size, subset_size=args.subset_size, seed=args.seed, device=device)
