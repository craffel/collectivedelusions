import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from data import get_multi_task_datasets

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def train_expert(dataset_name, epochs=5, lr=5e-4, weight_decay=1e-4, batch_size=64, device='cuda', seed=42):
    set_seed(seed)
    print(f"Loading datasets for task: {dataset_name}")
    train_datasets, test_datasets = get_multi_task_datasets(seed=seed)
    train_set = train_datasets[dataset_name]
    test_set = test_datasets[dataset_name]
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Initializing pre-trained ResNet-18...")
    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        # Fallback for older torchvision
        model = resnet18(pretrained=True)
        
    # Replace classification head for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # Enable dropout to prevent overfitting as per papers
    # Add a dropout layer before the final classifier
    model.fc = nn.Sequential(
        nn.Dropout(p=0.1),
        model.fc
    )
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    print(f"Starting training on {device}...")
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
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
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
            
    test_acc = 100.0 * correct / total
    print(f"Finished. Test Loss: {test_loss/len(test_loader.dataset):.4f} - Test Acc: {test_acc:.2f}%")
    
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f"checkpoints/{dataset_name}_expert.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Expert weights saved to {checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['mnist', 'fmnist', 'cifar10'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    train_expert(args.task, epochs=args.epochs, lr=args.lr, device=args.device, seed=args.seed)
