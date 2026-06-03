import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import argparse
from data import get_splits

def train_model(dataset_name, epochs=5, batch_size=128, device='cuda', save_dir='./models'):
    print(f"\n--- Training Expert for {dataset_name} ---")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    ft_ds, _, test_ds = get_splits(dataset_name)
    train_loader = DataLoader(ft_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize ResNet-18
    # Using the new weights parameter as of torchvision 0.13+
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    # Replace final classification head
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    # Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        
    # Evaluate on Test Set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    test_acc = 100.0 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Save the model
    save_path = os.path.join(save_dir, f"{dataset_name}_expert.pt")
    # Save full state dict
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all', choices=['mnist', 'fashion_mnist', 'cifar10', 'all'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    datasets = ['mnist', 'fashion_mnist', 'cifar10'] if args.dataset == 'all' else [args.dataset]
    for ds in datasets:
        train_model(ds, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
