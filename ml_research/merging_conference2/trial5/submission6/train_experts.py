import torch
import torch.nn as nn
import torchvision.models as models
from datasets_utils import get_dataloaders
import os

def train_one_epoch(model, loader, optimizer, criterion, device):
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

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    loaders = get_dataloaders(batch_size=128)
    
    # Save base model
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Linear(512, 10) # dummy head to keep structure identical
    torch.save(base_model.state_dict(), 'base_model.pt')
    print("Base model saved.")
    
    tasks = ['mnist', 'fashion', 'cifar']
    for task in tasks:
        print(f"\n--- Training Expert for {task.upper()} ---")
        
        # Load fresh ImageNet ResNet-18
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Replace head with 10 classes
        model.fc = nn.Linear(512, 10)
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        criterion = nn.CrossEntropyLoss()
        
        ft_loader = loaders[task]['ft']
        test_loader = loaders[task]['test']
        
        for epoch in range(1, 6):
            loss, acc = train_one_epoch(model, ft_loader, optimizer, criterion, device)
            scheduler.step()
            print(f"Epoch {epoch}/5 - Loss: {loss:.4f} - Train Acc: {acc*100:.2f}%")
            
        test_acc = evaluate(model, test_loader, device)
        print(f"Final Test Accuracy on {task.upper()}: {test_acc*100:.2f}%")
        
        # Save expert model state
        torch.save(model.state_dict(), f"{task}_expert.pt")
        print(f"Saved {task}_expert.pt")

if __name__ == '__main__':
    main()
