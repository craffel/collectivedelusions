import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from utils import get_dataloaders, evaluate_model

def train_expert(task_name, loaders, device):
    print("\n" + "="*50)
    print(f"Training expert for task: {task_name.upper()}")
    print("="*50)
    
    # 1. Initialize ImageNet pre-trained ResNet-18
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace the classification head for 10 classes
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    # 2. Get data loaders
    train_loader = loaders[task_name]['train']
    test_loader = loaders[task_name]['test']
    
    # 3. Setup optimizer, loss, scheduler
    # AdamW with lr=1e-4, weight_decay=1e-4 as in Paper 1
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
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
        epoch_acc = 100.0 * correct / total
        test_acc = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
    # Save the expert checkpoint
    os.makedirs('./checkpoints', exist_ok=True)
    checkpoint_path = f"./checkpoints/{task_name}_expert.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    return model

def main():
    # Disable cuDNN to bypass driver compatibility issues as noted in Paper 9
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Save the progenitor model (with 10-class head, un-finetuned)
    # This acts as our theta_init
    print("Saving progenitor network...")
    progenitor = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    progenitor.fc = nn.Linear(512, 10)
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(progenitor.state_dict(), './checkpoints/progenitor.pth')
    
    # Load loaders
    loaders = get_dataloaders(batch_size=256, num_workers=4)
    
    # Train experts
    for task in ['mnist', 'fmnist', 'cifar10']:
        train_expert(task, loaders, device)
        
    print("\nAll experts trained successfully!")

if __name__ == '__main__':
    main()
