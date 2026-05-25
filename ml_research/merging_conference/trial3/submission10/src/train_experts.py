import os
import torch
import torch.nn as nn
import torch.optim as optim

# Disable cuDNN due to local cluster compatibility issues
torch.backends.cudnn.enabled = False

from data import get_expert_dataloaders
from models import MultiTaskResNet18, get_base_state_dict

def train_expert(task_id, train_loader, device, num_epochs=3, lr=1e-4):
    print(f"\n--- Training Expert {task_id} ---")
    # Load fresh pre-trained model
    model = MultiTaskResNet18(num_tasks=3, num_classes=10).to(device)
    
    # We only train model.resnet (encoder) and model.heads[task_id] (head)
    optimizer = optim.Adam(
        list(model.resnet.parameters()) + list(model.heads[task_id].parameters()),
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs, task_id)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, predicted = logits.max(1)
            total += lbls.size(0)
            correct += predicted.eq(lbls).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    # Evaluate on the final train subset
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs, task_id)
            _, predicted = logits.max(1)
            total += lbls.size(0)
            correct += predicted.eq(lbls).sum().item()
    print(f"Final Train Accuracy: {100.0 * correct / total:.2f}%")
    
    # Save the encoder and head weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.resnet.state_dict(), f"checkpoints/expert_{task_id}_encoder.pt")
    torch.save(model.heads[task_id].state_dict(), f"checkpoints/expert_{task_id}_head.pt")
    print(f"Expert {task_id} weights saved successfully.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Get dataloaders
    mnist_loader, fmnist_loader, kmnist_loader = get_expert_dataloaders(img_size=32, batch_size=128, num_train_samples=10000)
    
    # Train experts
    train_expert(0, mnist_loader, device, num_epochs=3, lr=1e-4)
    train_expert(1, fmnist_loader, device, num_epochs=3, lr=1e-4)
    train_expert(2, kmnist_loader, device, num_epochs=3, lr=1e-4)
