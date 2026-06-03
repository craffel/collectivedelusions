import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_datasets
from models import MultiTaskResNet18, extract_expert
import os

torch.backends.cudnn.enabled = False

def train_expert(task_id, train_loader, test_loader, num_epochs=5, lr=5e-4, weight_decay=1e-4, device="cuda"):
    print(f"\n--- Training Expert for Task {task_id} ---")
    model = MultiTaskResNet18(pretrained=True)
    model.to(device)
    
    task_mapping = {0: 'mnist', 1: 'fmnist', 2: 'cifar10'}
    task_name = task_mapping[task_id]
    
    # We only train the backbone and the task-specific head.
    # To be extremely clean, we can freeze the other heads by setting their gradients to zero/not passing to optimizer,
    # or just only optimize parameters of backbone and the specific head.
    params_to_optimize = []
    for name, param in model.named_parameters():
        if "backbone" in name or f"heads.{task_name}" in name:
            param.requires_grad = True
            params_to_optimize.append(param)
        else:
            param.requires_grad = False
            
    optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits = model(imgs, task_id=task_id)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        # Eval on test
        test_acc = evaluate_model(model, test_loader, task_id, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
    return model

def evaluate_model(model, test_loader, task_id, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs, task_id=task_id)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Save the base model (ImageNet pre-trained weights) before any training
    base_model = MultiTaskResNet18(pretrained=True)
    torch.save(base_model.state_dict(), "./checkpoints/base_model.pth")
    print("Saved base pre-trained model.")
    
    # Load datasets (using 5000 images per task for higher accuracy, or 2000 for faster training)
    # Let's use 3000 images, which is a good sweet spot for solid performance and rapid training.
    num_train_samples = 3000
    dataloaders = load_datasets(data_dir="./data", num_train_samples=num_train_samples, seed=42)
    
    experts = {}
    task_mapping = {0: 'mnist', 1: 'fmnist', 2: 'cifar10'}
    
    for task_id in [0, 1, 2]:
        task_name = task_mapping[task_id]
        train_loader, test_loader = dataloaders[task_name]
        
        # Train
        expert_model = train_expert(
            task_id=task_id,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=5,
            lr=5e-4,
            weight_decay=1e-4,
            device=device
        )
        
        # Save expert state dict
        torch.save(expert_model.state_dict(), f"./checkpoints/expert_{task_name}.pth")
        print(f"Saved expert for task {task_name}.")
        experts[task_name] = expert_model

if __name__ == "__main__":
    main()
