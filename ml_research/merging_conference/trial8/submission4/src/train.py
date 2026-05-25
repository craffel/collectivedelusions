import os
import torch
import torch.optim as optim
import torch.nn as nn
from models import SimpleCNN
from data import download_datasets, get_expert_loaders

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

def train_expert(name, train_loader, test_loader, base_checkpoint_path, save_checkpoint_path, epochs=2, lr=1e-3, device="cpu"):
    print(f"\n--- Training Expert: {name.upper()} ---")
    model = SimpleCNN().to(device)
    # Load shared base weights
    model.load_state_dict(torch.load(base_checkpoint_path, map_location=device))
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
            
        train_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.4f}")
        
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, predicted = logits.max(1)
            test_correct += predicted.eq(y).sum().item()
            test_total += y.size(0)
            
    test_acc = test_correct / test_total
    print(f"Test Accuracy for {name.upper()} expert: {test_acc:.4f}")
    
    # Save checkpoint
    torch.save(model.state_dict(), save_checkpoint_path)
    print(f"Saved expert checkpoint to {save_checkpoint_path}")
    return test_acc

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs("checkpoints", exist_ok=True)
    base_checkpoint_path = "checkpoints/base_model.pt"
    
    # Save base model initialization
    base_model = SimpleCNN()
    torch.save(base_model.state_dict(), base_checkpoint_path)
    print(f"Saved shared base model initialization to {base_checkpoint_path}")
    
    # Download and load data
    datasets_dict = download_datasets()
    expert_loaders = get_expert_loaders(datasets_dict)
    
    # Train each expert
    train_expert("mnist", expert_loaders["mnist"]["train"], expert_loaders["mnist"]["test"], base_checkpoint_path, "checkpoints/mnist_expert.pt", device=device)
    train_expert("kmnist", expert_loaders["kmnist"]["train"], expert_loaders["kmnist"]["test"], base_checkpoint_path, "checkpoints/kmnist_expert.pt", device=device)
    train_expert("fmnist", expert_loaders["fmnist"]["train"], expert_loaders["fmnist"]["test"], base_checkpoint_path, "checkpoints/fmnist_expert.pt", device=device)

if __name__ == "__main__":
    main()
