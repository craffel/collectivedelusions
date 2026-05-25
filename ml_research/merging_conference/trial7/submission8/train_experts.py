import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from models import SimpleCNN
import os

# Set seed for reproducibility
torch.manual_seed(42)

def train_model(dataset_name, dataset_class, save_path, epochs=2, subset_size=10000):
    print(f"\nTraining {dataset_name} Expert...")
    # Load base model to ensure same starting point
    model = SimpleCNN()
    model.load_state_dict(torch.load("base_model.pt"))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = dataset_class(root='data', train=True, download=True, transform=transform)
    # Subset to subset_size
    train_subset = Subset(train_dataset, range(min(subset_size, len(train_dataset))))
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")
        
    torch.save(model.state_dict(), save_path)
    print(f"Saved {dataset_name} Expert to {save_path}")
    return model

def evaluate_model(model, dataset_name, dataset_class, subset_size=2000):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = dataset_class(root='data', train=False, download=True, transform=transform)
    test_subset = Subset(test_dataset, range(min(subset_size, len(test_dataset))))
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
    acc = 100. * correct / total
    print(f"Accuracy of {dataset_name} Expert on {dataset_name} test set: {acc:.2f}%")
    return acc

if __name__ == "__main__":
    # 1. Create and save base model
    base_model = SimpleCNN()
    torch.save(base_model.state_dict(), "base_model.pt")
    print("Base initialization saved to base_model.pt")
    
    # 2. Train MNIST expert
    mnist_model = train_model("MNIST", datasets.MNIST, "expert_mnist.pt")
    evaluate_model(mnist_model, "MNIST", datasets.MNIST)
    
    # 3. Train FashionMNIST expert
    fashion_model = train_model("FashionMNIST", datasets.FashionMNIST, "expert_fashion.pt")
    evaluate_model(fashion_model, "FashionMNIST", datasets.FashionMNIST)
    
    # 4. Train KMNIST expert
    kmnist_model = train_model("KMNIST", datasets.KMNIST, "expert_kmnist.pt")
    evaluate_model(kmnist_model, "KMNIST", datasets.KMNIST)
