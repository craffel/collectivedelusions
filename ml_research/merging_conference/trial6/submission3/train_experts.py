import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import os
import copy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Disabling cuDNN to bypass recurrent initialization errors as mentioned in IGGS-Merge
torch.backends.cudnn.enabled = False

def get_backbone():
    # Load pre-trained ResNet-18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Modify conv1 for 1-channel grayscale inputs
    old_conv1 = model.conv1
    new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Sum weights along channel dimension
    with torch.no_grad():
        new_conv1.weight.copy_(old_conv1.weight.sum(dim=1, keepdim=True))
    model.conv1 = new_conv1
    # Modify fc layer for 10 classes
    model.fc = nn.Linear(512, 10)
    return model

def get_datasets():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fit ResNet-18 expected input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    os.makedirs("./data", exist_ok=True)
    
    print("Loading datasets...")
    # MNIST
    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    # FashionMNIST
    fmnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    
    # KMNIST
    kmnist_train = torchvision.datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    
    return (mnist_train, mnist_test), (fmnist_train, fmnist_test), (kmnist_train, kmnist_test)

def train_expert(name, train_dataset, test_dataset):
    print(f"\n--- Training Expert: {name} ---")
    model = get_backbone().to(device)
    
    # Create a subset of 10,000 samples for training
    indices = list(range(10000))
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(4):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_subset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/4: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    test_acc = 100.0 * test_correct / test_total
    print(f"Expert {name} Test Accuracy: {test_acc:.2f}%")
    
    # Save checkpoint
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"./checkpoints/expert_{name}.pth")
    return model

def compute_fisher(name, model, train_dataset):
    print(f"Computing diagonal Fisher for {name}...")
    model.eval()
    
    # Use 500 samples for calibration
    cal_subset = Subset(train_dataset, list(range(500)))
    cal_loader = DataLoader(cal_subset, batch_size=1, shuffle=False)
    
    fisher_dict = {}
    for name_param, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name_param] = torch.zeros_like(param.data)
            
    criterion = nn.CrossEntropyLoss()
    
    for inputs, labels in cal_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        with torch.no_grad():
            for name_param, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name_param] += param.grad.data ** 2
                    
    # Average over calibration set size
    for name_param in fisher_dict:
        fisher_dict[name_param] /= len(cal_subset)
        
    torch.save(fisher_dict, f"./checkpoints/fisher_{name}.pth")
    print(f"Fisher computation for {name} complete.")

def main():
    (mnist_train, mnist_test), (fmnist_train, fmnist_test), (kmnist_train, kmnist_test) = get_datasets()
    
    # Save base model checkpoint first
    os.makedirs("./checkpoints", exist_ok=True)
    base_model = get_backbone()
    torch.save(base_model.state_dict(), "./checkpoints/base_model.pth")
    print("Saved base model.")
    
    # Train experts and compute their Fisher Information
    mnist_model = train_expert("mnist", mnist_train, mnist_test)
    compute_fisher("mnist", mnist_model, mnist_train)
    
    fmnist_model = train_expert("fmnist", fmnist_train, fmnist_test)
    compute_fisher("fmnist", fmnist_model, fmnist_train)
    
    kmnist_model = train_expert("kmnist", kmnist_train, kmnist_test)
    compute_fisher("kmnist", kmnist_model, kmnist_train)
    
    print("\nAll experts trained and Fisher matrices saved successfully!")

if __name__ == "__main__":
    main()
