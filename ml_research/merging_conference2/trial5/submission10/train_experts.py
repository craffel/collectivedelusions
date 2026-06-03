import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import os
import copy

# Ensure directories exist
os.makedirs("experts", exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset preprocessing and loading
# Grayscale images are resized to 32x32 and replicated to 3 channels to match CIFAR-10 dimensions
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # Replicate channels
])

transform_color = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_dataset(task, train=True):
    if task == "mnist":
        return torchvision.datasets.MNIST(root="./data", train=train, transform=transform_gray, download=False)
    elif task == "fashionmnist":
        return torchvision.datasets.FashionMNIST(root="./data", train=train, transform=transform_gray, download=False)
    elif task == "cifar10":
        return torchvision.datasets.CIFAR10(root="./data", train=train, transform=transform_color, download=False)
    else:
        raise ValueError("Unknown task")

tasks = ["mnist", "fashionmnist", "cifar10"]

# We will train experts on 3,000 deterministic samples
train_subset_size = 3000
test_subset_size = 10000 # Use full test set for evaluation

for task in tasks:
    print(f"\n--- Preparing training for {task.upper()} Expert ---")
    train_dataset = get_dataset(task, train=True)
    test_dataset = get_dataset(task, train=False)
    
    # Get deterministic subset indices
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset), generator=g).tolist()
    train_subset = Subset(train_dataset, indices[:train_subset_size])
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Initialize ResNet-18
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    # Replace the classification head
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    # Training Loop
    model.train()
    for epoch in range(5):
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
        
        scheduler.step()
        epoch_loss = running_loss / len(train_subset)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/5 | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    # Evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
    test_acc = 100. * test_correct / test_total
    print(f"{task.upper()} Expert Test Accuracy: {test_acc:.2f}%")
    
    # Save expert
    torch.save(model.state_dict(), f"experts/{task}_expert.pth")
    print(f"Saved {task}_expert.pth")
