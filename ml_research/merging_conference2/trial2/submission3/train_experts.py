import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os

# Disable cuDNN to avoid initialization errors
torch.backends.cudnn.enabled = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset directory
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

# Transforms
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_color = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define a function to get datasets
def get_dataset(name, train=True):
    if name == "mnist":
        return torchvision.datasets.MNIST(root=data_dir, train=train, download=True, transform=transform_gray)
    elif name == "fashionmnist":
        return torchvision.datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=transform_gray)
    elif name == "cifar10":
        return torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform_color)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# Training hyperparameters
epochs = 5
batch_size = 128
lr = 5e-4
weight_decay = 1e-4

datasets_list = ["mnist", "fashionmnist", "cifar10"]

for dataset_name in datasets_list:
    print(f"\n--- Training Expert for {dataset_name.upper()} ---")
    
    # Load data
    train_dataset = get_dataset(dataset_name, train=True)
    test_dataset = get_dataset(dataset_name, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Load ResNet-18 with default ImageNet weights
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    # Replace classification head
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
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
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    test_acc = 100.0 * correct / total
    print(f"Expert {dataset_name} Test Accuracy: {test_acc:.2f}%")
    
    # Save checkpoint
    checkpoint_path = f"expert_{dataset_name}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc
    }, checkpoint_path)
    print(f"Saved expert checkpoint to {checkpoint_path}")
