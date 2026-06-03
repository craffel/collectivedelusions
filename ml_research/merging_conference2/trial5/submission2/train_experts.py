import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms for all datasets (resizing to 32x32, converting to RGB, normalizing to 0.5 mean/std)
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load Datasets
print("Loading datasets...")
datasets_dict = {
    "mnist": datasets.MNIST(root="data", train=True, download=True, transform=data_transform),
    "fmnist": datasets.FashionMNIST(root="data", train=True, download=True, transform=data_transform),
    "cifar10": datasets.CIFAR10(root="data", train=True, download=True, transform=data_transform)
}

test_datasets_dict = {
    "mnist": datasets.MNIST(root="data", train=False, download=True, transform=data_transform),
    "fmnist": datasets.FashionMNIST(root="data", train=False, download=True, transform=data_transform),
    "cifar10": datasets.CIFAR10(root="data", train=False, download=True, transform=data_transform)
}

# Subsetting as per the paper guidelines
# Fine-Tuning: First 5,000 samples of the train set
# Calibration: Next 128 samples (5,000 to 5,128) of the train set
# Test: Full 10,000 samples of the test set
subsets = {}
for name, dataset in datasets_dict.items():
    subsets[name] = {
        "train": Subset(dataset, range(0, 5000)),
        "calib": Subset(dataset, range(5000, 5128)),
        "test": test_datasets_dict[name]
    }

print("Subsets created successfully.")

# Function to train an expert
def train_expert(task_name, train_set, test_set):
    print(f"\n--- Training Expert for Task: {task_name.upper()} ---")
    
    # Load pretrained ResNet-18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace the classification head
    # ResNet-18 fc input has 512 features
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    
    # Loss, Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    # Cosine annealing scheduler over 5 epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    epochs = 5
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
            
        scheduler.step()
        epoch_loss = running_loss / len(train_set)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}%")
        
    # Evaluate on test set
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100.0 * test_correct / test_total
    print(f"Task {task_name.upper()} Test Accuracy: {test_acc:.2f}%")
    
    # Save model state dict
    save_path = f"models/resnet18_{task_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

# Train the three experts
train_expert("mnist", subsets["mnist"]["train"], subsets["mnist"]["test"])
train_expert("fmnist", subsets["fmnist"]["train"], subsets["fmnist"]["test"])
train_expert("cifar10", subsets["cifar10"]["train"], subsets["cifar10"]["test"])

print("\nAll experts trained successfully!")
