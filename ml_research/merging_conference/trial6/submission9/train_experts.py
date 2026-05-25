import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# Disable cuDNN to prevent initialization errors on NVIDIA H100
torch.backends.cudnn.enabled = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Define transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),  # standard ResNet-18 size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets
print("Loading datasets...")
mnist_train = datasets.MNIST("data", train=True, download=True, transform=transform)
fashion_train = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
kmnist_train = datasets.KMNIST("data", train=True, download=True, transform=transform)

# Dataloaders
mnist_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
fashion_loader = DataLoader(fashion_train, batch_size=64, shuffle=True)
kmnist_loader = DataLoader(kmnist_train, batch_size=64, shuffle=True)

# Helper function to train an expert
def train_expert(name, loader, epochs=4):
    print(f"\n--- Training Expert: {name} ---")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace last layer
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    # Optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(loader)}], Loss: {running_loss/100:.4f}, Acc: {100.*correct/total:.2f}%")
                running_loss = 0.0
                
    # Save expert
    torch.save(model.state_dict(), f"models/expert_{name}.pth")
    print(f"Saved models/expert_{name}.pth")
    return model

# Train experts
mnist_model = train_expert("mnist", mnist_loader, epochs=4)
fashion_model = train_expert("fashionmnist", fashion_loader, epochs=4)
kmnist_model = train_expert("kmnist", kmnist_loader, epochs=4)

# Create and save base model
print("\nSaving base model...")
base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
base_model.fc = nn.Linear(512, 10)
# Use MNIST's fc initialization (or just save the initial pre-trained model with fresh fc)
torch.save(base_model.state_dict(), "models/base_model.pth")
print("Saved models/base_model.pth")
