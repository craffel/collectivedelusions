import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import copy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    print("cuDNN disabled to prevent initialization errors.")

# Hyperparameters
BATCH_SIZE = 128
LR = 1e-4
EPOCHS_MNIST = 3
EPOCHS_FASHION = 3
EPOCHS_CIFAR = 5

# Create directory for checkpoints and data
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Transforms
transform_mnist = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_cifar = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Datasets
print("Loading datasets...")
train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

train_fashion = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_mnist)
test_fashion = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_mnist)

train_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)

# Dataloaders
loader_train_mnist = DataLoader(train_mnist, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
loader_test_mnist = DataLoader(test_mnist, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

loader_train_fashion = DataLoader(train_fashion, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
loader_test_fashion = DataLoader(test_fashion, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

loader_train_cifar = DataLoader(train_cifar, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
loader_test_cifar = DataLoader(test_cifar, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

def get_resnet18_base():
    # Load pretrained resnet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace the FC layer with a 10-class head
    model.fc = nn.Linear(512, 10)
    return model

def train_expert(name, loader_train, loader_test, epochs):
    print(f"\n--- Training Expert: {name} ---")
    # Load the base model
    model = get_resnet18_base()
    model.load_state_dict(torch.load("checkpoints/base_model.pt"))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader_train:
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
            
        epoch_loss = running_loss / len(loader_train.dataset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}%")
        
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader_test:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    test_acc = 100.0 * correct / total
    print(f"Expert {name} Test Acc: {test_acc:.2f}%")
    
    # Save checkpoint
    torch.save(model.state_dict(), f"checkpoints/expert_{name}.pt")
    return test_acc

if __name__ == "__main__":
    # Create and save base model
    print("Initializing base model...")
    base_model = get_resnet18_base()
    torch.save(base_model.state_dict(), "checkpoints/base_model.pt")
    print("Base model saved to checkpoints/base_model.pt")
    
    # Train experts
    acc_mnist = train_expert("mnist", loader_train_mnist, loader_test_mnist, EPOCHS_MNIST)
    acc_fashion = train_expert("fashion", loader_train_fashion, loader_test_fashion, EPOCHS_FASHION)
    acc_cifar = train_expert("cifar", loader_train_cifar, loader_test_cifar, EPOCHS_CIFAR)
    
    print("\nTraining completed!")
    print(f"MNIST expert accuracy: {acc_mnist:.2f}%")
    print(f"FashionMNIST expert accuracy: {acc_fashion:.2f}%")
    print(f"CIFAR10 expert accuracy: {acc_cifar:.2f}%")
