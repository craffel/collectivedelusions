import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on this cluster environment
torch.backends.cudnn.enabled = False
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Transforms
transform_mnist = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3081, 0.3081, 0.3081))
])

transform_fashion = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.2860, 0.2860, 0.2860), std=(0.3530, 0.3530, 0.3530))
])

transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

# Datasets
train_mnist = datasets.MNIST('data', train=True, download=True, transform=transform_mnist)
test_mnist = datasets.MNIST('data', train=False, download=True, transform=transform_mnist)

train_fashion = datasets.FashionMNIST('data', train=True, download=True, transform=transform_fashion)
test_fashion = datasets.FashionMNIST('data', train=False, download=True, transform=transform_fashion)

train_cifar = datasets.CIFAR10('data', train=True, download=True, transform=transform_cifar)
test_cifar = datasets.CIFAR10('data', train=False, download=True, transform=transform_cifar)

# Dataloaders
loader_mnist_train = DataLoader(train_mnist, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
loader_mnist_test = DataLoader(test_mnist, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

loader_fashion_train = DataLoader(train_fashion, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
loader_fashion_test = DataLoader(test_fashion, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

loader_cifar_train = DataLoader(train_cifar, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
loader_cifar_test = DataLoader(test_cifar, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

def train_expert(name, train_loader, test_loader, epochs=5):
    print(f"\n--- Training Expert: {name} ---")
    
    # Instantiate standard pre-trained ResNet-18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace the classification head for 10 classes
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
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
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Save state dict
            torch.save(model.state_dict(), f'checkpoints/expert_{name}.pt')
            print(f"Saved best model with Val Acc: {val_acc:.2f}%")
            
    print(f"Finished training {name}. Best Val Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    # Save progenitor backbone weights for model merging reference
    prog_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Save the progenitor model's state_dict
    torch.save(prog_model.state_dict(), 'checkpoints/progenitor.pt')
    print("Saved progenitor ImageNet weights to checkpoints/progenitor.pt")
    
    # Train experts
    train_expert('mnist', loader_mnist_train, loader_mnist_test, epochs=5)
    train_expert('fashion', loader_fashion_train, loader_fashion_test, epochs=5)
    train_expert('cifar', loader_cifar_train, loader_cifar_test, epochs=5)
    
    print("\nAll experts trained successfully.")
