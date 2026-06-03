import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import copy

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
print(f"Using device: {device}")

# Helper for duplicating single-channel to 3-channel
class ToRGB(object):
    def __call__(self, pic):
        if pic.size(0) == 1:
            return pic.repeat(3, 1, 1)
        return pic

def get_dataloaders(batch_size=128):
    # Grayscale transforms (convert to 3 channels to match ResNet-18)
    gray_transform = transforms.Compose([
        transforms.Resize((32, 32)), # Resize to 32x32
        transforms.ToTensor(),
        ToRGB(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Color transforms (CIFAR-10 is already 32x32)
    color_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load MNIST
    train_mnist = datasets.MNIST(root="./data", train=True, download=True, transform=gray_transform)
    test_mnist = datasets.MNIST(root="./data", train=False, download=True, transform=gray_transform)
    
    # Load Fashion-MNIST
    train_fmnist = datasets.FashionMNIST(root="./data", train=True, download=True, transform=gray_transform)
    test_fmnist = datasets.FashionMNIST(root="./data", train=False, download=True, transform=gray_transform)

    # Load CIFAR-10
    train_cifar = datasets.CIFAR10(root="./data", train=True, download=True, transform=color_transform)
    test_cifar = datasets.CIFAR10(root="./data", train=False, download=True, transform=color_transform)

    loaders = {
        "mnist": {
            "train": DataLoader(train_mnist, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            "test": DataLoader(test_mnist, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        },
        "fmnist": {
            "train": DataLoader(train_fmnist, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            "test": DataLoader(test_fmnist, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        },
        "cifar10": {
            "train": DataLoader(train_cifar, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            "test": DataLoader(test_cifar, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        }
    }
    return loaders

def train_model(name, loader_dict, num_epochs=3):
    print(f"\n--- Training Expert for {name.upper()} ---")
    
    # Initialize from standard pretrained ResNet-18
    # Using modern weights initialization as weights=models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # All tasks have 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in loader_dict["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    # Evaluate on test set
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in loader_dict["test"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100.0 * test_correct / test_total
    print(f"Finished Training. Test Accuracy for {name.upper()}: {test_acc:.2f}%")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/expert_{name}.pt")
    print(f"Saved expert to models/expert_{name}.pt")

if __name__ == "__main__":
    loaders = get_dataloaders()
    train_model("mnist", loaders["mnist"], num_epochs=3)
    train_model("fmnist", loaders["fmnist"], num_epochs=3)
    train_model("cifar10", loaders["cifar10"], num_epochs=5)
