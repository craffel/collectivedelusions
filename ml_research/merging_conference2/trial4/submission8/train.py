import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# For grayscale datasets, replicate channel to 3 channels
class ReplicateChannel(object):
    def __call__(self, tensor):
        return tensor.repeat(3, 1, 1)

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32), antialias=True),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    ReplicateChannel(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets
train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=mnist_transform)
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=mnist_transform)

train_fashion = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=mnist_transform)
test_fashion = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=mnist_transform)

train_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

# Dataloaders
train_loaders = {
    'mnist': DataLoader(train_mnist, batch_size=128, shuffle=True, num_workers=4),
    'fashion': DataLoader(train_fashion, batch_size=128, shuffle=True, num_workers=4),
    'cifar': DataLoader(train_cifar, batch_size=128, shuffle=True, num_workers=4)
}

test_loaders = {
    'mnist': DataLoader(test_mnist, batch_size=256, shuffle=False, num_workers=4),
    'fashion': DataLoader(test_fashion, batch_size=256, shuffle=False, num_workers=4),
    'cifar': DataLoader(test_cifar, batch_size=256, shuffle=False, num_workers=4)
}

# Training function
def train_model(name, num_classes, train_loader, epochs=3):
    print(f"\nTraining {name} Expert Model...")
    # Load pretrained resnet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace the FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
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
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")
        
    return model

# Evaluate model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

if __name__ == '__main__':
    # Train experts
    mnist_expert = train_model('mnist', 10, train_loaders['mnist'], epochs=3)
    print(f"MNIST Test Accuracy: {evaluate_model(mnist_expert, test_loaders['mnist']):.4f}")
    torch.save(mnist_expert.state_dict(), 'mnist_expert.pth')
    
    fashion_expert = train_model('fashion', 10, train_loaders['fashion'], epochs=3)
    print(f"Fashion Test Accuracy: {evaluate_model(fashion_expert, test_loaders['fashion']):.4f}")
    torch.save(fashion_expert.state_dict(), 'fashion_expert.pth')
    
    cifar_expert = train_model('cifar', 10, train_loaders['cifar'], epochs=3)
    print(f"CIFAR Test Accuracy: {evaluate_model(cifar_expert, test_loaders['cifar']):.4f}")
    torch.save(cifar_expert.state_dict(), 'cifar_expert.pth')
    
    print("\nAll expert models trained and saved successfully!")
