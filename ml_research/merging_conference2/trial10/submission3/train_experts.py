import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import os
import time

# Create output directories if they don't exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preparation
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_mnist = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Convert to 3 channels
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_mnist)
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_mnist)

train_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform_mnist)
test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform_mnist)

train_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_cifar)
test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_cifar)

# Create dataloaders (use smaller batch sizes if needed, but 128 is standard)
batch_size = 128
loader_train_mnist = DataLoader(train_mnist, batch_size=batch_size, shuffle=True, num_workers=2)
loader_test_mnist = DataLoader(test_mnist, batch_size=batch_size, shuffle=False, num_workers=2)

loader_train_fmnist = DataLoader(train_fmnist, batch_size=batch_size, shuffle=True, num_workers=2)
loader_test_fmnist = DataLoader(test_fmnist, batch_size=batch_size, shuffle=False, num_workers=2)

loader_train_cifar = DataLoader(train_cifar, batch_size=batch_size, shuffle=True, num_workers=2)
loader_test_cifar = DataLoader(test_cifar, batch_size=batch_size, shuffle=False, num_workers=2)

def train_expert(task_name, train_loader, test_loader, num_epochs):
    print(f"\n--- Training Expert for {task_name} ---")
    
    # Load progenitor model (ResNet-18 pre-trained on ImageNet)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Replace classification head for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Fine-tune the backbone with a smaller learning rate and the head with a larger one
    optimizer = optim.AdamW([
        {'params': [p for name, p in model.named_parameters() if 'fc' not in name], 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
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
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Test Acc: {val_acc:.2f}% | Time: {time.time()-start_time:.1f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"checkpoints/{task_name}_expert.pt")
            print(f" Saved new best checkpoint with Test Acc: {val_acc:.2f}%")
            
    print(f"Finished training {task_name} expert. Best Test Acc: {best_acc:.2f}%")
    return best_acc

def main():
    print("Starting expert fine-tuning...")
    # Train MNIST expert
    mnist_acc = train_expert("mnist", loader_train_mnist, loader_test_mnist, num_epochs=2)
    
    # Train FashionMNIST expert
    fmnist_acc = train_expert("fmnist", loader_train_fmnist, loader_test_fmnist, num_epochs=3)
    
    # Train CIFAR-10 expert
    cifar_acc = train_expert("cifar10", loader_train_cifar, loader_test_cifar, num_epochs=5)
    
    print("\n--- Summary of Trained Experts ---")
    print(f"MNIST Expert: {mnist_acc:.2f}%")
    print(f"FashionMNIST Expert: {fmnist_acc:.2f}%")
    print(f"CIFAR-10 Expert: {cifar_acc:.2f}%")
    
if __name__ == "__main__":
    main()
