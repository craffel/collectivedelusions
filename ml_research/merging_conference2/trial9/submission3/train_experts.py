import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from models import ResNet18Backbone, MLPBackbone, CompleteModel

# Ensure checkpoints directory exists
os.makedirs('checkpoints', exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False
print(f"Using device: {device}")

def get_dataloaders(batch_size=256):
    # Grayscale transforms (Resize to 32x32, duplicate channels, normalize)
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # CIFAR-10 transforms (Resize to 32x32, normalize)
    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Datasets
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_gray)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_gray)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_gray)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_gray)
    
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_cifar)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_cifar)
    
    # Create dataloaders
    loaders = {
        'mnist': {
            'train': DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            'test': DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        },
        'fmnist': {
            'train': DataLoader(fmnist_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            'test': DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        },
        'cifar10': {
            'train': DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            'test': DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        }
    }
    return loaders

def train_model(model, train_loader, test_loader, epochs=5, lr=1e-3, weight_decay=1e-4):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        train_acc = 100.0 * correct / total
        train_loss = running_loss / total
        
        # Test eval
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = outputs.max(1)
                test_total += y.size(0)
                test_correct += predicted.eq(y).sum().item()
        test_acc = 100.0 * test_correct / test_total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
    return model

def main():
    loaders = get_dataloaders()
    
    # 1. ResNet-18 Progenitor
    print("\n--- Initializing ResNet-18 Progenitor ---")
    resnet_progenitor = ResNet18Backbone().to(device)
    torch.save(resnet_progenitor.state_dict(), 'checkpoints/resnet18_progenitor.pt')
    
    # Train ResNet-18 Experts
    for task_name in ['mnist', 'fmnist', 'cifar10']:
        ckpt_path = f'checkpoints/resnet18_{task_name}_backbone.pt'
        head_path = f'checkpoints/resnet18_{task_name}_head.pt'
        
        if os.path.exists(ckpt_path) and os.path.exists(head_path):
            print(f"ResNet-18 {task_name} expert already exists. Skipping training.")
            continue
            
        print(f"\n--- Training ResNet-18 Expert on {task_name.upper()} ---")
        # Load fresh progenitor backbone
        backbone = ResNet18Backbone()
        backbone.load_state_dict(torch.load('checkpoints/resnet18_progenitor.pt'))
        # Create head (512 features for ResNet-18 backbone)
        head = nn.Linear(512, 10)
        
        model = CompleteModel(backbone, head)
        model = train_model(model, loaders[task_name]['train'], loaders[task_name]['test'], epochs=5, lr=1e-3, weight_decay=1e-4)
        
        # Save backbone and head
        torch.save(backbone.state_dict(), ckpt_path)
        torch.save(head.state_dict(), head_path)
        print(f"Saved ResNet-18 {task_name} expert.")

    # 2. MLP Progenitor
    print("\n--- Initializing MLP Progenitor ---")
    mlp_progenitor = MLPBackbone().to(device)
    torch.save(mlp_progenitor.state_dict(), 'checkpoints/mlp_progenitor.pt')
    
    # Train MLP Experts
    for task_name in ['mnist', 'fmnist', 'cifar10']:
        ckpt_path = f'checkpoints/mlp_{task_name}_backbone.pt'
        head_path = f'checkpoints/mlp_{task_name}_head.pt'
        
        if os.path.exists(ckpt_path) and os.path.exists(head_path):
            print(f"MLP {task_name} expert already exists. Skipping training.")
            continue
            
        print(f"\n--- Training MLP Expert on {task_name.upper()} ---")
        # Load fresh progenitor backbone
        backbone = MLPBackbone()
        backbone.load_state_dict(torch.load('checkpoints/mlp_progenitor.pt'))
        # Create head (512 features for MLP backbone)
        head = nn.Linear(512, 10)
        
        model = CompleteModel(backbone, head)
        model = train_model(model, loaders[task_name]['train'], loaders[task_name]['test'], epochs=5, lr=1e-3, weight_decay=1e-4)
        
        # Save backbone and head
        torch.save(backbone.state_dict(), ckpt_path)
        torch.save(head.state_dict(), head_path)
        print(f"Saved MLP {task_name} expert.")

if __name__ == '__main__':
    main()
