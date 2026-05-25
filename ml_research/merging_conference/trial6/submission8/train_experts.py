import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import os

def main():
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs('experts', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # 1. Setup transforms
    # All inputs must be 3-channel 32x32 images
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 2. Download datasets
    print("Loading datasets...")
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_rgb)
    svhn_train = datasets.SVHN(root='./data', split='train', download=True, transform=transform_rgb)
    fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray)
    
    # Use 5000 samples for fast fine-tuning
    indices = list(range(5000))
    cifar_subset = Subset(cifar_train, indices)
    svhn_subset = Subset(svhn_train, indices)
    fmnist_subset = Subset(fmnist_train, indices)
    
    loaders = {
        'expert1_cifar': DataLoader(cifar_subset, batch_size=64, shuffle=True, num_workers=4),
        'expert2_svhn': DataLoader(svhn_subset, batch_size=64, shuffle=True, num_workers=4),
        'expert3_fmnist': DataLoader(fmnist_subset, batch_size=64, shuffle=True, num_workers=4)
    }
    
    # 3. Train each expert
    epochs = 5
    for name, loader in loaders.items():
        print(f"\n--- Training {name} ---")
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for x, y in loader:
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
                
            epoch_loss = running_loss / total
            epoch_acc = 100.0 * correct / total
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
            
        save_path = f"experts/{name}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Saved {name} to {save_path}")

if __name__ == '__main__':
    main()
