import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

def get_resnet_expert():
    # Load pre-trained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze layer4 and fc
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace classification head
    model.fc = nn.Linear(512, 10)
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model

def train_one_expert(dataset_name, train_loader, epochs=5):
    print(f"\n--- Training Expert for {dataset_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = get_resnet_expert().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / (i+1):.4f}, Acc: {100.0 * correct / total:.2f}%")
                
    # Save checkpoint
    save_path = f"expert_{dataset_name.lower()}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved {save_path}")
    return save_path

def main():
    import torch.backends.cudnn as cudnn
    cudnn.enabled = False
    
    # Transforms
    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # FashionMNIST transform (convert grayscale to 3 channels)
    transform_fmnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 1. CIFAR-10
    train_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_color)
    loader_cifar = DataLoader(train_cifar, batch_size=128, shuffle=True, num_workers=4)
    train_one_expert("CIFAR10", loader_cifar, epochs=4)
    
    # 2. SVHN
    train_svhn = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_color)
    loader_svhn = DataLoader(train_svhn, batch_size=128, shuffle=True, num_workers=4)
    train_one_expert("SVHN", loader_svhn, epochs=4)
    
    # 3. FashionMNIST
    train_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_fmnist)
    loader_fmnist = DataLoader(train_fmnist, batch_size=128, shuffle=True, num_workers=4)
    train_one_expert("FMNIST", loader_fmnist, epochs=4)
    
    print("\nAll experts trained and saved successfully!")

if __name__ == "__main__":
    main()
