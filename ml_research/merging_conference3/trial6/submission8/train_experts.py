import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

# Simple 3-layer CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 4x4
        )
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def get_transforms():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def train_expert(task_name, dataset_class, is_grayscale=False, num_epochs=15, num_train_samples=8192):
    print(f"--- Training Expert for {task_name} ---")
    
    # Custom transform for grayscale datasets to make them 3-channel
    if is_grayscale:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = get_transforms()
        
    # SVHN needs extra arguments for split
    if task_name == 'SVHN':
        train_dataset = dataset_class('./data', split='train', download=True, transform=transform)
    else:
        train_dataset = dataset_class('./data', train=True, download=True, transform=transform)
        
    # Subsample dataset to keep it ultra-fast and lightweight
    indices = torch.arange(num_train_samples)
    train_subset = torch.utils.data.Subset(train_dataset, indices)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/total:.4f} - Accuracy: {acc:.2f}%")
        
    end_time = time.time()
    print(f"Trained in {end_time - start_time:.2f} seconds.")
    
    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/{task_name.lower()}_expert.pth')
    print(f"Saved to checkpoints/{task_name.lower()}_expert.pth")

if __name__ == '__main__':
    # Train MNIST (grayscale)
    train_expert('MNIST', datasets.MNIST, is_grayscale=True)
    
    # Train FashionMNIST (grayscale)
    train_expert('FashionMNIST', datasets.FashionMNIST, is_grayscale=True)
    
    # Train CIFAR-10
    train_expert('CIFAR10', datasets.CIFAR10)
    
    # Train SVHN
    train_expert('SVHN', datasets.SVHN)
