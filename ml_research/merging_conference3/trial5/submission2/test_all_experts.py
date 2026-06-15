import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Deep12LayerCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.ModuleList([
            ConvBlock(3, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)
        
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

device = torch.device("cpu")

# Load datasets
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])
transform_color = transforms.Compose([
    transforms.ToTensor(),
])

datasets = {
    0: (torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_gray),
        torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_gray)),
    1: (torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform_gray),
        torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_gray)),
    2: (torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_color),
        torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_color)),
    3: (torchvision.datasets.SVHN(root='./data', split='train', download=False, transform=transform_color),
        torchvision.datasets.SVHN(root='./data', split='test', download=False, transform=transform_color))
}

for k in range(4):
    train_ds, test_ds = datasets[k]
    sub_train = Subset(train_ds, list(range(1000)))
    sub_test = Subset(test_ds, list(range(500)))
    
    train_loader = DataLoader(sub_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(sub_test, batch_size=64, shuffle=False)
    
    model = Deep12LayerCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for epoch in range(3):
        model.train()
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
    dur = time.time() - start_time
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total * 100.0
    print(f"Task {k} trained in {dur:.1f}s, Test Accuracy: {acc:.2f}%")
