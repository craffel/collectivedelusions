import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
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

transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])
transform_color = transforms.Compose([
    transforms.ToTensor(),
])

print("Loading datasets...")
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

print("Constructing subsets...")
sub_datasets = {}
for k in range(4):
    train_ds, test_ds = datasets[k]
    sub_datasets[k] = (Subset(train_ds, list(range(1000))), Subset(test_ds, list(range(500))))

# Pre-train base model on mixed dataset
print("\n--- Pre-training Base Model ---")
base_model = Deep12LayerCNN().to(device)
base_optimizer = optim.Adam(base_model.parameters(), lr=2e-3)

mixed_X = []
mixed_y = []
for k in range(4):
    train_sub, _ = sub_datasets[k]
    # Draw all 1000 samples
    loader = DataLoader(train_sub, batch_size=1000, shuffle=False)
    imgs, labels = next(iter(loader))
    mixed_X.append(imgs)
    mixed_y.append(labels)
    
mixed_dataset = TensorDataset(torch.cat(mixed_X), torch.cat(mixed_y))
mixed_loader = DataLoader(mixed_dataset, batch_size=64, shuffle=True)

start_time = time.time()
base_model.train()
for epoch in range(2):
    for imgs, labels in mixed_loader:
        base_optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(base_model(imgs), labels)
        loss.backward()
        base_optimizer.step()
print(f"Base model pre-training took {time.time() - start_time:.1f}s.")

# Fine-tune expert models
for k in range(4):
    print(f"\nFine-tuning Expert {k} on Task {k}...")
    expert = Deep12LayerCNN().to(device)
    expert.load_state_dict(base_model.state_dict())
    expert_optimizer = optim.Adam(expert.parameters(), lr=1e-3)
    
    train_sub, test_sub = sub_datasets[k]
    loader = DataLoader(train_sub, batch_size=32, shuffle=True)
    
    expert.train()
    start_time = time.time()
    for epoch in range(5):
        for imgs, labels in loader:
            expert_optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(expert(imgs), labels)
            loss.backward()
            expert_optimizer.step()
    dur = time.time() - start_time
    
    # Evaluate
    expert.eval()
    test_loader = DataLoader(test_sub, batch_size=64, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = expert(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total * 100.0
    print(f"Expert {k} Task {k} Test Accuracy: {acc:.2f}% (trained in {dur:.1f}s)")
