import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

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
mnist_train = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=False,
    transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
)
mnist_test = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=False,
    transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
)

sub_train = Subset(mnist_train, list(range(2000)))
sub_test = Subset(mnist_test, list(range(500)))

train_loader = DataLoader(sub_train, batch_size=32, shuffle=True)
test_loader = DataLoader(sub_test, batch_size=64, shuffle=False)

model = Deep12LayerCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Starting training...")
start_time = time.time()
for epoch in range(5):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

end_time = time.time()
print(f"Training took {end_time - start_time:.2f} seconds.")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"Test Accuracy: {correct/total*100.0:.2f}%")
