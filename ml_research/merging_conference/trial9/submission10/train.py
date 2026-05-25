import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Layer 2: Conv2D (3x3, Padding=1), Layer 3: BN, Layer 4: ReLU, Layer 5: MaxPool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        # Layer 6: Conv2D (3x3, Padding=1), Layer 7: BN, Layer 8: ReLU, Layer 9: MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        # Layer 11: Fully Connected, Layer 12: ReLU, Layer 13: Dropout
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        
        # Layer 14: Classifier (Linear)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_expert(model, train_loader, device, epochs=2):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {running_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%")
    return model

def evaluate(model, test_loader, device):
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
    return 100. * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Load Datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Loading datasets...")
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    
    mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=2)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False, num_workers=2)
    
    fmnist_train_loader = torch.utils.data.DataLoader(fmnist_train, batch_size=64, shuffle=True, num_workers=2)
    fmnist_test_loader = torch.utils.data.DataLoader(fmnist_test, batch_size=64, shuffle=False, num_workers=2)
    
    # Initialize Shared Base Model
    print("Initializing base model...")
    base_model = SimpleCNN()
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(base_model.state_dict(), "checkpoints/base.pt")
    
    # Train Expert 0 (MNIST)
    print("\n--- Training Expert 0 on MNIST ---")
    expert0 = SimpleCNN()
    expert0.load_state_dict(torch.load("checkpoints/base.pt"))
    expert0 = expert0.to(device)
    expert0 = train_expert(expert0, mnist_train_loader, device, epochs=2)
    acc0 = evaluate(expert0, mnist_test_loader, device)
    print(f"Expert 0 (MNIST) Test Accuracy: {acc0:.2f}%")
    torch.save(expert0.state_dict(), "checkpoints/expert0_mnist.pt")
    
    # Train Expert 1 (FashionMNIST)
    print("\n--- Training Expert 1 on FashionMNIST ---")
    expert1 = SimpleCNN()
    expert1.load_state_dict(torch.load("checkpoints/base.pt"))
    expert1 = expert1.to(device)
    expert1 = train_expert(expert1, fmnist_train_loader, device, epochs=2)
    acc1 = evaluate(expert1, fmnist_test_loader, device)
    print(f"Expert 1 (FashionMNIST) Test Accuracy: {acc1:.2f}%")
    torch.save(expert1.state_dict(), "checkpoints/expert1_fmnist.pt")

if __name__ == "__main__":
    main()
