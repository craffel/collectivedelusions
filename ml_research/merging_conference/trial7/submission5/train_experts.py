import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ExpertCNN(nn.Module):
    def __init__(self):
        super(ExpertCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def extract_features(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.bn3(self.fc1(x)))
        return x
        
    def forward(self, x):
        feat = self.extract_features(x)
        return self.fc2(feat)

def train_and_eval(dataset_name, save_path, epochs=5, batch_size=64, lr=0.001):
    print(f"\n--- Training Expert on {dataset_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    elif dataset_name == "KMNIST":
        train_dataset = datasets.KMNIST(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.KMNIST(root="data", train=False, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = ExpertCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
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
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100.0 * test_correct / test_total
    print(f"Finished. Test Accuracy on {dataset_name}: {test_acc:.2f}%")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return test_acc

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train_and_eval("MNIST", "checkpoints/mnist_expert.pth")
    train_and_eval("KMNIST", "checkpoints/kmnist_expert.pth")
    train_and_eval("FashionMNIST", "checkpoints/fashionmnist_expert.pth")
