import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os

# Ensure reproducible random seed for base model initialization
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.50)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(self.bn3(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train_expert(dataset_name, save_path, epochs=1, device='cpu'):
    print(f"\n--- Training Expert on {dataset_name} ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    elif dataset_name == "KMNIST":
        train_dataset = datasets.KMNIST(root='./data', train=True, transform=transform)
        test_dataset = datasets.KMNIST(root='./data', train=False, transform=transform)
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform)
        
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Initialize from the shared random seed to obtain base model
    set_seed(42)
    model = SimpleCNN().to(device)
    
    # Let's save the initial base model parameters if not saved already
    base_path = "base_model.pt"
    if not os.path.exists(base_path):
        torch.save(model.state_dict(), base_path)
        print("Saved base model initialized from seed 42.")
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        train_acc = 100. * correct / total
        print(f"Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}, Train Acc = {train_acc:.2f}%")
        
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        test_acc = 100. * test_correct / test_total
        print(f"Test Accuracy: {test_acc:.2f}%")
        
    # Save the expert checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"Saved expert to {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Train experts for 2 epochs
    train_expert("MNIST", "expert_mnist.pt", epochs=2, device=device)
    train_expert("KMNIST", "expert_kmnist.pt", epochs=2, device=device)
    train_expert("FashionMNIST", "expert_fashion.pt", epochs=2, device=device)
