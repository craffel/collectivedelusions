import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(dataset_name, save_path, base_weights_path=None, epochs=2):
    device = torch.device("cpu")
    print(f"Training {dataset_name} on {device}...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name == "MNIST":
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        valset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        valset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)
    
    model = SimpleCNN().to(device)
    if base_weights_path and os.path.exists(base_weights_path):
        print(f"Loading base weights from {base_weights_path}")
        model.load_state_dict(torch.load(base_weights_path, map_location=device))
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(trainloader):.4f} | Val Acc: {accuracy:.2f}%")
        
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    
    # 1. Save shared base model
    base_model = SimpleCNN()
    base_path = "./checkpoints/base_model.pt"
    torch.save(base_model.state_dict(), base_path)
    print(f"Saved initial base model to {base_path}")
    
    # 2. Train Expert 0 on MNIST
    train_model("MNIST", "./checkpoints/expert_0.pt", base_weights_path=base_path, epochs=2)
    
    # 3. Train Expert 1 on FashionMNIST
    train_model("FashionMNIST", "./checkpoints/expert_1.pt", base_weights_path=base_path, epochs=2)
    
    # 4. Trigger download of KMNIST so it is cached
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    print("Downloaded KMNIST successfully.")
