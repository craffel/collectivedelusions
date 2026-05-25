import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, use_cosface=False, s=30.0, m=0.35):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        
        self.use_cosface = use_cosface
        self.s = s
        self.m = m
        
        if self.use_cosface:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, 128))
            nn.init.xavier_uniform_(self.weight)
        else:
            self.fc2 = nn.Linear(128, num_classes)

    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        return x

    def forward(self, x, label=None):
        features = self.get_features(x)
        features_drop = self.dropout(features)
        
        if self.use_cosface:
            logits = F.linear(F.normalize(features_drop), F.normalize(self.weight))
            if self.training and label is not None:
                one_hot = torch.zeros_like(logits)
                one_hot.scatter_(1, label.view(-1, 1), 1.0)
                logits = logits - one_hot * self.m
                logits = logits * self.s
            else:
                logits = logits * self.s
            return logits
        else:
            return self.fc2(features_drop)

def train_model(dataset_name, use_cosface, base_weights_path, epochs=2, batch_size=64, device="cuda"):
    print(f"Training {dataset_name} (CosFace: {use_cosface}) on {device} starting from {base_weights_path}...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name == "MNIST":
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset")
        
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = SimpleCNN(use_cosface=use_cosface).to(device)
    # Load shared pre-trained/base initialization weights to keep them in the same loss basin!
    model.load_state_dict(torch.load(base_weights_path, map_location=device))
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Test Acc: {acc:.2f}%")
        
    # Save the model
    suffix = "cosface" if use_cosface else "standard"
    model_path = f"models/{dataset_name.lower()}_{suffix}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        import torch.backends.cudnn as cudnn
        cudnn.enabled = False
        
    os.makedirs("models", exist_ok=True)
    
    # 1. Initialize and save standard base initialization
    print("Saving standard base initialization...")
    torch.manual_seed(42)
    base_std = SimpleCNN(use_cosface=False)
    base_std_path = "models/base_standard.pt"
    torch.save(base_std.state_dict(), base_std_path)
    
    # 2. Initialize and save CosFace base initialization
    print("Saving CosFace base initialization...")
    torch.manual_seed(42)
    base_cos = SimpleCNN(use_cosface=True)
    base_cos_path = "models/base_cosface.pt"
    torch.save(base_cos.state_dict(), base_cos_path)
    
    # Train standard experts starting from shared standard base weights
    train_model("MNIST", use_cosface=False, base_weights_path=base_std_path, device=device)
    train_model("FashionMNIST", use_cosface=False, base_weights_path=base_std_path, device=device)
    
    # Train CosFace experts starting from shared CosFace base weights
    train_model("MNIST", use_cosface=True, base_weights_path=base_cos_path, device=device)
    train_model("FashionMNIST", use_cosface=True, base_weights_path=base_cos_path, device=device)
