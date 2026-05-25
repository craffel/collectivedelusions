import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os

class SimpleCNN(nn.Module):
    def __init__(self, use_cosface=False, s=30.0, m=0.35):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.25)
        
        self.use_cosface = use_cosface
        if use_cosface:
            self.classifier_weight = nn.Parameter(torch.randn(10, 128))
            self.s = s
            self.m = m
        else:
            self.classifier = nn.Linear(128, 10)
            
    def extract_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.bn3(self.fc1(x))))
        return x
        
    def forward(self, x, labels=None):
        features = self.extract_features(x)
        if self.use_cosface:
            f_norm = F.normalize(features, p=2, dim=1)
            w_norm = F.normalize(self.classifier_weight, p=2, dim=1)
            cosine = F.linear(f_norm, w_norm)
            if self.training and labels is not None:
                one_hot = torch.zeros_like(cosine)
                one_hot.scatter_(1, labels.view(-1, 1), 1.0)
                output = self.s * (cosine - one_hot * self.m)
            else:
                output = self.s * cosine
        else:
            output = self.classifier(features)
        return output

def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load MNIST
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_val = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Load FashionMNIST
    fashion_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fashion_val = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # KMNIST for OOD/Novel domain
    kmnist_val = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Subsample training sets to 10,000 samples for fast training
    mnist_train_sub = Subset(mnist_train, list(range(10000)))
    fashion_train_sub = Subset(fashion_train, list(range(10000)))
    
    loaders = {
        'mnist_train': DataLoader(mnist_train_sub, batch_size=64, shuffle=True),
        'mnist_val': DataLoader(mnist_val, batch_size=64, shuffle=False),
        'fashion_train': DataLoader(fashion_train_sub, batch_size=64, shuffle=True),
        'fashion_val': DataLoader(fashion_val, batch_size=64, shuffle=False),
        'kmnist_val': DataLoader(kmnist_val, batch_size=64, shuffle=False)
    }
    return loaders

def train_model(name, model, train_loader, val_loader, device, epochs=2):
    print(f"\n--- Training {name} ---")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x, y)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        train_loss /= len(train_loader.dataset)
        train_acc = 100.0 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
    return val_acc

def compute_prototypes(model, val_loader, device):
    model.eval()
    model.to(device)
    
    features_list = [[] for _ in range(10)]
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            features = model.extract_features(x) # (B, 128)
            for f_i, y_i in zip(features, y):
                features_list[y_i.item()].append(f_i.cpu())
                
    prototypes = []
    for c in range(10):
        if len(features_list[c]) > 0:
            c_feats = torch.stack(features_list[c])
            # compute mean prototype
            proto = c_feats.mean(dim=0)
        else:
            proto = torch.zeros(128)
        prototypes.append(proto)
        
    return torch.stack(prototypes) # (10, 128)

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loaders = get_dataloaders()
    
    # 1. Train Standard MNIST Expert
    mnist_std = SimpleCNN(use_cosface=False)
    train_model("MNIST Standard Expert", mnist_std, loaders['mnist_train'], loaders['mnist_val'], device, epochs=2)
    mnist_std_protos = compute_prototypes(mnist_std, loaders['mnist_val'], device)
    
    # 2. Train Standard Fashion Expert
    fashion_std = SimpleCNN(use_cosface=False)
    train_model("Fashion Standard Expert", fashion_std, loaders['fashion_train'], loaders['fashion_val'], device, epochs=2)
    fashion_std_protos = compute_prototypes(fashion_std, loaders['fashion_val'], device)
    
    # 3. Train CosFace MNIST Expert
    mnist_cos = SimpleCNN(use_cosface=True)
    train_model("MNIST CosFace Expert", mnist_cos, loaders['mnist_train'], loaders['mnist_val'], device, epochs=2)
    mnist_cos_protos = compute_prototypes(mnist_cos, loaders['mnist_val'], device)
    
    # 4. Train CosFace Fashion Expert
    fashion_cos = SimpleCNN(use_cosface=True)
    train_model("Fashion CosFace Expert", fashion_cos, loaders['fashion_train'], loaders['fashion_val'], device, epochs=2)
    fashion_cos_protos = compute_prototypes(fashion_cos, loaders['fashion_val'], device)
    
    # Save checkpoints including prototypes
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({'state_dict': mnist_std.cpu().state_dict(), 'prototypes': mnist_std_protos}, "checkpoints/expert_mnist_std.pt")
    torch.save({'state_dict': fashion_std.cpu().state_dict(), 'prototypes': fashion_std_protos}, "checkpoints/expert_fashion_std.pt")
    torch.save({'state_dict': mnist_cos.cpu().state_dict(), 'prototypes': mnist_cos_protos}, "checkpoints/expert_mnist_cos.pt")
    torch.save({'state_dict': fashion_cos.cpu().state_dict(), 'prototypes': fashion_cos_protos}, "checkpoints/expert_fashion_cos.pt")
    print("\nSaved all checkpoints!")

if __name__ == "__main__":
    main()
