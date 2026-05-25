import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import copy

from models import SharedEncoder, ClassificationHead

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    os.makedirs("./data", exist_ok=True)
    
    # MNIST
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    # FashionMNIST
    fmnist_train = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    
    # KMNIST
    kmnist_train = datasets.KMNIST("./data", train=True, download=True, transform=transform)
    kmnist_test = datasets.KMNIST("./data", train=False, download=True, transform=transform)
    
    loaders = {
        "mnist": {
            "train": DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2),
            "test": DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2)
        },
        "fmnist": {
            "train": DataLoader(fmnist_train, batch_size=batch_size, shuffle=True, num_workers=2),
            "test": DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, num_workers=2)
        },
        "kmnist": {
            "train": DataLoader(kmnist_train, batch_size=batch_size, shuffle=True, num_workers=2),
            "test": DataLoader(kmnist_test, batch_size=batch_size, shuffle=False, num_workers=2)
        }
    }
    return loaders

def train_expert(name, train_loader, test_loader, init_encoder_state, device):
    print(f"\n--- Training Expert for {name.upper()} ---")
    encoder = SharedEncoder().to(device)
    encoder.load_state_dict(copy.deepcopy(init_encoder_state))
    
    head = ClassificationHead().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )
    
    for epoch in range(5):
        encoder.train()
        head.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            features = encoder(x)
            outputs = head(features)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}%")
        
    # Evaluate
    encoder.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            features = encoder(x)
            outputs = head(features)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    test_acc = 100.0 * correct / total
    print(f"Test Accuracy for {name.upper()}: {test_acc:.2f}%")
    
    # Save checkpoints
    torch.save(encoder.state_dict(), f"encoder_{name}.pth")
    torch.save(head.state_dict(), f"head_{name}.pth")
    print(f"Saved encoder_{name}.pth and head_{name}.pth")

def main():
    torch.manual_seed(42)
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create the initial shared encoder to be saved
    init_encoder = SharedEncoder()
    torch.save(init_encoder.state_dict(), "encoder_init.pth")
    print("Saved initial encoder base to encoder_init.pth")
    
    init_encoder_state = init_encoder.state_dict()
    
    loaders = get_dataloaders()
    
    train_expert("mnist", loaders["mnist"]["train"], loaders["mnist"]["test"], init_encoder_state, device)
    train_expert("fmnist", loaders["fmnist"]["train"], loaders["fmnist"]["test"], init_encoder_state, device)
    train_expert("kmnist", loaders["kmnist"]["train"], loaders["kmnist"]["test"], init_encoder_state, device)

if __name__ == "__main__":
    main()
