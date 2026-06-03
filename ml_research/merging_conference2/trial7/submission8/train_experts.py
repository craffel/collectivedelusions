import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

# Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED on this cluster
torch.backends.cudnn.enabled = False

# Ensure checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define dataset transforms (Resizing to 32x32 and making 3 channels for ResNet-18 compatibility)
mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

fmnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cifar10_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataset(name, train=True):
    if name == "mnist":
        return datasets.MNIST(root="./data", train=train, download=True, transform=mnist_transform)
    elif name == "fmnist":
        return datasets.FashionMNIST(root="./data", train=train, download=True, transform=fmnist_transform)
    elif name == "cifar10":
        return datasets.CIFAR10(root="./data", train=train, download=True, transform=cifar10_transform)
    else:
        raise ValueError("Unknown dataset " + name)

def train_expert(name, epochs=3, batch_size=128, lr=1e-4):
    print(f"\n--- Training {name.upper()} Expert ---")
    
    # Load dataset
    train_set = get_dataset(name, train=True)
    test_set = get_dataset(name, train=False)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize Pretrained ResNet-18
    # Replace the FC layer to output 10 classes
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
        test_acc = 100.0 * test_correct / test_total
        print(f"Epoch {epoch+1} summary: Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
    # Save checkpoint
    save_path = f"checkpoints/{name}_expert.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved {name} expert to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", choices=["mnist", "fmnist", "cifar10", "all"])
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    if args.dataset == "all":
        for d in ["mnist", "fmnist", "cifar10"]:
            train_expert(d, epochs=args.epochs)
    else:
        train_expert(args.dataset, epochs=args.epochs)
