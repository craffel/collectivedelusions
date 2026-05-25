import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Create models directory
os.makedirs("./models", exist_ok=True)

# Disable cuDNN to bypass initialization errors on this cluster
torch.backends.cudnn.enabled = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def get_resnet18_expert():
    # Load pretrained resnet18
    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except:
        model = resnet18(pretrained=True)
        
    # Sum conv1 weights across channel dimension to accept 1-channel grayscale input
    pretrained_conv1 = model.conv1.weight.data
    new_conv1_weight = pretrained_conv1.sum(dim=1, keepdim=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = new_conv1_weight
    
    # Modify fc layer to output 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def train_expert(dataset_name, train_dataset, epochs=4, lr=1e-4, weight_decay=1e-2):
    print(f"\n--- Training Expert for {dataset_name} ---")
    model = get_resnet18_expert().to(device)
    
    # We use the first 10,000 samples for fine-tuning
    subset_indices = list(range(min(10000, len(train_dataset))))
    subset_dataset = Subset(train_dataset, subset_indices)
    dataloader = DataLoader(subset_dataset, batch_size=64, shuffle=True, num_workers=2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(dataloader):
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
            
        epoch_loss = running_loss / len(subset_dataset)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    # Save expert
    save_path = f"./models/expert_{dataset_name.lower()}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved expert to {save_path}")
    return model

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # MNIST
    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_expert("MNIST", mnist_train)
    
    # FashionMNIST
    fmnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    train_expert("FashionMNIST", fmnist_train)
    
    # KMNIST
    kmnist_train = torchvision.datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
    train_expert("KMNIST", kmnist_train)
    
    print("\nAll experts trained and saved!")
