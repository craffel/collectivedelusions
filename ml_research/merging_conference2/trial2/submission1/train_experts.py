import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.enabled = False


# Define transforms
transform_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_fmnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataset(name, train=True):
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(root="./data", train=train, download=True, transform=transform_rgb)
    elif name == "svhn":
        split = "train" if train else "test"
        return torchvision.datasets.SVHN(root="./data", split=split, download=True, transform=transform_rgb)
    elif name == "fmnist":
        return torchvision.datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform_fmnist)
    else:
        raise ValueError(f"Unknown dataset {name}")

def train_expert(task_name):
    print(f"\n--- Training Expert for {task_name} ---")
    
    # Load ImageNet pre-trained ResNet-18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Modify classification head
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    train_dataset = get_dataset(task_name, train=True)
    test_dataset = get_dataset(task_name, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(2):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/2 | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    test_acc = 100. * correct / total
    print(f"Final Test Accuracy for {task_name}: {test_acc:.2f}%")
    
    # Save checkpoint
    torch.save(model.state_dict(), f"expert_{task_name}.pt")
    print(f"Saved expert_{task_name}.pt")

if __name__ == "__main__":
    # Create data directory
    os.makedirs("./data", exist_ok=True)
    
    # Save pre-trained base model weights
    base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(512, 10) # modify head to match architecture
    torch.save(base_model.state_dict(), "base_model.pt")
    print("Saved pre-trained base_model.pt")
    
    # Train the three experts
    for task in ["cifar10", "svhn", "fmnist"]:
        train_expert(task)
