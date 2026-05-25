import os
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def train_expert(task_name, dataset_class):
    print(f"\n--- Training Expert for {task_name} ---")
    
    # Load dataset
    train_dataset = dataset_class(root='./data', train=True, download=False, transform=transform)
    test_dataset = dataset_class(root='./data', train=False, download=False, transform=transform)
    
    # Take a subset of 10,000 training samples
    torch.manual_seed(42)
    indices = torch.randperm(len(train_dataset))[:10000].tolist()
    train_subset = Subset(train_dataset, indices)
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Initialize model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(4):
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
            
        epoch_loss = running_loss / len(train_subset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/4 - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    # Evaluate on test set
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
    print(f"Test Accuracy for {task_name}: {test_acc:.2f}%")
    
    # Save expert
    torch.save(model.state_dict(), f"expert_{task_name.lower()}.pt")
    print(f"Saved expert_{task_name.lower()}.pt")
    return test_acc

def main():
    # Save clean base backbone
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # We replace the fc layer to have the correct state_dict structure
    base_model.fc = nn.Linear(512, 10)
    torch.save(base_model.state_dict(), "base_model.pt")
    print("Saved base_model.pt")
    
    accuracies = {}
    accuracies['MNIST'] = train_expert('MNIST', torchvision.datasets.MNIST)
    accuracies['FashionMNIST'] = train_expert('FashionMNIST', torchvision.datasets.FashionMNIST)
    accuracies['KMNIST'] = train_expert('KMNIST', torchvision.datasets.KMNIST)
    
    print("\nTraining summary:")
    for task, acc in accuracies.items():
        print(f"{task}: {acc:.2f}%")

if __name__ == "__main__":
    main()
