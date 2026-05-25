import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import CNNBackbone, TaskHead

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Datasets mapping
dataset_classes = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Initialize and save the shared base backbone
base_backbone = CNNBackbone()
torch.save(base_backbone.state_dict(), "checkpoints/backbone_init.pt")
print("Saved base backbone initialization to checkpoints/backbone_init.pt")

def train_expert(task_name, epochs=5, batch_size=128, lr=1e-3):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    
    # Load dataset
    dataset_cls = dataset_classes[task_name]
    train_dataset = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_dataset = dataset_cls(root="./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with shared backbone init
    backbone = CNNBackbone()
    backbone.load_state_dict(torch.load("checkpoints/backbone_init.pt"))
    head = TaskHead()
    
    backbone.to(device)
    head.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        backbone.train()
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            features = backbone(images)
            outputs = head(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = total_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    # Evaluate Expert
    backbone.eval()
    head.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = backbone(images)
            outputs = head(features)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100. * test_correct / test_total
    print(f"Finished. Test Accuracy on {task_name.upper()}: {test_acc:.2f}%")
    
    # Save checkpoints
    torch.save(backbone.state_dict(), f"checkpoints/{task_name}_backbone.pt")
    torch.save(head.state_dict(), f"checkpoints/{task_name}_head.pt")
    print(f"Saved expert checkpoints to checkpoints/{task_name}_backbone.pt and checkpoints/{task_name}_head.pt")
    return test_acc

if __name__ == "__main__":
    results = {}
    for task in ["mnist", "fashionmnist", "kmnist"]:
        results[task] = train_expert(task, epochs=5)
    print("\n--- Summary of Expert Training ---")
    for task, acc in results.items():
        print(f"Expert {task.upper()}: {acc:.2f}%")
