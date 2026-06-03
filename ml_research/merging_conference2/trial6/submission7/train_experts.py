import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.enabled = False

# Define data transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# Load full CIFAR-100 datasets
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform_test)

def get_task_indices(dataset, task_id, num_classes_per_task=10):
    start_class = task_id * num_classes_per_task
    end_class = start_class + num_classes_per_task
    indices = [i for i, label in enumerate(dataset.targets) if start_class <= label < end_class]
    return indices

def train_expert(task_id, epochs=5, num_classes_per_task=10):
    print(f"\n--- Fine-tuning Expert {task_id} (Classes {task_id*num_classes_per_task} to {task_id*num_classes_per_task + num_classes_per_task - 1}) ---")
    
    # Get task subsets
    train_idx = get_task_indices(train_dataset, task_id, num_classes_per_task)
    test_idx = get_task_indices(test_dataset, task_id, num_classes_per_task)
    
    train_sub = Subset(train_dataset, train_idx)
    test_sub = Subset(test_dataset, test_idx)
    
    train_loader = DataLoader(train_sub, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_sub, batch_size=128, shuffle=False, num_workers=4)
    
    # Load progenitor model
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(torch.load("progenitor.pt"))
    model = model.to(device)
    
    # Define optimizer, scheduler, loss
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
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
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        
    # Evaluate on task test subset
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    test_acc = 100. * correct / total
    print(f"Expert {task_id} Final Task Test Acc: {test_acc:.2f}%")
    
    # Save expert
    torch.save(model.state_dict(), f"expert_{task_id}.pt")
    return test_acc

def create_progenitor():
    print("Creating and saving progenitor model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Reinitialize the fc layer to output 100 classes
    model.fc = nn.Linear(model.fc.in_features, 100)
    torch.save(model.state_dict(), "progenitor.pt")
    print("Progenitor model saved.")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    if not os.path.exists("progenitor.pt"):
        create_progenitor()
        
    accuracies = []
    for k in range(10):
        acc = train_expert(k, epochs=5)
        accuracies.append(acc)
        
    print("\nAll experts trained!")
    for k, acc in enumerate(accuracies):
        print(f"Expert {k}: {acc:.2f}%")
