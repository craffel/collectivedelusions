import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_gray = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Define tasks and datasets
tasks = {
    "cifar10": {
        "dataset_class": datasets.CIFAR10,
        "transform": transform_rgb,
        "train_kwargs": {"download": True, "train": True},
        "test_kwargs": {"download": True, "train": False}
    },
    "svhn": {
        "dataset_class": datasets.SVHN,
        "transform": transform_rgb,
        "train_kwargs": {"download": True, "split": "train"},
        "test_kwargs": {"download": True, "split": "test"}
    },
    "mnist": {
        "dataset_class": datasets.MNIST,
        "transform": transform_gray,
        "train_kwargs": {"download": True, "train": True},
        "test_kwargs": {"download": True, "train": False}
    }
}

# Number of classes per task
NUM_CLASSES = 10
SUBSET_SIZE_TRAIN = 2000  # Small size for rapid GPU fine-tuning (token and time efficient)
SUBSET_SIZE_TEST = 500

def get_dataloaders(task_name):
    task_info = tasks[task_name]
    
    # Train dataset
    train_dataset = task_info["dataset_class"](
        root="./data", transform=task_info["transform"], **task_info["train_kwargs"]
    )
    # Subset train dataset
    indices = torch.randperm(len(train_dataset))[:SUBSET_SIZE_TRAIN]
    train_subset = Subset(train_dataset, indices)
    
    # Test dataset
    test_dataset = task_info["dataset_class"](
        root="./data", transform=task_info["transform"], **task_info["test_kwargs"]
    )
    # Subset test dataset
    indices_test = torch.randperm(len(test_dataset))[:SUBSET_SIZE_TEST]
    test_subset = Subset(test_dataset, indices_test)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# Save pre-trained base model weights
print("Saving pre-trained base model...")
base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
torch.save(base_model.state_dict(), "models/resnet18_pretrained.pt")

# Train each expert
for task_name in tasks.keys():
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    train_loader, test_loader = get_dataloaders(task_name)
    
    # Load fresh pre-trained model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Replace FC layer
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    
    # We fine-tune the entire model to get full task vectors (standard task arithmetic)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    # Fine-tune for 2 epochs (very fast with subset)
    model.train()
    for epoch in range(2):
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
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/2 | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
    # Evaluate expert
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
    print(f"Expert {task_name.upper()} Test Accuracy: {test_acc:.2f}%")
    
    # Save expert weights
    torch.save(model.state_dict(), f"models/expert_{task_name}.pt")
    print(f"Saved expert_{task_name}.pt")

print("\n--- Expert Training Complete ---")
