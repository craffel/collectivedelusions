import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import timm
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ImageNet normalization
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_grayscale = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1 channel to 3 channels
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define datasets
def get_dataset(name, train=True):
    if name == "mnist":
        return torchvision.datasets.MNIST(root="./data", train=train, download=True, transform=transform_grayscale)
    elif name == "fmnist":
        return torchvision.datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform_grayscale)
    elif name == "cifar10":
        return torchvision.datasets.CIFAR10(root="./data", train=train, download=True, transform=transform_rgb)
    elif name == "svhn":
        split = "train" if train else "test"
        return torchvision.datasets.SVHN(root="./data", split=split, download=True, transform=transform_rgb)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# Train function
def train_expert(name, epochs=5, num_samples=5000, batch_size=128):
    print(f"\n--- Training Expert for {name.upper()} ---")
    
    # Load dataset
    full_dataset = get_dataset(name, train=True)
    
    # Create subset for faster training
    indices = torch.randperm(len(full_dataset))[:num_samples].tolist()
    dataset = Subset(full_dataset, indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Load test set for evaluation
    test_dataset = get_dataset(name, train=False)
    test_indices = torch.randperm(len(test_dataset))[:1000].tolist()  # 1000 samples for test evaluation
    test_subset = Subset(test_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10)
    model = model.to(device)
    
    # Optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
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
            
        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        epoch_acc = 100.0 * correct / total
        
        # Test evaluation
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
        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f} | Train Acc = {epoch_acc:.2f}% | Test Acc = {test_acc:.2f}%")
        
    # Save checkpoint
    checkpoint_path = f"checkpoints/expert_{name}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Expert model saved to {checkpoint_path}")

if __name__ == "__main__":
    # Seed for reproducibility
    torch.manual_seed(42)
    
    datasets = ["mnist", "fmnist", "cifar10", "svhn"]
    for ds in datasets:
        train_expert(ds, epochs=5, num_samples=5000, batch_size=128)
