import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors on this cluster
torch.backends.cudnn.enabled = False

def get_resnet18_model():
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except ImportError:
        from torchvision.models import resnet18
        model = resnet18(pretrained=True)
    # Replace the classification head for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
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
    return running_loss / len(loader.dataset), 100.0 * correct / total

def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs("models", exist_ok=True)
    
    # Save the original pretrained base model
    print("Saving pretrained base model...")
    base_model = get_resnet18_model()
    # Save only the state_dict of the encoder (excluding fc)
    encoder_state = {k: v for k, v in base_model.state_dict().items() if not k.startswith("fc.")}
    torch.save(encoder_state, "models/pretrained_base.pth")
    
    # Define standard transforms
    # MNIST/FashionMNIST/KMNIST are grayscale, so we convert them to 3 channels and normalize
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    tasks = {
        "mnist": datasets.MNIST,
        "fashionmnist": datasets.FashionMNIST,
        "kmnist": datasets.KMNIST
    }
    
    for task_name, dataset_cls in tasks.items():
        print(f"\n--- Fine-tuning on {task_name.upper()} ---")
        # Load dataset
        train_dataset = dataset_cls(root="./data", train=True, download=True, transform=transform)
        test_dataset = dataset_cls(root="./data", train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        # Load fresh pretrained model
        model = get_resnet18_model().to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Train for 3 epochs
        epochs = 3
        for epoch in range(1, epochs + 1):
            loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_acc = evaluate_model(model, test_loader, device)
            print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}, Train Acc: {acc:.2f}%, Test Acc: {test_acc:.2f}%")
            
        # Save expert model
        torch.save(model.state_dict(), f"models/{task_name}_expert.pth")
        print(f"Saved {task_name} expert to models/{task_name}_expert.pth")

if __name__ == "__main__":
    main()
