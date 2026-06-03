import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# Ensure checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED error.")

# Hyperparameters
BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-4

# Transform to convert to RGB, resize, and normalize to ImageNet standards
transform_rgb = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloader(task_name, is_train):
    if task_name == "mnist":
        dataset = torchvision.datasets.MNIST(root='./data', train=is_train, download=False, transform=transform_gray)
    elif task_name == "fashion":
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=is_train, download=False, transform=transform_gray)
    elif task_name == "cifar":
        dataset = torchvision.datasets.CIFAR10(root='./data', train=is_train, download=False, transform=transform_rgb)
    else:
        raise ValueError(f"Unknown task {task_name}")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=is_train, num_workers=4, pin_memory=True)
    return loader

def train_expert(task_name):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    train_loader = get_dataloader(task_name, is_train=True)
    test_loader = get_dataloader(task_name, is_train=False)
    
    # Load pretrained model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
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
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
        
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    final_acc = 100.0 * correct / total
    print(f"Final Test Accuracy for {task_name.upper()}: {final_acc:.2f}%")
    
    # Save checkpoint
    checkpoint_path = f"checkpoints/{task_name}_expert.pt"
    torch.save({
        'fc_state_dict': model.fc.state_dict(),
        'backbone_state_dict': {k: v for k, v in model.state_dict().items() if not k.startswith('fc.')},
        'accuracy': final_acc
    }, checkpoint_path)
    print(f"Saved expert checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train_expert("mnist")
    train_expert("fashion")
    train_expert("cifar")
