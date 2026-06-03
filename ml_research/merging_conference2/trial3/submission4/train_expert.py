import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on some cluster nodes
torch.backends.cudnn.enabled = False

# Define transforms
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_color = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class ExpertModel(nn.Module):
    def __init__(self, task_name):
        super().__init__()
        # Load pre-trained ResNet-18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(in_features, 10)
        self.task_name = task_name
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

def train(task, epochs=5, lr=5e-4, weight_decay=1e-4, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training expert for task: {task} on device: {device}")
    
    # Setup dataset
    if task == 'mnist':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_gray)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray)
    elif task == 'fashion':
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray)
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)
    elif task == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_color)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_color)
    else:
        raise ValueError(f"Unknown task {task}")
        
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    model = ExpertModel(task).to(device)
    
    # Save pre-trained backbone weights (only once)
    pretrained_path = "resnet18_pretrained.pth"
    if not os.path.exists(pretrained_path):
        torch.save(model.backbone.state_dict(), pretrained_path)
        print(f"Saved pre-trained backbone weights to {pretrained_path}")
        
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
            
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    # Evaluation
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
            
    test_acc = 100.0 * correct / total
    print(f"Final Test Accuracy for {task}: {test_acc:.2f}%")
    
    # Save expert checkpoint
    checkpoint_path = f"expert_{task}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc
    }, checkpoint_path)
    print(f"Saved expert model checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an expert on a task")
    parser.add_argument("--task", type=str, required=True, choices=['mnist', 'fashion', 'cifar10'], help="Task to train on")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    args = parser.parse_args()
    
    train(args.task, epochs=args.epochs)
