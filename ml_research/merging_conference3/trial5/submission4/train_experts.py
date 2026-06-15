import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import timm
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Directory to save expert checkpoints
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
transform_gray = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_dataset(task_name, train=True):
    if task_name == "MNIST":
        return torchvision.datasets.MNIST(root=DATA_DIR, train=train, download=True, transform=transform_gray)
    elif task_name == "FashionMNIST":
        return torchvision.datasets.FashionMNIST(root=DATA_DIR, train=train, download=True, transform=transform_gray)
    elif task_name == "CIFAR10":
        return torchvision.datasets.CIFAR10(root=DATA_DIR, train=train, download=True, transform=transform_rgb)
    elif task_name == "SVHN":
        split = "train" if train else "test"
        return torchvision.datasets.SVHN(root=DATA_DIR, split=split, download=True, transform=transform_rgb)
    else:
        raise ValueError(f"Unknown task: {task_name}")

def train_expert(task_name):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"expert_{task_name.lower()}.pth")
    if os.path.exists(checkpoint_path):
        print(f"Expert for {task_name} already exists at {checkpoint_path}. Skipping training.")
        return

    print(f"\n--- Training Expert for {task_name} ---")
    
    # Load dataset
    train_dataset = get_dataset(task_name, train=True)
    # Use standard subset of train set if we want to speed up training,
    # but the paper says "trained to high convergence... 15 epochs".
    # To ensure high convergence, let's train on the full training set.
    # Note: On a p5 GPU (H100), full epoch training on vit_tiny is incredibly fast (~1-2s/epoch).
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    
    # Load base model from timm
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10)
    model.to(device)
    
    # Dual learning rate optimization setup
    backbone_params = [p for name, p in model.named_parameters() if "head" not in name]
    head_params = model.head.parameters()
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 2e-5},
        {"params": head_params, "lr": 1e-3}
    ], weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    epochs = 15
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total * 100.0
        print(f"Epoch {epoch} complete. Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
    # Save the expert model state dict
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved expert model to {checkpoint_path}")

if __name__ == "__main__":
    tasks = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]
    for task in tasks:
        train_expert(task)
    print("All expert training complete!")
