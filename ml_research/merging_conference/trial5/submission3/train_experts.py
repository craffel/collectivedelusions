import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import os
import random
import numpy as np

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  # Disable cuDNN to avoid initialization errors

from models import ResNetEncoder, ClassificationHead

def get_subsets(seed=42):
    set_seed(seed)
    
    # Standard ResNet-18 ImageNet transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10
    print("Loading CIFAR-10...")
    cifar_full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    cifar_full_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Load SVHN
    print("Loading SVHN...")
    svhn_full_train = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    svhn_full_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    
    # Deterministic subsets of 5,000 training images
    g = torch.Generator().manual_seed(seed)
    
    cifar_indices = torch.randperm(len(cifar_full_train), generator=g)[:5000].tolist()
    cifar_train_subset = Subset(cifar_full_train, cifar_indices)
    
    svhn_indices = torch.randperm(len(svhn_full_train), generator=g)[:5000].tolist()
    svhn_train_subset = Subset(svhn_full_train, svhn_indices)
    
    # Validation/Test subsets of 1,024 images
    cifar_val_indices = torch.randperm(len(cifar_full_test), generator=g)[:1024].tolist()
    cifar_val_subset = Subset(cifar_full_test, cifar_val_indices)
    
    svhn_val_indices = torch.randperm(len(svhn_full_test), generator=g)[:1024].tolist()
    svhn_val_subset = Subset(svhn_full_test, svhn_val_indices)
    
    return cifar_train_subset, cifar_val_subset, svhn_train_subset, svhn_val_subset

def train_one_expert(name, train_dataset, val_dataset, epochs=3, lr=3e-4, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining expert for {name} on device: {device}")
    
    encoder = ResNetEncoder().to(device)
    head = ClassificationHead().to(device)
    
    # Optimizer and loss
    # Fine-tune entire model (both encoder and head)
    optimizer = optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    for epoch in range(epochs):
        encoder.train()
        head.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            features = encoder(images)
            outputs = head(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100.0 * correct / total
        
        # Validation
        encoder.eval()
        head.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                features = encoder(images)
                outputs = head(features)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
    # Save checkpoint
    checkpoint = {
        'encoder': encoder.state_dict(),
        'head': head.state_dict(),
        'val_acc': val_acc
    }
    torch.save(checkpoint, f"expert_{name.lower()}.pth")
    print(f"Saved expert_{name.lower()}.pth with validation accuracy: {val_acc:.2f}%")
    return val_acc

def main():
    set_seed(42)
    cifar_train, cifar_val, svhn_train, svhn_val = get_subsets()
    
    # Train both experts
    cifar_acc = train_one_expert("CIFAR10", cifar_train, cifar_val)
    svhn_acc = train_one_expert("SVHN", svhn_train, svhn_val)
    
    print("\nTraining completed successfully!")
    print(f"CIFAR-10 Expert Val Acc: {cifar_acc:.2f}%")
    print(f"SVHN Expert Val Acc: {svhn_acc:.2f}%")

if __name__ == "__main__":
    main()
