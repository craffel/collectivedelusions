import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

def get_transform(is_gray=False):
    if is_gray:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataset(name, root='./data', train=True):
    os.makedirs(root, exist_ok=True)
    if name == 'MNIST':
        return datasets.MNIST(root=root, train=train, download=True, transform=get_transform(is_gray=True))
    elif name == 'FashionMNIST':
        return datasets.FashionMNIST(root=root, train=train, download=True, transform=get_transform(is_gray=True))
    elif name == 'CIFAR10':
        return datasets.CIFAR10(root=root, train=train, download=True, transform=get_transform(is_gray=False))
    elif name == 'SVHN':
        split = 'train' if train else 'test'
        return datasets.SVHN(root=root, split=split, download=True, transform=get_transform(is_gray=False))
    else:
        raise ValueError(f"Unknown dataset {name}")

def train_expert(task_name, device):
    print(f"\n--- Training Expert for {task_name} ---")
    train_dataset = get_dataset(task_name, train=True)
    test_dataset = get_dataset(task_name, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.head = nn.Linear(model.num_features, 10)
    model = model.to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Train for 2 epochs
    epochs = 2
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
        epoch_acc = correct / total
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    test_acc = correct / total
    print(f"Test Accuracy for {task_name}: {test_acc:.4f}")
    
    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f'checkpoints/{task_name}_expert.pt'
    torch.save({
        'state_dict': model.state_dict(),
        'test_acc': test_acc
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tasks = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    for task in tasks:
        checkpoint_path = f'checkpoints/{task}_expert.pt'
        if os.path.exists(checkpoint_path):
            print(f"Checkpoint for {task} already exists at {checkpoint_path}. Skipping training.")
        else:
            train_expert(task, device)
            
    # Save the base model weights with a placeholder head for convenience
    base_checkpoint_path = 'checkpoints/base_model.pt'
    if not os.path.exists(base_checkpoint_path):
        base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        base_model.head = nn.Linear(base_model.num_features, 10)
        torch.save({
            'state_dict': base_model.state_dict()
        }, base_checkpoint_path)
        print(f"Saved base model to {base_checkpoint_path}")
