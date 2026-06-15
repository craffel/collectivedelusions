import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

def get_transforms(dataset_name):
    if dataset_name in ['MNIST', 'FashionMNIST']:
        return transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def get_dataset(dataset_name, split='train'):
    transform = get_transforms(dataset_name)
    download = True
    root = './data'
    
    if dataset_name == 'MNIST':
        return datasets.MNIST(root=root, train=(split == 'train'), download=download, transform=transform)
    elif dataset_name == 'FashionMNIST':
        return datasets.FashionMNIST(root=root, train=(split == 'train'), download=download, transform=transform)
    elif dataset_name == 'CIFAR10':
        return datasets.CIFAR10(root=root, train=(split == 'train'), download=download, transform=transform)
    elif dataset_name == 'SVHN':
        svhn_split = 'train' if split == 'train' else 'test'
        return datasets.SVHN(root=root, split=svhn_split, download=download, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def train_expert(dataset_name, epochs=2, batch_size=256, lr=5e-4):
    save_path = f'checkpoints/{dataset_name.lower()}_expert.pth'
    if os.path.exists(save_path):
        print(f"\nExpert checkpoint for {dataset_name} already exists at {save_path}. Skipping training.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining expert for {dataset_name} on {device}...")
    
    train_dataset = get_dataset(dataset_name, split='train')
    val_dataset = get_dataset(dataset_name, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize pretrained ViT-Tiny model
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    # Replace the classification head for 10 classes
    model.head = nn.Linear(model.head.in_features, 10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_loss = train_loss / total
        epoch_acc = 100.0 * correct / total
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
    os.makedirs('checkpoints', exist_ok=True)
    save_path = f'checkpoints/{dataset_name.lower()}_expert.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Saved expert checkpoint to {save_path}")

if __name__ == '__main__':
    datasets_to_train = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    for ds in datasets_to_train:
        train_expert(ds, epochs=2)
