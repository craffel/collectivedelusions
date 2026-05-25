import os
import torch
# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on this cluster
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories for saving models
    os.makedirs("models", exist_ok=True)

    # Image transformations for ResNet-18
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    # Load CIFAR-10
    cifar_train_full = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
    # Subset of 5000 images for fast fine-tuning (as in LFWA)
    cifar_subset_indices = list(range(5000))
    cifar_train = Subset(cifar_train_full, cifar_subset_indices)
    cifar_loader = DataLoader(cifar_train, batch_size=64, shuffle=True, num_workers=4)

    # Load SVHN
    svhn_train_full = torchvision.datasets.SVHN(root='./data', split='train', transform=transform, download=False)
    # Subset of 5000 images for fast fine-tuning
    svhn_subset_indices = list(range(5000))
    svhn_train = Subset(svhn_train_full, svhn_subset_indices)
    svhn_loader = DataLoader(svhn_train, batch_size=64, shuffle=True, num_workers=4)

    # Load base pre-trained ResNet-18
    print("Saving base pre-trained ResNet-18 backbone...")
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Save the base model state dict (backbone only, but we can save full state dict as reference)
    torch.save(base_model.state_dict(), "models/base_pretrained.pt")

    # Train CIFAR-10 Expert
    print("\n--- Training CIFAR-10 Expert ---")
    cifar_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    cifar_model.fc = nn.Linear(512, 10)
    cifar_model = cifar_model.to(device)

    # Fine-tune backbone and head
    optimizer = optim.AdamW([
        {'params': cifar_model.fc.parameters(), 'lr': 1e-3},
        {'params': [p for name, p in cifar_model.named_parameters() if 'fc' not in name], 'lr': 1e-4}
    ], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    cifar_model.train()
    for epoch in range(3):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in cifar_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = cifar_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/3 | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    torch.save(cifar_model.state_dict(), "models/cifar10_expert.pt")
    print("CIFAR-10 expert saved.")

    # Train SVHN Expert
    print("\n--- Training SVHN Expert ---")
    svhn_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    svhn_model.fc = nn.Linear(512, 10)
    svhn_model = svhn_model.to(device)

    # Fine-tune backbone and head
    optimizer = optim.AdamW([
        {'params': svhn_model.fc.parameters(), 'lr': 1e-3},
        {'params': [p for name, p in svhn_model.named_parameters() if 'fc' not in name], 'lr': 1e-4}
    ], weight_decay=1e-4)

    svhn_model.train()
    for epoch in range(3):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in svhn_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = svhn_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/3 | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    torch.save(svhn_model.state_dict(), "models/svhn_expert.pt")
    print("SVHN expert saved.")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
