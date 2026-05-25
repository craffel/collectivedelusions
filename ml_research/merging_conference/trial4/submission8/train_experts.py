import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import os

torch.backends.cudnn.enabled = False

def get_datasets():
    # Transforms for ResNet-18 (ImageNet resolution)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # CIFAR-10
    print("Loading CIFAR-10...")
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # SVHN
    print("Loading SVHN...")
    svhn_train = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    svhn_test = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)

    # Subsets for fast and efficient training/evaluation
    # 5,000 for training, 1,000 for test
    cifar_train_sub = Subset(cifar_train, list(range(5000)))
    cifar_test_sub = Subset(cifar_test, list(range(1000)))
    cifar_calib_sub = Subset(cifar_train, list(range(5000, 5500))) # 500 for calibration

    svhn_train_sub = Subset(svhn_train, list(range(5000)))
    svhn_test_sub = Subset(svhn_test, list(range(1000)))
    svhn_calib_sub = Subset(svhn_train, list(range(5000, 5500))) # 500 for calibration

    return (cifar_train_sub, cifar_test_sub, cifar_calib_sub), (svhn_train_sub, svhn_test_sub, svhn_calib_sub)

def train_expert(model, train_loader, device, epochs=3):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    (cifar_train, cifar_test, cifar_calib), (svhn_train, svhn_test, svhn_calib) = get_datasets()

    # Load shared pre-trained ResNet-18
    print("Loading base ResNet-18 model...")
    base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Save base model encoder weights (we exclude the fc layer as heads are task-specific)
    base_encoder_state = {k: v.cpu() for k, v in base_model.state_dict().items() if not k.startswith("fc.")}
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(base_encoder_state, "checkpoints/base_encoder.pt")
    print("Saved base encoder to checkpoints/base_encoder.pt")

    # Expert 1: CIFAR-10
    print("\n--- Training Expert 1 (CIFAR-10) ---")
    expert1 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    expert1.fc = nn.Linear(expert1.fc.in_features, 10)
    train_loader = DataLoader(cifar_train, batch_size=128, shuffle=True)
    train_expert(expert1, train_loader, device, epochs=3)
    torch.save(expert1.state_dict(), "checkpoints/expert_cifar10.pt")
    print("Saved Expert 1 to checkpoints/expert_cifar10.pt")

    # Expert 2: SVHN
    print("\n--- Training Expert 2 (SVHN) ---")
    expert2 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    expert2.fc = nn.Linear(expert2.fc.in_features, 10)
    train_loader = DataLoader(svhn_train, batch_size=128, shuffle=True)
    train_expert(expert2, train_loader, device, epochs=3)
    torch.save(expert2.state_dict(), "checkpoints/expert_svhn.pt")
    print("Saved Expert 2 to checkpoints/expert_svhn.pt")

    # Save calibration and test datasets for the merging phase
    print("\nSaving test and calibration datasets...")
    torch.save(cifar_calib, "checkpoints/cifar_calib.pt")
    torch.save(svhn_calib, "checkpoints/svhn_calib.pt")
    torch.save(cifar_test, "checkpoints/cifar_test.pt")
    torch.save(svhn_test, "checkpoints/svhn_test.pt")
    print("Saved datasets to checkpoints/")

if __name__ == "__main__":
    main()
