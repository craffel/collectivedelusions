import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

def main():
    print("Setting up training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    os.makedirs('experts', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Transforms
    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Datasets
    print("Loading datasets...")
    mnist_train_full = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_mnist)
    mnist_test = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_mnist)

    fmnist_train_full = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform_mnist)
    fmnist_test = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform_mnist)

    cifar_train_full = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_cifar)
    cifar_test = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_cifar)

    # Deterministic Subsets of 3,000 samples
    mnist_train = Subset(mnist_train_full, list(range(3000)))
    fmnist_train = Subset(fmnist_train_full, list(range(3000)))
    cifar_train = Subset(cifar_train_full, list(range(3000)))

    # DataLoaders
    train_loaders = {
        'mnist': DataLoader(mnist_train, batch_size=128, shuffle=True, num_workers=2),
        'fmnist': DataLoader(fmnist_train, batch_size=128, shuffle=True, num_workers=2),
        'cifar10': DataLoader(cifar_train, batch_size=128, shuffle=True, num_workers=2)
    }

    test_loaders = {
        'mnist': DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=2),
        'fmnist': DataLoader(fmnist_test, batch_size=128, shuffle=False, num_workers=2),
        'cifar10': DataLoader(cifar_test, batch_size=128, shuffle=False, num_workers=2)
    }

    # Load Base ImageNet Pre-trained Model
    print("Loading base model...")
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except ImportError:
        base_model = models.resnet18(pretrained=True)
    
    # Save base model state dict
    torch.save(base_model.state_dict(), 'experts/base.pt')
    print("Saved base model to experts/base.pt")

    tasks = ['mnist', 'fmnist', 'cifar10']
    
    for task in tasks:
        print(f"\n--- Fine-tuning Expert on {task.upper()} ---")
        
        # Reload base model for each expert to ensure fresh initialization
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except ImportError:
            model = models.resnet18(pretrained=True)
            
        model.fc = nn.Linear(512, 10)
        model = model.to(device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        criterion = nn.CrossEntropyLoss()

        # Training Loop
        for epoch in range(5):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loaders[task]:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
            
            scheduler.step()
            epoch_loss = total_loss / total
            epoch_acc = 100.0 * correct / total
            print(f"Epoch {epoch+1}/5 | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loaders[task]:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()
                test_total += labels.size(0)
        
        test_acc = 100.0 * test_correct / test_total
        print(f"Finished {task.upper()} | Test Acc: {test_acc:.2f}%")
        
        # Save expert weights
        torch.save(model.state_dict(), f'experts/{task}.pt')
        print(f"Saved {task} expert to experts/{task}.pt")

if __name__ == '__main__':
    main()
