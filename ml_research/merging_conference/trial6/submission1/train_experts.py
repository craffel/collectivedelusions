import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import argparse

# Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED error
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = True

def get_resnet18_32x32():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model

def train(dataset_name, epochs, batch_size, lr, gpu_id):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Training {dataset_name} on device: {device}")
    
    os.makedirs("./checkpoints", exist_ok=True)
    data_dir = "./data"
    
    # Setup transforms
    if dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)
        
    elif dataset_name == "svhn":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        trainset = torchvision.datasets.SVHN(root=data_dir, split="train", download=False, transform=transform_train)
        testset = torchvision.datasets.SVHN(root=data_dir, split="test", download=False, transform=transform_test)
        
    elif dataset_name == "fmnist":
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=transform_test)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = get_resnet18_32x32().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        train_loss /= total
        train_acc = 100. * correct / total
        
        # Eval
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        test_loss /= test_total
        test_acc = 100. * test_correct / test_total
        
        print(f"[{dataset_name}] Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"./checkpoints/expert_{dataset_name}.pth")
            print(f"Saved best model for {dataset_name} with accuracy {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "svhn", "fmnist"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    
    train(args.dataset, args.epochs, args.batch_size, args.lr, args.gpu_id)
