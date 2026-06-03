import os
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

def get_dataloaders(batch_size=128):
    class RepeatChannels:
        def __call__(self, x):
            return x.repeat(3, 1, 1)

    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        RepeatChannels(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=transform_gray)
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=False, transform=transform_gray)

    fmnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=False, transform=transform_gray)
    fmnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=False, transform=transform_gray)

    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_color)
    cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_color)

    loaders = {
        "mnist": (
            DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        ),
        "fmnist": (
            DataLoader(fmnist_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        ),
        "cifar10": (
            DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        ),
    }
    return loaders

def train_expert(task_name, train_loader, test_loader, epochs, device):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

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

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        test_acc = 100.0 * test_correct / test_total
        print(f"Test Acc: {test_acc:.2f}%")

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/expert_{task_name}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved expert checkpoint to {checkpoint_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    loaders = get_dataloaders()

    train_expert("mnist", loaders["mnist"][0], loaders["mnist"][1], epochs=2, device=device)
    train_expert("fmnist", loaders["fmnist"][0], loaders["fmnist"][1], epochs=2, device=device)
    train_expert("cifar10", loaders["cifar10"][0], loaders["cifar10"][1], epochs=5, device=device)

if __name__ == "__main__":
    main()
