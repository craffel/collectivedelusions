import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import torch
torch.backends.cudnn.enabled = False


def get_resnet18_grayscale():
    model = models.resnet18(weights='IMAGENET1K_V1')
    # Modify conv1: kernel_size=3, stride=1, padding=1, 1 input channel, no bias
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(512, 10)
    return model

def train_expert(dataset_name, train_loader, test_loader, save_path, epochs=3, device="cuda"):
    print(f"\n--- Training Expert on {dataset_name} ---")
    model = get_resnet18_grayscale().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
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
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        
    # Evaluation
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
    print(f"Finished Training. Test Acc on {dataset_name}: {test_acc:.2f}%")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint to {save_path}")
    return test_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download datasets
    print("Loading datasets...")
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    kmnist_train = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_train_loader = DataLoader(mnist_train, batch_size=256, shuffle=True, num_workers=2)
    mnist_test_loader = DataLoader(mnist_test, batch_size=256, shuffle=False, num_workers=2)
    
    kmnist_train_loader = DataLoader(kmnist_train, batch_size=256, shuffle=True, num_workers=2)
    kmnist_test_loader = DataLoader(kmnist_test, batch_size=256, shuffle=False, num_workers=2)
    
    acc_mnist = train_expert("MNIST", mnist_train_loader, mnist_test_loader, "./checkpoints/expert_mnist.pth", epochs=3, device=device)
    acc_kmnist = train_expert("KMNIST", kmnist_train_loader, kmnist_test_loader, "./checkpoints/expert_kmnist.pth", epochs=3, device=device)
    
    print("\nTraining completed successfully!")
    print(f"MNIST expert test accuracy: {acc_mnist:.2f}%")
    print(f"KMNIST expert test accuracy: {acc_kmnist:.2f}%")

if __name__ == "__main__":
    main()
