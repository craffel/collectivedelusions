import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader

def get_resnet18_grayscale():
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(512, 10)
    return model

def evaluate(model, dataset, device="cuda"):
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=False, transform=transform)
    
    model1 = get_resnet18_grayscale().to(device)
    model2 = get_resnet18_grayscale().to(device)
    
    model1.load_state_dict(torch.load("./checkpoints/expert_mnist.pth", map_location=device))
    model2.load_state_dict(torch.load("./checkpoints/expert_kmnist.pth", map_location=device))
    
    acc1_mnist = evaluate(model1, mnist_test, device=device)
    acc1_kmnist = evaluate(model1, kmnist_test, device=device)
    acc2_mnist = evaluate(model2, mnist_test, device=device)
    acc2_kmnist = evaluate(model2, kmnist_test, device=device)
    
    print(f"Expert 1 (MNIST) accuracy on MNIST test: {acc1_mnist:.2f}%")
    print(f"Expert 1 (MNIST) accuracy on KMNIST test: {acc1_kmnist:.2f}%")
    print(f"Expert 2 (KMNIST) accuracy on MNIST test: {acc2_mnist:.2f}%")
    print(f"Expert 2 (KMNIST) accuracy on KMNIST test: {acc2_kmnist:.2f}%")

if __name__ == "__main__":
    main()
