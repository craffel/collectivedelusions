import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from evaluate_ttmm import SimpleCNN, set_bn_mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load experts
model_mnist = SimpleCNN().to(device)
model_mnist.load_state_dict(torch.load("expert_mnist.pth", map_location=device))
model_mnist.eval()

model_fashion = SimpleCNN().to(device)
model_fashion.load_state_dict(torch.load("expert_fashion.pth", map_location=device))
model_fashion.eval()

# Datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)

loader_mnist_clean = DataLoader(Subset(mnist_test, list(range(0, 640))), batch_size=64, shuffle=False)
loader_mnist_noisy = DataLoader(Subset(mnist_test, list(range(640, 1280))), batch_size=64, shuffle=False)
loader_fashion_clean = DataLoader(Subset(fmnist_test, list(range(0, 640))), batch_size=64, shuffle=False)
loader_fashion_noisy = DataLoader(Subset(fmnist_test, list(range(640, 1280))), batch_size=64, shuffle=False)

def evaluate_loader(model, loader, noise_std=0.0, use_ttbn=False):
    model.eval()
    if use_ttbn:
        set_bn_mode(model, train=True)
    else:
        set_bn_mode(model, train=False)
        
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if noise_std > 0:
                noise = torch.randn_like(x) * noise_std
                x = torch.clamp(x + noise, -1.0, 1.0)
            out = model(x)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

print("--- MNIST Expert ---")
print(f"Clean MNIST (no TTBN): {evaluate_loader(model_mnist, loader_mnist_clean):.2f}%")
print(f"Clean MNIST (with TTBN): {evaluate_loader(model_mnist, loader_mnist_clean, use_ttbn=True):.2f}%")
print(f"Noisy MNIST (no TTBN): {evaluate_loader(model_mnist, loader_mnist_noisy, noise_std=0.6):.2f}%")
print(f"Noisy MNIST (with TTBN): {evaluate_loader(model_mnist, loader_mnist_noisy, noise_std=0.6, use_ttbn=True):.2f}%")

print("\n--- FashionMNIST Expert ---")
print(f"Clean Fashion (no TTBN): {evaluate_loader(model_fashion, loader_fashion_clean):.2f}%")
print(f"Clean Fashion (with TTBN): {evaluate_loader(model_fashion, loader_fashion_clean, use_ttbn=True):.2f}%")
print(f"Noisy Fashion (no TTBN): {evaluate_loader(model_fashion, loader_fashion_noisy, noise_std=0.6):.2f}%")
print(f"Noisy Fashion (with TTBN): {evaluate_loader(model_fashion, loader_fashion_noisy, noise_std=0.6, use_ttbn=True):.2f}%")
