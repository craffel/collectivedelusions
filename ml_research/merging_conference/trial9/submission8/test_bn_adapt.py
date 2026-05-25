import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os

# Define the SimpleCNN architecture exactly as in run_experiments.py
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.fc = nn.Linear(64 * 3 * 3, 10)
        self.relu = nn.ReLU()

    def forward(self, x, return_features=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        feat = self.pool3(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        if return_features:
            return out, feat
        return out

def main():
    device = torch.device("cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # Load dataset
    print("Loading datasets...")
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True)
    fashion_loader = torch.utils.data.DataLoader(fashion_test, batch_size=64, shuffle=True)
    
    # Load Expert 0 (MNIST)
    expert0 = SimpleCNN().to(device)
    expert0.load_state_dict(torch.load("expert0.pt", map_location=device))
    expert0.eval()
    
    # Load Expert 1 (Fashion)
    expert1 = SimpleCNN().to(device)
    expert1.load_state_dict(torch.load("expert1.pt", map_location=device))
    expert1.eval()
    
    # Let's get one batch of Clean MNIST and Noisy MNIST
    mnist_iter = iter(mnist_loader)
    clean_mnist_inputs, mnist_targets = next(mnist_iter)
    noisy_mnist_inputs = clean_mnist_inputs + 0.6 * torch.randn_like(clean_mnist_inputs)
    noisy_mnist_inputs = torch.clamp(noisy_mnist_inputs, -1.0, 1.0)
    
    # Let's get one batch of Clean Fashion and Noisy Fashion
    fashion_iter = iter(fashion_loader)
    clean_fashion_inputs, fashion_targets = next(fashion_iter)
    noisy_fashion_inputs = clean_fashion_inputs + 0.6 * torch.randn_like(clean_fashion_inputs)
    noisy_fashion_inputs = torch.clamp(noisy_fashion_inputs, -1.0, 1.0)
    
    def eval_batch(model, inputs, targets, bn_mode="eval"):
        # Configure BN
        if bn_mode == "eval":
            model.eval()
        else:
            model.eval()
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train() # use batch statistics
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            acc = 100.0 * correct / targets.size(0)
        return acc

    print("\n--- EXPERT 0 (MNIST) EVALUATION ---")
    print(f"Clean MNIST (BN eval): {eval_batch(expert0, clean_mnist_inputs, mnist_targets, 'eval'):.2f}%")
    print(f"Clean MNIST (BN train/adapt): {eval_batch(expert0, clean_mnist_inputs, mnist_targets, 'train'):.2f}%")
    print(f"Noisy MNIST (BN eval): {eval_batch(expert0, noisy_mnist_inputs, mnist_targets, 'eval'):.2f}%")
    print(f"Noisy MNIST (BN train/adapt): {eval_batch(expert0, noisy_mnist_inputs, mnist_targets, 'train'):.2f}%")
    
    print("\n--- EXPERT 1 (FashionMNIST) EVALUATION ---")
    print(f"Clean Fashion (BN eval): {eval_batch(expert1, clean_fashion_inputs, fashion_targets, 'eval'):.2f}%")
    print(f"Clean Fashion (BN train/adapt): {eval_batch(expert1, clean_fashion_inputs, fashion_targets, 'train'):.2f}%")
    print(f"Noisy Fashion (BN eval): {eval_batch(expert1, noisy_fashion_inputs, fashion_targets, 'eval'):.2f}%")
    print(f"Noisy Fashion (BN train/adapt): {eval_batch(expert1, noisy_fashion_inputs, fashion_targets, 'train'):.2f}%")

if __name__ == "__main__":
    main()
