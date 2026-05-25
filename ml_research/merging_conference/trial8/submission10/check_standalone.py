import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Same architecture
class SharedBackbone(nn.Module):
    def __init__(self):
        super(SharedBackbone, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 7 * 7, 128)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ExpertHead(nn.Module):
    def __init__(self):
        super(ExpertHead, self).__init__()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, feats):
        return self.fc2(self.relu(feats))

def train_shared_moe(mnist_dataset, kmnist_dataset, epochs=2):
    backbone = SharedBackbone()
    mnist_head = ExpertHead()
    kmnist_head = ExpertHead()
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(mnist_head.parameters()) + list(kmnist_head.parameters()),
        lr=0.003
    )
    criterion = nn.CrossEntropyLoss()
    indices = list(range(12000))
    mnist_loader = DataLoader(Subset(mnist_dataset, indices), batch_size=256, shuffle=True)
    kmnist_loader = DataLoader(Subset(kmnist_dataset, indices), batch_size=256, shuffle=True)
    backbone.train()
    mnist_head.train()
    kmnist_head.train()
    for epoch in range(epochs):
        for (x_m, y_m), (x_k, y_k) in zip(mnist_loader, kmnist_loader):
            optimizer.zero_grad()
            feats_m = backbone(x_m)
            out_m = mnist_head(feats_m)
            loss_m = criterion(out_m, y_m)
            feats_k = backbone(x_k)
            out_k = kmnist_head(feats_k)
            loss_k = criterion(out_k, y_k)
            loss = loss_m + loss_k
            loss.backward()
            optimizer.step()
    return backbone, mnist_head, kmnist_head

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_train = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    backbone, mnist_head, kmnist_head = train_shared_moe(mnist_train, kmnist_train, epochs=2)
    backbone.eval()
    mnist_head.eval()
    kmnist_head.eval()
    
    mnist_test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)
    kmnist_test_loader = DataLoader(kmnist_test, batch_size=128, shuffle=False)
    
    mnist_iter = iter(mnist_test_loader)
    kmnist_iter = iter(kmnist_test_loader)
    
    # Evaluate Clean MNIST
    accs_clean_m = []
    with torch.no_grad():
        for _ in range(10):
            x, y = next(mnist_iter)
            feats = backbone(x)
            out = mnist_head(feats)
            pred = out.argmax(dim=1)
            accs_clean_m.append(pred.eq(y).sum().item() / x.size(0))
            
    # Evaluate Noisy MNIST (std=1.2)
    accs_noisy_m = []
    with torch.no_grad():
        for _ in range(10):
            x, y = next(mnist_iter)
            noisy_x = x + 0.9 * torch.randn_like(x)
            feats = backbone(noisy_x)
            out = mnist_head(feats)
            pred = out.argmax(dim=1)
            accs_noisy_m.append(pred.eq(y).sum().item() / x.size(0))
            
    # Evaluate Clean KMNIST
    accs_clean_k = []
    with torch.no_grad():
        for _ in range(10):
            x, y = next(kmnist_iter)
            feats = backbone(x)
            out = kmnist_head(feats)
            pred = out.argmax(dim=1)
            accs_clean_k.append(pred.eq(y).sum().item() / x.size(0))
            
    # Evaluate Noisy KMNIST (std=1.2)
    accs_noisy_k = []
    with torch.no_grad():
        for _ in range(10):
            x, y = next(kmnist_iter)
            noisy_x = x + 0.9 * torch.randn_like(x)
            feats = backbone(noisy_x)
            out = kmnist_head(feats)
            pred = out.argmax(dim=1)
            accs_noisy_k.append(pred.eq(y).sum().item() / x.size(0))
            
    print(f"Clean MNIST standalone: {100.0 * np.mean(accs_clean_m):.2f}%")
    print(f"Noisy MNIST standalone (std=1.2): {100.0 * np.mean(accs_noisy_m):.2f}%")
    print(f"Clean KMNIST standalone: {100.0 * np.mean(accs_clean_k):.2f}%")
    print(f"Noisy KMNIST standalone (std=1.2): {100.0 * np.mean(accs_noisy_k):.2f}%")
