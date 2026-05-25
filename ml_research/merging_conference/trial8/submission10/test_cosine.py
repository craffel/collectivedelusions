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
    
    mnist_cal_loader = DataLoader(Subset(mnist_train, list(range(500))), batch_size=500, shuffle=False)
    kmnist_cal_loader = DataLoader(Subset(kmnist_train, list(range(500))), batch_size=500, shuffle=False)
    with torch.no_grad():
        for x, _ in mnist_cal_loader:
            mnist_feats = backbone(x)
            proto_mnist = mnist_feats.mean(dim=0)
        for x, _ in kmnist_cal_loader:
            kmnist_feats = backbone(x)
            proto_kmnist = kmnist_feats.mean(dim=0)
            
    mnist_test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)
    kmnist_test_loader = DataLoader(kmnist_test, batch_size=128, shuffle=False)
    
    mnist_iter = iter(mnist_test_loader)
    kmnist_iter = iter(kmnist_test_loader)
    
    # Get a clean batch and a noisy batch
    x_clean_m, _ = next(mnist_iter)
    x_noisy_m = x_clean_m + 0.6 * torch.randn_like(x_clean_m)
    x_noisy_m_severe = x_clean_m + 1.2 * torch.randn_like(x_clean_m)
    
    x_clean_k, _ = next(kmnist_iter)
    x_noisy_k = x_clean_k + 0.6 * torch.randn_like(x_clean_k)
    
    with torch.no_grad():
        feats_clean_m = backbone(x_clean_m)
        feats_noisy_m = backbone(x_noisy_m)
        feats_noisy_m_severe = backbone(x_noisy_m_severe)
        feats_clean_k = backbone(x_clean_k)
        
        bm_clean_m = feats_clean_m.mean(dim=0)
        bm_noisy_m = feats_noisy_m.mean(dim=0)
        bm_noisy_m_severe = feats_noisy_m_severe.mean(dim=0)
        bm_clean_k = feats_clean_k.mean(dim=0)
        
        # Helper to compute cosine similarity
        def cos_sim(a, b):
            return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
            
        print("--- Cosine Similarity of Batch Means to Prototypes ---")
        print(f"Clean MNIST BM to Proto MNIST: {cos_sim(bm_clean_m, proto_mnist).item():.4f}")
        print(f"Clean MNIST BM to Proto KMNIST: {cos_sim(bm_clean_m, proto_kmnist).item():.4f}")
        print()
        print(f"Noisy MNIST (0.6) BM to Proto MNIST: {cos_sim(bm_noisy_m, proto_mnist).item():.4f}")
        print(f"Noisy MNIST (0.6) BM to Proto KMNIST: {cos_sim(bm_noisy_m, proto_kmnist).item():.4f}")
        print()
        print(f"Noisy MNIST (1.2) BM to Proto MNIST: {cos_sim(bm_noisy_m_severe, proto_mnist).item():.4f}")
        print(f"Noisy MNIST (1.2) BM to Proto KMNIST: {cos_sim(bm_noisy_m_severe, proto_kmnist).item():.4f}")
        print()
        print(f"Clean KMNIST BM to Proto MNIST: {cos_sim(bm_clean_k, proto_mnist).item():.4f}")
        print(f"Clean KMNIST BM to Proto KMNIST: {cos_sim(bm_clean_k, proto_kmnist).item():.4f}")
        
        # Compare with raw Euclidean distances of batch means
        def dist(a, b):
            return torch.norm(a - b).item()
            
        print("\n--- Euclidean Distance of Batch Means to Prototypes ---")
        print(f"Clean MNIST BM to Proto MNIST: {dist(bm_clean_m, proto_mnist):.4f}")
        print(f"Clean MNIST BM to Proto KMNIST: {dist(bm_clean_m, proto_kmnist):.4f}")
        print()
        print(f"Noisy MNIST (0.6) BM to Proto MNIST: {dist(bm_noisy_m, proto_mnist):.4f}")
        print(f"Noisy MNIST (0.6) BM to Proto KMNIST: {dist(bm_noisy_m, proto_kmnist):.4f}")
        print()
        print(f"Noisy MNIST (1.2) BM to Proto MNIST: {dist(bm_noisy_m_severe, proto_mnist):.4f}")
        print(f"Noisy MNIST (1.2) BM to Proto KMNIST: {dist(bm_noisy_m_severe, proto_kmnist):.4f}")
