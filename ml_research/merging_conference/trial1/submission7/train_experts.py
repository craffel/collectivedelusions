import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model Definitions
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 7 * 7, 128)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc(x))
        return x

class Head(nn.Module):
    def __init__(self, num_classes=10):
        super(Head, self).__init__()
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        return self.fc(x)

# Function to get datasets (with robust synthetic fallback)
def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    try:
        print("Attempting to download MNIST, FashionMNIST, and KMNIST...")
        mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        
        fmnist_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        fmnist_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
        
        kmnist_train = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
        kmnist_test = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
        
        print("Successfully downloaded all real datasets.")
        return (mnist_train, mnist_test), (fmnist_train, fmnist_test), (kmnist_train, kmnist_test), False
    except Exception as e:
        print(f"Error downloading datasets: {e}. Falling back to procedurally generated synthetic task datasets...")
        # Create synthetic datasets
        # Task 1: MNIST-like (smooth blobs)
        # Task 2: FashionMNIST-like (geometric stripes and patterns)
        # Task 3: KMNIST-like (random squiggles)
        torch.manual_seed(42)
        
        def create_synthetic_task(pattern_type, num_samples=1000):
            images = []
            labels = []
            for i in range(num_samples):
                img = torch.randn(1, 28, 28) * 0.1
                label = i % 10
                # Add specific class-dependent patterns
                if pattern_type == "blobs":
                    # Add circular blobs at class-dependent coordinates
                    cx, cy = 5 + label * 2, 5 + label * 2
                    img[0, cx-2:cx+2, cy-2:cy+2] += 0.8
                elif pattern_type == "stripes":
                    # Add vertical/horizontal lines based on class
                    if label % 2 == 0:
                        img[0, label + 4, :] += 0.8
                    else:
                        img[0, :, label + 4] += 0.8
                elif pattern_type == "squiggles":
                    # Add random class-dependent patterns
                    img[0, :, :] += torch.sin(torch.linspace(0, label, 28)).view(1, -1) * 0.5
                
                images.append(img)
                labels.append(label)
            return TensorDataset(torch.stack(images), torch.tensor(labels))
            
        mnist_train = create_synthetic_task("blobs", 2000)
        mnist_test = create_synthetic_task("blobs", 500)
        
        fmnist_train = create_synthetic_task("stripes", 2000)
        fmnist_test = create_synthetic_task("stripes", 500)
        
        kmnist_train = create_synthetic_task("squiggles", 2000)
        kmnist_test = create_synthetic_task("squiggles", 500)
        
        print("Generated synthetic datasets successfully.")
        return (mnist_train, mnist_test), (fmnist_train, fmnist_test), (kmnist_train, kmnist_test), True

def train_epoch(encoder, head, loader, optimizer, criterion):
    encoder.train()
    head.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        features = encoder(images)
        outputs = head(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / total, correct / total

def eval_model(encoder, head, loader):
    encoder.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            features = encoder(images)
            outputs = head(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    
    (m_tr, m_te), (f_tr, f_te), (k_tr, k_te), is_synthetic = get_datasets()
    
    # 1. Pre-train Shared Base Encoder theta_0
    print("--- Phase 1: Pre-training Base Shared Encoder ---")
    # Pool training sets
    pooled_dataset = ConcatDataset([m_tr, f_tr, k_tr])
    pooled_loader = DataLoader(pooled_dataset, batch_size=64, shuffle=True)
    
    base_encoder = Encoder().to(device)
    base_head = Head(num_classes=10).to(device)
    
    optimizer = optim.Adam(list(base_encoder.parameters()) + list(base_head.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 3 if is_synthetic else 5
    for epoch in range(epochs):
        loss, acc = train_epoch(base_encoder, base_head, pooled_loader, optimizer, criterion)
        print(f"Base Pre-training Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.4f}")
        
    # Save the base encoder theta_0
    torch.save(base_encoder.state_dict(), "./models/base_encoder.pt")
    print("Saved base encoder to models/base_encoder.pt")
    
    # 2. Fine-tune Task Experts
    tasks = [
        ("MNIST", m_tr, m_te),
        ("FashionMNIST", f_tr, f_te),
        ("KMNIST", k_tr, k_te)
    ]
    
    for name, tr_set, te_set in tasks:
        print(f"\n--- Phase 2: Fine-tuning Expert for {name} ---")
        tr_loader = DataLoader(tr_set, batch_size=64, shuffle=True)
        te_loader = DataLoader(te_set, batch_size=128, shuffle=False)
        
        # Initialize from base pre-trained encoder
        expert_encoder = Encoder().to(device)
        expert_encoder.load_state_dict(torch.load("./models/base_encoder.pt"))
        expert_head = Head(num_classes=10).to(device)
        
        optimizer = optim.Adam(list(expert_encoder.parameters()) + list(expert_head.parameters()), lr=0.001)
        
        epochs = 3 if is_synthetic else 5
        for epoch in range(epochs):
            loss, acc = train_epoch(expert_encoder, expert_head, tr_loader, optimizer, criterion)
            val_acc = eval_model(expert_encoder, expert_head, te_loader)
            print(f"Expert {name} Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Train Acc: {acc:.4f} | Val Acc: {val_acc:.4f}")
            
        # Save expert encoder and head
        torch.save(expert_encoder.state_dict(), f"./models/expert_encoder_{name}.pt")
        torch.save(expert_head.state_dict(), f"./models/expert_head_{name}.pt")
        print(f"Saved expert weights for {name}")
