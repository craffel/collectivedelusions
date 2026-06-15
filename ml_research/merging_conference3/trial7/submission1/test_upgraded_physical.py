import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")

# Dataset preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading datasets...")
datasets = {
    0: torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_gray),
    1: torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray),
    2: torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
    3: torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
}

test_datasets = {
    0: torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray),
    1: torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray),
    2: torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
    3: torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
}

# Define TinyCNN expert
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 4 * 4, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Train experts properly on 500 samples
K = 4
task_ids = [0, 1, 2, 3]
experts = {}

for k_idx, task_id in enumerate(task_ids):
    print(f"Training Expert {k_idx} on Task {task_id} (500 samples)...")
    subset_indices = list(range(500))
    subset = torch.utils.data.Subset(datasets[task_id], subset_indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)
    
    model = TinyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(15):
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    experts[k_idx] = model
    
    # Test accuracy of individual expert on its own test set
    test_subset = torch.utils.data.Subset(test_datasets[task_id], list(range(200)))
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=200, shuffle=False)
    for x_t, y_t in test_loader:
        with torch.no_grad():
            out_t = model(x_t)
            acc_t = (out_t.argmax(dim=-1) == y_t).float().mean().item() * 100.0
            print(f"  Expert {k_idx} Single-Task Test Accuracy: {acc_t:.2f}%")
