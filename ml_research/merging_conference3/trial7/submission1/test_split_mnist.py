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

# Dataset loading (28x28 grayscale)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Loading MNIST...")
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Helper to filter subsets
def get_task_subset(dataset, classes, size):
    targets = dataset.targets
    indices = torch.zeros_like(targets, dtype=torch.bool)
    for c in classes:
        indices |= (targets == c)
    subset_indices = torch.where(indices)[0]
    # Keep only the requested size
    subset_indices = subset_indices[:size]
    return torch.utils.data.Subset(dataset, subset_indices)

# Define TinyCNN expert
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 3 * 3, 10)
        
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

# Tasks: 0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]
tasks = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6, 7]
}
K = 4
experts = {}

# Shared initialization
base_model = TinyCNN().to(device)

for k in range(K):
    classes = tasks[k]
    print(f"Training Expert {k} on classes {classes}...")
    subset = get_task_subset(mnist_train, classes, 512)
    loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)
    
    model = TinyCNN().to(device)
    model.load_state_dict(base_model.state_dict())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(12):
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    experts[k] = model
    
    # Test expert on its own classes
    test_subset = get_task_subset(mnist_test, classes, 200)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=200, shuffle=False)
    for x_t, y_t in test_loader:
        with torch.no_grad():
            out_t = model(x_t)
            acc_t = (out_t.argmax(dim=-1) == y_t).float().mean().item() * 100.0
            print(f"  Expert {k} Single-Task Accuracy: {acc_t:.2f}%")
