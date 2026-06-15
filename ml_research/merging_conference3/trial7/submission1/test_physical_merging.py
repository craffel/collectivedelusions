import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Define dataset loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Replicate grayscale to 3 channels for MNIST and FashionMNIST
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

# Define the 4-layer CNN expert model
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

# Train experts
experts = {}
K = 4
for k in range(K):
    print(f"Training Expert {k}...")
    # Get a subset of the dataset
    subset_indices = list(range(100)) # 100 samples to train fast on CPU
    subset = torch.utils.data.Subset(datasets[k], subset_indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=16, shuffle=True)
    
    model = TinyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(10): # 10 epochs
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    experts[k] = model
    print(f"Expert {k} trained successfully!")

# Let's test dynamic routing
class DynamicRouter(nn.Module):
    def __init__(self, L, d, K, routing_type="layer-wise"):
        super().__init__()
        self.L = L
        self.d = d
        self.K = K
        self.routing_type = routing_type
        
        if routing_type == "layer-wise":
            self.W = nn.Parameter(torch.randn(L, d, K) * 0.01)
            self.B = nn.Parameter(torch.zeros(L, K))
        elif routing_type == "global":
            self.W = nn.Parameter(torch.randn(1, d, K) * 0.01)
            self.B = nn.Parameter(torch.zeros(1, K))
            
    def forward(self, psi):
        # psi shape: (B, d)
        B_size = psi.shape[0]
        if self.routing_type == "layer-wise":
            logits = torch.einsum("bd,ldk->lbk", psi, self.W) + self.B.unsqueeze(1)
        elif self.routing_type == "global":
            logits = torch.einsum("bd,gdk->gbk", psi, self.W) + self.B.unsqueeze(1)
            logits = logits.repeat(self.L, 1, 1)
            
        alpha = 0.3 * torch.sigmoid(logits)
        return alpha

# Frozen random projection matrix
P = torch.randn(4, 3072) # project flattened 3*32*32 to d=4

# We define functional forward of merged weights
def forward_merged(x, alpha, experts_dict):
    # alpha has shape (L, B, K)
    # Since we can process sample by sample for on-the-fly merging:
    outputs = []
    for idx in range(x.size(0)):
        img = x[idx:idx+1]
        a = alpha[:, idx, :] # shape (L, K)
        
        # Merge Layer 1
        w1 = sum(a[0, j] * experts_dict[j].conv1.weight for j in range(K))
        b1 = sum(a[0, j] * experts_dict[j].conv1.bias for j in range(K))
        h = F.conv2d(img, w1, b1, padding=1)
        h = F.relu(h)
        h = F.max_pool2d(h, 2)
        
        # Merge Layer 2
        w2 = sum(a[1, j] * experts_dict[j].conv2.weight for j in range(K))
        b2 = sum(a[1, j] * experts_dict[j].conv2.bias for j in range(K))
        h = F.conv2d(h, w2, b2, padding=1)
        h = F.relu(h)
        h = F.max_pool2d(h, 2)
        
        # Merge Layer 3
        w3 = sum(a[2, j] * experts_dict[j].conv3.weight for j in range(K))
        b3 = sum(a[2, j] * experts_dict[j].conv3.bias for j in range(K))
        h = F.conv2d(h, w3, b3, padding=1)
        h = F.relu(h)
        h = F.max_pool2d(h, 2)
        
        # Merge Layer 4
        h = h.view(h.size(0), -1)
        w4 = sum(a[3, j] * experts_dict[j].fc.weight for j in range(K))
        b4 = sum(a[3, j] * experts_dict[j].fc.bias for j in range(K))
        out = F.linear(h, w4, b4)
        outputs.append(out)
        
    return torch.cat(outputs, dim=0)

# Calibrate Layer-wise Router
print("Calibrating Layer-wise Router...")
router = DynamicRouter(L=4, d=4, K=4, routing_type="layer-wise")
optimizer = torch.optim.Adam(router.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Gather calibration dataset (16 samples per task, total 64)
cal_x = []
cal_y = []
for k in range(K):
    indices = list(range(100, 116))
    for idx in indices:
        x, y = datasets[k][idx]
        cal_x.append(x.unsqueeze(0))
        cal_y.append(torch.tensor([y]))
        
cal_x = torch.cat(cal_x, dim=0)
cal_y = torch.cat(cal_y, dim=0)

# Calibration training loop
for step in range(50):
    optimizer.zero_grad()
    
    # Project inputs to d=4
    flat_x = cal_x.view(cal_x.size(0), -1)
    psi = torch.mm(flat_x, P.t()) # shape (64, 4)
    
    # Get dynamic routing weights
    alpha = router(psi) # shape (L, B, K)
    
    # Run forward merged model
    logits = forward_merged(cal_x, alpha, experts)
    
    # We want each sample to be predicted correctly under task classification
    loss = criterion(logits, cal_y)
    
    # Add L2 decay manually
    loss_reg = sum(0.5 * 1e-4 * torch.sum(p**2) for p in router.parameters() if p.requires_grad)
    total_loss = loss + loss_reg
    
    total_loss.backward()
    optimizer.step()
    
    if (step+1) % 10 == 0:
        print(f"Step {step+1}/50 | Loss: {loss.item():.4f}")

print("Calibration complete!")
