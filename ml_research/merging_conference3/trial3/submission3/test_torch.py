import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import copy

# Set random seed
torch.manual_seed(42)

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 32)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class MultiTaskCNN(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head1 = nn.Linear(32, 10) # MNIST Head
        self.head2 = nn.Linear(32, 10) # FashionMNIST Head
        self.head3 = nn.Linear(32, 10) # KMNIST Head
        
    def forward(self, x, task_id):
        features = self.backbone(x)
        if isinstance(task_id, int):
            if task_id == 0:
                return self.head1(features)
            elif task_id == 1:
                return self.head2(features)
            else:
                return self.head3(features)
        else:
            out1 = self.head1(features)
            out2 = self.head2(features)
            out3 = self.head3(features)
            
            out = torch.zeros_like(out1)
            mask0 = (task_id == 0).unsqueeze(-1)
            mask1 = (task_id == 1).unsqueeze(-1)
            mask2 = (task_id == 2).unsqueeze(-1)
            
            out = out + mask0 * out1 + mask1 * out2 + mask2 * out3
            return out

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 1. Load data
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
fashion_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
kmnist_train = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)

mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
fashion_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

# 2. Pre-training: train on a mixture of all three tasks
print("Pre-training base model on joint mixture...")
base_backbone = CNNBackbone()
base_model = MultiTaskCNN(base_backbone)
base_model.train()

# Combine subsets for pre-training
sub_m = Subset(mnist_train, list(range(2000)))
sub_f = Subset(fashion_train, list(range(2000)))
sub_k = Subset(kmnist_train, list(range(2000)))

# We will create a custom dataset that yields (img, label, task_id)
class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, ds_list):
        self.ds_list = ds_list
        self.lens = [len(ds) for ds in ds_list]
        self.total_len = sum(self.lens)
        
    def __len__(self):
        return self.total_len
        
    def __getitem__(self, idx):
        if idx < self.lens[0]:
            img, label = self.ds_list[0][idx]
            return img, label, 0
        elif idx < self.lens[0] + self.lens[1]:
            img, label = self.ds_list[1][idx - self.lens[0]]
            return img, label, 1
        else:
            img, label = self.ds_list[2][idx - self.lens[0] - self.lens[1]]
            return img, label, 2

pretrain_ds = MultiTaskDataset([sub_m, sub_f, sub_k])
pretrain_loader = DataLoader(pretrain_ds, batch_size=128, shuffle=True)

optimizer = optim.Adam(base_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(2):
    for x, y, tid in pretrain_loader:
        optimizer.zero_grad()
        out = base_model(x, tid)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    print(f"Pretrain Epoch {epoch+1} finished.")

# Save pre-trained base model state
base_backbone_state = copy.deepcopy(base_backbone.state_dict())

# 3. Fine-tune experts from the pre-trained backbone
print("Fine-tuning Expert 1 on MNIST...")
model_mnist = MultiTaskCNN(copy.deepcopy(base_backbone))
model_mnist.load_state_dict(base_model.state_dict())
model_mnist.train()
optimizer = optim.Adam(model_mnist.parameters(), lr=2e-4) # lower LR to stay in basin
loader_m = DataLoader(Subset(mnist_train, list(range(3000))), batch_size=128, shuffle=True)
for x, y in loader_m:
    optimizer.zero_grad()
    loss = criterion(model_mnist(x, 0), y)
    loss.backward()
    optimizer.step()

print("Fine-tuning Expert 2 on FashionMNIST...")
model_fashion = MultiTaskCNN(copy.deepcopy(base_backbone))
model_fashion.load_state_dict(base_model.state_dict())
model_fashion.train()
optimizer = optim.Adam(model_fashion.parameters(), lr=2e-4)
loader_f = DataLoader(Subset(fashion_train, list(range(3000))), batch_size=128, shuffle=True)
for x, y in loader_f:
    optimizer.zero_grad()
    loss = criterion(model_fashion(x, 1), y)
    loss.backward()
    optimizer.step()

print("Fine-tuning Expert 3 on KMNIST...")
model_kmnist = MultiTaskCNN(copy.deepcopy(base_backbone))
model_kmnist.load_state_dict(base_model.state_dict())
model_kmnist.train()
optimizer = optim.Adam(model_kmnist.parameters(), lr=2e-4)
loader_k = DataLoader(Subset(kmnist_train, list(range(3000))), batch_size=128, shuffle=True)
for x, y in loader_k:
    optimizer.zero_grad()
    loss = criterion(model_kmnist(x, 2), y)
    loss.backward()
    optimizer.step()

# 4. Evaluate individual experts on test sets
model_mnist.eval()
model_fashion.eval()
model_kmnist.eval()

correct_m, total_m = 0, 0
for x, y in DataLoader(Subset(mnist_test, list(range(500))), batch_size=128):
    with torch.no_grad():
        preds = torch.argmax(model_mnist(x, 0), dim=-1)
        correct_m += (preds == y).sum().item()
        total_m += len(y)
print(f"Expert 1 MNIST Test Acc: {100.0 * correct_m / total_m:.2f}%")

correct_f, total_f = 0, 0
for x, y in DataLoader(Subset(fashion_test, list(range(500))), batch_size=128):
    with torch.no_grad():
        preds = torch.argmax(model_fashion(x, 1), dim=-1)
        correct_f += (preds == y).sum().item()
        total_f += len(y)
print(f"Expert 2 FashionMNIST Test Acc: {100.0 * correct_f / total_f:.2f}%")

correct_k, total_k = 0, 0
for x, y in DataLoader(Subset(kmnist_test, list(range(500))), batch_size=128):
    with torch.no_grad():
        preds = torch.argmax(model_kmnist(x, 2), dim=-1)
        correct_k += (preds == y).sum().item()
        total_k += len(y)
print(f"Expert 3 KMNIST Test Acc: {100.0 * correct_k / total_k:.2f}%")

# 5. Extract task vectors
task_vector_m = {k: model_mnist.backbone.state_dict()[k] - base_backbone_state[k] for k in base_backbone_state}
task_vector_f = {k: model_fashion.backbone.state_dict()[k] - base_backbone_state[k] for k in base_backbone_state}
task_vector_k = {k: model_kmnist.backbone.state_dict()[k] - base_backbone_state[k] for k in base_backbone_state}

layer_groups = [
    ["conv1.weight", "conv1.bias"],
    ["conv2.weight", "conv2.bias"],
    ["conv3.weight", "conv3.bias"],
    ["fc1.weight", "fc1.bias"],
    ["fc2.weight", "fc2.bias"]
]

# Differentiable weight reconstruction and forward pass
def functional_forward(x, lambdas):
    reconstructed_state = {}
    for g_idx, keys in enumerate(layer_groups):
        for k in keys:
            val = base_backbone_state[k].clone()
            val += lambdas[0, g_idx] * task_vector_m[k]
            val += lambdas[1, g_idx] * task_vector_f[k]
            val += lambdas[2, g_idx] * task_vector_k[k]
            reconstructed_state[k] = val
            
    out = F.conv2d(x, reconstructed_state["conv1.weight"], reconstructed_state["conv1.bias"], padding=1)
    out = F.relu(out)
    out = F.max_pool2d(out, 2, 2)
    
    out = F.conv2d(out, reconstructed_state["conv2.weight"], reconstructed_state["conv2.bias"], padding=1)
    out = F.relu(out)
    out = F.max_pool2d(out, 2, 2)
    
    out = F.conv2d(out, reconstructed_state["conv3.weight"], reconstructed_state["conv3.bias"], padding=1)
    out = F.relu(out)
    out = F.max_pool2d(out, 2, 2)
    
    out = out.view(-1, 64 * 3 * 3)
    
    out = F.linear(out, reconstructed_state["fc1.weight"], reconstructed_state["fc1.bias"])
    out = F.relu(out)
    
    out = F.linear(out, reconstructed_state["fc2.weight"], reconstructed_state["fc2.bias"])
    out = F.relu(out)
    
    return out

# Task Arithmetic: Blend with lambda = 0.3 for all tasks
ta_lambdas = torch.full((3, 5), 0.3)

correct_ta_m, correct_ta_f, correct_ta_k = 0, 0, 0
for x, y in DataLoader(Subset(mnist_test, list(range(500))), batch_size=128):
    with torch.no_grad():
        features = functional_forward(x, ta_lambdas)
        preds = torch.argmax(model_mnist.head1(features), dim=-1)
        correct_ta_m += (preds == y).sum().item()
        
for x, y in DataLoader(Subset(fashion_test, list(range(500))), batch_size=128):
    with torch.no_grad():
        features = functional_forward(x, ta_lambdas)
        preds = torch.argmax(model_fashion.head2(features), dim=-1)
        correct_ta_f += (preds == y).sum().item()
        
for x, y in DataLoader(Subset(kmnist_test, list(range(500))), batch_size=128):
    with torch.no_grad():
        features = functional_forward(x, ta_lambdas)
        preds = torch.argmax(model_kmnist.head3(features), dim=-1)
        correct_ta_k += (preds == y).sum().item()

print(f"Task Arithmetic MNIST Acc: {100.0 * correct_ta_m / 500:.2f}%")
print(f"Task Arithmetic Fashion Acc: {100.0 * correct_ta_f / 500:.2f}%")
print(f"Task Arithmetic KMNIST Acc: {100.0 * correct_ta_k / 500:.2f}%")
print(f"Task Arithmetic Joint Acc: {100.0 * (correct_ta_m + correct_ta_f + correct_ta_k) / 1500:.2f}%")
