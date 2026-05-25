import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

random.seed(2026)
np.random.seed(2026)
torch.manual_seed(2026)
torch.backends.cudnn.enabled = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_grayscale_resnet18(num_classes=10):
    resnet = models.resnet18(weights=None)
    old_conv = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).conv1
    new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None)
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
    resnet.conv1 = new_conv
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet

# Load Experts
mnist_expert = get_grayscale_resnet18()
kmnist_expert = get_grayscale_resnet18()
fashion_expert = get_grayscale_resnet18()

mnist_expert.load_state_dict(torch.load('models/mnist_expert.pt', map_location=device))
kmnist_expert.load_state_dict(torch.load('models/kmnist_expert.pt', map_location=device))
fashion_expert.load_state_dict(torch.load('models/fashion_expert.pt', map_location=device))

base_model = get_grayscale_resnet18()
base_model.to(device)

experts = [mnist_expert, kmnist_expert, fashion_expert]
for exp in experts:
    exp.to(device)
    exp.eval()

task_vectors = []
for k in range(3):
    tv = {}
    expert_state = experts[k].state_dict()
    base_state = base_model.state_dict()
    for name in base_state.keys():
        if base_state[name].dtype.is_floating_point:
            tv[name] = expert_state[name] - base_state[name]
        else:
            tv[name] = expert_state[name].clone()
    task_vectors.append(tv)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
train_kmnist = torchvision.datasets.KMNIST(root='./data', train=True, transform=transform, download=False)
train_fashion = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=False)

test_mnist = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)
test_kmnist = torchvision.datasets.KMNIST(root='./data', train=False, transform=transform, download=False)
test_fashion = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=False)

def get_feature_extractor(model):
    class FeatureExtractor(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.backbone = nn.Sequential(*list(original_model.children())[:-1])
        def forward(self, x):
            feat = self.backbone(x)
            return feat.view(feat.size(0), -1)
    return FeatureExtractor(model)

static_model = get_grayscale_resnet18()
static_model_state = static_model.state_dict()
base_state = base_model.state_dict()
with torch.no_grad():
    for name in static_model_state.keys():
        if static_model_state[name].dtype.is_floating_point:
            static_model_state[name].copy_(base_state[name] + (task_vectors[0][name] + task_vectors[1][name] + task_vectors[2][name]) / 3.0)
static_model.load_state_dict(static_model_state)
static_model.to(device)
static_model.eval()

static_feat_extractor = get_feature_extractor(static_model)
static_feat_extractor.eval()

cal_size = 200
datasets = [train_mnist, train_kmnist, train_fashion]
mu_k = []
pi_kc = []

for k in range(3):
    cal_subset, _ = torch.utils.data.random_split(datasets[k], [cal_size, len(datasets[k]) - cal_size])
    loader = torch.utils.data.DataLoader(cal_subset, batch_size=32, shuffle=False)
    feats = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f_x = static_feat_extractor(x)
            feats.append(f_x)
            labels.append(y)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    mu = feats.mean(dim=0)
    mu_k.append(mu)
    centered_feats = feats - mu
    class_protos = {}
    for c in range(10):
        mask = (labels == c)
        if mask.sum() > 0:
            class_protos[c] = centered_feats[mask].mean(dim=0)
        else:
            class_protos[c] = torch.zeros(512, device=device)
    pi_kc.append(class_protos)

mu_static = sum(mu_k) / 3.0

mnist_loader = torch.utils.data.DataLoader(test_mnist, batch_size=64, shuffle=False)
kmnist_loader = torch.utils.data.DataLoader(test_kmnist, batch_size=64, shuffle=False)
fashion_loader = torch.utils.data.DataLoader(test_fashion, batch_size=64, shuffle=False)

mnist_batches = list(mnist_loader)[:5]
kmnist_batches = list(kmnist_loader)[:5]
fashion_batches = list(fashion_loader)[:5]

for corruption in ["clean", "gaussian", "contrast"]:
    print(f"\n--- Diagnostic: Corruption = {corruption.upper()} ---")
    for domain_name, batches in [("MNIST (Known)", mnist_batches), ("KMNIST (Known)", kmnist_batches), ("FashionMNIST (Novel)", fashion_batches)]:
        cohesions = []
        for x, _ in batches:
            if corruption == "gaussian":
                noise = torch.randn_like(x) * 0.2
                x = torch.clamp(x + noise, -1.0, 1.0)
            elif corruption == "contrast":
                x = torch.clamp(x * 0.3, -1.0, 1.0)
            
            x = x.to(device)
            with torch.no_grad():
                z_anchor = static_feat_extractor(x) - mu_static
            
            cohesion_k = []
            for k in range(2):
                max_sims = []
                for i in range(len(x)):
                    sims = []
                    for c in range(10):
                        proto = pi_kc[k][c]
                        sim = torch.dot(z_anchor[i], proto) / (torch.norm(z_anchor[i]) * torch.norm(proto) + 1e-8)
                        sims.append(sim.item())
                    max_sims.append(max(sims))
                cohesion_k.append(np.mean(max_sims))
            cohesions.append(cohesion_k)
        
        cohesions = np.array(cohesions)
        print(f"Domain: {domain_name}")
        print(f"  Cohesion MNIST: {cohesions[:, 0].mean():.4f} | KMNIST: {cohesions[:, 1].mean():.4f} | Max Cohesion: {cohesions.max(axis=1).mean():.4f}")
