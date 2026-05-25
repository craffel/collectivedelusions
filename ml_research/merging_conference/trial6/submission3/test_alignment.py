import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.func as func
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_base_model():
    model = resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 10)
    return model

# Load models
base_model = get_base_model().to(device)
base_model.load_state_dict(torch.load("./checkpoints/base_model.pth", map_location=device))

expert_mnist = get_base_model().to(device)
expert_mnist.load_state_dict(torch.load("./checkpoints/expert_mnist.pth", map_location=device))

expert_kmnist = get_base_model().to(device)
expert_kmnist.load_state_dict(torch.load("./checkpoints/expert_kmnist.pth", map_location=device))

class ResNet18FeatureWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.fc(feats)
        return feats, logits

wrapped_base = ResNet18FeatureWrapper(base_model).to(device)
wrapped_base.eval()

# Task vectors
base_params = {name: param.clone().detach() for name, param in base_model.named_parameters()}
base_buffers = {name: buf.clone().detach() for name, buf in base_model.named_buffers()}

task_vectors = {}
for name, param in base_model.named_parameters():
    v1 = expert_mnist.state_dict()[name] - base_params[name]
    v2 = expert_kmnist.state_dict()[name] - base_params[name]
    task_vectors[name] = [v1.detach(), v2.detach()]

task_vector_buffers = {}
for name, buf in base_model.named_buffers():
    v1 = expert_mnist.state_dict()[name] - base_buffers[name]
    v2 = expert_kmnist.state_dict()[name] - base_buffers[name]
    task_vector_buffers[name] = [v1.detach(), v2.detach()]

# Static merged parameters
static_params = {}
for name, param in base_model.named_parameters():
    static_params[name] = base_params[name] + 0.5 * task_vectors[name][0] + 0.5 * task_vectors[name][1]
static_buffers = {}
for name, buf in base_model.named_buffers():
    static_buffers[name] = base_buffers[name] + 0.5 * task_vector_buffers[name][0] + 0.5 * task_vector_buffers[name][1]
static_merged_all = {**static_params, **static_buffers}

def merged_forward(merged_params, inputs):
    return func.functional_call(wrapped_base, merged_params, inputs)

# Load data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
kmnist_dataset = torchvision.datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
fmnist_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# Compute prototypes & means USING STATIC MERGED MODEL
def compute_protos_static(dataset):
    loader = DataLoader(Subset(dataset, list(range(500))), batch_size=64, shuffle=False)
    feats_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            feats, _ = merged_forward(static_merged_all, inputs)
            feats_list.append(feats.cpu())
            labels_list.append(labels)
    feats = torch.cat(feats_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    mean = feats.mean(dim=0).to(device)
    protos = {}
    for c in range(10):
        protos[c] = feats[labels == c].mean(dim=0).to(device)
    return mean, protos

mean_mnist, protos_mnist = compute_protos_static(mnist_dataset)
mean_kmnist, protos_kmnist = compute_protos_static(kmnist_dataset)

# Center prototypes
for c in range(10):
    protos_mnist[c] = protos_mnist[c] - mean_mnist
    protos_kmnist[c] = protos_kmnist[c] - mean_kmnist

# Test cohesion on a batch using static merged model
mnist_batch = next(iter(DataLoader(Subset(mnist_dataset, list(range(64))), batch_size=64)))[0].to(device)
kmnist_batch = next(iter(DataLoader(Subset(kmnist_dataset, list(range(64))), batch_size=64)))[0].to(device)
fmnist_batch = next(iter(DataLoader(Subset(fmnist_dataset, list(range(64))), batch_size=64)))[0].to(device)

def get_cohesion(batch, mean_d, protos_d):
    with torch.no_grad():
        feats, _ = merged_forward(static_merged_all, batch)
        feats_centered = feats - mean_d
        
    cohesion_list = []
    for feat in feats_centered:
        sims = [F.cosine_similarity(feat, protos_d[c], dim=0).item() for c in range(10)]
        cohesion_list.append(max(sims))
    return np.mean(cohesion_list)

print("Unified Static Space Cohesion:")
print("  MNIST batch vs MNIST protos: ", get_cohesion(mnist_batch, mean_mnist, protos_mnist))
print("  MNIST batch vs KMNIST protos:", get_cohesion(mnist_batch, mean_kmnist, protos_kmnist))
print("  KMNIST batch vs KMNIST protos:", get_cohesion(kmnist_batch, mean_kmnist, protos_kmnist))
print("  KMNIST batch vs MNIST protos: ", get_cohesion(kmnist_batch, mean_mnist, protos_mnist))
print("  FMNIST batch vs MNIST protos: ", get_cohesion(fmnist_batch, mean_mnist, protos_mnist))
print("  FMNIST batch vs KMNIST protos:", get_cohesion(fmnist_batch, mean_kmnist, protos_kmnist))
