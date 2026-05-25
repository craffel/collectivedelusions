import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.func import functional_call
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

class ResNetBackbone(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

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
        x = torch.flatten(x, 1)
        return x

def get_dataset(name):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if name == 'mnist':
        return torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    elif name == 'fashion':
        return torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    else:
        return torchvision.datasets.KMNIST(root='./data', train=False, download=False, transform=transform)

# Load base model
base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
base_backbone = ResNetBackbone(base_model).to(device)
base_backbone_params = {k: v.clone().detach() for k, v in base_backbone.named_parameters()}

expert_heads = []
task_vectors = []

for i in range(3):
    ckpt = torch.load(f'checkpoints/expert_{i}.pt', map_location=device)
    eb = ResNetBackbone(base_model).to(device)
    eb.load_state_dict(ckpt['backbone_state_dict'])
    eh = nn.Linear(512, 10).to(device)
    eh.load_state_dict(ckpt['head_state_dict'])
    expert_heads.append(eh)
    
    eb_params = {k: v.clone().detach() for k, v in eb.named_parameters()}
    tv = {k: eb_params[k] - base_backbone_params[k] for k in base_backbone_params.keys()}
    task_vectors.append(tv)

# Datasets and loaders
mnist_test = get_dataset('mnist')
fashion_test = get_dataset('fashion')
kmnist_test = get_dataset('kmnist')

mnist_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)
fashion_loader = DataLoader(fashion_test, batch_size=32, shuffle=False)
kmnist_loader = DataLoader(kmnist_test, batch_size=32, shuffle=False)

mnist_batches = []
for i, batch in enumerate(mnist_loader):
    if i >= 50: break
    mnist_batches.append((batch, 0))
    
fashion_batches = []
for i, batch in enumerate(fashion_loader):
    if i >= 50: break
    fashion_batches.append((batch, 1))
    
kmnist_batches = []
for i, batch in enumerate(kmnist_loader):
    if i >= 50: break
    kmnist_batches.append((batch, 2))

alternating_batches = []
for i in range(50):
    alternating_batches.append(mnist_batches[i])
    alternating_batches.append(fashion_batches[i])
    alternating_batches.append(kmnist_batches[i])

correct_detections = 0
total_detections = 0

print("Evaluating task detection via prediction entropy on Alternating stream...")
for step, ((inputs, targets), true_task_idx) in enumerate(alternating_batches):
    inputs = inputs.to(device)
    
    # Measure prediction entropy for each anchor setting
    entropies = []
    for k in range(3):
        # Temp merged params for anchor k (100% task k vector)
        temp_lambda = [0.0, 0.0, 0.0]
        temp_lambda[k] = 1.0
        
        merged_params = {}
        for param_key in base_backbone_params.keys():
            merged_params[param_key] = base_backbone_params[param_key] + sum(
                temp_lambda[i] * task_vectors[i][param_key] for i in range(3)
            )
            
        with torch.no_grad():
            features = functional_call(base_backbone, merged_params, inputs)
            logits = expert_heads[k](features)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().item()
            entropies.append(entropy)
            
    detected_task_idx = entropies.index(min(entropies))
    if detected_task_idx == true_task_idx:
        correct_detections += 1
    total_detections += 1
    
    if step < 15:
        print(f"Step {step:02d} | True Task: {true_task_idx} | Entropies: {[f'{e:.4f}' for e in entropies]} | Detected: {detected_task_idx}")

print(f"\nTask Detection Accuracy: {100.0 * correct_detections / total_detections:.2f}% ({correct_detections}/{total_detections})")
