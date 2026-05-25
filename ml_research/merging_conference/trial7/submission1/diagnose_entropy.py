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

experts = [mnist_expert, kmnist_expert, fashion_expert]
for exp in experts:
    exp.to(device)
    exp.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_mnist = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)
test_kmnist = torchvision.datasets.KMNIST(root='./data', train=False, transform=transform, download=False)
test_fashion = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=False)

mnist_loader = torch.utils.data.DataLoader(test_mnist, batch_size=64, shuffle=False)
kmnist_loader = torch.utils.data.DataLoader(test_kmnist, batch_size=64, shuffle=False)
fashion_loader = torch.utils.data.DataLoader(test_fashion, batch_size=64, shuffle=False)

mnist_batches = list(mnist_loader)[:5]
kmnist_batches = list(kmnist_loader)[:5]
fashion_batches = list(fashion_loader)[:5]

for corruption in ["clean", "gaussian", "contrast"]:
    print(f"\n--- Entropy Diagnostic: Corruption = {corruption.upper()} ---")
    for domain_name, batches in [("MNIST (Known)", mnist_batches), ("KMNIST (Known)", kmnist_batches), ("FashionMNIST (Novel)", fashion_batches)]:
        min_entropies = []
        for x, _ in batches:
            if corruption == "gaussian":
                noise = torch.randn_like(x) * 0.2
                x = torch.clamp(x + noise, -1.0, 1.0)
            elif corruption == "contrast":
                x = torch.clamp(x * 0.3, -1.0, 1.0)
            
            x = x.to(device)
            ents = []
            for k in range(2): # Known experts: MNIST (0) and KMNIST (1)
                with torch.no_grad():
                    logits = experts[k](x)
                    probs = F.softmax(logits, dim=-1)
                    ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
                    ents.append(ent)
            min_entropies.append(min(ents))
        print(f"Domain: {domain_name} | Min Entropy among known experts: {np.mean(min_entropies):.4f}")
