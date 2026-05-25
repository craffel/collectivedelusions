import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# Set reproducibility seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ResNet18Custom(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.base_model.fc(feat)
        return feat, logits

def get_resnet18_1channel():
    model = resnet18()
    conv1_new = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = conv1_new
    model.fc = nn.Linear(512, 10)
    return model

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Load datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    mnist_loader = DataLoader(Subset(mnist_test, range(200)), batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(Subset(kmnist_test, range(200)), batch_size=64, shuffle=False)
    fmnist_loader = DataLoader(Subset(fmnist_test, range(200)), batch_size=64, shuffle=False)

    loaders = {'mnist': mnist_loader, 'kmnist': kmnist_loader, 'fashionmnist': fmnist_loader}

    # Load experts
    expert_paths = {
        'mnist': 'mnist_expert.pt',
        'kmnist': 'kmnist_expert.pt',
        'fashionmnist': 'fashionmnist_expert.pt'
    }
    experts = {}
    for name, path in expert_paths.items():
        model = get_resnet18_1channel()
        model.load_state_dict(torch.load(path, map_location=device))
        model = ResNet18Custom(model).to(device)
        model.eval()
        experts[name] = model

    # Evaluate each expert on each dataset
    for data_name, loader in loaders.items():
        print(f"\nEvaluating dataset: {data_name.upper()}")
        for exp_name, expert in experts.items():
            correct = 0
            total = 0
            all_entropies = []
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    _, logits = expert(x)
                    probs = torch.softmax(logits, dim=-1)
                    ent = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
                    all_entropies.append(ent.cpu().numpy())
                    
                    _, preds = logits.max(1)
                    correct += preds.eq(y).sum().item()
                    total += y.size(0)
            avg_acc = (correct / total) * 100
            avg_ent = np.mean(np.concatenate(all_entropies))
            print(f"  Expert: {exp_name:<15} | Acc: {avg_acc:>6.2f}% | Avg Entropy: {avg_ent:>6.4f}")

if __name__ == '__main__':
    main()
