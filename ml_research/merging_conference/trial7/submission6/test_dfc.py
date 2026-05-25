import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_train = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

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

    K = len(experts)
    expert_names = ['mnist', 'kmnist', 'fashionmnist']

    # Precompute static model
    static_model = get_resnet18_1channel()
    static_state_dict = {}
    for name in static_model.state_dict().keys():
        w_sum = sum(experts[e_name].base_model.state_dict()[name].float() for e_name in expert_names)
        static_state_dict[name] = w_sum / K
    static_model.load_state_dict(static_state_dict)
    static_model = ResNet18Custom(static_model).to(device)
    static_model.eval()

    # Precompute mean features (clean)
    calib_size = 500
    calib_subsets = {
        'mnist': Subset(mnist_train, list(range(calib_size))),
        'kmnist': Subset(kmnist_train, list(range(calib_size))),
        'fashionmnist': Subset(fmnist_train, list(range(calib_size)))
    }
    mu_static_domain_clean = {}
    for e_name in expert_names:
        loader = DataLoader(calib_subsets[e_name], batch_size=100, shuffle=False)
        feats_list = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                feat, _ = static_model(x)
                feats_list.append(feat)
        mu_static_domain_clean[e_name] = torch.cat(feats_list, dim=0).mean(dim=0)

    # Precompute prototypes with offline clean mean
    prototypes_clean = {0: {}, 1: {}}
    known_datasets = ['mnist', 'kmnist']
    for k, e_name in enumerate(known_datasets):
        loader = DataLoader(calib_subsets[e_name], batch_size=100, shuffle=False)
        feats_all = []
        labels_all = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                feat, _ = static_model(x)
                feats_all.append(feat)
                labels_all.append(y.to(device))
        feats_all = torch.cat(feats_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        z_all = feats_all - mu_static_domain_clean[e_name]
        for c in range(10):
            mask = (labels_all == c)
            if mask.sum() > 0:
                pi_c = z_all[mask].mean(dim=0)
                pi_c = pi_c / (pi_c.norm(p=2) + 1e-12)
                prototypes_clean[k][c] = pi_c
            else:
                prototypes_clean[k][c] = torch.zeros_like(mu_static_domain_clean[e_name])

    # Let's inspect cohesion scores with DFC
    mnist_test_loader = DataLoader(Subset(mnist_test, list(range(128))), batch_size=64, shuffle=False)
    kmnist_test_loader = DataLoader(Subset(kmnist_test, list(range(128))), batch_size=64, shuffle=False)
    fmnist_test_loader = DataLoader(Subset(fmnist_test, list(range(128))), batch_size=64, shuffle=False)

    def compute_cohesion_dfc(batch_X, use_dfc=True):
        batch_X = batch_X.to(device)
        with torch.no_grad():
            anchor_feats, _ = static_model(batch_X)
        
        cohesions = []
        for k in range(2):
            e_name = known_datasets[k]
            if use_dfc:
                # DFC: center with current batch mean
                batch_mean = anchor_feats.mean(dim=0)
                z_anchor_k = anchor_feats - batch_mean
            else:
                # Standard: center with offline clean mean
                z_anchor_k = anchor_feats - mu_static_domain_clean[e_name]
                
            max_sims = []
            for i in range(batch_X.size(0)):
                z_i = z_anchor_k[i]
                z_i_norm = z_i / (z_i.norm(p=2) + 1e-12)
                max_sim = max(torch.dot(z_i_norm, prototypes_clean[k][c]) for c in range(10))
                max_sims.append(max_sim)
            cohesions.append(sum(max_sims)/len(max_sims))
        return [c.item() for c in cohesions]

    def apply_noise(batch_X):
        return torch.clamp(batch_X + torch.randn_like(batch_X) * 0.2, -1.0, 1.0)
        
    def apply_contrast(batch_X):
        return torch.clamp(batch_X * 0.3, -1.0, 1.0)

    print("=== COHESION WITH DFC ===")
    for name, loader in [('MNIST', mnist_test_loader), ('KMNIST', kmnist_test_loader), ('F-MNIST', fmnist_test_loader)]:
        batch_X, _ = next(iter(loader))
        
        # Clean
        c_clean = compute_cohesion_dfc(batch_X, use_dfc=True)
        # Noise
        c_noise = compute_cohesion_dfc(apply_noise(batch_X), use_dfc=True)
        # Contrast
        c_contrast = compute_cohesion_dfc(apply_contrast(batch_X), use_dfc=True)
        
        print(f"\nDataset: {name} (DFC)")
        print(f"  Clean    | MNIST Cohesion: {c_clean[0]:.4f} | KMNIST Cohesion: {c_clean[1]:.4f} | Max: {max(c_clean):.4f}")
        print(f"  Noise    | MNIST Cohesion: {c_noise[0]:.4f} | KMNIST Cohesion: {c_noise[1]:.4f} | Max: {max(c_noise):.4f}")
        print(f"  Contrast | MNIST Cohesion: {c_contrast[0]:.4f} | KMNIST Cohesion: {c_contrast[1]:.4f} | Max: {max(c_contrast):.4f}")

if __name__ == '__main__':
    main()
