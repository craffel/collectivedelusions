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
        
        # Center features with the domain mean (from static_model)
        domain_mean = feats_all.mean(dim=0)
        z_all = feats_all - domain_mean
        for c in range(10):
            mask = (labels_all == c)
            if mask.sum() > 0:
                pi_c = z_all[mask].mean(dim=0)
                pi_c = pi_c / (pi_c.norm(p=2) + 1e-12)
                prototypes_clean[k][c] = pi_c
            else:
                prototypes_clean[k][c] = torch.zeros_like(domain_mean)

    # Let's construct the test streams
    test_size_per_task = 30 * 64
    mnist_test_subset = Subset(mnist_test, list(range(test_size_per_task)))
    kmnist_test_subset = Subset(kmnist_test, list(range(test_size_per_task)))
    fmnist_test_subset = Subset(fmnist_test, list(range(test_size_per_task)))

    mnist_loader = DataLoader(mnist_test_subset, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(kmnist_test_subset, batch_size=64, shuffle=False)
    fmnist_loader = DataLoader(fmnist_test_subset, batch_size=64, shuffle=False)

    mnist_batches = [b for b in mnist_loader]
    kmnist_batches = [b for b in kmnist_loader]
    fmnist_batches = [b for b in fmnist_loader]

    seq_stream_clean = mnist_batches[:30] + kmnist_batches[:30] + fmnist_batches[:30]
    seq_domains = [0]*30 + [1]*30 + [2]*30

    def apply_noise(batch_X):
        return torch.clamp(batch_X + torch.randn_like(batch_X) * 0.2, -1.0, 1.0)
        
    def apply_contrast(batch_X):
        return torch.clamp(batch_X * 0.3, -1.0, 1.0)

    streams = {
        'Sequential_Clean': (seq_stream_clean, seq_domains),
        'Sequential_Noise': ([(apply_noise(x), y) for x, y in seq_stream_clean], seq_domains),
        'Sequential_Contrast': ([(apply_contrast(x), y) for x, y in seq_stream_clean], seq_domains)
    }

    # Define expert-specific thresholds based on clean calibration cohesions
    # MNIST expert clean cohesion is ~0.82, KMNIST is ~0.63
    # Let's set beta = 0.82
    beta = 0.82
    tau_N_k = {
        0: beta * 0.82,  # MNIST threshold: ~0.67
        1: beta * 0.63   # KMNIST threshold: ~0.516
    }
    print(f"Using expert-specific thresholds: {tau_N_k}")

    for stream_name, (stream, domains) in streams.items():
        novel_actual = 0
        novel_detected = 0
        known_actual = 0
        known_detected_novel = 0

        for t, (batch_X, batch_y) in enumerate(stream):
            batch_X = batch_X.to(device)
            true_dom = domains[t]
            is_novel_actual = (true_dom == 2)

            with torch.no_grad():
                anchor_feats, _ = static_model(batch_X)
            
            # DFC: center with active batch mean
            batch_mean = anchor_feats.mean(dim=0)
            z_anchor = anchor_feats - batch_mean

            cohesions = []
            for k in range(2):
                max_sims = []
                for i in range(batch_X.size(0)):
                    z_i = z_anchor[i]
                    z_i_norm = z_i / (z_i.norm(p=2) + 1e-12)
                    max_sim = max(torch.dot(z_i_norm, prototypes_clean[k][c]) for c in range(10))
                    max_sims.append(max_sim)
                cohesion_k = sum(max_sims) / len(max_sims)
                cohesions.append(cohesion_k.item())

            # Novelty detection with expert-specific thresholds
            # Flagged as novel if BOTH cohesions are below their respective thresholds
            is_novel_pred = (cohesions[0] < tau_N_k[0]) and (cohesions[1] < tau_N_k[1])

            if is_novel_actual:
                novel_actual += 1
                if is_novel_pred:
                    novel_detected += 1
            else:
                known_actual += 1
                if is_novel_pred:
                    known_detected_novel += 1

        ndr = (novel_detected / novel_actual) * 100.0 if novel_actual > 0 else 100.0
        fpr = (known_detected_novel / known_actual) * 100.0 if known_actual > 0 else 0.0
        print(f"Stream: {stream_name:<20} | NDR: {ndr:>6.2f}% | FPR: {fpr:>6.2f}%")

if __name__ == '__main__':
    main()
