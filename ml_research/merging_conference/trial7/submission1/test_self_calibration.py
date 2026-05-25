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

mnist_loader = torch.utils.data.DataLoader(test_mnist, batch_size=64, shuffle=False)
kmnist_loader = torch.utils.data.DataLoader(test_kmnist, batch_size=64, shuffle=False)
fashion_loader = torch.utils.data.DataLoader(test_fashion, batch_size=64, shuffle=False)

mnist_batches = list(mnist_loader)[:30]
kmnist_batches = list(kmnist_loader)[:30]
fashion_batches = list(fashion_loader)[:30]

# Precomputed expected clean max cohesions
# For MNIST (domain 0): 0.8173
# For KMNIST (domain 1): 0.6357
expected_clean_cohesions = {0: 0.8173, 1: 0.6357}

for s in ["sequential"]:
    for c in ["contrast"]:
        print(f"\nEvaluating stream type={s}, corruption={c}")
        
        # Build stream
        if s == "sequential":
            stream = mnist_batches + kmnist_batches + fashion_batches
            domain_labels = [0]*30 + [1]*30 + [2]*30
            
        # Initialize self-calibrating tracker
        gamma_bar = 1.0
        phi = 0.20 # momentum
        tau_N_base = 0.59
        
        novel_detected = 0
        false_positives = 0
        
        for t, (x, y) in enumerate(stream):
            if c == "gaussian":
                noise = torch.randn_like(x) * 0.2
                x = torch.clamp(x + noise, -1.0, 1.0)
            elif c == "contrast":
                x = torch.clamp(x * 0.3, -1.0, 1.0)
            
            x = x.to(device)
            with torch.no_grad():
                feats = static_feat_extractor(x)
                z_anchor = feats - feats.mean(dim=0)
            
            cohesion = []
            for k in range(2):
                max_sims = []
                for i in range(len(x)):
                    sims = []
                    for c_idx in range(10):
                        proto = pi_kc[k][c_idx]
                        sim = torch.dot(z_anchor[i], proto) / (torch.norm(z_anchor[i]) * torch.norm(proto) + 1e-8)
                        sims.append(sim.item())
                    max_sims.append(max(sims))
                cohesion.append(np.mean(max_sims))
            
            max_coh = max(cohesion)
            k_star = np.argmax(cohesion)
            
            # If first batch, initialize gamma_bar immediately!
            if t == 0:
                gamma_bar = max_coh / expected_clean_cohesions[k_star]
                print(f"Batch {t} (MNIST start): Max Coh: {max_coh:.4f} | expected clean: {expected_clean_cohesions[k_star]:.4f} | initialized gamma_bar: {gamma_bar:.4f}")
                
            # Dynamic thresholding
            tau_N = tau_N_base * gamma_bar
            is_novel = max_coh < tau_N
            
            true_domain = domain_labels[t]
            
            if t in [0, 1, 2, 29, 30, 31, 32, 59, 60, 61, 62]:
                print(f"Batch {t}: true_domain={true_domain} | max_coh={max_coh:.4f} | tau_N={tau_N:.4f} | is_novel={is_novel} | gamma_bar={gamma_bar:.4f}")
                
            if is_novel:
                if true_domain == 2:
                    novel_detected += 1
            else:
                if true_domain == 2:
                    false_positives += 1
                # Update gamma_bar using the known batch's scale factor
                inst_gamma = max_coh / expected_clean_cohesions[k_star]
                gamma_bar = (1 - phi) * gamma_bar + phi * inst_gamma
                
        ndr = 100.0 * novel_detected / 30.0
        fpr = 100.0 * false_positives / 60.0
        print(f"  NDR: {ndr:.2f}% | FPR: {fpr:.2f}% (Final Gamma Bar: {gamma_bar:.4f})")
