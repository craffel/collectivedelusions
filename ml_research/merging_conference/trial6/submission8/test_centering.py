import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from run_eval import ResNetWithFeatures, compute_prototypes

def main():
    device = 'cpu'
    experts = []
    for name in ['expert1_cifar', 'expert2_svhn', 'expert3_fmnist']:
        model = ResNetWithFeatures()
        model.base_model.load_state_dict(torch.load(f"experts/{name}.pt", map_location=device))
        model.eval()
        experts.append(model)
        
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb)
    svhn_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform_rgb)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    cifar_val = Subset(datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_rgb), list(range(500)))
    svhn_val = Subset(datasets.SVHN(root='./data', split='train', download=True, transform=transform_rgb), list(range(500)))
    
    p1 = compute_prototypes(experts[0], cifar_val, num_samples=500, device=device)
    p2 = compute_prototypes(experts[1], svhn_val, num_samples=500, device=device)
    
    mu1 = torch.zeros(512)
    mu2 = torch.zeros(512)
    for c in range(10):
        mu1 += p1[c]
        mu2 += p2[c]
    mu1 = F.normalize(mu1 / 10.0, p=2, dim=0)
    mu2 = F.normalize(mu2 / 10.0, p=2, dim=0)
    
    # Center prototypes with their respective means
    p1_centered = {c: F.normalize(p1[c] - mu1, p=2, dim=0) for c in range(10)}
    p2_centered = {c: F.normalize(p2[c] - mu2, p=2, dim=0) for c in range(10)}
    
    test_batches = [
        (next(iter(DataLoader(Subset(cifar_test, list(range(64))), batch_size=64)))[0], 'Task A (CIFAR-10)'),
        (next(iter(DataLoader(Subset(svhn_test, list(range(64))), batch_size=64)))[0], 'Task B (SVHN)'),
        (next(iter(DataLoader(Subset(fmnist_test, list(range(64))), batch_size=64)))[0], 'Task C (FashionMNIST)')
    ]
    
    print("--- Testing Fixed-Expert Cohesion (No feedback loop!) ---")
    for x, task in test_batches:
        # Pass through Expert 1
        with torch.no_grad():
            f1, _ = experts[0](x)
            f1_centered = F.normalize(f1 - mu1, p=2, dim=-1)
            c1_scores = []
            for i in range(len(x)):
                sims = [torch.dot(f1_centered[i], p1_centered[c]).item() for c in range(10)]
                c1_scores.append(max(sims))
            c1_score = np.mean(c1_scores)
            
            # Pass through Expert 2
            f2, _ = experts[1](x)
            f2_centered = F.normalize(f2 - mu2, p=2, dim=-1)
            c2_scores = []
            for i in range(len(x)):
                sims = [torch.dot(f2_centered[i], p2_centered[c]).item() for c in range(10)]
                c2_scores.append(max(sims))
            c2_score = np.mean(c2_scores)
            
        print(f"[{task}] cohesion_exp1 = {c1_score:.4f}, cohesion_exp2 = {c2_score:.4f}")

if __name__ == '__main__':
    main()
