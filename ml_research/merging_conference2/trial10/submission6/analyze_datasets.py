import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
])

fmnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860, 0.2860, 0.2860], std=[0.3530, 0.3530, 0.3530])
])

cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

test_sets = {
    'mnist': datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform),
    'fmnist': datasets.FashionMNIST(root='./data', train=False, download=True, transform=fmnist_transform),
    'cifar10': datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
}

correct_dds = 0
total = 0

for gt_idx, name in enumerate(['mnist', 'fmnist', 'cifar10']):
    loader = DataLoader(Subset(test_sets[name], range(200)), batch_size=32, shuffle=False)
    for inputs, _ in loader:
        for x in inputs:
            total += 1
            # 1. Check channel difference for CIFAR10
            channel_diff = (x[0] - x[1]).abs().max().item()
            if channel_diff > 1e-4:
                pred_task = 2 # cifar10
            else:
                # Grayscale, separate MNIST and FMNIST based on background pixel normalization signatures
                # MNIST background is normalized to -0.4242
                # FMNIST background is normalized to -0.8102
                mnist_bg_count = ((x - (-0.4242)).abs() < 1e-2).sum().item()
                fmnist_bg_count = ((x - (-0.8102)).abs() < 1e-2).sum().item()
                
                if mnist_bg_count > fmnist_bg_count:
                    pred_task = 0 # mnist
                else:
                    pred_task = 1 # fmnist
                    
            if pred_task == gt_idx:
                correct_dds += 1

print(f"DDS Routing Accuracy: {correct_dds / total * 100.0:.2f}% ({correct_dds}/{total})")
