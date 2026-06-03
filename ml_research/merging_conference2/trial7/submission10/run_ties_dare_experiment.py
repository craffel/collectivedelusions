import os
import copy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Import helpers from experiment
from experiment import merge_models, evaluate, get_subset, apply_spttbc, run_task_specific_bn_calibration

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    transform_mnist = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])

    transform_fmnist = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
    ])

    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

    train_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_fmnist)
    test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_fmnist)

    train_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
    test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
    
    train_mnist_sub = get_subset(train_mnist)
    train_fmnist_sub = get_subset(train_fmnist)
    train_cifar_sub = get_subset(train_cifar)
    
    test_loader_mnist = DataLoader(test_mnist, batch_size=256, shuffle=False)
    test_loader_fmnist = DataLoader(test_fmnist, batch_size=256, shuffle=False)
    test_loader_cifar = DataLoader(test_cifar, batch_size=256, shuffle=False)
    
    print("Preparing models...")
    progenitor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    expert_mnist = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    expert_mnist.fc = nn.Linear(512, 10)
    
    expert_fmnist = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    expert_fmnist.fc = nn.Linear(512, 10)
    
    expert_cifar = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    expert_cifar.fc = nn.Linear(512, 10)
    
    if os.path.exists("expert_mnist_severe.pth") and os.path.exists("expert_fmnist_severe.pth") and os.path.exists("expert_cifar_severe.pth"):
        print("Loading pre-trained severe-drift experts from disk...")
        expert_mnist.load_state_dict(torch.load("expert_mnist_severe.pth", map_location=device))
        expert_fmnist.load_state_dict(torch.load("expert_fmnist_severe.pth", map_location=device))
        expert_cifar.load_state_dict(torch.load("expert_cifar_severe.pth", map_location=device))
    else:
        raise FileNotFoundError("Pre-trained severe-drift experts not found on disk!")
    
    experts = [expert_mnist, expert_fmnist, expert_cifar]
    dataloaders = [test_loader_mnist, test_loader_fmnist, test_loader_cifar]
    calib_loaders = [
        DataLoader(train_mnist_sub, batch_size=64, shuffle=True),
        DataLoader(train_fmnist_sub, batch_size=64, shuffle=True),
        DataLoader(train_cifar_sub, batch_size=64, shuffle=True)
    ]
    
    results = {
        "merging": []
    }
    
    # Sweep lambdas for TIES and DARE
    lambdas = [0.3, 0.5, 0.7, 1.0]
    merge_configs = []
    for method in ['TIES', 'DARE']:
        for l in lambdas:
            merge_configs.append((method, l))
            
    for method, lam in merge_configs:
        config_name = f"{method} (lambda={lam})"
        print(f"\n==================== Evaluating {config_name} ====================")
        
        # 1. No Calibration
        m_model = merge_models(progenitor, experts, method=method, lam=lam)
        m_model.fc = nn.Linear(512, 10)
        accs_no_calib = []
        for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
            m_model.fc = exp.fc
            acc = evaluate(m_model, loader, device)
            accs_no_calib.append(acc)
        avg_no_calib = sum(accs_no_calib) / 3
        print(f"No Calibration - MNIST: {accs_no_calib[0]:.4%} | FMNIST: {accs_no_calib[1]:.4%} | CIFAR-10: {accs_no_calib[2]:.4%} | Avg: {avg_no_calib:.4%}")
        
        # 2. SP-TTBC
        m_model = merge_models(progenitor, experts, method=method, lam=lam)
        apply_spttbc(m_model, alpha=1.0)
        accs_spttbc = []
        for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
            m_model.fc = exp.fc
            acc = evaluate(m_model, loader, device)
            accs_spttbc.append(acc)
        avg_spttbc = sum(accs_spttbc) / 3
        print(f"SP-TTBC (1.0)  - MNIST: {accs_spttbc[0]:.4%} | FMNIST: {accs_spttbc[1]:.4%} | CIFAR-10: {accs_spttbc[2]:.4%} | Avg: {avg_spttbc:.4%}")
        
        # 3. TS-BN Calib (Ours)
        accs_ts_bc = []
        for i, (loader, exp) in enumerate(zip(dataloaders, experts)):
            m_model = merge_models(progenitor, experts, method=method, lam=lam)
            m_model.fc = nn.Linear(512, 10)
            run_task_specific_bn_calibration(m_model, calib_loaders[i], num_batches=20, device=device)
            m_model.fc = exp.fc
            acc = evaluate(m_model, loader, device)
            accs_ts_bc.append(acc)
        avg_ts_bc = sum(accs_ts_bc) / 3
        print(f"TS-BN Calib    - MNIST: {accs_ts_bc[0]:.4%} | FMNIST: {accs_ts_bc[1]:.4%} | CIFAR-10: {accs_ts_bc[2]:.4%} | Avg: {avg_ts_bc:.4%}")

        results["merging"].append({
            "method": method,
            "lambda": lam,
            "No_Calib": {"MNIST": accs_no_calib[0], "Fashion-MNIST": accs_no_calib[1], "CIFAR-10": accs_no_calib[2], "Average": avg_no_calib},
            "SP-TTBC": {"MNIST": accs_spttbc[0], "Fashion-MNIST": accs_spttbc[1], "CIFAR-10": accs_spttbc[2], "Average": avg_spttbc},
            "TS_BC": {"MNIST": accs_ts_bc[0], "Fashion-MNIST": accs_ts_bc[1], "CIFAR-10": accs_ts_bc[2], "Average": avg_ts_bc}
        })
        
    with open("ties_dare_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nTIES & DARE results successfully saved to ties_dare_results.json!")

if __name__ == '__main__':
    main()