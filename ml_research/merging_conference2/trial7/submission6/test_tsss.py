import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.enabled = False
print(f"Using device: {device}")

# Define a customized ResNet18 model with dropout
class MergibleResNet18(nn.Module):
    def __init__(self, num_classes=10, dropout=0.1):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

def get_datasets():
    gray_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    rgb_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    mnist_train = torchvision.datasets.MNIST(root="data", train=True, transform=gray_transform)
    mnist_test = torchvision.datasets.MNIST(root="data", train=False, transform=gray_transform)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root="data", train=True, transform=gray_transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root="data", train=False, transform=gray_transform)
    
    cifar_train = torchvision.datasets.CIFAR10(root="data", train=True, transform=rgb_transform)
    cifar_test = torchvision.datasets.CIFAR10(root="data", train=False, transform=rgb_transform)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create 128-sample subsets for calibration
    mnist_calib = Subset(mnist_train, np.random.choice(len(mnist_train), 128, replace=False))
    fmnist_calib = Subset(fmnist_train, np.random.choice(len(fmnist_train), 128, replace=False))
    cifar_calib = Subset(cifar_train, np.random.choice(len(cifar_train), 128, replace=False))
    
    return {
        "mnist": {"test": mnist_test, "calib": mnist_calib},
        "fmnist": {"test": fmnist_test, "calib": fmnist_calib},
        "cifar10": {"test": cifar_test, "calib": cifar_calib}
    }

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return correct / total

def calibrate_bn(model, calib_loader):
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.momentum = 1.0
            
    with torch.no_grad():
        for x, _ in calib_loader:
            x = x.to(device)
            _ = model(x)
            break

def merge_wa(expert_state_dicts):
    merged = {}
    keys = expert_state_dicts[0].keys()
    for key in keys:
        tensors = [d[key] for d in expert_state_dicts]
        if tensors[0].is_floating_point():
            merged[key] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            merged[key] = tensors[0].clone()
    return merged

def merge_sps_tsss(expert_state_dicts, wa_state_dict, task_idx):
    sps_state_dict = {}
    for key in wa_state_dict.keys():
        w_wa = wa_state_dict[key].clone()
        
        if ("weight" in key) and (w_wa.ndim >= 2) and any(kw in key for kw in ["conv", "fc", "downsample.0"]):
            expert_tensor = expert_state_dicts[task_idx][key]
            orig_shape = w_wa.shape
            
            if w_wa.ndim > 2:
                w_wa_2d = w_wa.view(orig_shape[0], -1)
                expert_2d = expert_tensor.view(orig_shape[0], -1)
            else:
                w_wa_2d = w_wa
                expert_2d = expert_tensor
                
            try:
                U_wa, S_wa, V_wa = torch.linalg.svd(w_wa_2d, full_matrices=False)
                _, S_exp, _ = torch.linalg.svd(expert_2d, full_matrices=False)
                
                # Reconstruct using WA's left and right singular vectors, but the specific expert's singular values
                w_sps_2d = U_wa @ torch.diag(S_exp) @ V_wa
                sps_state_dict[key] = w_sps_2d.view(orig_shape)
            except Exception as e:
                sps_state_dict[key] = w_wa
        else:
            sps_state_dict[key] = w_wa
            
    return sps_state_dict

def main():
    data_dict = get_datasets()
    
    mnist_test_loader = DataLoader(data_dict["mnist"]["test"], batch_size=128, shuffle=False)
    fmnist_test_loader = DataLoader(data_dict["fmnist"]["test"], batch_size=128, shuffle=False)
    cifar_test_loader = DataLoader(data_dict["cifar10"]["test"], batch_size=128, shuffle=False)
    
    expert_paths = {
        "mnist": "experts/expert_mnist.pt",
        "fmnist": "experts/expert_fmnist.pt",
        "cifar10": "experts/expert_cifar10.pt"
    }
    
    expert_state_dicts = []
    for task, path in expert_paths.items():
        expert_state_dicts.append(torch.load(path, map_location=device))
        
    wa_state = merge_wa(expert_state_dicts)
    
    test_loaders = {
        "mnist": mnist_test_loader,
        "fmnist": fmnist_test_loader,
        "cifar10": cifar_test_loader
    }
    
    # 1. Evaluate standard WA + TS-BNC
    wa_bnc_accs = {}
    for t, loader in test_loaders.items():
        m = MergibleResNet18().to(device)
        m.load_state_dict(wa_state)
        
        # Load task-specific classification head
        task_idx = {"mnist": 0, "fmnist": 1, "cifar10": 2}[t]
        m.resnet.fc[1].weight.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.weight"])
        m.resnet.fc[1].bias.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.bias"])
        
        task_calib_loader = DataLoader(data_dict[t]["calib"], batch_size=128, shuffle=True)
        calibrate_bn(m, task_calib_loader)
        
        wa_bnc_accs[t] = evaluate_model(m, loader)
    
    print("\n--- WA + TS-BNC (Baseline) ---")
    for t, acc in wa_bnc_accs.items():
        print(f"  {t}: {acc * 100:.2f}%")
    print(f"  Average: {np.mean(list(wa_bnc_accs.values())) * 100:.2f}%")
    
    # 2. Evaluate SPS-Merge + TSSS + TS-BNC
    tsss_bnc_accs = {}
    for t, loader in test_loaders.items():
        task_idx = {"mnist": 0, "fmnist": 1, "cifar10": 2}[t]
        
        # Generate the TSSS-reconstructed weight for this specific task
        sps_tsss_state = merge_sps_tsss(expert_state_dicts, wa_state, task_idx)
        
        m = MergibleResNet18().to(device)
        m.load_state_dict(sps_tsss_state)
        
        # Load task-specific classification head
        m.resnet.fc[1].weight.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.weight"])
        m.resnet.fc[1].bias.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.bias"])
        
        task_calib_loader = DataLoader(data_dict[t]["calib"], batch_size=128, shuffle=True)
        calibrate_bn(m, task_calib_loader)
        
        tsss_bnc_accs[t] = evaluate_model(m, loader)
        
    print("\n--- SPS-Merge with TSSS + TS-BNC (Proposed) ---")
    for t, acc in tsss_bnc_accs.items():
        print(f"  {t}: {acc * 100:.2f}%")
    print(f"  Average: {np.mean(list(tsss_bnc_accs.values())) * 100:.2f}%")

if __name__ == "__main__":
    main()
