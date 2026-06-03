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

def get_datasets(calib_sizes):
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
    
    # Subsample the test datasets to 1000 samples for fast CPU evaluation
    mnist_test_sub = Subset(mnist_test, np.random.choice(len(mnist_test), 1000, replace=False))
    fmnist_test_sub = Subset(fmnist_test, np.random.choice(len(fmnist_test), 1000, replace=False))
    cifar_test_sub = Subset(cifar_test, np.random.choice(len(cifar_test), 1000, replace=False))
    
    res = {
        "mnist": {"test": mnist_test_sub, "calibs": {}},
        "fmnist": {"test": fmnist_test_sub, "calibs": {}},
        "cifar10": {"test": cifar_test_sub, "calibs": {}}
    }
    
    for sz in calib_sizes:
        res["mnist"]["calibs"][sz] = Subset(mnist_train, np.random.choice(len(mnist_train), sz, replace=False))
        res["fmnist"]["calibs"][sz] = Subset(fmnist_train, np.random.choice(len(fmnist_train), sz, replace=False))
        res["cifar10"]["calibs"][sz] = Subset(cifar_train, np.random.choice(len(cifar_train), sz, replace=False))
        
    return res

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
                
                w_sps_2d = U_wa @ torch.diag(S_exp) @ V_wa
                sps_state_dict[key] = w_sps_2d.view(orig_shape)
            except Exception as e:
                sps_state_dict[key] = w_wa
        else:
            sps_state_dict[key] = w_wa
            
    return sps_state_dict

def main():
    calib_sizes = [16, 32, 64, 128, 256, 512]
    data_dict = get_datasets(calib_sizes)
    
    # We use batch size of 256 for test evaluation to speed up
    mnist_test_loader = DataLoader(data_dict["mnist"]["test"], batch_size=256, shuffle=False)
    fmnist_test_loader = DataLoader(data_dict["fmnist"]["test"], batch_size=256, shuffle=False)
    cifar_test_loader = DataLoader(data_dict["cifar10"]["test"], batch_size=256, shuffle=False)
    
    test_loaders = {
        "mnist": mnist_test_loader,
        "fmnist": fmnist_test_loader,
        "cifar10": cifar_test_loader
    }
    
    expert_paths = {
        "mnist": "experts_strong/expert_mnist.pt",
        "fmnist": "experts_strong/expert_fmnist.pt",
        "cifar10": "experts_strong/expert_cifar10.pt"
    }
    
    expert_state_dicts = []
    for task, path in expert_paths.items():
        expert_state_dicts.append(torch.load(path, map_location=device))
        
    wa_state = merge_wa(expert_state_dicts)
    
    results = {
        "calib_sizes": calib_sizes,
        "wa_bnc": {},
        "sps_tsss_bnc": {}
    }
    
    for sz in calib_sizes:
        print(f"\n--- Evaluating calibration size = {sz} ---")
        
        # 1. WA + TS-BNC
        wa_accs = {}
        for t, loader in test_loaders.items():
            m = MergibleResNet18().to(device)
            m.load_state_dict(wa_state)
            task_idx = {"mnist": 0, "fmnist": 1, "cifar10": 2}[t]
            m.resnet.fc[1].weight.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.weight"])
            m.resnet.fc[1].bias.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.bias"])
            
            task_calib_loader = DataLoader(data_dict[t]["calibs"][sz], batch_size=sz, shuffle=True)
            calibrate_bn(m, task_calib_loader)
            
            acc = evaluate_model(m, loader)
            wa_accs[t] = acc
        wa_avg = np.mean(list(wa_accs.values()))
        results["wa_bnc"][sz] = {"accs": wa_accs, "avg": float(wa_avg)}
        print(f"  WA + TS-BNC Average: {wa_avg * 100:.2f}%")
        
        # 2. SPS-Merge + TSSS + TS-BNC
        tsss_accs = {}
        for t, loader in test_loaders.items():
            task_idx = {"mnist": 0, "fmnist": 1, "cifar10": 2}[t]
            sps_tsss_state = merge_sps_tsss(expert_state_dicts, wa_state, task_idx)
            
            m = MergibleResNet18().to(device)
            m.load_state_dict(sps_tsss_state)
            m.resnet.fc[1].weight.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.weight"])
            m.resnet.fc[1].bias.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.bias"])
            
            task_calib_loader = DataLoader(data_dict[t]["calibs"][sz], batch_size=sz, shuffle=True)
            calibrate_bn(m, task_calib_loader)
            
            acc = evaluate_model(m, loader)
            tsss_accs[t] = acc
        tsss_avg = np.mean(list(tsss_accs.values()))
        results["sps_tsss_bnc"][sz] = {"accs": tsss_accs, "avg": float(tsss_avg)}
        print(f"  SPS-Merge + TSSS + TS-BNC Average: {tsss_avg * 100:.2f}%")
        
    with open("results_calib_sizes.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nAblation study results saved to results_calib_sizes.json.")

if __name__ == "__main__":
    main()
