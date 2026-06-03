import os
import json
import time
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

# Define model architecture
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

# Standard Full SVD SSSS-Merge
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
                # Full SVD
                U_wa, S_wa, V_wa = torch.linalg.svd(w_wa_2d, full_matrices=False)
                _, S_exp, _ = torch.linalg.svd(expert_2d, full_matrices=False)
                
                w_sps_2d = U_wa @ torch.diag(S_exp) @ V_wa
                sps_state_dict[key] = w_sps_2d.view(orig_shape)
            except Exception as e:
                sps_state_dict[key] = w_wa
        else:
            sps_state_dict[key] = w_wa
    return sps_state_dict

# Randomized SVD implementation
def randomized_svd(A, k, p=5, q=1):
    m, n = A.shape
    r = min(k + p, m, n)
    Omega = torch.randn(n, r, device=A.device, dtype=A.dtype)
    Y = A @ Omega
    for _ in range(q):
        Q_qr, _ = torch.linalg.qr(Y, mode="reduced")
        Y = A @ (A.T @ Q_qr)
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.T @ A
    U_tilde, S, V_h = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    return U[:, :k], S[:k], V_h[:k, :]

# Randomized SVD SPS-Merge (R-SPS-Merge) with TSSS
def merge_r_sps_tsss(expert_state_dicts, wa_state_dict, task_idx, rank_fraction=1.0):
    r_sps_state_dict = {}
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
                
            m, n = w_wa_2d.shape
            full_rank = min(m, n)
            # Target rank
            k = max(int(rank_fraction * full_rank), 1)
            
            try:
                # Randomized SVD of WA matrix
                U_wa, S_wa, V_wa = randomized_svd(w_wa_2d, k, p=5, q=1)
                # Randomized SVD of expert matrix (to get singular values)
                _, S_exp, _ = randomized_svd(expert_2d, k, p=5, q=1)
                
                # S_exp is of length k
                # Reconstruct weight using the approximated singular vectors and values
                w_sps_2d = U_wa @ torch.diag(S_exp) @ V_wa
                r_sps_state_dict[key] = w_sps_2d.view(orig_shape)
            except Exception as e:
                # Fallback to WA if SVD fails
                r_sps_state_dict[key] = w_wa
        else:
            r_sps_state_dict[key] = w_wa
    return r_sps_state_dict

def main():
    data_dict = get_datasets()
    
    mnist_test_loader = DataLoader(data_dict["mnist"]["test"], batch_size=128, shuffle=False)
    fmnist_test_loader = DataLoader(data_dict["fmnist"]["test"], batch_size=128, shuffle=False)
    cifar_test_loader = DataLoader(data_dict["cifar10"]["test"], batch_size=128, shuffle=False)
    
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
    
    # Load strong experts
    expert_state_dicts = []
    for task, path in expert_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert model for {task} not found at {path}. Please run training first.")
        expert_state_dicts.append(torch.load(path, map_location=device))
        
    print("Loaded strong experts successfully.")
    wa_state = merge_wa(expert_state_dicts)
    
    # Benchmark 1: Full SVD SPS-Merge + TSSS
    print("\n--- Benchmarking Standard Full SVD SPS-Merge + TSSS ---")
    start_time = time.time()
    
    # Generate the 3 task-specific merged states (MNIST, F-MNIST, CIFAR-10)
    full_sps_states = []
    for idx in range(3):
        full_sps_states.append(merge_sps_tsss(expert_state_dicts, wa_state, idx))
        
    full_svd_time = time.time() - start_time
    print(f"Full SVD merging wall-clock time: {full_svd_time:.4f} seconds")
    
    # Evaluate Full SVD accuracies
    full_svd_accs = {}
    for t, loader in test_loaders.items():
        task_idx = {"mnist": 0, "fmnist": 1, "cifar10": 2}[t]
        state = full_sps_states[task_idx]
        
        m = MergibleResNet18().to(device)
        m.load_state_dict(state)
        # Copy task classifier head
        m.resnet.fc[1].weight.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.weight"])
        m.resnet.fc[1].bias.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.bias"])
        
        # Calibrate
        task_calib_loader = DataLoader(data_dict[t]["calib"], batch_size=128, shuffle=True)
        calibrate_bn(m, task_calib_loader)
        
        full_svd_accs[t] = evaluate_model(m, loader)
        
    full_svd_avg = float(np.mean(list(full_svd_accs.values())))
    print(f"Full SVD Accuracies:")
    for t, acc in full_svd_accs.items():
        print(f"  {t}: {acc*100:.2f}%")
    print(f"  Average: {full_svd_avg*100:.2f}%")
    
    # Benchmark 2: Randomized SVD SPS-Merge + TSSS (R-SPS-Merge) across various rank fractions
    rank_fractions = [1.0, 0.8, 0.6, 0.4, 0.2]
    r_sps_results = {}
    
    for rf in rank_fractions:
        print(f"\n--- Benchmarking R-SPS-Merge + TSSS (Rank Fraction = {rf}) ---")
        start_time = time.time()
        
        r_sps_states = []
        for idx in range(3):
            r_sps_states.append(merge_r_sps_tsss(expert_state_dicts, wa_state, idx, rank_fraction=rf))
            
        r_svd_time = time.time() - start_time
        print(f"R-SVD merging wall-clock time: {r_svd_time:.4f} seconds (Speedup vs Full: {full_svd_time/r_svd_time:.2f}x)")
        
        # Calculate Relative Frobenius Norm Error vs Full SVD states
        frob_errors = []
        for idx, key in enumerate(wa_state.keys()):
            if ("weight" in key) and (wa_state[key].ndim >= 2) and any(kw in key for kw in ["conv", "fc", "downsample.0"]):
                w_full = full_sps_states[0][key]  # use first task-specific model as a proxy, or average across tasks
                # Let's average relative errors across all 3 tasks
                for task_idx in range(3):
                    w_full_t = full_sps_states[task_idx][key]
                    w_rand_t = r_sps_states[task_idx][key]
                    err = torch.linalg.norm(w_rand_t - w_full_t) / (torch.linalg.norm(w_full_t) + 1e-8)
                    frob_errors.append(err.item())
                    
        avg_frob_error = float(np.mean(frob_errors))
        print(f"Mean Relative Frobenius Reconstruction Error vs Full SVD: {avg_frob_error*100:.4f}%")
        
        # Evaluate R-SVD accuracies
        r_svd_accs = {}
        for t, loader in test_loaders.items():
            task_idx = {"mnist": 0, "fmnist": 1, "cifar10": 2}[t]
            state = r_sps_states[task_idx]
            
            m = MergibleResNet18().to(device)
            m.load_state_dict(state)
            # Copy task classifier head
            m.resnet.fc[1].weight.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.weight"])
            m.resnet.fc[1].bias.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.bias"])
            
            # Calibrate
            task_calib_loader = DataLoader(data_dict[t]["calib"], batch_size=128, shuffle=True)
            calibrate_bn(m, task_calib_loader)
            
            r_svd_accs[t] = evaluate_model(m, loader)
            
        r_svd_avg = float(np.mean(list(r_svd_accs.values())))
        print(f"R-SVD Accuracies:")
        for t, acc in r_svd_accs.items():
            print(f"  {t}: {acc*100:.2f}%")
        print(f"  Average: {r_svd_avg*100:.2f}%")
        
        r_sps_results[str(rf)] = {
            "time": r_svd_time,
            "speedup": full_svd_time / r_svd_time,
            "frob_error": avg_frob_error,
            "accs": r_svd_accs,
            "avg_acc": r_svd_avg
        }
        
    # Write a summary json file
    summary = {
        "full_svd": {
            "time": full_svd_time,
            "accs": full_svd_accs,
            "avg_acc": full_svd_avg
        },
        "r_svd": r_sps_results
    }
    
    with open("results_randomized_svd.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("\nBenchmarking complete! Results saved to results_randomized_svd.json.")

if __name__ == "__main__":
    main()
