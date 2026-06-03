import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

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
    
    mnist_train_sub = Subset(mnist_train, np.random.choice(len(mnist_train), 5000, replace=False))
    fmnist_train_sub = Subset(fmnist_train, np.random.choice(len(fmnist_train), 5000, replace=False))
    cifar_train_sub = Subset(cifar_train, np.random.choice(len(cifar_train), 5000, replace=False))
    
    mnist_calib = Subset(mnist_train, np.random.choice(len(mnist_train), 128, replace=False))
    fmnist_calib = Subset(fmnist_train, np.random.choice(len(fmnist_train), 128, replace=False))
    cifar_calib = Subset(cifar_train, np.random.choice(len(cifar_train), 128, replace=False))
    
    return {
        "mnist": {"train": mnist_train_sub, "test": mnist_test, "calib": mnist_calib},
        "fmnist": {"train": fmnist_train_sub, "test": fmnist_test, "calib": fmnist_calib},
        "cifar10": {"train": cifar_train_sub, "test": cifar_test, "calib": cifar_calib}
    }

def train_model(model, train_loader, epochs=10, lr=1e-3):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
    return model

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

def merge_sps(expert_state_dicts, wa_state_dict):
    sps_state_dict = {}
    for key in wa_state_dict.keys():
        w_wa = wa_state_dict[key].clone()
        
        if ("weight" in key) and (w_wa.ndim >= 2) and any(kw in key for kw in ["conv", "fc", "downsample.0"]):
            expert_tensors = [d[key] for d in expert_state_dicts]
            orig_shape = w_wa.shape
            
            if w_wa.ndim > 2:
                w_wa_2d = w_wa.view(orig_shape[0], -1)
                expert_tensors_2d = [t.view(orig_shape[0], -1) for t in expert_tensors]
            else:
                w_wa_2d = w_wa
                expert_tensors_2d = expert_tensors
            
            try:
                U_wa, S_wa, V_wa = torch.linalg.svd(w_wa_2d, full_matrices=False)
                S_experts = []
                for t_2d in expert_tensors_2d:
                    _, S_exp, _ = torch.linalg.svd(t_2d, full_matrices=False)
                    S_experts.append(S_exp)
                S_avg = torch.stack(S_experts, dim=0).mean(dim=0)
                
                w_sps_2d = U_wa @ torch.diag(S_avg) @ V_wa
                sps_state_dict[key] = w_sps_2d.view(orig_shape)
            except Exception as e:
                sps_state_dict[key] = w_wa
        else:
            sps_state_dict[key] = w_wa
    return sps_state_dict

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

def analyze_spectra(expert_state_dicts, wa_state_dict, sps_state_dict):
    target_key = "resnet.layer4.1.conv2.weight"
    orig_shape = wa_state_dict[target_key].shape
    
    def get_sv(tensor):
        t_2d = tensor.view(orig_shape[0], -1)
        _, S, _ = torch.linalg.svd(t_2d, full_matrices=False)
        return S.cpu().numpy()
        
    sv_experts = [get_sv(d[target_key]) for d in expert_state_dicts]
    sv_wa = get_sv(wa_state_dict[target_key])
    sv_sps = get_sv(sps_state_dict[target_key])
    
    plt.figure(figsize=(8, 5))
    for i, sv in enumerate(sv_experts):
        plt.plot(sv, label=f"Expert {i+1}", alpha=0.5, linestyle="--")
    plt.plot(sv_wa, label="Weight Averaging (WA)", color="red", linewidth=2)
    plt.plot(sv_sps, label="SPS-Merge (Ours)", color="green", linewidth=2)
    plt.title(f"Singular Value Spectrum of {target_key} (Strong Experts)")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.savefig("singular_value_spectra_strong.png", dpi=300)
    plt.close()
    
    return {
        "mean_sv_experts": [float(sv.mean()) for sv in sv_experts],
        "mean_sv_wa": float(sv_wa.mean()),
        "mean_sv_sps": float(sv_sps.mean())
    }

def main():
    os.makedirs("experts_strong", exist_ok=True)
    data_dict = get_datasets()
    
    mnist_train_loader = DataLoader(data_dict["mnist"]["train"], batch_size=128, shuffle=True)
    fmnist_train_loader = DataLoader(data_dict["fmnist"]["train"], batch_size=128, shuffle=True)
    cifar_train_loader = DataLoader(data_dict["cifar10"]["train"], batch_size=128, shuffle=True)
    
    mnist_test_loader = DataLoader(data_dict["mnist"]["test"], batch_size=128, shuffle=False)
    fmnist_test_loader = DataLoader(data_dict["fmnist"]["test"], batch_size=128, shuffle=False)
    cifar_test_loader = DataLoader(data_dict["cifar10"]["test"], batch_size=128, shuffle=False)
    
    prog_path = "experts/progenitor.pt"
    if not os.path.exists(prog_path):
        model_init = MergibleResNet18().to(device)
        torch.save(model_init.state_dict(), prog_path)
    
    expert_paths = {
        "mnist": "experts_strong/expert_mnist.pt",
        "fmnist": "experts_strong/expert_fmnist.pt",
        "cifar10": "experts_strong/expert_cifar10.pt"
    }
    
    # Train strong experts if they don't exist
    for task, path in expert_paths.items():
        if not os.path.exists(path):
            print(f"\nTraining strong expert model for {task}...")
            m = MergibleResNet18().to(device)
            m.load_state_dict(torch.load(prog_path, map_location=device))

            loader = {
                "mnist": mnist_train_loader,
                "fmnist": fmnist_train_loader,
                "cifar10": cifar_train_loader
            }[task]

            # Train for 10 epochs, learning rate 1e-3 (10x larger!)
            m = train_model(m, loader, epochs=10, lr=1e-3)
            torch.save(m.state_dict(), path)
            print(f"Strong expert model for {task} saved.")
        else:
            print(f"Found existing strong expert model for {task} at {path}.")

    expert_state_dicts = []
    for task, path in expert_paths.items():
        expert_state_dicts.append(torch.load(path, map_location=device))

    # Evaluate standalone
    experts_eval = {}
    test_loaders = {
        "mnist": mnist_test_loader,
        "fmnist": fmnist_test_loader,
        "cifar10": cifar_test_loader
    }
    for task, path in expert_paths.items():
        m = MergibleResNet18().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        acc = evaluate_model(m, test_loaders[task])
        experts_eval[task] = acc
        print(f"Strong Expert {task} standalone accuracy: {acc * 100:.2f}%")
    print("Strong Experts Average Accuracy:", np.mean(list(experts_eval.values())) * 100)
    
    results = {
        "experts": experts_eval,
        "experts_avg": float(np.mean(list(experts_eval.values())))
    }
    
    # 1. Evaluate uncalibrated WA
    print("\n--- Running Method: WA (Uncalibrated) ---")
    wa_state = merge_wa(expert_state_dicts)
    
    def eval_merged_state_uncalibrated(state_dict, name):
        accs = {}
        for t, loader in test_loaders.items():
            m = MergibleResNet18().to(device)
            m.load_state_dict(state_dict)
            task_idx = {"mnist": 0, "fmnist": 1, "cifar10": 2}[t]
            m.resnet.fc[1].weight.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.weight"])
            m.resnet.fc[1].bias.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.bias"])
            accs[t] = evaluate_model(m, loader)
        avg = float(np.mean(list(accs.values())))
        print(f"Evaluation of {name} (Uncalibrated):")
        for t, acc in accs.items():
            print(f"  {t}: {acc * 100:.2f}%")
        print(f"  Average: {avg * 100:.2f}%")
        return {"accs": accs, "avg": avg}

    def eval_merged_state_calibrated(state_dict, name):
        accs = {}
        for t, loader in test_loaders.items():
            m = MergibleResNet18().to(device)
            m.load_state_dict(state_dict)
            task_idx = {"mnist": 0, "fmnist": 1, "cifar10": 2}[t]
            m.resnet.fc[1].weight.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.weight"])
            m.resnet.fc[1].bias.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.bias"])
            
            task_calib_loader = DataLoader(data_dict[t]["calib"], batch_size=128, shuffle=True)
            calibrate_bn(m, task_calib_loader)
            
            accs[t] = evaluate_model(m, loader)
        avg = float(np.mean(list(accs.values())))
        print(f"Evaluation of {name} (TS-Calibrated):")
        for t, acc in accs.items():
            print(f"  {t}: {acc * 100:.2f}%")
        print(f"  Average: {avg * 100:.2f}%")
        return {"accs": accs, "avg": avg}
        
    results["wa"] = eval_merged_state_uncalibrated(wa_state, "WA")
    
    # 2. Evaluate WA + TS-BNC
    print("\n--- Running Method: WA + TS-BNC ---")
    results["wa_bnc"] = eval_merged_state_calibrated(wa_state, "WA + TS-BNC")
    
    # 3. Evaluate SPS-Merge (Ours, Data-Free)
    print("\n--- Running Method: SPS-Merge (Ours, Data-Free) ---")
    sps_state = merge_sps(expert_state_dicts, wa_state)
    results["sps"] = eval_merged_state_uncalibrated(sps_state, "SPS-Merge")
    
    # 4. Evaluate SPS-Merge + TS-BNC
    print("\n--- Running Method: SPS-Merge + TS-BNC ---")
    results["sps_bnc"] = eval_merged_state_calibrated(sps_state, "SPS-Merge + TS-BNC")
    
    # 5. Evaluate SPS-Merge + TSSS + TS-BNC (Proposed)
    print("\n--- Running Method: SPS-Merge with TSSS + TS-BNC ---")
    tsss_accs = {}
    for t, loader in test_loaders.items():
        task_idx = {"mnist": 0, "fmnist": 1, "cifar10": 2}[t]
        sps_tsss_state = merge_sps_tsss(expert_state_dicts, wa_state, task_idx)
        
        m = MergibleResNet18().to(device)
        m.load_state_dict(sps_tsss_state)
        m.resnet.fc[1].weight.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.weight"])
        m.resnet.fc[1].bias.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.bias"])
        
        task_calib_loader = DataLoader(data_dict[t]["calib"], batch_size=128, shuffle=True)
        calibrate_bn(m, task_calib_loader)
        
        tsss_accs[t] = evaluate_model(m, loader)
    avg_tsss = float(np.mean(list(tsss_accs.values())))
    print(f"Evaluation of SPS-Merge with TSSS + TS-BNC (Ours):")
    for t, acc in tsss_accs.items():
        print(f"  {t}: {acc * 100:.2f}%")
    print(f"  Average: {avg_tsss * 100:.2f}%")
    results["sps_tsss_bnc"] = {"accs": tsss_accs, "avg": avg_tsss}
    
    # Analyze spectra
    print("\n--- Analyzing Spectra ---")
    spectral_metrics = analyze_spectra(expert_state_dicts, wa_state, sps_state)
    results["spectral"] = spectral_metrics
    
    with open("results_strong.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to results_strong.json.")

if __name__ == "__main__":
    main()
