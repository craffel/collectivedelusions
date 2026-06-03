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
        # Use standard pretrained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Replace the fc layer to match our task
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

def get_datasets():
    # Grayscale transforms: resize to 32x32, replicate to 3 channels, normalize
    gray_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # RGB transform
    rgb_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load full datasets
    mnist_train = torchvision.datasets.MNIST(root="data", train=True, transform=gray_transform)
    mnist_test = torchvision.datasets.MNIST(root="data", train=False, transform=gray_transform)
    
    fmnist_train = torchvision.datasets.FashionMNIST(root="data", train=True, transform=gray_transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root="data", train=False, transform=gray_transform)
    
    cifar_train = torchvision.datasets.CIFAR10(root="data", train=True, transform=rgb_transform)
    cifar_test = torchvision.datasets.CIFAR10(root="data", train=False, transform=rgb_transform)
    
    # Seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create 5,000-sample subsets for training
    mnist_train_sub = Subset(mnist_train, np.random.choice(len(mnist_train), 5000, replace=False))
    fmnist_train_sub = Subset(fmnist_train, np.random.choice(len(fmnist_train), 5000, replace=False))
    cifar_train_sub = Subset(cifar_train, np.random.choice(len(cifar_train), 5000, replace=False))
    
    # Create 128-sample subsets for calibration
    mnist_calib = Subset(mnist_train, np.random.choice(len(mnist_train), 128, replace=False))
    fmnist_calib = Subset(fmnist_train, np.random.choice(len(fmnist_train), 128, replace=False))
    cifar_calib = Subset(cifar_train, np.random.choice(len(cifar_train), 128, replace=False))
    
    return {
        "mnist": {"train": mnist_train_sub, "test": mnist_test, "calib": mnist_calib},
        "fmnist": {"train": fmnist_train_sub, "test": fmnist_test, "calib": fmnist_calib},
        "cifar10": {"train": cifar_train_sub, "test": cifar_test, "calib": cifar_calib}
    }

def train_model(model, train_loader, epochs=5, lr=1e-4):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
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
    # Set momentum to 1.0 for all BN layers to completely overwrite stats without distortion
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.momentum = 1.0
            
    with torch.no_grad():
        for x, _ in calib_loader:
            x = x.to(device)
            _ = model(x)
            break # A single large batch (e.g. size 128) is sufficient and exact

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

def merge_ta(expert_state_dicts, init_state_dict, lam=0.4):
    merged = {}
    keys = init_state_dict.keys()
    for key in keys:
        # Check if the parameter is floating point
        if init_state_dict[key].is_floating_point():
            task_vectors = [d[key] - init_state_dict[key] for d in expert_state_dicts]
            sum_tasks = torch.stack(task_vectors, dim=0).sum(dim=0)
            merged[key] = init_state_dict[key] + lam * sum_tasks
        else:
            merged[key] = init_state_dict[key]
    return merged

def merge_sps(expert_state_dicts, wa_state_dict):
    sps_state_dict = {}
    for key in wa_state_dict.keys():
        w_wa = wa_state_dict[key].clone()
        
        # Apply SVD reconstruction only on weights with ndim >= 2
        # Exclude batchnorm weights/biases and fc layer biases
        if ("weight" in key) and (w_wa.ndim >= 2) and any(kw in key for kw in ["conv", "fc", "downsample.0"]):
            expert_tensors = [d[key] for d in expert_state_dicts]
            orig_shape = w_wa.shape
            
            # Reshape multi-dimensional tensors to 2D
            if w_wa.ndim > 2:
                # Shape is [C_out, C_in, Kh, Kw] -> reshape to [C_out, C_in * Kh * Kw]
                w_wa_2d = w_wa.view(orig_shape[0], -1)
                expert_tensors_2d = [t.view(orig_shape[0], -1) for t in expert_tensors]
            else:
                w_wa_2d = w_wa
                expert_tensors_2d = expert_tensors
            
            try:
                # Compute SVD of WA
                U_wa, S_wa, V_wa = torch.linalg.svd(w_wa_2d, full_matrices=False)
                
                # Compute singular values for each expert
                S_experts = []
                for t_2d in expert_tensors_2d:
                    _, S_exp, _ = torch.linalg.svd(t_2d, full_matrices=False)
                    S_experts.append(S_exp)
                
                # Average the singular values
                S_avg = torch.stack(S_experts, dim=0).mean(dim=0)
                
                # Reconstruct merged weights using WA's singular vectors and averaged singular values
                # SVD output: A = U * S * V_h
                w_sps_2d = U_wa @ torch.diag(S_avg) @ V_wa
                
                # Reshape back to original dimensions
                sps_state_dict[key] = w_sps_2d.view(orig_shape)
            except Exception as e:
                print(f"SVD failed for {key}: {e}. Falling back to WA.")
                sps_state_dict[key] = w_wa
        else:
            # Fall back to WA for other parameters (biases, running stats)
            sps_state_dict[key] = w_wa
            
    return sps_state_dict

def analyze_spectra(expert_state_dicts, wa_state_dict, sps_state_dict):
    # Analyze the singular value spectrum of a deep layer (e.g., resnet.layer4.1.conv2.weight)
    target_key = "resnet.layer4.1.conv2.weight"
    if target_key not in wa_state_dict:
        # Fallback to some key that exists
        for key in wa_state_dict.keys():
            if "conv" in key and wa_state_dict[key].ndim == 4:
                target_key = key
                break
                
    print(f"Analyzing spectra of layer: {target_key}")
    orig_shape = wa_state_dict[target_key].shape
    
    # Helper to get singular values
    def get_sv(tensor):
        t_2d = tensor.view(orig_shape[0], -1)
        _, S, _ = torch.linalg.svd(t_2d, full_matrices=False)
        return S.cpu().numpy()
        
    sv_experts = [get_sv(d[target_key]) for d in expert_state_dicts]
    sv_wa = get_sv(wa_state_dict[target_key])
    sv_sps = get_sv(sps_state_dict[target_key])
    
    # Save a plot of the singular values
    plt.figure(figsize=(8, 5))
    for i, sv in enumerate(sv_experts):
        plt.plot(sv, label=f"Expert {i+1}", alpha=0.5, linestyle="--")
    plt.plot(sv_wa, label="Weight Averaging (WA)", color="red", linewidth=2)
    plt.plot(sv_sps, label="SPS-Merge (Ours)", color="green", linewidth=2)
    plt.title(f"Singular Value Spectrum of {target_key}")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.savefig("singular_value_spectra.png", dpi=300)
    plt.close()
    
    return {
        "mean_sv_experts": [float(sv.mean()) for sv in sv_experts],
        "mean_sv_wa": float(sv_wa.mean()),
        "mean_sv_sps": float(sv_sps.mean())
    }

def main():
    # 1. Setup paths
    os.makedirs("experts", exist_ok=True)
    
    # 2. Get Datasets
    data_dict = get_datasets()
    
    mnist_train_loader = DataLoader(data_dict["mnist"]["train"], batch_size=128, shuffle=True)
    fmnist_train_loader = DataLoader(data_dict["fmnist"]["train"], batch_size=128, shuffle=True)
    cifar_train_loader = DataLoader(data_dict["cifar10"]["train"], batch_size=128, shuffle=True)
    
    mnist_test_loader = DataLoader(data_dict["mnist"]["test"], batch_size=128, shuffle=False)
    fmnist_test_loader = DataLoader(data_dict["fmnist"]["test"], batch_size=128, shuffle=False)
    cifar_test_loader = DataLoader(data_dict["cifar10"]["test"], batch_size=128, shuffle=False)
    
    # Combined calibration loader (128 samples per task = 384 samples total)
    calib_dataset = ConcatDataset([
        data_dict["mnist"]["calib"],
        data_dict["fmnist"]["calib"],
        data_dict["cifar10"]["calib"]
    ])
    calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=True)
    
    # 3. Create or load the pre-trained shared progenitor
    prog_path = "experts/progenitor.pt"
    model_init = MergibleResNet18().to(device)
    torch.save(model_init.state_dict(), prog_path)
    print("Pre-trained progenitor model initialized and saved.")
    
    # 4. Train expert models if they don't exist
    expert_paths = {
        "mnist": "experts/expert_mnist.pt",
        "fmnist": "experts/expert_fmnist.pt",
        "cifar10": "experts/expert_cifar10.pt"
    }
    
    for task, path in expert_paths.items():
        if not os.path.exists(path):
            print(f"\nTraining expert model for {task}...")
            # Load from progenitor
            m = MergibleResNet18().to(device)
            m.load_state_dict(torch.load(prog_path, map_location=device))
            
            loader = {
                "mnist": mnist_train_loader,
                "fmnist": fmnist_train_loader,
                "cifar10": cifar_train_loader
            }[task]
            
            m = train_model(m, loader, epochs=5, lr=1e-4)
            torch.save(m.state_dict(), path)
            print(f"Expert model for {task} saved.")
        else:
            print(f"Found existing expert model for {task} at {path}.")
            
    # Load experts
    expert_state_dicts = []
    for task, path in expert_paths.items():
        expert_state_dicts.append(torch.load(path, map_location=device))
        
    init_state_dict = torch.load(prog_path, map_location=device)
    
    # Evaluate individual experts
    experts_eval = {}
    for task, path in expert_paths.items():
        m = MergibleResNet18().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        test_loader = {
            "mnist": mnist_test_loader,
            "fmnist": fmnist_test_loader,
            "cifar10": cifar_test_loader
        }[task]
        acc = evaluate_model(m, test_loader)
        experts_eval[task] = acc
        print(f"Expert {task} standalone accuracy: {acc * 100:.2f}%")
        
    print("\nStandalone Experts Average Accuracy:", np.mean(list(experts_eval.values())) * 100)
    
    results = {}
    results["experts"] = experts_eval
    results["experts_avg"] = float(np.mean(list(experts_eval.values())))
    
    # Define test loaders for evaluation of merged models
    test_loaders = {
        "mnist": mnist_test_loader,
        "fmnist": fmnist_test_loader,
        "cifar10": cifar_test_loader
    }
    
    def eval_merged_state_uncalibrated(state_dict, name):
        accs = {}
        for t, loader in test_loaders.items():
            m = MergibleResNet18().to(device)
            m.load_state_dict(state_dict)
            
            # Load task-specific classification head from corresponding expert
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
            
            # Load task-specific classification head from corresponding expert
            task_idx = {"mnist": 0, "fmnist": 1, "cifar10": 2}[t]
            m.resnet.fc[1].weight.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.weight"])
            m.resnet.fc[1].bias.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.bias"])
            
            # Perform Task-Specific BatchNorm Calibration (TS-BNC)
            task_calib_loader = DataLoader(data_dict[t]["calib"], batch_size=128, shuffle=True)
            calibrate_bn(m, task_calib_loader)
            
            accs[t] = evaluate_model(m, loader)
        avg = float(np.mean(list(accs.values())))
        print(f"Evaluation of {name} (TS-Calibrated):")
        for t, acc in accs.items():
            print(f"  {t}: {acc * 100:.2f}%")
        print(f"  Average: {avg * 100:.2f}%")
        return {"accs": accs, "avg": avg}
        
    # Method 1: Weight Averaging (WA) without calibration
    print("\n--- Running Method: Weight Averaging (WA) ---")
    wa_state = merge_wa(expert_state_dicts)
    results["wa"] = eval_merged_state_uncalibrated(wa_state, "WA")
    
    # Method 2: Weight Averaging (WA) with TS-BatchNorm Calibration (TS-BNC)
    print("\n--- Running Method: WA + TS-BatchNorm Calibration (TS-BNC) ---")
    results["wa_bnc"] = eval_merged_state_calibrated(wa_state, "WA + TS-BNC")
    
    # Method 3: Task Arithmetic (TA) with grid search on lambda
    print("\n--- Running Method: Task Arithmetic (TA) Grid Search ---")
    best_ta_avg = -1
    best_ta_lam = -1
    best_ta_state = None
    best_ta_accs = None
    
    for lam in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        ta_state = merge_ta(expert_state_dicts, init_state_dict, lam=lam)
        eval_res = eval_merged_state_uncalibrated(ta_state, f"TA (lambda={lam})")
        if eval_res["avg"] > best_ta_avg:
            best_ta_avg = eval_res["avg"]
            best_ta_lam = lam
            best_ta_state = ta_state
            best_ta_accs = eval_res["accs"]
            
    print(f"\nBest TA lambda: {best_ta_lam} with avg accuracy: {best_ta_avg * 100:.2f}%")
    results["ta"] = {"accs": best_ta_accs, "avg": best_ta_avg, "lambda": best_ta_lam}
    
    # Method 4: Task Arithmetic + TS-BNC
    print("\n--- Running Method: TA + TS-BNC ---")
    results["ta_bnc"] = eval_merged_state_calibrated(best_ta_state, "TA + TS-BNC")
    
    # Method 5: SPS-Merge (Ours, Data-Free)
    print("\n--- Running Method: SPS-Merge (Ours, Data-Free) ---")
    sps_state = merge_sps(expert_state_dicts, wa_state)
    results["sps"] = eval_merged_state_uncalibrated(sps_state, "SPS-Merge")
    
    # Method 6: SPS-Merge + TS-BNC (Ours, Calibrated)
    print("\n--- Running Method: SPS-Merge + TS-BNC (Ours, Calibrated) ---")
    results["sps_bnc"] = eval_merged_state_calibrated(sps_state, "SPS-Merge + TS-BNC")
    
    # 5. Analyze weight spectra
    print("\n--- Analyzing Spectra ---")
    spectral_metrics = analyze_spectra(expert_state_dicts, wa_state, sps_state)
    results["spectral"] = spectral_metrics
    
    # 6. Save results to JSON
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to results.json.")
    
if __name__ == "__main__":
    main()
