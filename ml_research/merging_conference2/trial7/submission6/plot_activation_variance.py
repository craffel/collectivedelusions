import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# Define device (prefer CPU since it's just a forward pass on a few batches)
device = torch.device("cpu")
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
    
    # 128 samples for forward pass
    mnist_calib = Subset(mnist_train, np.random.choice(len(mnist_train), 128, replace=False))
    fmnist_calib = Subset(fmnist_train, np.random.choice(len(fmnist_train), 128, replace=False))
    cifar_calib = Subset(cifar_train, np.random.choice(len(cifar_train), 128, replace=False))
    
    return {
        "mnist": {"calib": mnist_calib},
        "fmnist": {"calib": fmnist_calib},
        "cifar10": {"calib": cifar_calib}
    }

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

class ActivationHook:
    def __init__(self):
        self.outputs = []
        
    def hook_fn(self, module, input, output):
        # We compute the variance over all dimensions except channel dimension if preferred,
        # but to get a single scalar variance for the layer, we compute over all dimensions of the output tensor.
        self.outputs.append(output.detach().clone())

def get_layer_variances(model, calib_loader):
    # Register hooks on all Conv2d layers sequentially
    hooks = []
    hook_objects = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hook_obj = ActivationHook()
            hook_handle = module.register_forward_hook(hook_obj.hook_fn)
            hooks.append(hook_handle)
            hook_objects.append((name, hook_obj))
            
    # Run a single batch through the model
    model.eval()
    with torch.no_grad():
        for x, _ in calib_loader:
            x = x.to(device)
            _ = model(x)
            break
            
    # Remove hooks
    for h in hooks:
        h.remove()
        
    # Extract variances
    variances = []
    names = []
    for name, hook_obj in hook_objects:
        # hook_obj.outputs contains outputs of this layer
        if len(hook_obj.outputs) > 0:
            out_tensor = hook_obj.outputs[0]
            var = torch.var(out_tensor).item()
            variances.append(var)
            names.append(name)
            
    return names, variances

def main():
    data_dict = get_datasets()
    
    expert_paths = {
        "mnist": "experts_strong/expert_mnist.pt",
        "fmnist": "experts_strong/expert_fmnist.pt",
        "cifar10": "experts_strong/expert_cifar10.pt"
    }
    
    expert_state_dicts = []
    for task, path in expert_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert model for {task} not found at {path}.")
        expert_state_dicts.append(torch.load(path, map_location=device))
        
    print("Loaded expert models successfully.")
    wa_state = merge_wa(expert_state_dicts)
    
    # We will evaluate activation variance for each task
    tasks = ["mnist", "fmnist", "cifar10"]
    task_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    
    all_results = {}
    
    for task_idx, (t, t_name) in enumerate(zip(tasks, task_names)):
        print(f"\n--- Analyzing Activation Variances for {t_name} ---")
        calib_loader = DataLoader(data_dict[t]["calib"], batch_size=128, shuffle=False)
        
        # 1. Standalone Expert
        m_expert = MergibleResNet18().to(device)
        m_expert.load_state_dict(expert_state_dicts[task_idx])
        layer_names, var_expert = get_layer_variances(m_expert, calib_loader)
        
        # 2. WA (Uncalibrated)
        m_wa_uncal = MergibleResNet18().to(device)
        m_wa_uncal.load_state_dict(wa_state)
        m_wa_uncal.resnet.fc[1].weight.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.weight"])
        m_wa_uncal.resnet.fc[1].bias.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.bias"])
        _, var_wa_uncal = get_layer_variances(m_wa_uncal, calib_loader)
        
        # 3. WA + TS-BNC (Calibrated)
        m_wa_cal = MergibleResNet18().to(device)
        m_wa_cal.load_state_dict(wa_state)
        m_wa_cal.resnet.fc[1].weight.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.weight"])
        m_wa_cal.resnet.fc[1].bias.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.bias"])
        calibrate_bn(m_wa_cal, calib_loader)
        _, var_wa_cal = get_layer_variances(m_wa_cal, calib_loader)
        
        # 4. SPS-Merge + TSSS (Uncalibrated)
        sps_state = merge_sps_tsss(expert_state_dicts, wa_state, task_idx)
        m_sps_uncal = MergibleResNet18().to(device)
        m_sps_uncal.load_state_dict(sps_state)
        m_sps_uncal.resnet.fc[1].weight.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.weight"])
        m_sps_uncal.resnet.fc[1].bias.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.bias"])
        _, var_sps_uncal = get_layer_variances(m_sps_uncal, calib_loader)
        
        # 5. SPS-Merge + TSSS + TS-BNC (Ours, Calibrated)
        m_sps_cal = MergibleResNet18().to(device)
        m_sps_cal.load_state_dict(sps_state)
        m_sps_cal.resnet.fc[1].weight.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.weight"])
        m_sps_cal.resnet.fc[1].bias.data.copy_(expert_state_dicts[task_idx]["resnet.fc.1.bias"])
        calibrate_bn(m_sps_cal, calib_loader)
        _, var_sps_cal = get_layer_variances(m_sps_cal, calib_loader)
        
        all_results[t] = {
            "layer_names": layer_names,
            "expert": var_expert,
            "wa_uncal": var_wa_uncal,
            "wa_cal": var_wa_cal,
            "sps_uncal": var_sps_uncal,
            "sps_cal": var_sps_cal
        }
        
        # Print a small summary comparing first and last conv layers
        print(f"First Conv layer variance:")
        print(f"  Expert: {var_expert[0]:.6f} | WA (Uncal): {var_wa_uncal[0]:.6f} | WA (Cal): {var_wa_cal[0]:.6f} | Ours (Cal): {var_sps_cal[0]:.6f}")
        print(f"Deepest Conv layer variance:")
        print(f"  Expert: {var_expert[-1]:.6f} | WA (Uncal): {var_wa_uncal[-1]:.6f} | WA (Cal): {var_wa_cal[-1]:.6f} | Ours (Cal): {var_sps_cal[-1]:.6f}")
        
    # Now let's plot a beautiful combined figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (t, t_name) in enumerate(zip(tasks, task_names)):
        res = all_results[t]
        x = np.arange(len(res["layer_names"]))
        
        axes[idx].plot(x, res["expert"], label="Standalone Expert", color="black", linestyle="--", linewidth=2)
        axes[idx].plot(x, res["wa_uncal"], label="WA (Uncalibrated)", color="red", alpha=0.7)
        axes[idx].plot(x, res["sps_uncal"], label="SPS-Merge+TSSS (Uncal)", color="orange", alpha=0.7)
        axes[idx].plot(x, res["wa_cal"], label="WA + TS-BNC (Calibrated)", color="blue", linewidth=1.5)
        axes[idx].plot(x, res["sps_cal"], label="SPS-Merge+TSSS+TS-BNC (Ours)", color="green", linewidth=2)
        
        axes[idx].set_title(f"{t_name} Activation Variance", fontsize=12)
        axes[idx].set_xlabel("Layer Index (Sequential Conv Layers)", fontsize=10)
        axes[idx].set_ylabel("Variance", fontsize=10)
        axes[idx].set_yscale("log")
        axes[idx].grid(True, which="both", linestyle=":", alpha=0.5)
        
    axes[2].legend(fontsize=9, loc="lower left")
    plt.tight_layout()
    plt.savefig("activation_variance_comparison.png", dpi=300)
    plt.close()
    print("\nPlot saved as activation_variance_comparison.png!")
    
    # Save raw results to json
    with open("results_activation_variance.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("Results saved to results_activation_variance.json.")

if __name__ == "__main__":
    main()
