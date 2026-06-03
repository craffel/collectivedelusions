import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import copy
import numpy as np

# Import helper functions from run_experiments
from run_experiments import (
    get_resnet18_expert,
    fold_resnet18,
    get_calibration_layer_names,
    get_preceding_conv_name,
    set_module_by_name,
    get_module_by_name,
    collect_activations,
    evaluate_model
)

def run_selective_sscc(folded_model, cal_layers, selective_layers, expert_acts, merged_acts, k_size, gamma_max, device):
    """
    Runs SSCC calibration only on the selected layers.
    """
    cal_model = copy.deepcopy(folded_model)
    with torch.no_grad():
        for layer in cal_layers:
            if layer not in selective_layers:
                continue
                
            # 1. Compute scale map identical to FDSA
            exp_spectrals = []
            for name in expert_acts.keys():
                X = expert_acts[name][layer].to(device)
                spect = torch.fft.fft2(X, dim=(2, 3)).abs().mean(dim=(0, 1))
                exp_spectrals.append(spect.cpu().numpy())
            target_spect = torch.tensor(np.mean(exp_spectrals, axis=0))
            
            X_merged = merged_acts[layer].to(device)
            merged_spect = torch.fft.fft2(X_merged, dim=(2, 3)).abs().mean(dim=(0, 1)).cpu()
            
            scale_map = target_spect / (merged_spect + 1e-5)
            scale_map = torch.clamp(scale_map, 1.0 / gamma_max, gamma_max)
            
            # 2. Get 2D spatial kernel via IFFT2D
            k_full = torch.fft.ifft2(scale_map).real
            k_centered = torch.fft.fftshift(k_full)
            
            # Truncate to k_size local filter
            H_map, W_map = scale_map.shape
            if H_map < k_size or W_map < k_size:
                k_trunc = torch.zeros(k_size, k_size, device=k_centered.device)
                h_offset = (k_size - H_map) // 2
                w_offset = (k_size - W_map) // 2
                k_trunc[h_offset:h_offset+H_map, w_offset:w_offset+W_map] = k_centered
            else:
                start_h = H_map // 2 - k_size // 2
                start_w = W_map // 2 - k_size // 2
                k_trunc = k_centered[start_h:start_h + k_size, start_w:start_w + k_size]
            
            # Normalize k_trunc
            dc_target = scale_map[0, 0].item()
            k_trunc_sum = k_trunc.sum().item()
            if abs(k_trunc_sum) > 1e-8:
                k_trunc = k_trunc * (dc_target / k_trunc_sum)
            else:
                k_trunc = torch.zeros_like(k_trunc)
                k_trunc[k_size//2, k_size//2] = dc_target
                
            # Convert to depthwise filter weight for PyTorch
            C = merged_acts[layer].shape[1]
            w_dw = k_trunc.view(1, 1, k_size, k_size).repeat(C, 1, 1, 1).to(device)
            
            # 3. Fuse this depthwise filter back into the preceding Conv2d
            conv_name = get_preceding_conv_name(layer)
            conv = get_module_by_name(cal_model, conv_name)
            
            w1 = conv.weight
            b1 = conv.bias
            
            C_out, C_in, k1, _ = w1.shape
            k2 = k_size
            p1 = conv.padding[0]
            p2 = k2 // 2
            
            # Vectorized fusion
            w1_reshaped = w1.view(1, C_out * C_in, k1, k1)
            w_dw_expanded = w_dw.unsqueeze(1).repeat(1, C_in, 1, 1, 1).view(C_out * C_in, 1, k2, k2)
            kernel_flipped = torch.flip(w_dw_expanded, dims=[2, 3])
            
            fused_conv = F.conv2d(w1_reshaped, kernel_flipped, padding=k2 - 1, groups=C_out * C_in)
            w_fused = fused_conv.view(C_out, C_in, k1 + k2 - 1, k1 + k2 - 1)
            
            b_fused = b1 * w_dw.sum(dim=(2, 3)).squeeze(1)
                
            new_conv = nn.Conv2d(
                C_in, C_out,
                kernel_size=k1 + k2 - 1,
                padding=p1 + p2,
                stride=conv.stride,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True
            ).to(device)
            new_conv.weight.data.copy_(w_fused)
            new_conv.bias.data.copy_(b_fused)
            
            set_module_by_name(cal_model, conv_name, new_conv)
            
    return cal_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running layer ablation on {device}")
    
    gray_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    color_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load test sets
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=gray_transform)
    fmnist_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=gray_transform)
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=color_transform)
    
    task_test_loaders = {
        "mnist": DataLoader(mnist_test, batch_size=256, shuffle=False, num_workers=2),
        "fmnist": DataLoader(fmnist_test, batch_size=256, shuffle=False, num_workers=2),
        "cifar10": DataLoader(cifar_test, batch_size=256, shuffle=False, num_workers=2)
    }
    
    # Load calibration datasets
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=gray_transform)
    fmnist_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=gray_transform)
    cifar_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=color_transform)
    
    N = 64
    cal_indices = list(range(5000, 5000 + N))
    mnist_cal = Subset(mnist_train, cal_indices)
    fmnist_cal = Subset(fmnist_train, cal_indices)
    cifar_cal = Subset(cifar_train, cal_indices)
    
    cal_loaders = {
        "mnist": DataLoader(mnist_cal, batch_size=64, shuffle=False),
        "fmnist": DataLoader(fmnist_cal, batch_size=64, shuffle=False),
        "cifar10": DataLoader(cifar_cal, batch_size=64, shuffle=False)
    }
    
    # Load weights
    base_sd = torch.load("base_model.pt", map_location=device)
    mnist_sd = torch.load("mnist_expert.pt", map_location=device)
    fmnist_sd = torch.load("fmnist_expert.pt", map_location=device)
    cifar_sd = torch.load("cifar10_expert.pt", map_location=device)
    
    # Construct WA model
    merged_model = get_resnet18_expert().to(device)
    fused_sd = copy.deepcopy(base_sd)
    for key in base_sd.keys():
        if "fc" not in key:
            fused_sd[key] = (mnist_sd[key] + fmnist_sd[key] + cifar_sd[key]) / 3.0
    merged_model.load_state_dict(fused_sd)
    folded_merged = fold_resnet18(merged_model)
    
    # Folded experts for stats
    expert_models = {}
    for name, sd in [("mnist", mnist_sd), ("fmnist", fmnist_sd), ("cifar10", cifar_sd)]:
        exp_m = get_resnet18_expert().to(device)
        exp_m.load_state_dict(sd)
        expert_models[name] = fold_resnet18(exp_m)
        
    cal_layers = get_calibration_layer_names()
    
    print("Collecting expert calibration statistics...")
    expert_acts = {}
    for name, exp_m in expert_models.items():
        expert_acts[name] = collect_activations(exp_m, cal_loaders[name], cal_layers, device)
        
    print("Collecting merged model calibration statistics...")
    merged_acts = {layer: [] for layer in cal_layers}
    for name, loader in cal_loaders.items():
        acts = collect_activations(folded_merged, loader, cal_layers, device)
        for layer in cal_layers:
            merged_acts[layer].append(acts[layer])
    for layer in cal_layers:
        merged_acts[layer] = torch.cat(merged_acts[layer], dim=0)
        
    # Categories of layers
    shallow_layers = [l for l in cal_layers if "layer1" in l or "layer2" in l or l == "bn1"]
    deep_layers = [l for l in cal_layers if "layer3" in l or "layer4" in l]
    
    print(f"\nTotal calibration layers: {len(cal_layers)}")
    print(f"Shallow layers: {len(shallow_layers)} ({shallow_layers})")
    print(f"Deep layers: {len(deep_layers)} ({deep_layers})")
    
    configurations = {
        "None (Uncalibrated)": [],
        "All Layers": cal_layers,
        "Shallow Layers Only": shallow_layers,
        "Deep Layers Only": deep_layers
    }
    
    k_size = 3
    gamma_max = 5.0
    
    print("\nEvaluating configurations...")
    for name, selective_list in configurations.items():
        if len(selective_list) == 0:
            cal_model = copy.deepcopy(folded_merged)
        else:
            cal_model = run_selective_sscc(
                folded_merged, cal_layers, selective_list,
                expert_acts, merged_acts, k_size, gamma_max, device
            )
        accs = evaluate_model(cal_model, task_test_loaders, device)
        avg_acc = np.mean(list(accs.values()))
        print(f"  {name:<25}: MNIST={accs['mnist']:.2f}%, FMNIST={accs['fmnist']:.2f}%, CIFAR={accs['cifar10']:.2f}%, Avg={avg_acc:.2f}%")

if __name__ == '__main__':
    main()
