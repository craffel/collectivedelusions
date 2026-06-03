import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import os
import copy
import numpy as np

torch.backends.cudnn.enabled = False

def get_resnet18_expert():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 10)
    return model

def fold_batchnorm(conv, bn):
    with torch.no_grad():
        w_conv = conv.weight
        b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=w_conv.device)
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        w_bn = bn.weight
        b_bn = bn.bias
        
        scale = w_bn / torch.sqrt(var + eps)
        w_folded = w_conv * scale.view(-1, 1, 1, 1)
        b_folded = (b_conv - mean) * scale + b_bn
        
        folded_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True
        ).to(w_conv.device)
        folded_conv.weight.copy_(w_folded)
        folded_conv.bias.copy_(b_folded)
        return folded_conv

def fold_resnet18(model):
    folded_model = copy.deepcopy(model)
    folded_model.eval()
    
    folded_model.conv1 = fold_batchnorm(folded_model.conv1, folded_model.bn1)
    folded_model.bn1 = nn.Identity()
    
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(folded_model, layer_name)
        for block in layer:
            block.conv1 = fold_batchnorm(block.conv1, block.bn1)
            block.bn1 = nn.Identity()
            block.conv2 = fold_batchnorm(block.conv2, block.bn2)
            block.bn2 = nn.Identity()
            if block.downsample is not None:
                block.downsample[0] = fold_batchnorm(block.downsample[0], block.downsample[1])
                block.downsample[1] = nn.Identity()
    return folded_model

# Names of BatchNorm modules in the original model
def get_calibration_layer_names():
    names = ["bn1"]
    for l in [1, 2, 3, 4]:
        for b in [0, 1]:
            names.append(f"layer{l}.{b}.bn1")
            names.append(f"layer{l}.{b}.bn2")
            if b == 0 and l > 1:
                names.append(f"layer{l}.0.downsample.1")
    return names

# Find corresponding preceding Conv2d layer in folded model
def get_preceding_conv_name(bn_name):
    if bn_name == "bn1":
        return "conv1"
    parts = bn_name.split(".")
    # e.g., layer1.0.bn1 -> layer1.0.conv1
    if parts[-1] == "bn1":
        return f"{parts[0]}.{parts[1]}.conv1"
    elif parts[-1] == "bn2":
        return f"{parts[0]}.{parts[1]}.conv2"
    elif parts[-1] == "1" and parts[-2] == "downsample":
        return f"{parts[0]}.{parts[1]}.downsample.0"
    return None

def set_module_by_name(model, name, new_module):
    parts = name.split(".")
    curr = model
    for part in parts[:-1]:
        if part.isdigit():
            curr = curr[int(part)]
        else:
            curr = getattr(curr, part)
    if parts[-1].isdigit():
        curr[int(parts[-1])] = new_module
    else:
        setattr(curr, parts[-1], new_module)

def get_module_by_name(model, name):
    parts = name.split(".")
    curr = model
    for part in parts:
        if part.isdigit():
            curr = curr[int(part)]
        else:
            curr = getattr(curr, part)
    return curr

def collect_activations(model, dataloader, target_layers, device):
    model.eval()
    activations = {layer: [] for layer in target_layers}
    hooks = []
    
    def get_hook(layer_name):
        def hook(module, input, output):
            activations[layer_name].append(output.detach().cpu())
        return hook
        
    for layer in target_layers:
        module = get_module_by_name(model, layer)
        hooks.append(module.register_forward_hook(get_hook(layer)))
        
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)
            
    for hook in hooks:
        hook.remove()
        
    for layer in target_layers:
        activations[layer] = torch.cat(activations[layer], dim=0)
    return activations

# Frequency domain magnitude scaling hook class
class FDSAHook:
    def __init__(self, scale_map, gamma_max=5.0):
        self.scale_map = scale_map.cuda()
        self.gamma_max = gamma_max
        
    def __call__(self, module, input, output):
        # output shape: (B, C, H, W)
        # Apply 2D FFT along spatial dimensions
        fft_out = torch.fft.fft2(output, dim=(2, 3))
        # Multiply magnitudes by scale_map, keep phase
        scaled_fft = fft_out * self.scale_map.unsqueeze(0) # scale_map shape: (H, W) or (C, H, W)
        # Apply inverse FFT
        cal_out = torch.fft.ifft2(scaled_fft, dim=(2, 3)).real
        return cal_out

def evaluate_model(model, task_loaders, device):
    model.eval()
    accuracies = {}
    with torch.no_grad():
        for name, test_loader in task_loaders.items():
            # Set the task-specific classification head
            expert_sd = torch.load(f"{name}_expert.pt", map_location=device)
            model.fc.weight.copy_(expert_sd["fc.weight"])
            model.fc.bias.copy_(expert_sd["fc.bias"])
            
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            accuracies[name] = 100.0 * correct / total
    return accuracies

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiments on {device}")
    
    # Define transforms
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
    
    # Download test sets
    print("Loading test datasets...")
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=gray_transform)
    fmnist_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=gray_transform)
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=color_transform)
    
    task_test_loaders = {
        "mnist": DataLoader(mnist_test, batch_size=256, shuffle=False, num_workers=2),
        "fmnist": DataLoader(fmnist_test, batch_size=256, shuffle=False, num_workers=2),
        "cifar10": DataLoader(cifar_test, batch_size=256, shuffle=False, num_workers=2)
    }
    
    # Load raw datasets for calibration
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=gray_transform)
    fmnist_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=gray_transform)
    cifar_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=color_transform)
    
    # Target calibration layers (BatchNorms in folded model, which are nn.Identity)
    cal_layers = get_calibration_layer_names()
    
    # We will sweep N
    N_sweeps = [16, 64, 128]
    merge_methods = ["WA", "TA"] # Weight Averaging and Task Arithmetic
    cal_methods = ["None", "SP-TAAC", "TAAC", "FDSA", "SSCC"]
    
    results = []
    
    for N in N_sweeps:
        print(f"\n====================== N = {N} ======================")
        # Construct calibration data loaders
        # Use indices 5,000 to 5,000+N (disjoint from fine-tuning)
        cal_indices = list(range(5000, 5000 + N))
        mnist_cal = Subset(mnist_train, cal_indices)
        fmnist_cal = Subset(fmnist_train, cal_indices)
        cifar_cal = Subset(cifar_train, cal_indices)
        
        cal_loaders = {
            "mnist": DataLoader(mnist_cal, batch_size=64, shuffle=False),
            "fmnist": DataLoader(fmnist_cal, batch_size=64, shuffle=False),
            "cifar10": DataLoader(cifar_cal, batch_size=64, shuffle=False)
        }
        
        # Load expert and base models to perform merging
        base_sd = torch.load("base_model.pt", map_location=device)
        mnist_sd = torch.load("mnist_expert.pt", map_location=device)
        fmnist_sd = torch.load("fmnist_expert.pt", map_location=device)
        cifar_sd = torch.load("cifar10_expert.pt", map_location=device)
        
        for merge_mode in merge_methods:
            print(f"\n--- Merge Mode: {merge_mode} ---")
            
            # 1. Merge models
            merged_model = get_resnet18_expert().to(device)
            if merge_mode == "WA":
                # Average expert state dicts (backbone only, fc is task-specific)
                fused_sd = copy.deepcopy(base_sd)
                for key in base_sd.keys():
                    if "fc" not in key:
                        fused_sd[key] = (mnist_sd[key] + fmnist_sd[key] + cifar_sd[key]) / 3.0
                merged_model.load_state_dict(fused_sd)
            elif merge_mode == "TA":
                # Task Arithmetic with lambda = 0.3
                lam = 0.3
                fused_sd = copy.deepcopy(base_sd)
                for key in base_sd.keys():
                    if "fc" not in key:
                        task_vector_mnist = mnist_sd[key] - base_sd[key]
                        task_vector_fmnist = fmnist_sd[key] - base_sd[key]
                        task_vector_cifar = cifar_sd[key] - base_sd[key]
                        fused_sd[key] = base_sd[key] + lam * (task_vector_mnist + task_vector_fmnist + task_vector_cifar)
                merged_model.load_state_dict(fused_sd)
                
            # Fold BatchNorm layers for clean mathematical analysis and fusion!
            folded_merged = fold_resnet18(merged_model)
            
            # Also load folded experts for statistics collection
            expert_models = {}
            for name, sd in [("mnist", mnist_sd), ("fmnist", fmnist_sd), ("cifar10", cifar_sd)]:
                exp_m = get_resnet18_expert().to(device)
                exp_m.load_state_dict(sd)
                expert_models[name] = fold_resnet18(exp_m)
                
            # 2. Collect activations for experts
            print("Collecting expert calibration statistics...")
            expert_acts = {}
            for name, exp_m in expert_models.items():
                expert_acts[name] = collect_activations(exp_m, cal_loaders[name], cal_layers, device)
                
            # 3. Collect activations for merged model
            print("Collecting merged model calibration statistics...")
            # Joint calibration dataset is the union of all task calibration sets
            # We can run each task's loader through the merged model to get its joint activations
            merged_acts = {layer: [] for layer in cal_layers}
            for name, loader in cal_loaders.items():
                acts = collect_activations(folded_merged, loader, cal_layers, device)
                for layer in cal_layers:
                    merged_acts[layer].append(acts[layer])
            for layer in cal_layers:
                merged_acts[layer] = torch.cat(merged_acts[layer], dim=0)
                
            # Evaluate all calibration methods
            for cal_method in cal_methods:
                # Working copy of folded merged model
                cal_model = copy.deepcopy(folded_merged)
                
                if cal_method == "None":
                    pass
                elif cal_method == "SP-TAAC":
                    # Global layer-wise scaling
                    with torch.no_grad():
                        for layer in cal_layers:
                            # Target global std: average of expert global stds
                            exp_stds = []
                            for name in expert_acts.keys():
                                exp_stds.append(expert_acts[name][layer].std().item())
                            target_std = np.mean(exp_stds)
                            merged_std = merged_acts[layer].std().item()
                            
                            scale = target_std / (merged_std + 1e-8)
                            
                            # Scale preceding Conv weights and bias
                            conv_name = get_preceding_conv_name(layer)
                            conv = get_module_by_name(cal_model, conv_name)
                            conv.weight.data *= scale
                            conv.bias.data *= scale
                            
                elif cal_method == "TAAC":
                    # Channel-wise scaling and shift
                    with torch.no_grad():
                        for layer in cal_layers:
                            # Target mean and std
                            exp_stds = []
                            exp_means = []
                            for name in expert_acts.keys():
                                exp_means.append(expert_acts[name][layer].mean(dim=(0, 2, 3)).numpy())
                                exp_stds.append(expert_acts[name][layer].std(dim=(0, 2, 3)).numpy())
                            target_mean = torch.tensor(np.mean(exp_means, axis=0)).to(device)
                            target_std = torch.tensor(np.mean(exp_stds, axis=0)).to(device)
                            
                            merged_mean = merged_acts[layer].mean(dim=(0, 2, 3)).to(device)
                            merged_std = merged_acts[layer].std(dim=(0, 2, 3)).to(device)
                            
                            # Scaling vector s and shift vector b_cal
                            s = target_std / (merged_std + 1e-8)
                            b_cal = target_mean - s * merged_mean
                            
                            # Scale preceding Conv weights and bias
                            conv_name = get_preceding_conv_name(layer)
                            conv = get_module_by_name(cal_model, conv_name)
                            conv.weight.data *= s.view(-1, 1, 1, 1)
                            conv.bias.data = s * conv.bias.data + b_cal
                            
                elif cal_method == "FDSA":
                    # Frequency-domain spectral magnitude alignment via forward hooks
                    hooks = []
                    for layer in cal_layers:
                        # Target spectral profile (L-FDSA: average over batch and channel)
                        exp_spectrals = []
                        for name in expert_acts.keys():
                            X = expert_acts[name][layer].to(device)
                            spect = torch.fft.fft2(X, dim=(2, 3)).abs().mean(dim=(0, 1))
                            exp_spectrals.append(spect.cpu().numpy())
                        target_spect = torch.tensor(np.mean(exp_spectrals, axis=0))
                        
                        # Merged spectral profile
                        X_merged = merged_acts[layer].to(device)
                        merged_spect = torch.fft.fft2(X_merged, dim=(2, 3)).abs().mean(dim=(0, 1)).cpu()
                        
                        scale_map = target_spect / (merged_spect + 1e-5)
                        scale_map = torch.clamp(scale_map, 1.0/5.0, 5.0)
                        
                        # Register Hook
                        module = get_module_by_name(cal_model, layer) # This is Identity in folded model
                        hook_fn = FDSAHook(scale_map)
                        hooks.append(module.register_forward_hook(hook_fn))
                        
                elif cal_method == "SSCC":
                    # Our method: Translate FDSA frequency maps to 3x3 spatial convolutions and fuse!
                    with torch.no_grad():
                        for layer in cal_layers:
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
                            scale_map = torch.clamp(scale_map, 1.0/5.0, 5.0) # shape (H, W)
                            
                            # 2. Get 2D spatial kernel via IFFT2D
                            # scale_map represents frequency magnitudes.
                            # Compute full-size circular spatial filter k_full
                            k_full = torch.fft.ifft2(scale_map).real
                            k_centered = torch.fft.fftshift(k_full) # shift DC center to physical center
                            
                            # Truncate to 3x3 local filter (or pad if map is smaller)
                            k_size = 3
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
                            
                            # Normalize k_trunc to match DC scale (which is scale_map[0, 0] or k_full.sum())
                            # This exactly preserves the low-frequency scaling factor
                            dc_target = scale_map[0, 0].item()
                            k_trunc_sum = k_trunc.sum().item()
                            if abs(k_trunc_sum) > 1e-8:
                                k_trunc = k_trunc * (dc_target / k_trunc_sum)
                            else:
                                k_trunc = torch.zeros_like(k_trunc)
                                k_trunc[k_size//2, k_size//2] = dc_target
                                
                            # Convert to depthwise filter weight for PyTorch
                            C = merged_acts[layer].shape[1] # number of channels
                            # Shape (C, 1, 3, 3), replicated across channels
                            w_dw = k_trunc.view(1, 1, k_size, k_size).repeat(C, 1, 1, 1).to(device)
                            
                            # 3. Fuse this depthwise filter back into the preceding Conv2d!
                            conv_name = get_preceding_conv_name(layer)
                            conv = get_module_by_name(cal_model, conv_name)
                            
                            w1 = conv.weight # (C, C_in, k1, k1)
                            b1 = conv.bias # (C)
                            
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
                            # Create new Conv2d with kernel size 5 and padding increased by 1
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
                            
                            # Replace in model
                            set_module_by_name(cal_model, conv_name, new_conv)
                            
                # Evaluate model
                accs = evaluate_model(cal_model, task_test_loaders, device)
                avg_acc = np.mean(list(accs.values()))
                print(f"  {cal_method}: MNIST={accs['mnist']:.2f}%, FMNIST={accs['fmnist']:.2f}%, CIFAR={accs['cifar10']:.2f}%, Avg={avg_acc:.2f}%")
                
                results.append({
                    "N": N,
                    "Merge": merge_mode,
                    "Cal": cal_method,
                    "mnist": accs['mnist'],
                    "fmnist": accs['fmnist'],
                    "cifar10": accs['cifar10'],
                    "avg": avg_acc
                })
                
                # Cleanup FDSA hooks
                if cal_method == "FDSA":
                    for hook in hooks:
                        hook.remove()

    print("\n\n====================== Summary Table ======================")
    print(f"{'N':<5} | {'Merge':<5} | {'Calibration':<10} | {'MNIST':<8} | {'FMNIST':<8} | {'CIFAR10':<8} | {'Average':<8}")
    print("-" * 75)
    for r in results:
        print(f"{r['N']:<5} | {r['Merge']:<5} | {r['Cal']:<10} | {r['mnist']:.2f}% | {r['fmnist']:.2f}% | {r['cifar10']:.2f}% | {r['avg']:.2f}%")

if __name__ == '__main__':
    main()
