import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False  # Disable cuDNN for stability on this cluster

# Define Dataset transform helper
def get_transforms(dataset_name):
    if dataset_name in ["mnist", "fashion"]:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_name == "cifar10":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# Get Dataset
def get_dataset(dataset_name, train=True):
    transform = get_transforms(dataset_name)
    if dataset_name == "mnist":
        return torchvision.datasets.MNIST(root="./data", train=train, download=False, transform=transform)
    elif dataset_name == "fashion":
        return torchvision.datasets.FashionMNIST(root="./data", train=train, download=False, transform=transform)
    elif dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10(root="./data", train=train, download=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# Load expert models
def load_experts():
    experts = {}
    for ds in ["mnist", "fashion", "cifar10"]:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, 10)
        save_path = f"./checkpoints/{ds}_expert.pt"
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Expert model for {ds} not found at {save_path}. Please train it first.")
        model.load_state_dict(torch.load(save_path, map_location=device))
        model = model.to(device)
        model.eval()
        experts[ds] = model
    return experts

# Create merged model from experts (averaging backbones)
def create_merged_model(experts):
    merged = models.resnet18(weights=None)
    merged.fc = nn.Linear(512, 10)
    merged = merged.to(device)
    
    # Average backbone weights
    merged_state = merged.state_dict()
    expert_states = {ds: exp.state_dict() for ds, exp in experts.items()}
    
    for key in merged_state.keys():
        if key.startswith("fc."):
            # Classification heads are task-specific and not merged
            continue
        # Average the backbone weights
        merged_state[key] = (
            expert_states["mnist"][key] +
            expert_states["fashion"][key] +
            expert_states["cifar10"][key]
        ) / 3.0
        
    merged.load_state_dict(merged_state)
    merged.eval()
    return merged

# Evaluate a model on a dataset
def evaluate_model(model, dataset_name, fc_head):
    # Swap out the classification head to the task-specific head
    original_fc = model.fc
    model.fc = fc_head
    model.eval()
    
    test_dataset = get_dataset(dataset_name, train=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    # Restore original classification head
    model.fc = original_fc
    return 100. * correct / total

# Run evaluation of a model across all three tasks
def evaluate_all(model, experts):
    results = {}
    for ds in ["mnist", "fashion", "cifar10"]:
        acc = evaluate_model(model, ds, experts[ds].fc)
        results[ds] = acc
    results["average"] = sum(results.values()) / 3.0
    return results

# Sequential layer finder helper with names
def get_conv_bn_pairs_with_names(model):
    pairs = []
    # Initial layer
    if hasattr(model, 'conv1') and hasattr(model, 'bn1'):
        pairs.append(("conv1", model.conv1, model.bn1))
    
    # Basic blocks in layer1, layer2, layer3, layer4
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            for b_idx, block in enumerate(layer):
                if hasattr(block, 'conv1') and hasattr(block, 'bn1'):
                    pairs.append((f"{layer_name}.{b_idx}.conv1", block.conv1, block.bn1))
                if hasattr(block, 'conv2') and hasattr(block, 'bn2'):
                    pairs.append((f"{layer_name}.{b_idx}.conv2", block.conv2, block.bn2))
                if hasattr(block, 'downsample') and block.downsample is not None:
                    if len(block.downsample) >= 2:
                        conv = block.downsample[0]
                        bn = block.downsample[1]
                        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                            pairs.append((f"{layer_name}.{b_idx}.downsample", conv, bn))
    return pairs

# ----------------- CALIBRATION METHODS -----------------

# 1. SP-TAAC (Sparsity-Preserving Task-Agnostic Activation Calibration)
def calibrate_sp_taac(model, experts, calibration_sets):
    print("Calibrating via SP-TAAC...")
    calibrated_model = copy.deepcopy(model)
    calibrated_model.eval()
    
    # We find all BatchNorm modules in the model and experts
    merged_bn_modules = [m for m in calibrated_model.modules() if isinstance(m, nn.BatchNorm2d)]
    expert_bn_modules = {ds: [m for m in experts[ds].modules() if isinstance(m, nn.BatchNorm2d)] for ds in experts}
    
    # We calibrate sequentially layer-by-layer
    for l, merged_bn in enumerate(merged_bn_modules):
        # We collect standard deviations of activations at the output of this BN layer
        expert_stds = []
        
        # Collect expert target standard deviations
        for ds in experts:
            # Run calibration pass for this expert
            loader = DataLoader(calibration_sets[ds], batch_size=len(calibration_sets[ds]), shuffle=False)
            x_batch = next(iter(loader))[0].to(device)
            
            # Setup a temporary hook to capture the output of BN
            expert_bn = expert_bn_modules[ds][l]
            bn_output = []
            def hook_fn(module, input, output):
                bn_output.append(output.detach())
            handle = expert_bn.register_forward_hook(hook_fn)
            with torch.no_grad():
                experts[ds](x_batch)
            handle.remove()
            
            # Compute global standard deviation
            out = bn_output[0]
            # std over batch and spatial dimensions
            expert_std = torch.std(out)
            expert_stds.append(expert_std.item())
            
        target_std = sum(expert_stds) / 3.0
        
        # Collect merged model activations standard deviation
        joint_ds = torch.utils.data.ConcatDataset(list(calibration_sets.values()))
        loader = DataLoader(joint_ds, batch_size=len(joint_ds), shuffle=False)
        x_joint = next(iter(loader))[0].to(device)
        
        merged_output = []
        def merged_hook(module, input, output):
            merged_output.append(output.detach())
        handle = merged_bn.register_forward_hook(merged_hook)
        with torch.no_grad():
            calibrated_model(x_joint)
        handle.remove()
        
        merged_std = torch.std(merged_output[0]).item()
        
        # Compute scaling factor
        gamma = target_std / (merged_std + 1e-8)
        
        # Modify the BatchNorm parameters in place
        merged_bn.weight.data *= gamma
        merged_bn.bias.data *= gamma
        
    return calibrated_model

# 2. N-TAAC (Native Joint Calibration)
def calibrate_n_taac(model, calibration_sets):
    print("Calibrating via N-TAAC...")
    calibrated_model = copy.deepcopy(model)
    calibrated_model.train()  # Set to train mode to update BatchNorm running stats
    
    # Create joint calibration set
    joint_dataset = torch.utils.data.ConcatDataset(list(calibration_sets.values()))
    joint_loader = DataLoader(joint_dataset, batch_size=64, shuffle=True)
    
    # Run a few steps (forward passes only) to update running stats
    with torch.no_grad():
        for inputs, _ in joint_loader:
            inputs = inputs.to(device)
            calibrated_model(inputs)
            
    calibrated_model.eval()
    return calibrated_model

# 3. TCAC (Task-Conditional Activation Calibration - Channel-wise)
def calibrate_tcac_single(model, expert, calibration_set):
    # This evaluates TCAC on a single expert and dataset (useful for comparison)
    calibrated_model = copy.deepcopy(model)
    calibrated_model.eval()
    
    merged_bn_modules = [m for m in calibrated_model.modules() if isinstance(m, nn.BatchNorm2d)]
    expert_bn_modules = [m for m in expert.modules() if isinstance(m, nn.BatchNorm2d)]
    
    loader = DataLoader(calibration_set, batch_size=len(calibration_set), shuffle=False)
    x_batch = next(iter(loader))[0].to(device)
    
    for l, merged_bn in enumerate(merged_bn_modules):
        expert_bn = expert_bn_modules[l]
        
        # Collect expert activations
        expert_output = []
        handle = expert_bn.register_forward_hook(lambda m, i, o: expert_output.append(o.detach()))
        with torch.no_grad():
            expert(x_batch)
        handle.remove()
        
        # Collect merged activations
        merged_output = []
        handle = merged_bn.register_forward_hook(lambda m, i, o: merged_output.append(o.detach()))
        with torch.no_grad():
            calibrated_model(x_batch)
        handle.remove()
        
        # Channel-wise mean and standard deviation
        # shape is (B, C, H, W). We take mean over (B, H, W) -> dims (0, 2, 3)
        mu_expert = expert_output[0].mean(dim=(0, 2, 3))
        sigma_expert = expert_output[0].std(dim=(0, 2, 3))
        
        mu_merged = merged_output[0].mean(dim=(0, 2, 3))
        sigma_merged = merged_output[0].std(dim=(0, 2, 3))
        
        # Adjust weight and bias
        # Under ReLU, many channels have near-zero activations leading to the "Sparsity Trap"
        # We add a small eps for numerical stability
        scale = sigma_expert / (sigma_merged + 1e-5)
        
        merged_bn.weight.data *= scale
        merged_bn.bias.data = scale * (merged_bn.bias.data - mu_merged) + mu_expert
        
    return calibrated_model

# 4. L-FDSA (Layer-wise Frequency-Domain Spectral Alignment)
class FDSAHook:
    def __init__(self, gamma_max=5.0):
        self.gamma_max = gamma_max
        self.gamma_star = None
        self.active = False
        
    def collect_expert_stats(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        # Compute 2D FFT along spatial dimensions
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        # Average magnitude across batch and channels
        mag = torch.mean(torch.abs(x_fft), dim=(0, 1)) # shape: (H, W)
        return mag
        
    def compute_gamma(self, mag_target, x_merged):
        # x_merged shape: (B, C, H, W)
        B, C, H, W = x_merged.shape
        x_fft = torch.fft.fft2(x_merged, dim=(-2, -1))
        mag_merged = torch.mean(torch.abs(x_fft), dim=(0, 1)) # shape: (H, W)
        
        gamma = mag_target / (mag_merged + 1e-5)
        self.gamma_star = torch.clamp(gamma, 1.0 / self.gamma_max, self.gamma_max)
        self.active = True
        
    def __call__(self, module, input, output):
        if not self.active or self.gamma_star is None:
            return output
        # Apply 2D scaling map in Fourier domain
        x_fft = torch.fft.fft2(output, dim=(-2, -1))
        # Multiply magnitude by gamma_star (broadcasted)
        # gamma_star shape is (H, W)
        x_fft_calibrated = x_fft * self.gamma_star.view(1, 1, output.shape[2], output.shape[3])
        # Transform back to spatial domain
        output_calibrated = torch.real(torch.fft.ifft2(x_fft_calibrated, dim=(-2, -1)))
        return output_calibrated

def calibrate_l_fdsa(model, experts, calibration_sets):
    print("Calibrating via L-FDSA...")
    calibrated_model = copy.deepcopy(model)
    calibrated_model.eval()
    
    # We find all BatchNorm modules in the model and experts
    merged_bn_modules = [m for m in calibrated_model.modules() if isinstance(m, nn.BatchNorm2d)]
    expert_bn_modules = {ds: [m for m in experts[ds].modules() if isinstance(m, nn.BatchNorm2d)] for ds in experts}
    
    hooks = []
    for l, merged_bn in enumerate(merged_bn_modules):
        fdsa_hook = FDSAHook()
        
        # 1. Collect target statistics from experts
        expert_mags = []
        for ds in experts:
            loader = DataLoader(calibration_sets[ds], batch_size=len(calibration_sets[ds]), shuffle=False)
            x_batch = next(iter(loader))[0].to(device)
            
            # Setup a temporary hook to capture the output of BN
            expert_bn = expert_bn_modules[ds][l]
            bn_output = []
            handle = expert_bn.register_forward_hook(lambda m, i, o: bn_output.append(o.detach()))
            with torch.no_grad():
                experts[ds](x_batch)
            handle.remove()
            
            # Get 2D spectral magnitude profile
            mag = fdsa_hook.collect_expert_stats(bn_output[0])
            expert_mags.append(mag)
            
        mag_target = torch.stack(expert_mags).mean(dim=0)
        
        # 2. Collect merged model activations and estimate gamma
        joint_ds = torch.utils.data.ConcatDataset(list(calibration_sets.values()))
        loader = DataLoader(joint_ds, batch_size=len(joint_ds), shuffle=False)
        x_joint = next(iter(loader))[0].to(device)
        
        merged_output = []
        handle = merged_bn.register_forward_hook(lambda m, i, o: merged_output.append(o.detach()))
        with torch.no_grad():
            calibrated_model(x_joint)
        handle.remove()
        
        fdsa_hook.compute_gamma(mag_target, merged_output[0])
        
        # 3. Register the active hook for evaluation
        handle = merged_bn.register_forward_hook(fdsa_hook)
        hooks.append(handle)
        
    return calibrated_model, hooks

# 5. SLR-WBC (SVD Low-Rank Weight and BatchNorm Calibration, our proposed method)
def calibrate_slr_wbc(model, experts, calibration_sets, rank=4, reg=1e-1):
    print(f"Calibrating via Hybrid SP-TAAC + SLR-WBC (Rank={rank}, Reg={reg})...")
    calibrated_model = copy.deepcopy(model)
    calibrated_model.eval()
    
    # Get sequential (Conv, BN) pairs with names
    merged_pairs = get_conv_bn_pairs_with_names(calibrated_model)
    expert_pairs = {ds: get_conv_bn_pairs_with_names(experts[ds]) for ds in experts}
    
    # We calibrate sequentially layer-by-layer
    for l, (name, merged_conv, merged_bn) in enumerate(merged_pairs):
        is_deep = "layer3" in name or "layer4" in name
        
        expert_outputs = []
        expert_inverted_outputs = []
        
        # 1. Collect target statistics from experts
        for ds in ["mnist", "fashion", "cifar10"]:
            loader = DataLoader(calibration_sets[ds], batch_size=len(calibration_sets[ds]), shuffle=False)
            x_batch = next(iter(loader))[0].to(device)
            
            # Hook to collect output of expert's BatchNorm
            exp_name, exp_conv, exp_bn = expert_pairs[ds][l]
            exp_bn_out = []
            handle = exp_bn.register_forward_hook(lambda m, i, o: exp_bn_out.append(o.detach()))
            with torch.no_grad():
                experts[ds](x_batch)
            handle.remove()
            
            H_exp = exp_bn_out[0] # shape: (N, C_out, H_out, W_out)
            expert_outputs.append(H_exp)
            
            # If doing low-rank correction, we also need inverted targets (Conv output level)
            if is_deep:
                B_exp, C_out, H_out, W_out = H_exp.shape
                H_exp_matrix = H_exp.transpose(0, 1).reshape(C_out, -1) # shape: (C_out, N * L_spatial)
                
                # Expert BN parameters
                mu_exp = exp_bn.running_mean.view(C_out, 1)
                var_exp = exp_bn.running_var.view(C_out, 1)
                w_exp = exp_bn.weight.view(C_out, 1)
                b_exp = exp_bn.bias.view(C_out, 1)
                eps_exp = exp_bn.eps
                
                w_exp_sign = torch.sign(w_exp)
                w_exp_sign = torch.where(w_exp_sign == 0, torch.ones_like(w_exp_sign), w_exp_sign)
                w_exp_stable = torch.where(w_exp.abs() > 1e-5, w_exp, w_exp_sign * 1e-5)
                
                V_target_exp = (H_exp_matrix - b_exp) / w_exp_stable * torch.sqrt(var_exp + eps_exp) + mu_exp
                V_target_exp = V_target_exp.reshape(C_out, B_exp, H_out, W_out).transpose(0, 1)
                expert_inverted_outputs.append(V_target_exp)
            
        # Target activations at BN output: concatenate across experts to match joint dataset
        H_target = torch.cat(expert_outputs, dim=0) # shape: (3N, C_out, H_out, W_out)
        
        # Feed the joint calibration data through the current partially calibrated merged model
        # to collect the input to the Conv layer
        joint_dataset = torch.utils.data.ConcatDataset(list(calibration_sets.values()))
        joint_loader = DataLoader(joint_dataset, batch_size=len(joint_dataset), shuffle=False)
        x_joint = next(iter(joint_loader))[0].to(device)
        
        merged_conv_in = []
        handle = merged_conv.register_forward_hook(lambda m, i, o: merged_conv_in.append(i[0].detach()))
        with torch.no_grad():
            calibrated_model(x_joint)
        handle.remove()
        
        X_input = merged_conv_in[0] # shape: (3N, C_in, H_in, W_in)
        
        if not is_deep:
            # Apply SP-TAAC style calibration for early layers:
            # Scale merged_bn weight and bias by gamma = target_std / merged_std
            expert_stds = [torch.std(h).item() for h in expert_outputs]
            target_std = sum(expert_stds) / 3.0
            
            merged_output = []
            handle = merged_bn.register_forward_hook(lambda m, i, o: merged_output.append(o.detach()))
            with torch.no_grad():
                calibrated_model(x_joint)
            handle.remove()
            
            merged_std = torch.std(merged_output[0]).item()
            gamma = target_std / (merged_std + 1e-8)
            
            merged_bn.weight.data *= gamma
            merged_bn.bias.data *= gamma
            
        else:
            V_target_joint = torch.cat(expert_inverted_outputs, dim=0) # shape: (3N, C_out, H_out, W_out)
            
            # 2. Unfold and flatten matrices
            kernel_size = merged_conv.kernel_size
            stride = merged_conv.stride
            padding = merged_conv.padding
            dilation = merged_conv.dilation
            
            X_unfolded = F.unfold(X_input, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
            d_in = X_unfolded.shape[1]
            X_matrix = X_unfolded.transpose(0, 1).reshape(d_in, -1) # shape: (d_in, 3N * L_spatial)
            
            C_out = V_target_joint.shape[1]
            V_matrix = V_target_joint.transpose(0, 1).reshape(C_out, -1) # shape: (C_out, 3N * L_spatial)
            
            # 3. Compute Low-Rank Weight Correction for the Conv layer
            W_curr = merged_conv.weight.flatten(1) # shape: (C_out, d_in)
            
            # Conv error (uncentered)
            E = V_matrix - W_curr @ X_matrix
            
            # Solve Ridge Regression: Delta_W* = E @ X_matrix^T @ (X_matrix @ X_matrix^T + reg_effective * I)^-1
            M_total = X_matrix.shape[1]
            cov_X = X_matrix @ X_matrix.transpose(0, 1)
            reg_effective = reg * M_total
            reg_I = reg_effective * torch.eye(d_in, device=device)
            try:
                # Solve A Y^T = B^T for Y^T where A = cov_X + reg_I and B = E @ X_matrix^T
                rhs = (E @ X_matrix.transpose(0, 1)).transpose(0, 1)
                Delta_W_star_T = torch.linalg.solve(cov_X + reg_I, rhs)
                Delta_W_star = Delta_W_star_T.transpose(0, 1)
            except RuntimeError:
                cov_inv = torch.inverse(cov_X + reg_I)
                Delta_W_star = (E @ X_matrix.transpose(0, 1)) @ cov_inv
                
            # Truncate to rank r
            if rank > 0:
                U, S, Vh = torch.linalg.svd(Delta_W_star, full_matrices=False)
                r = min(rank, len(S))
                Delta_W_r = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
            else:
                Delta_W_r = torch.zeros_like(Delta_W_star)
                
            # 4. Update Conv weights in-place
            merged_conv.weight.data += Delta_W_r.view_as(merged_conv.weight)
            
            # 5. Pass calibration inputs through updated Conv to update BatchNorm running stats
            with torch.no_grad():
                V_new = merged_conv(X_input) # shape: (3N, C_out, H_out, W_out)
                
            # Flatten V_new to compute statistics
            C_out = V_new.shape[1]
            V_new_matrix = V_new.transpose(0, 1).reshape(C_out, -1) # shape: (C_out, 3N * L_spatial)
            eps = merged_bn.eps
            
            # Update BN running statistics
            emp_mean = V_new_matrix.mean(dim=1)
            emp_var = V_new_matrix.var(dim=1)
            
            merged_bn.running_mean.copy_(emp_mean)
            merged_bn.running_var.copy_(emp_var)
            
            # 6. Align BN affine parameters (scale and shift) channel-wise using least-squares linear regression
            V_norm = (V_new_matrix - emp_mean.view(C_out, 1)) / torch.sqrt(emp_var.view(C_out, 1) + eps)
            
            # Target activations at BN output
            H_matrix = H_target.transpose(0, 1).reshape(C_out, -1) # shape: (C_out, 3N * L_spatial)
            
            s = torch.mean(V_norm * H_matrix, dim=1)
            d = torch.mean(H_matrix, dim=1)
            
            # To be stable, clip the scaling factor s
            s = torch.clamp(s, 0.1, 10.0)
            
            # Update BatchNorm weight and bias parameters
            merged_bn.weight.data.copy_(s)
            merged_bn.bias.data.copy_(d)
        
    return calibrated_model


# ----------------- MAIN RUN -----------------
def run_experiment():
    print("Loading experts...")
    experts = load_experts()
    
    # 1. Evaluate Oracle experts
    print("\n--- Oracle Experts (Upper Bound) ---")
    oracle_results = {}
    for ds in ["mnist", "fashion", "cifar10"]:
        acc = evaluate_model(experts[ds], ds, experts[ds].fc)
        oracle_results[ds] = acc
    oracle_results["average"] = sum(oracle_results.values()) / 3.0
    print(f"Oracle: MNIST={oracle_results['mnist']:.2f}%, Fashion={oracle_results['fashion']:.2f}%, CIFAR10={oracle_results['cifar10']:.2f}%, Avg={oracle_results['average']:.2f}%")
    
    # Create the base merged model (uncalibrated)
    base_merged = create_merged_model(experts)
    
    # Evaluate Uncalibrated Weight Averaging
    print("\n--- Uncalibrated Weight Averaging ---")
    uncal_results = evaluate_all(base_merged, experts)
    print(f"Uncalibrated WA: MNIST={uncal_results['mnist']:.2f}%, Fashion={uncal_results['fashion']:.2f}%, CIFAR10={uncal_results['cifar10']:.2f}%, Avg={uncal_results['average']:.2f}%")
    
    # Prepare Calibration Sets for different calibration sizes N
    results_table = []
    
    for N in [16, 64, 128]:
        print(f"\n=======================================================")
        print(f"RUNNING CALIBRATIONS WITH BUDGET N={N} SAMPLES PER TASK")
        print(f"=======================================================")
        
        # Deterministically extract the calibration subsets from training set
        calibration_sets = {}
        for ds in ["mnist", "fashion", "cifar10"]:
            full_train = get_dataset(ds, train=True)
            # Calibration set is indices 5,000 to 5,000 + N
            calibration_sets[ds] = Subset(full_train, list(range(5000, 5000 + N)))
            
        # --- 2. Evaluate SP-TAAC ---
        model_sp = calibrate_sp_taac(base_merged, experts, calibration_sets)
        sp_res = evaluate_all(model_sp, experts)
        print(f"SP-TAAC (N={N}): MNIST={sp_res['mnist']:.2f}%, Fashion={sp_res['fashion']:.2f}%, CIFAR10={sp_res['cifar10']:.2f}%, Avg={sp_res['average']:.2f}%")
        
        # --- 3. Evaluate N-TAAC ---
        model_ntaac = calibrate_n_taac(base_merged, calibration_sets)
        ntaac_res = evaluate_all(model_ntaac, experts)
        print(f"N-TAAC (N={N}): MNIST={ntaac_res['mnist']:.2f}%, Fashion={ntaac_res['fashion']:.2f}%, CIFAR10={ntaac_res['cifar10']:.2f}%, Avg={ntaac_res['average']:.2f}%")
        
        # --- 4. Evaluate TCAC ---
        # Evaluate TCAC on each dataset separately (since it's task-conditional)
        print("Calibrating via TCAC (Task-Conditional)...")
        tcac_res = {}
        for ds in ["mnist", "fashion", "cifar10"]:
            model_tcac = calibrate_tcac_single(base_merged, experts[ds], calibration_sets[ds])
            acc = evaluate_model(model_tcac, ds, experts[ds].fc)
            tcac_res[ds] = acc
        tcac_res["average"] = sum(tcac_res.values()) / 3.0
        print(f"TCAC (N={N}): MNIST={tcac_res['mnist']:.2f}%, Fashion={tcac_res['fashion']:.2f}%, CIFAR10={tcac_res['cifar10']:.2f}%, Avg={tcac_res['average']:.2f}%")
        
        # --- 5. Evaluate L-FDSA ---
        model_fdsa, hooks = calibrate_l_fdsa(base_merged, experts, calibration_sets)
        fdsa_res = evaluate_all(model_fdsa, experts)
        # Remove hooks to clean up
        for h in hooks:
            h.remove()
        print(f"L-FDSA (N={N}): MNIST={fdsa_res['mnist']:.2f}%, Fashion={fdsa_res['fashion']:.2f}%, CIFAR10={fdsa_res['cifar10']:.2f}%, Avg={fdsa_res['average']:.2f}%")
        
        # --- 6. Evaluate our SLR-WBC ---
        # Sweep some ranks and regularization values
        for reg_val in [1e-2, 5e-2, 1e-1, 5e-1]:
            for rank in [1, 2, 4]:
                model_slr = calibrate_slr_wbc(base_merged, experts, calibration_sets, rank=rank, reg=reg_val)
                slr_res = evaluate_all(model_slr, experts)
                print(f"SLR-WBC (Rank={rank}, Reg={reg_val}, N={N}): MNIST={slr_res['mnist']:.2f}%, Fashion={slr_res['fashion']:.2f}%, CIFAR10={slr_res['cifar10']:.2f}%, Avg={slr_res['average']:.2f}%")
                results_table.append({
                    "N": N,
                    "Method": f"SLR-WBC (Ours, Rank={rank}, Reg={reg_val})",
                    "MNIST": slr_res['mnist'],
                    "Fashion": slr_res['fashion'],
                    "CIFAR10": slr_res['cifar10'],
                    "Avg": slr_res['average']
                })
            
        results_table.append({"N": N, "Method": "SP-TAAC", "MNIST": sp_res['mnist'], "Fashion": sp_res['fashion'], "CIFAR10": sp_res['cifar10'], "Avg": sp_res['average']})
        results_table.append({"N": N, "Method": "N-TAAC", "MNIST": ntaac_res['mnist'], "Fashion": ntaac_res['fashion'], "CIFAR10": ntaac_res['cifar10'], "Avg": ntaac_res['average']})
        results_table.append({"N": N, "Method": "TCAC (Task-ID req.)", "MNIST": tcac_res['mnist'], "Fashion": tcac_res['fashion'], "CIFAR10": tcac_res['cifar10'], "Avg": tcac_res['average']})
        results_table.append({"N": N, "Method": "L-FDSA", "MNIST": fdsa_res['mnist'], "Fashion": fdsa_res['fashion'], "CIFAR10": fdsa_res['cifar10'], "Avg": fdsa_res['average']})
        
    print("\n\n================ FINAL RESULTS SUMMARY ================")
    print("| Calibration N | Method | MNIST | Fashion-MNIST | CIFAR-10 | Average Accuracy |")
    print("|---|---|---|---|---|---|")
    print(f"| N/A | Oracle Experts (Upper Bound) | {oracle_results['mnist']:.2f}% | {oracle_results['fashion']:.2f}% | {oracle_results['cifar10']:.2f}% | {oracle_results['average']:.2f}% |")
    print(f"| N/A | Uncalibrated Weight Averaging | {uncal_results['mnist']:.2f}% | {uncal_results['fashion']:.2f}% | {uncal_results['cifar10']:.2f}% | {uncal_results['average']:.2f}% |")
    for r in results_table:
        print(f"| {r['N']} | {r['Method']} | {r['MNIST']:.2f}% | {r['Fashion']:.2f}% | {r['CIFAR10']:.2f}% | {r['Avg']:.2f}% |")

if __name__ == "__main__":
    run_experiment()
