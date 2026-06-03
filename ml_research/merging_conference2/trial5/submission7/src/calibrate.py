import torch
import torch.nn as nn
import torch.fft as fft

class ActivationCollector:
    def __init__(self):
        self.activations = []
        
    def hook_fn(self, module, input, output):
        # Store activations on CPU to save GPU memory
        self.activations.append(output.detach().cpu())
        
    def get_all(self):
        if not self.activations:
            return None
        return torch.cat(self.activations, dim=0)
    
    def clear(self):
        self.activations = []

def get_bn_layers(model):
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))
    return bn_layers

def collect_activations(model, loader, bn_layer_names, device='cuda'):
    model.eval()
    collectors = {name: ActivationCollector() for name in bn_layer_names}
    hooks = []
    
    # Register temporary collectors
    for name, module in model.named_modules():
        if name in collectors:
            h = module.register_forward_hook(collectors[name].hook_fn)
            hooks.append(h)
            
    # Run forward pass on calibration data
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            _ = model(images)
            
    # Remove hooks
    for h in hooks:
        h.remove()
        
    # Get activations
    activations = {name: col.get_all() for name, col in collectors.items()}
    return activations

def collect_target_stats(expert_models, calib_loaders, device='cuda'):
    # expert_models is a dict: {'mnist': model, 'fmnist': model, 'cifar10': model}
    # calib_loaders is a dict: {'mnist': loader, 'fmnist': loader, 'cifar10': loader}
    
    target_stats = {}
    tasks = list(expert_models.keys())
    
    # Get BN layer names from one model
    sample_model = next(iter(expert_models.values()))
    bn_layers = get_bn_layers(sample_model)
    bn_names = [name for name, _ in bn_layers]
    
    for task in tasks:
        print(f"Collecting target stats from expert for task {task}...")
        model = expert_models[task]
        loader = calib_loaders[task]
        
        # Collect activations for this expert on its task
        activations = collect_activations(model, loader, bn_names, device=device)
        
        for name in bn_names:
            act = activations[name] # [N, C, H, W]
            if name not in target_stats:
                target_stats[name] = {}
                
            # Compute SP-TAAC target statistic: standard deviation globally across all batch, channel, spatial dims
            target_stats[name][f'{task}_std'] = float(torch.std(act))
            
            # Compute FDSA/WRSA target statistic: 2D Fourier magnitude spectrum
            # FFT of each sample along height and width
            # act has shape [N, C, H, W]
            fft_act = fft.fft2(act, dim=(-2, -1))
            mag_act = torch.abs(fft_act) # [N, C, H, W]
            
            # L-FDSA: average across both batch and channel dimensions to get a [H, W] spectrum
            target_stats[name][f'{task}_spectrum'] = torch.mean(mag_act, dim=(0, 1))
            
    # Compute final combined target statistics (averages across tasks)
    final_targets = {}
    for name in bn_names:
        stds = [target_stats[name][f'{task}_std'] for task in tasks]
        spectra = [target_stats[name][f'{task}_spectrum'] for task in tasks]
        
        final_targets[name] = {
            'std_target': sum(stds) / len(tasks),
            'spectrum_target': torch.mean(torch.stack(spectra), dim=0)
        }
        
    return final_targets

def calibrate_sp_taac(merged_model, target_stats, joint_loader, device='cuda'):
    # Step 1: Collect uncalibrated merged statistics on the joint calibration dataset
    bn_layers = get_bn_layers(merged_model)
    bn_names = [name for name, _ in bn_layers]
    
    print("Collecting merged model activations for SP-TAAC...")
    merged_activations = collect_activations(merged_model, joint_loader, bn_names, device=device)
    
    # Step 2: For each BN layer, compute the global scaling factor and fuse it
    print("Calibrating SP-TAAC and performing Zero-Inference-Overhead parameter fusion...")
    for name, module in bn_layers:
        act = merged_activations[name]
        merged_std = float(torch.std(act))
        target_std = target_stats[name]['std_target']
        
        # Scaling factor gamma
        gamma = target_std / (merged_std + 1e-8)
        
        # Fuse in-place: w' = gamma * w, b' = gamma * b
        with torch.no_grad():
            module.weight.copy_(gamma * module.weight)
            module.bias.copy_(gamma * module.bias)
            
    print("SP-TAAC calibration and fusion complete.")
    return merged_model

class SpectralCalibrationHook:
    def __init__(self, scaling_map):
        # scaling_map is a [H, W] tensor of scaling factors
        self.scaling_map = scaling_map
        
    def hook_fn(self, module, input, output):
        # output is a [B, C, H, W] spatial tensor
        device = output.device
        self.scaling_map = self.scaling_map.to(device)
        
        # FFT along spatial dimensions
        fft_out = fft.fft2(output, dim=(-2, -1))
        
        # Apply scaling map (broadcasting along batch and channel dimensions)
        scaled_fft = fft_out * self.scaling_map
        
        # Inverse FFT
        calibrated_out = torch.real(fft.ifft2(scaled_fft, dim=(-2, -1)))
        return calibrated_out

def apply_spectral_calibration(merged_model, target_stats, joint_loader, method='fdsa', c_val=0.1, device='cuda'):
    # method can be 'fdsa' or 'wrsa'
    bn_layers = get_bn_layers(merged_model)
    bn_names = [name for name, _ in bn_layers]
    
    print(f"Collecting merged model activations for {method.upper()}...")
    merged_activations = collect_activations(merged_model, joint_loader, bn_names, device=device)
    
    hooks = []
    
    print(f"Registering {method.upper()} active inference hooks...")
    for name, module in bn_layers:
        act = merged_activations[name]
        
        # Compute merged spectrum [H, W]
        fft_act = fft.fft2(act, dim=(-2, -1))
        mag_act = torch.abs(fft_act)
        merged_spectrum = torch.mean(mag_act, dim=(0, 1))
        
        target_spectrum = target_stats[name]['spectrum_target']
        
        if method == 'fdsa':
            # FDSA pointwise division with clamping
            scaling_map = target_spectrum / (merged_spectrum + 1e-5)
            scaling_map = torch.clamp(scaling_map, 1.0/5.0, 5.0)
        elif method == 'wrsa':
            # WRSA (Wiener-Regularized Spectral Alignment)
            # Formula: (Target * Merged) / (Merged^2 + c^2 * Target^2)
            scaling_map = (target_spectrum * merged_spectrum) / (merged_spectrum**2 + (c_val**2) * target_spectrum**2 + 1e-8)
        else:
            raise ValueError(f"Unknown spectral method: {method}")
            
        # Register the forward hook
        calib_hook = SpectralCalibrationHook(scaling_map)
        h = module.register_forward_hook(calib_hook.hook_fn)
        hooks.append(h)
        
    print(f"{method.upper()} active inference hooks registered successfully.")
    return merged_model, hooks
