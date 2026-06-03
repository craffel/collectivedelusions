import torch
import torch.nn as nn
import torchvision.models as models
import math

# Fourier-domain Hook for FDSA
class FDSAHook:
    def __init__(self, gamma):
        # gamma is a tensor of shape (H, W) or (C, H, W)
        self.gamma = gamma

    def __call__(self, module, input, output):
        # output shape: (B, C, H, W)
        device = output.device
        O_fft = torch.fft.fft2(output)
        
        # Broadcast gamma to (B, C, H, W)
        if len(self.gamma.shape) == 2:
            g = self.gamma.view(1, 1, *self.gamma.shape).to(device)
        else:
            g = self.gamma.unsqueeze(0).to(device)
            
        O_fft_calibrated = O_fft * g
        return torch.fft.ifft2(O_fft_calibrated).real

def get_bn_modules(model):
    bn_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_modules.append((name, module))
    return bn_modules

def map_bn_to_conv(model):
    bn_to_conv = {}
    last_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_name = name
            bn_to_conv[name] = last_conv_name
        elif isinstance(module, nn.BatchNorm2d):
            bn_to_conv[name] = last_conv_name
    return bn_to_conv

def collect_activations(model, loader, device, target_layer_name, max_samples=None):
    captured = []
    samples_collected = 0
    
    def hook_fn(module, input, output):
        captured.append(output.detach().cpu())
        
    target_module = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_module = module
            break
            
    assert target_module is not None, f"Layer {target_layer_name} not found"
    hook_handle = target_module.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for images, _ in loader:
            if max_samples is not None and samples_collected >= max_samples:
                break
            needed = max_samples - samples_collected if max_samples is not None else images.size(0)
            if images.size(0) > needed:
                images = images[:needed]
            images = images.to(device)
            _ = model(images)
            samples_collected += images.size(0)
            
    hook_handle.remove()
    if len(captured) == 0:
        return torch.tensor([])
    return torch.cat(captured, dim=0)

def merge_models(merge_mode='wa', lambda_val=0.3):
    # Load state dicts
    mnist_state = torch.load('mnist_expert.pt', map_location='cpu')
    fashion_state = torch.load('fashion_expert.pt', map_location='cpu')
    cifar_state = torch.load('cifar_expert.pt', map_location='cpu')
    
    # Base model for Task Arithmetic
    base_state = torch.load('base_model.pt', map_location='cpu')
    
    merged_state = {}
    for key in base_state.keys():
        if 'fc' in key:
            # Keep heads task-specific (handled dynamically at evaluation time)
            merged_state[key] = base_state[key].clone()
        else:
            if merge_mode == 'wa':
                merged_state[key] = (mnist_state[key] + fashion_state[key] + cifar_state[key]) / 3.0
            elif merge_mode == 'ta':
                tau_mnist = mnist_state[key] - base_state[key]
                tau_fashion = fashion_state[key] - base_state[key]
                tau_cifar = cifar_state[key] - base_state[key]
                merged_state[key] = base_state[key] + lambda_val * (tau_mnist + tau_fashion + tau_cifar)
                
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(merged_state)
    return model

def calibrate_sequential(model, experts, loaders, method, device, cal_size=128):
    """
    Sequentially calibrates the merged model in-place.
    For standard calibrations (SP-TAAC, TAAC, DWSS), values are fused directly into BatchNorm weights/biases (zero overhead!).
    For FDSA, forward hooks are registered because they are frequency-domain operations.
    """
    model.eval()
    for exp in experts:
        exp.eval()
        
    bn_modules = get_bn_modules(model)
    bn_to_conv = map_bn_to_conv(model)
    
    # We will save registered hooks list so we can manage them
    fdsa_hook_handles = []
    
    # Dictionary of loaders by task
    # task names are: 'mnist', 'fashion', 'cifar'
    tasks = ['mnist', 'fashion', 'cifar']
    
    if method == 'pbr':
        print(f"Calibrating model using PARALLEL BatchNorm Recalibration (N={cal_size})...")
        # Register hooks on ALL BN layers simultaneously
        captured_inputs = {bn_name: [] for bn_name, _ in bn_modules}
        
        def make_hook(name):
            def hook_fn(module, input, output):
                captured_inputs[name].append(input[0].detach().cpu())
            return hook_fn
            
        handles = []
        for bn_name, bn_module in bn_modules:
            handles.append(bn_module.register_forward_hook(make_hook(bn_name)))
            
        # Run forward pass on joint calibration dataset
        with torch.no_grad():
            for task in tasks:
                cal_loader = loaders[task]['cal']
                samples_collected = 0
                for images, _ in cal_loader:
                    if cal_size is not None and samples_collected >= cal_size:
                        break
                    needed = cal_size - samples_collected if cal_size is not None else images.size(0)
                    if images.size(0) > needed:
                        images = images[:needed]
                    images = images.to(device)
                    _ = model(images)
                    samples_collected += images.size(0)
                    
        for h in handles:
            h.remove()
            
        # Now update all BN running statistics in-place
        for bn_name, bn_module in bn_modules:
            inputs_tensor = torch.cat(captured_inputs[bn_name], dim=0)
            mu_new = inputs_tensor.mean(dim=(0, 2, 3))
            var_new = inputs_tensor.var(dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                bn_module.running_mean.copy_(mu_new.to(bn_module.running_mean.device))
                bn_module.running_var.copy_(var_new.to(bn_module.running_var.device))
                
        return model, []
        
    print(f"Calibrating model using sequential {method.upper()} (N={cal_size})...")
    
    for layer_idx, (bn_name, bn_module) in enumerate(bn_modules):
        # Find corresponding expert modules
        expert_bn_modules = []
        for exp in experts:
            target_module = None
            for name, module in exp.named_modules():
                if name == bn_name:
                    target_module = module
                    break
            expert_bn_modules.append(target_module)
            
        # Check preceding conv weight for DWSS
        preceding_conv_name = bn_to_conv.get(bn_name)
        
        # METHOD-SPECIFIC CALIBRATION
        if method in ['sp_taac', 'taac']:
            # 1. Collect statistics
            target_stds = []
            target_means = []
            
            # For each expert, run task-specific calibration set
            for m, task in enumerate(tasks):
                exp = experts[m]
                cal_loader = loaders[task]['cal']
                # Collect activations
                acts = collect_activations(exp, cal_loader, device, bn_name, max_samples=cal_size)
                # Compute channel-wise mean and std
                mean_m = acts.mean(dim=(0, 2, 3))
                std_m = acts.std(dim=(0, 2, 3), unbiased=False)
                target_means.append(mean_m)
                target_stds.append(std_m)
                
            # Average target statistics
            mu_target = torch.stack(target_means).mean(dim=0)
            sigma_target = torch.stack(target_stds).mean(dim=0)
            
            # Now, pass joint calibration data through merged model
            # To construct the joint calibration loader, we can just run sequential forward passes on each loader
            merged_acts = []
            for task in tasks:
                cal_loader = loaders[task]['cal']
                acts = collect_activations(model, cal_loader, device, bn_name, max_samples=cal_size)
                merged_acts.append(acts)
            merged_acts = torch.cat(merged_acts, dim=0)
            
            mu_merged = merged_acts.mean(dim=(0, 2, 3))
            sigma_merged = merged_acts.std(dim=(0, 2, 3), unbiased=False)
            
            # 2. Compute calibration factors
            if method == 'taac':
                # Channel-wise scale and bias
                s = sigma_target / (sigma_merged + 1e-5)
                bcal = mu_target - s * mu_merged
                # Clamp to prevent division-by-zero noise amplification
                s = torch.clamp(s, 1.0/5.0, 5.0)
                bcal = torch.clamp(bcal, -5.0, 5.0)
                
                # Move to GPU device
                s = s.to(bn_module.weight.device)
                bcal = bcal.to(bn_module.bias.device)
                
                # Fuse in-place (ZIO-CF style!)
                with torch.no_grad():
                    bn_module.weight.copy_(s * bn_module.weight)
                    bn_module.bias.copy_(s * bn_module.bias + bcal)
                    
            elif method == 'sp_taac':
                # Global scale ratio
                gamma = sigma_target.mean() / (sigma_merged.mean() + 1e-5)
                gamma = torch.clamp(gamma, 1.0/5.0, 5.0)
                
                # Move to GPU device
                gamma = gamma.to(bn_module.weight.device)
                
                # Fuse in-place
                with torch.no_grad():
                    bn_module.weight.copy_(gamma * bn_module.weight)
                    bn_module.bias.copy_(gamma * bn_module.bias)
                    
        elif method in ['l_fdsa', 'c_fdsa']:
            # Fourier spectral alignment
            target_profiles = []
            
            for m, task in enumerate(tasks):
                exp = experts[m]
                cal_loader = loaders[task]['cal']
                acts = collect_activations(exp, cal_loader, device, bn_name, max_samples=cal_size)
                # Compute FFT
                O_fft = torch.fft.fft2(acts)
                O_mag = O_fft.abs()
                if method == 'l_fdsa':
                    # Average over batch and channel
                    profile_m = O_mag.mean(dim=(0, 1)) # (H, W)
                else:
                    # Average over batch only
                    profile_m = O_mag.mean(dim=0) # (C, H, W)
                target_profiles.append(profile_m)
                
            M_target = torch.stack(target_profiles).mean(dim=0)
            
            # Merged spectral profile
            merged_profiles = []
            for task in tasks:
                cal_loader = loaders[task]['cal']
                acts = collect_activations(model, cal_loader, device, bn_name, max_samples=cal_size)
                O_fft = torch.fft.fft2(acts)
                O_mag = O_fft.abs()
                if method == 'l_fdsa':
                    profile_merged = O_mag.mean(dim=(0, 1))
                else:
                    profile_merged = O_mag.mean(dim=0)
                merged_profiles.append(profile_merged)
            M_merged = torch.stack(merged_profiles).mean(dim=0)
            
            Gamma = M_target / (M_merged + 1e-5)
            Gamma_star = torch.clamp(Gamma, 1.0/5.0, 5.0)
            
            # Register forward hook for FDSA (can't be fused!)
            hook_obj = FDSAHook(Gamma_star)
            handle = bn_module.register_forward_hook(hook_obj)
            fdsa_hook_handles.append(handle)
            
        elif method in ['l_dwss', 'c_dwss']:
            # DATA-FREE DIRECT WEIGHT-SIMILARITY SCALING (Ours!)
            # Retrieve preceding conv layer weights for all 3 experts
            expert_convs = []
            for exp in experts:
                target_module = None
                for name, module in exp.named_modules():
                    if name == preceding_conv_name:
                        target_module = module
                        break
                expert_convs.append(target_module)
                
            if expert_convs[0] is not None:
                # Get expert weights shape: (C_out, C_in, K_H, K_W)
                w1 = expert_convs[0].weight.data
                w2 = expert_convs[1].weight.data
                w3 = expert_convs[2].weight.data
                
                C_out = w1.shape[0]
                # Flatten filters to vectors of size D = C_in * K_H * K_W
                w1_flat = w1.view(C_out, -1)
                w2_flat = w2.view(C_out, -1)
                w3_flat = w3.view(C_out, -1)
                
                # Compute cosine similarities for each channel
                # dot_prod / (norm1 * norm2)
                eps = 1e-8
                norm1 = torch.norm(w1_flat, dim=1)
                norm2 = torch.norm(w2_flat, dim=1)
                norm3 = torch.norm(w3_flat, dim=1)
                
                rho12 = torch.sum(w1_flat * w2_flat, dim=1) / (norm1 * norm2 + eps)
                rho13 = torch.sum(w1_flat * w3_flat, dim=1) / (norm1 * norm3 + eps)
                rho23 = torch.sum(w2_flat * w3_flat, dim=1) / (norm2 * norm3 + eps)
                
                # Average similarity per channel
                rho_bar_c = (rho12 + rho13 + rho23) / 3.0
                # Clamp similarity to valid range [-1.0, 1.0]
                rho_bar_c = torch.clamp(rho_bar_c, -1.0, 1.0)
                
                # Expected variance scaling: V_c = (3 + 6 * rho_bar_c) / 9 = (1 + 2 * rho_bar_c) / 3
                # We add a small eps for numerical stability
                V_c = (1.0 + 2.0 * rho_bar_c) / 3.0
                V_c = torch.clamp(V_c, min=1e-3)
                
                # Scale factors s_c
                s_c = 1.0 / torch.sqrt(V_c)
                s_c = torch.clamp(s_c, 1.0/5.0, 5.0)
                
                if method == 'l_dwss':
                    # Layer-wise: average scaling factor across all channels
                    s = s_c.mean()
                    with torch.no_grad():
                        bn_module.weight.copy_(s * bn_module.weight)
                        bn_module.bias.copy_(s * bn_module.bias)
                else:
                    # Channel-wise: apply distinct scale per channel
                    # s_c is 1D tensor of size C_out
                    with torch.no_grad():
                        bn_module.weight.copy_(s_c * bn_module.weight)
                        bn_module.bias.copy_(s_c * bn_module.bias)
            else:
                print(f"Warning: Preceding conv layer {preceding_conv_name} not found for {bn_name}")
                
        elif method == 'abs':
            # ANALYTICAL BATCHNORM SCALING (Ours!)
            # Scale down the running variance by M=3
            with torch.no_grad():
                bn_module.running_var.copy_(bn_module.running_var / 3.0)
                
        elif method == 'sbr':
            # SEQUENTIAL BATCHNORM RECALIBRATION (Ours!)
            # Capture the input activations to the BatchNorm layer
            captured_inputs = []
            def input_hook_fn(module, input, output):
                captured_inputs.append(input[0].detach().cpu())
                
            handle = bn_module.register_forward_hook(input_hook_fn)
            
            # Run forward pass on joint calibration dataset
            with torch.no_grad():
                for task in tasks:
                    cal_loader = loaders[task]['cal']
                    samples_collected = 0
                    for images, _ in cal_loader:
                        if cal_size is not None and samples_collected >= cal_size:
                            break
                        needed = cal_size - samples_collected if cal_size is not None else images.size(0)
                        if images.size(0) > needed:
                            images = images[:needed]
                        images = images.to(device)
                        _ = model(images)
                        samples_collected += images.size(0)
                        
            handle.remove()
            
            # Compute exact mean and variance over the joint dataset
            inputs_tensor = torch.cat(captured_inputs, dim=0) # (N_joint, C, H, W)
            mu_new = inputs_tensor.mean(dim=(0, 2, 3))
            var_new = inputs_tensor.var(dim=(0, 2, 3), unbiased=False)
            
            # Update running statistics in-place
            with torch.no_grad():
                bn_module.running_mean.copy_(mu_new.to(bn_module.running_mean.device))
                bn_module.running_var.copy_(var_new.to(bn_module.running_var.device))
                
    return model, fdsa_hook_handles
