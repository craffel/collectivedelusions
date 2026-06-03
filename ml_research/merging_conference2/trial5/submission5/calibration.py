import torch
import torch.nn as nn
import copy
import torchvision.models as models

class CalibrationHook:
    def __init__(self, mode='none', spectral_mode='layer-wise', gamma_max=5.0):
        self.mode = mode  # 'collect_spatial', 'collect_spectral', 'apply_spatial', 'apply_spectral', 'apply_joint', 'none'
        self.spectral_mode = spectral_mode  # 'layer-wise' or 'channel-wise'
        self.gamma_max = gamma_max
        self.epsilon = 1e-5
        
        # Collected stats
        self.spatial_means = []
        self.spatial_stds = []
        self.spectral_mags = []
        
        # Computed calibration parameters
        self.s = None  # Spatial scaling vector
        self.b_cal = None  # Spatial shift vector
        self.gamma_star = None  # Spectral scaling map
        
    def reset(self):
        self.spatial_means = []
        self.spatial_stds = []
        self.spectral_mags = []

    def set_calibration_params(self, s, b_cal, gamma_star):
        self.s = s
        self.b_cal = b_cal
        self.gamma_star = gamma_star

    def hook_fn(self, module, input, output):
        # output shape is (B, C, H, W)
        if self.mode == 'none':
            return output
            
        if self.mode == 'collect_spatial':
            with torch.no_grad():
                # Compute mean and std across batch, height, width
                # shape: (C,)
                mean = output.mean(dim=(0, 2, 3))
                std = output.std(dim=(0, 2, 3), unbiased=False)
                self.spatial_means.append(mean.cpu())
                self.spatial_stds.append(std.cpu())
            return output
            
        elif self.mode == 'collect_spectral':
            with torch.no_grad():
                # Compute 2D FFT along spatial dimensions
                out_fft = torch.fft.fft2(output, dim=(-2, -1))
                mag = torch.abs(out_fft)
                if self.spectral_mode == 'layer-wise':
                    # Average across batch and channel dimensions
                    # shape: (H, W)
                    mag_profile = mag.mean(dim=(0, 1))
                else:
                    # Average across batch dimension only
                    # shape: (C, H, W)
                    mag_profile = mag.mean(dim=0)
                self.spectral_mags.append(mag_profile.cpu())
            return output
            
        elif self.mode == 'apply_spatial':
            if self.s is None or self.b_cal is None:
                return output
            if self.s.device != output.device:
                self.s = self.s.to(output.device)
            if self.b_cal.device != output.device:
                self.b_cal = self.b_cal.to(output.device)
            s = self.s.view(1, -1, 1, 1)
            b_cal = self.b_cal.view(1, -1, 1, 1)
            return s * output + b_cal
            
        elif self.mode == 'apply_spectral':
            if self.gamma_star is None:
                return output
            if self.gamma_star.device != output.device:
                self.gamma_star = self.gamma_star.to(output.device)
            gamma = self.gamma_star
            if self.spectral_mode == 'layer-wise':
                gamma = gamma.view(1, 1, gamma.shape[0], gamma.shape[1])
            else:
                gamma = gamma.view(1, gamma.shape[0], gamma.shape[1], gamma.shape[2])
            out_fft = torch.fft.fft2(output, dim=(-2, -1))
            out_fft_cal = out_fft * gamma
            out_cal = torch.fft.ifft2(out_fft_cal, dim=(-2, -1))
            return torch.real(out_cal)
            
        elif self.mode == 'apply_joint':
            # 1. Apply spatial calibration
            cal_output = output
            if self.s is not None and self.b_cal is not None:
                if self.s.device != output.device:
                    self.s = self.s.to(output.device)
                if self.b_cal.device != output.device:
                    self.b_cal = self.b_cal.to(output.device)
                s = self.s.view(1, -1, 1, 1)
                b_cal = self.b_cal.view(1, -1, 1, 1)
                cal_output = s * output + b_cal
            
            # 2. Apply spectral calibration
            if self.gamma_star is not None:
                if self.gamma_star.device != cal_output.device:
                    self.gamma_star = self.gamma_star.to(cal_output.device)
                gamma = self.gamma_star
                if self.spectral_mode == 'layer-wise':
                    gamma = gamma.view(1, 1, gamma.shape[0], gamma.shape[1])
                else:
                    gamma = gamma.view(1, gamma.shape[0], gamma.shape[1], gamma.shape[2])
                out_fft = torch.fft.fft2(cal_output, dim=(-2, -1))
                out_fft_cal = out_fft * gamma
                out_cal = torch.fft.ifft2(out_fft_cal, dim=(-2, -1))
                cal_output = torch.real(out_cal)
                
            return cal_output
            
        return output

def register_hooks(model, mode='none', spectral_mode='channel-wise', gamma_max=5.0):
    hooks = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hook = CalibrationHook(mode=mode, spectral_mode=spectral_mode, gamma_max=gamma_max)
            handle = module.register_forward_hook(hook.hook_fn)
            hooks[name] = hook
            handles.append(handle)
    return hooks, handles

def merge_models(expert_state_dicts, base_state_dict=None, mode='WA', lambda_val=0.3):
    """
    Merges multiple state dicts.
    mode='WA': Weight Averaging
    mode='TA': Task Arithmetic (requires base_state_dict)
    """
    merged_state = copy.deepcopy(expert_state_dicts[0])
    keys = list(merged_state.keys())
    
    if mode == 'WA':
        for key in keys:
            # Check if parameter is a tensor (non-tensor metadata is skipped)
            if isinstance(merged_state[key], torch.Tensor):
                device = merged_state[key].device
                # Sum over all experts
                tensor_sum = expert_state_dicts[0][key].clone().float().to(device)
                for i in range(1, len(expert_state_dicts)):
                    tensor_sum += expert_state_dicts[i][key].clone().float().to(device)
                # Average and cast back to original type
                merged_state[key] = (tensor_sum / len(expert_state_dicts)).to(merged_state[key].dtype)
                
    elif mode == 'TA':
        if base_state_dict is None:
            raise ValueError("base_state_dict is required for Task Arithmetic (TA)")
        for key in keys:
            if isinstance(merged_state[key], torch.Tensor):
                device = merged_state[key].device
                base_t = base_state_dict[key].clone().float().to(device)
                task_vector_sum = torch.zeros_like(base_t)
                for i in range(len(expert_state_dicts)):
                    exp_t = expert_state_dicts[i][key].clone().float().to(device)
                    task_vector_sum += (exp_t - base_t)
                merged_state[key] = (base_t + lambda_val * task_vector_sum).to(merged_state[key].dtype)
                
    return merged_state

def fuse_calibration_to_bn(merged_model, hooks):
    """
    Fuses spatial calibration parameters (s, b_cal) directly into preceding BatchNorm weights and biases.
    Uses ZIO-CF reparameterization.
    """
    fused_model = copy.deepcopy(merged_model)
    with torch.no_grad():
        for name, module in fused_model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in hooks:
                hook = hooks[name]
                if hook.s is not None and hook.b_cal is not None:
                    # Update learnable weight and bias of BatchNorm
                    s = hook.s.to(module.weight.device)
                    b_cal = hook.b_cal.to(module.bias.device)
                    
                    module.weight.copy_(s * module.weight)
                    module.bias.copy_(s * module.bias + b_cal)
    return fused_model

if __name__ == '__main__':
    # Simple verification code on CPU
    print("Verifying merger and hook logic on dummy ResNet-18...")
    base_model = models.resnet18()
    expert1 = models.resnet18()
    expert2 = models.resnet18()
    
    # Merge
    merged_state = merge_models([expert1.state_dict(), expert2.state_dict()], base_model.state_dict(), mode='TA', lambda_val=0.3)
    base_model.load_state_dict(merged_state)
    
    # Hooks
    hooks, handles = register_hooks(base_model, mode='collect_spatial')
    print(f"Successfully registered {len(hooks)} calibration hooks.")
    
    x = torch.randn(2, 3, 32, 32)
    y = base_model(x)
    print("Forward pass successful.")
    
    # Clean up hooks
    for h in handles:
        h.remove()
    print("Hooks removed successfully.")
