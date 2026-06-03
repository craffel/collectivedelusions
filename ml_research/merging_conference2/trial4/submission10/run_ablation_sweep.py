import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import copy
from eval_merging import (
    load_experts, merge_models, evaluate_merged, 
    remove_hooks, get_dataset, clip_scale
)

# Define a customizable FDSA hook for ablation
class Ablation_FDSA_Hook:
    def __init__(self, max_val=5.0):
        self.calibrating_target = False
        self.calibrating_merged = False
        self.active = False
        self.expert_mags = []
        self.target_mag = None # Will be of shape (H, W)
        self.merged_mag = None # Will be of shape (H, W)
        self.gamma = 1.0 # Will be of shape (H, W)
        self.max_val = max_val

    def start_expert_calibration(self):
        self.calibrating_target = True
        self.expert_mags = []

    def finalize_target(self):
        self.calibrating_target = False
        self.target_mag = torch.stack(self.expert_mags).mean(dim=0)

    def start_merged_calibration(self):
        self.calibrating_merged = True

    def finalize_merged(self, mag):
        self.calibrating_merged = False
        self.merged_mag = mag
        self.gamma = self.target_mag / (self.merged_mag + 1e-5)
        # Clamp gamma values using our custom max_val!
        self.gamma = clip_scale(self.gamma, self.max_val)
        self.active = True

    def deactivate(self):
        self.active = False

    def hook_fn(self, module, input, output):
        if self.calibrating_target:
            # Compute 2D FFT along spatial dimensions (-2, -1)
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
            # Average across batch (dim 0) and channel (dim 1)
            mean_mag = torch.mean(mag, dim=(0, 1)) # shape (H, W)
            self.expert_mags.append(mean_mag.detach().cpu())
            return output
            
        if self.calibrating_merged:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
            mean_mag = torch.mean(mag, dim=(0, 1)) # shape (H, W)
            self.finalize_merged(mean_mag.detach().to(output.device))
            return output
            
        if self.active:
            # Perform Frequency Domain Spectral Alignment
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            # Broadcast gamma of shape (H, W) to (B, C, H, W)
            scaled_fft = fft_out * self.gamma.view(1, 1, self.gamma.shape[0], self.gamma.shape[1])
            # Inverse FFT and take real part
            calibrated_out = torch.real(torch.fft.ifft2(scaled_fft, dim=(-2, -1)))
            return calibrated_out
            
        return output


def run_calibration_ablation(model, experts, datasets_list, N, device, max_val=5.0):
    # Register our custom Ablation_FDSA_Hook with custom max_val!
    hooks = []
    hook_objs = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hook_obj = Ablation_FDSA_Hook(max_val=max_val)
            h = module.register_forward_hook(hook_obj.hook_fn)
            hooks.append(h)
            hook_objs.append(hook_obj)
            
    # Phase 1: Collect Target statistics from Experts on task-specific calibration sets
    for i, ds in enumerate(datasets_list):
        train_dataset = get_dataset(ds, train=True)
        generator = torch.Generator().manual_seed(100 + i) # different seed per task
        calib_indices = torch.randperm(len(train_dataset), generator=generator)[:N].tolist()
        calib_subset = Subset(train_dataset, calib_indices)
        calib_loader = DataLoader(calib_subset, batch_size=N, shuffle=False)
        
        expert = experts[ds]
        exp_hooks = []
        exp_hook_objs = []
        for name, module in expert.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hook_obj = Ablation_FDSA_Hook(max_val=max_val)
                h = module.register_forward_hook(hook_obj.hook_fn)
                exp_hooks.append(h)
                exp_hook_objs.append(hook_obj)
                
        for h_obj in exp_hook_objs:
            h_obj.start_expert_calibration()
            
        inputs, _ = next(iter(calib_loader))
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = expert(inputs)
            
        for exp_h_obj, main_h_obj in zip(exp_hook_objs, hook_objs):
            main_h_obj.expert_mags.append(exp_h_obj.expert_mags[0])
            
        remove_hooks(exp_hooks)
        
    for main_h_obj in hook_objs:
        main_h_obj.finalize_target()
        
    # Phase 2: Collect Merged statistics on the Joint calibration set
    joint_inputs = []
    for i, ds in enumerate(datasets_list):
        train_dataset = get_dataset(ds, train=True)
        generator = torch.Generator().manual_seed(100 + i)
        calib_indices = torch.randperm(len(train_dataset), generator=generator)[:N].tolist()
        calib_subset = Subset(train_dataset, calib_indices)
        calib_loader = DataLoader(calib_subset, batch_size=N, shuffle=False)
        inputs, _ = next(iter(calib_loader))
        joint_inputs.append(inputs)
        
    joint_inputs = torch.cat(joint_inputs, dim=0).to(device)
    
    for main_h_obj in hook_objs:
        main_h_obj.start_merged_calibration()
        
    # Forward pass on joint calibration set
    with torch.no_grad():
        _ = model(joint_inputs)
        
    return hooks, hook_objs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for ablation sweep: {device}")
    
    # Load expert models and heads
    experts, heads, base_model = load_experts(device)
    datasets_list = ["mnist", "fmnist", "cifar10"]
    
    N = 64  # standard calibration budget
    ablation_results = {}
    
    for max_val in [2.0, 5.0, 10.0]:
        ablation_results[max_val] = {}
        print(f"\n================= Evaluating FDSA Ablation (γ_max = {max_val}) =================")
        
        for merge_method in ["wa", "ta"]:
            lam = 0.3 if merge_method == "ta" else 0.0
            print(f"Running {merge_method.upper()}...")
            
            cal_model = merge_models(experts, base_model, method=merge_method, lam=lam)
            hooks, hook_objs = run_calibration_ablation(
                cal_model, experts, datasets_list, N, device, max_val=max_val
            )
            accs = evaluate_merged(cal_model, heads, device)
            remove_hooks(hooks)
            
            ablation_results[max_val][merge_method] = accs
            print(f"  {merge_method.upper()}: MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR10: {accs['cifar10']:.2f}%, Avg: {accs['average']:.2f}%")
            
    # Write summary of ablation results to a text file
    with open("ablation_results.txt", "w") as f:
        f.write("Ablation Study: Clamping Threshold γ_max in FDSA (N=64)\n")
        f.write("===================================================\n\n")
        for max_val in [2.0, 5.0, 10.0]:
            f.write(f"Clamping Threshold γ_max = {max_val}:\n")
            for merge_method in ["wa", "ta"]:
                f.write(f"  {merge_method.upper()} Merge:\n")
                f.write(f"    MNIST: {ablation_results[max_val][merge_method]['mnist']:.2f}%\n")
                f.write(f"    FMNIST: {ablation_results[max_val][merge_method]['fmnist']:.2f}%\n")
                f.write(f"    CIFAR10: {ablation_results[max_val][merge_method]['cifar10']:.2f}%\n")
                f.write(f"    Average: {ablation_results[max_val][merge_method]['average']:.2f}%\n")
            f.write("\n")
            
    print("\nSaved ablation results to ablation_results.txt!")
