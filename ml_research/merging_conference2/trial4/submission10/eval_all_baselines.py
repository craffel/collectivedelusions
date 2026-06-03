import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import copy
import numpy as np

# Set random seeds and disable cuDNN for stability
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled = False

def get_transforms(dataset_name):
    if dataset_name in ["mnist", "fmnist"]:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else: # cifar10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transform

def get_dataset(name, train=True):
    transform = get_transforms(name)
    if name == "mnist":
        dataset = datasets.MNIST(root="./data", train=train, download=False, transform=transform)
    elif name == "fmnist":
        dataset = datasets.FashionMNIST(root="./data", train=train, download=False, transform=transform)
    elif name == "cifar10":
        dataset = datasets.CIFAR10(root="./data", train=train, download=False, transform=transform)
    return dataset

# 1. Load Expert Models and classification heads
def load_experts(device):
    datasets_list = ["mnist", "fmnist", "cifar10"]
    experts = {}
    heads = {}
    
    # Load base model to get the state dict structure
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(512, 10)
    
    for ds in datasets_list:
        model = copy.deepcopy(base_model)
        model.load_state_dict(torch.load(f"weights/expert_{ds}.pth", map_location=device))
        model = model.to(device)
        model.eval()
        experts[ds] = model
        # Extract classification head
        heads[ds] = copy.deepcopy(model.fc)
        
    return experts, heads, base_model.to(device)

# 2. Perform Merging
def merge_models(experts, base_model, method="wa", lam=0.3):
    merged_model = copy.deepcopy(base_model)
    merged_state = merged_model.state_dict()
    
    expert_states = {k: v.state_dict() for k, v in experts.items()}
    base_state = base_model.state_dict()
    
    # We do not merge the 'fc' classification heads
    keys_to_merge = [k for k in merged_state.keys() if "fc" not in k]
    
    if method == "wa":
        for key in keys_to_merge:
            tensors = [expert_states[ds][key] for ds in experts]
            if tensors[0].dtype.is_floating_point or tensors[0].dtype.is_complex:
                merged_state[key] = torch.stack(tensors).mean(dim=0)
            else:
                merged_state[key] = tensors[0]
    elif method == "ta":
        for key in keys_to_merge:
            tensors = [expert_states[ds][key] for ds in experts]
            if tensors[0].dtype.is_floating_point or tensors[0].dtype.is_complex:
                task_vectors = [expert_states[ds][key] - base_state[key] for ds in experts]
                merged_state[key] = base_state[key] + lam * sum(task_vectors)
            else:
                merged_state[key] = base_state[key]
            
    merged_model.load_state_dict(merged_state)
    return merged_model

# 3. Evaluation on all test sets
def evaluate_merged(model, heads, device):
    datasets_list = ["mnist", "fmnist", "cifar10"]
    accuracies = {}
    
    for ds in datasets_list:
        test_dataset = get_dataset(ds, train=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
        
        # Swap head
        model.fc = copy.deepcopy(heads[ds]).to(device)
        model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        acc = 100.0 * correct / total
        accuracies[ds] = acc
        
    accuracies["average"] = sum(accuracies.values()) / len(datasets_list)
    return accuracies

def clip_scale(scale, max_val=5.0):
    if isinstance(scale, torch.Tensor):
        return torch.clamp(scale, 1.0/max_val, max_val)
    return min(max(scale, 1.0/max_val), max_val)

# 4. Calibration Hook Classes

# Baselines:
class SPTAAC_Hook:
    """ Layer-wise Spatial Scaling """
    def __init__(self):
        self.calibrating_target = False
        self.calibrating_merged = False
        self.active = False
        self.expert_stds = []
        self.target_std = None
        self.merged_std = None
        self.gamma = 1.0

    def start_expert_calibration(self):
        self.calibrating_target = True
        self.expert_stds = []

    def save_expert_std(self, std):
        self.expert_stds.append(std)

    def finalize_target(self):
        self.calibrating_target = False
        self.target_std = sum(self.expert_stds) / len(self.expert_stds)

    def start_merged_calibration(self):
        self.calibrating_merged = True

    def finalize_merged(self, std):
        self.calibrating_merged = False
        self.merged_std = std
        self.gamma = self.target_std / (self.merged_std + 1e-5)
        self.gamma = clip_scale(self.gamma)
        self.active = True

    def deactivate(self):
        self.active = False

    def hook_fn(self, module, input, output):
        if self.calibrating_target:
            std = torch.sqrt(torch.var(output, dim=(0, 1, 2, 3), unbiased=False) + 1e-5).item()
            self.save_expert_std(std)
            return output
            
        if self.calibrating_merged:
            std = torch.sqrt(torch.var(output, dim=(0, 1, 2, 3), unbiased=False) + 1e-5).item()
            self.finalize_merged(std)
            return output
            
        if self.active:
            return output * self.gamma
            
        return output


class TCAC_Hook:
    """ Channel-wise Spatial Scaling (Task-specific Channel-wise Activation Calibration) """
    def __init__(self):
        self.calibrating_target = False
        self.calibrating_merged = False
        self.active = False
        self.expert_stds = []
        self.target_std = None
        self.merged_std = None
        self.gamma = 1.0

    def start_expert_calibration(self):
        self.calibrating_target = True
        self.expert_stds = []

    def save_expert_std(self, std):
        self.expert_stds.append(std)

    def finalize_target(self):
        self.calibrating_target = False
        self.target_std = torch.stack(self.expert_stds).mean(dim=0)

    def start_merged_calibration(self):
        self.calibrating_merged = True

    def finalize_merged(self, std):
        self.calibrating_merged = False
        self.merged_std = std
        self.gamma = self.target_std.to(std.device) / (self.merged_std + 1e-5)
        self.gamma = clip_scale(self.gamma)
        self.active = True

    def deactivate(self):
        self.active = False

    def hook_fn(self, module, input, output):
        if self.calibrating_target:
            std = torch.sqrt(torch.var(output, dim=(0, 2, 3), unbiased=False) + 1e-5) # shape (C,)
            self.save_expert_std(std.detach().cpu())
            return output
            
        if self.calibrating_merged:
            std = torch.sqrt(torch.var(output, dim=(0, 2, 3), unbiased=False) + 1e-5)
            self.finalize_merged(std.detach().to(output.device))
            return output
            
        if self.active:
            return output * self.gamma.view(1, -1, 1, 1)
            
        return output


class STCAC_Hook:
    """ Shrinkage TCAC: interpolates between channel-wise and layer-wise scaling to avoid Sparsity Trap """
    def __init__(self, alpha=0.5):
        self.calibrating_target = False
        self.calibrating_merged = False
        self.active = False
        self.expert_stds = []
        self.target_std = None
        self.merged_std = None
        self.gamma = 1.0
        self.alpha = alpha

    def start_expert_calibration(self):
        self.calibrating_target = True
        self.expert_stds = []

    def save_expert_std(self, std):
        self.expert_stds.append(std)

    def finalize_target(self):
        self.calibrating_target = False
        self.target_std = torch.stack(self.expert_stds).mean(dim=0)

    def start_merged_calibration(self):
        self.calibrating_merged = True

    def finalize_merged(self, std):
        self.calibrating_merged = False
        self.merged_std = std
        
        # Move target_std to correct device
        target_std_dev = self.target_std.to(std.device)
        
        # TCAC channel-wise scaling
        gamma_tcac = target_std_dev / (self.merged_std + 1e-5)
        gamma_tcac = clip_scale(gamma_tcac)
        
        # Layer-wise scaling
        target_std_layer = target_std_dev.mean()
        merged_std_layer = self.merged_std.mean()
        gamma_layer = target_std_layer / (merged_std_layer + 1e-5)
        gamma_layer = clip_scale(gamma_layer)
        
        # Shrinkage interpolation
        self.gamma = (1 - self.alpha) * gamma_tcac + self.alpha * gamma_layer
        self.active = True

    def deactivate(self):
        self.active = False

    def hook_fn(self, module, input, output):
        if self.calibrating_target:
            std = torch.sqrt(torch.var(output, dim=(0, 2, 3), unbiased=False) + 1e-5)
            self.save_expert_std(std.detach().cpu())
            return output
            
        if self.calibrating_merged:
            std = torch.sqrt(torch.var(output, dim=(0, 2, 3), unbiased=False) + 1e-5)
            self.finalize_merged(std.detach().to(output.device))
            return output
            
        if self.active:
            return output * self.gamma.view(1, -1, 1, 1)
            
        return output


# Proposed Frequency Domain Methods:
class LFDSA_Hook:
    """ Layer-wise FDSA: shared magnitude spectral scaling across channels """
    def __init__(self):
        self.calibrating_target = False
        self.calibrating_merged = False
        self.active = False
        self.expert_mags = []
        self.target_mag = None # shape (H, W)
        self.merged_mag = None # shape (H, W)
        self.gamma = 1.0 # shape (H, W)

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
        self.gamma = self.target_mag.to(mag.device) / (self.merged_mag + 1e-5)
        self.gamma = clip_scale(self.gamma)
        self.active = True

    def deactivate(self):
        self.active = False

    def hook_fn(self, module, input, output):
        if self.calibrating_target:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
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
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            scaled_fft = fft_out * self.gamma.view(1, 1, self.gamma.shape[0], self.gamma.shape[1])
            calibrated_out = torch.real(torch.fft.ifft2(scaled_fft, dim=(-2, -1)))
            return calibrated_out
            
        return output


class CFDSA_Hook:
    """ Channel-wise FDSA: independent magnitude spectral scaling per channel """
    def __init__(self):
        self.calibrating_target = False
        self.calibrating_merged = False
        self.active = False
        self.expert_mags = []
        self.target_mag = None # shape (C, H, W)
        self.merged_mag = None # shape (C, H, W)
        self.gamma = 1.0 # shape (C, H, W)

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
        self.gamma = self.target_mag.to(mag.device) / (self.merged_mag + 1e-5)
        self.gamma = clip_scale(self.gamma)
        self.active = True

    def deactivate(self):
        self.active = False

    def hook_fn(self, module, input, output):
        if self.calibrating_target:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
            mean_mag = torch.mean(mag, dim=0) # shape (C, H, W)
            self.expert_mags.append(mean_mag.detach().cpu())
            return output
            
        if self.calibrating_merged:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
            mean_mag = torch.mean(mag, dim=0) # shape (C, H, W)
            self.finalize_merged(mean_mag.detach().to(output.device))
            return output
            
        if self.active:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            scaled_fft = fft_out * self.gamma.unsqueeze(0)
            calibrated_out = torch.real(torch.fft.ifft2(scaled_fft, dim=(-2, -1)))
            return calibrated_out
            
        return output


class LFDSA_PR_Hook:
    """ Layer-wise FDSA with Phase Realignment (L-FDSA-PR) """
    def __init__(self):
        self.calibrating_target = False
        self.calibrating_merged = False
        self.active = False
        self.expert_mags = []
        self.expert_phasors = []
        self.target_mag = None # shape (H, W)
        self.target_phasor = None # shape (H, W), complex
        self.merged_mag = None # shape (H, W)
        self.merged_phasor = None # shape (H, W), complex
        self.gamma = 1.0 # shape (H, W)
        self.R = 1.0 # shape (H, W), complex

    def start_expert_calibration(self):
        self.calibrating_target = True
        self.expert_mags = []
        self.expert_phasors = []

    def finalize_target(self):
        self.calibrating_target = False
        self.target_mag = torch.stack(self.expert_mags).mean(dim=0)
        avg_phasor = torch.stack(self.expert_phasors).mean(dim=0)
        self.target_phasor = avg_phasor / (torch.abs(avg_phasor) + 1e-5)

    def start_merged_calibration(self):
        self.calibrating_merged = True

    def finalize_merged(self, mag, phasor):
        self.calibrating_merged = False
        self.merged_mag = mag
        self.merged_phasor = phasor / (torch.abs(phasor) + 1e-5)
        self.gamma = self.target_mag.to(mag.device) / (self.merged_mag + 1e-5)
        self.gamma = clip_scale(self.gamma)
        R_raw = self.target_phasor.to(mag.device) * torch.conj(self.merged_phasor)
        self.R = R_raw / (torch.abs(R_raw) + 1e-5)
        self.active = True

    def deactivate(self):
        self.active = False

    def hook_fn(self, module, input, output):
        if self.calibrating_target:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
            mean_mag = torch.mean(mag, dim=(0, 1)) # shape (H, W)
            self.expert_mags.append(mean_mag.detach().cpu())
            
            phasor = fft_out / (mag + 1e-5)
            mean_phasor = torch.mean(phasor, dim=(0, 1)) # shape (H, W)
            self.expert_phasors.append(mean_phasor.detach().cpu())
            return output
            
        if self.calibrating_merged:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
            mean_mag = torch.mean(mag, dim=(0, 1)) # shape (H, W)
            
            phasor = fft_out / (mag + 1e-5)
            mean_phasor = torch.mean(phasor, dim=(0, 1)) # shape (H, W)
            
            self.finalize_merged(
                mean_mag.detach().to(output.device),
                mean_phasor.detach().to(output.device)
            )
            return output
            
        if self.active:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            g_bcast = self.gamma.view(1, 1, self.gamma.shape[0], self.gamma.shape[1])
            r_bcast = self.R.view(1, 1, self.R.shape[0], self.R.shape[1])
            scaled_fft = fft_out * g_bcast * r_bcast
            calibrated_out = torch.real(torch.fft.ifft2(scaled_fft, dim=(-2, -1)))
            return calibrated_out
            
        return output


class CFDSA_PR_Hook:
    """ Channel-wise FDSA with Phase Realignment (C-FDSA-PR) """
    def __init__(self):
        self.calibrating_target = False
        self.calibrating_merged = False
        self.active = False
        self.expert_mags = []
        self.expert_phasors = []
        self.target_mag = None # shape (C, H, W)
        self.target_phasor = None # shape (C, H, W), complex
        self.merged_mag = None # shape (C, H, W)
        self.merged_phasor = None # shape (C, H, W), complex
        self.gamma = 1.0 # shape (C, H, W)
        self.R = 1.0 # shape (C, H, W), complex

    def start_expert_calibration(self):
        self.calibrating_target = True
        self.expert_mags = []
        self.expert_phasors = []

    def finalize_target(self):
        self.calibrating_target = False
        self.target_mag = torch.stack(self.expert_mags).mean(dim=0)
        avg_phasor = torch.stack(self.expert_phasors).mean(dim=0)
        self.target_phasor = avg_phasor / (torch.abs(avg_phasor) + 1e-5)

    def start_merged_calibration(self):
        self.calibrating_merged = True

    def finalize_merged(self, mag, phasor):
        self.calibrating_merged = False
        self.merged_mag = mag
        self.merged_phasor = phasor / (torch.abs(phasor) + 1e-5)
        self.gamma = self.target_mag.to(mag.device) / (self.merged_mag + 1e-5)
        self.gamma = clip_scale(self.gamma)
        R_raw = self.target_phasor.to(mag.device) * torch.conj(self.merged_phasor)
        self.R = R_raw / (torch.abs(R_raw) + 1e-5)
        self.active = True

    def deactivate(self):
        self.active = False

    def hook_fn(self, module, input, output):
        if self.calibrating_target:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
            mean_mag = torch.mean(mag, dim=0) # shape (C, H, W)
            self.expert_mags.append(mean_mag.detach().cpu())
            
            phasor = fft_out / (mag + 1e-5)
            mean_phasor = torch.mean(phasor, dim=0) # shape (C, H, W)
            self.expert_phasors.append(mean_phasor.detach().cpu())
            return output
            
        if self.calibrating_merged:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            mag = torch.abs(fft_out)
            mean_mag = torch.mean(mag, dim=0) # shape (C, H, W)
            
            phasor = fft_out / (mag + 1e-5)
            mean_phasor = torch.mean(phasor, dim=0) # shape (C, H, W)
            
            self.finalize_merged(
                mean_mag.detach().to(output.device),
                mean_phasor.detach().to(output.device)
            )
            return output
            
        if self.active:
            fft_out = torch.fft.fft2(output, dim=(-2, -1))
            scaled_fft = fft_out * self.gamma.unsqueeze(0) * self.R.unsqueeze(0)
            calibrated_out = torch.real(torch.fft.ifft2(scaled_fft, dim=(-2, -1)))
            return calibrated_out
            
        return output


# Helper to register and handle hooks
def register_calibration_hooks(model, hook_class):
    hooks = []
    hook_objects = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hook_obj = hook_class()
            h = module.register_forward_hook(hook_obj.hook_fn)
            hooks.append(h)
            hook_objects.append(hook_obj)
    return hooks, hook_objects

def remove_hooks(hooks):
    for h in hooks:
        h.remove()

# Calibration execution
def run_calibration(model, experts, heads, datasets_list, N, device, calibration_type="sp-taac"):
    if calibration_type == "sp-taac":
        hooks, hook_objs = register_calibration_hooks(model, SPTAAC_Hook)
    elif calibration_type == "tcac":
        hooks, hook_objs = register_calibration_hooks(model, TCAC_Hook)
    elif calibration_type == "s-tcac":
        hooks, hook_objs = register_calibration_hooks(model, STCAC_Hook)
    elif calibration_type == "l-fdsa":
        hooks, hook_objs = register_calibration_hooks(model, LFDSA_Hook)
    elif calibration_type == "c-fdsa":
        hooks, hook_objs = register_calibration_hooks(model, CFDSA_Hook)
    elif calibration_type == "l-fdsa-pr":
        hooks, hook_objs = register_calibration_hooks(model, LFDSA_PR_Hook)
    elif calibration_type == "c-fdsa-pr":
        hooks, hook_objs = register_calibration_hooks(model, CFDSA_PR_Hook)
    else:
        raise ValueError(f"Unknown calibration type {calibration_type}")
        
    # Phase 1: Collect Target statistics from Experts on task-specific calibration sets
    for i, ds in enumerate(datasets_list):
        train_dataset = get_dataset(ds, train=True)
        generator = torch.Generator().manual_seed(100 + i) # different seed per task
        calib_indices = torch.randperm(len(train_dataset), generator=generator)[:N].tolist()
        calib_subset = Subset(train_dataset, calib_indices)
        calib_loader = DataLoader(calib_subset, batch_size=N, shuffle=False)
        
        expert = experts[ds]
        # Attach temporary hooks to the expert to collect statistics
        if calibration_type == "sp-taac":
            exp_hooks, exp_hook_objs = register_calibration_hooks(expert, SPTAAC_Hook)
        elif calibration_type == "tcac":
            exp_hooks, exp_hook_objs = register_calibration_hooks(expert, TCAC_Hook)
        elif calibration_type == "s-tcac":
            exp_hooks, exp_hook_objs = register_calibration_hooks(expert, STCAC_Hook)
        elif calibration_type == "l-fdsa":
            exp_hooks, exp_hook_objs = register_calibration_hooks(expert, LFDSA_Hook)
        elif calibration_type == "c-fdsa":
            exp_hooks, exp_hook_objs = register_calibration_hooks(expert, CFDSA_Hook)
        elif calibration_type == "l-fdsa-pr":
            exp_hooks, exp_hook_objs = register_calibration_hooks(expert, LFDSA_PR_Hook)
        elif calibration_type == "c-fdsa-pr":
            exp_hooks, exp_hook_objs = register_calibration_hooks(expert, CFDSA_PR_Hook)
            
        for h_obj in exp_hook_objs:
            h_obj.start_expert_calibration()
            
        inputs, _ = next(iter(calib_loader))
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = expert(inputs)
            
        # Transfer gathered stats to our main merged model hooks
        for exp_h_obj, main_h_obj in zip(exp_hook_objs, hook_objs):
            if calibration_type in ["sp-taac", "tcac", "s-tcac"]:
                main_h_obj.expert_stds.append(exp_h_obj.expert_stds[0])
            else:
                main_h_obj.expert_mags.append(exp_h_obj.expert_mags[0])
                if hasattr(exp_h_obj, "expert_phasors"):
                    main_h_obj.expert_phasors.append(exp_h_obj.expert_phasors[0])
                
        # Remove expert hooks
        remove_hooks(exp_hooks)
        
    # Finalize target statistics
    for main_h_obj in hook_objs:
        main_h_obj.finalize_target()
        
    # Phase 2: Collect Merged statistics on the Joint calibration set
    # Joint dataset size is 3 * N
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
    
    # Enable calibration gathering on merged model
    for main_h_obj in hook_objs:
        main_h_obj.start_merged_calibration()
        
    # Forward pass of joint calibration set
    # Swap head of merged model to some dummy/first head during stats collection
    model.fc = copy.deepcopy(heads[datasets_list[0]]).to(device)
    with torch.no_grad():
        _ = model(joint_inputs)
        
    return hooks, hook_objs

# 5. Head Adaptation via Supervised SFT (from REDA)
def run_head_sft(model, heads, datasets_list, N, device, epochs=15):
    adapted_heads = {}
    model.eval()
    
    for i, ds in enumerate(datasets_list):
        train_dataset = get_dataset(ds, train=True)
        generator = torch.Generator().manual_seed(100 + i)
        calib_indices = torch.randperm(len(train_dataset), generator=generator)[:N].tolist()
        calib_subset = Subset(train_dataset, calib_indices)
        calib_loader = DataLoader(calib_subset, batch_size=32, shuffle=True)
        
        # Clone expert classification head
        adapted_head = copy.deepcopy(heads[ds]).to(device)
        adapted_head.train()
        
        optimizer = torch.optim.AdamW(adapted_head.parameters(), lr=1e-3, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for inputs, labels in calib_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                # Extract features through the backbone with hooks active
                with torch.no_grad():
                    # Temporarily replace fc with Identity to get features
                    model.fc = nn.Identity()
                    features = model(inputs)
                    
                outputs = adapted_head(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        adapted_heads[ds] = adapted_head.eval()
        
    return adapted_heads


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load expert models and heads
    experts, heads, base_model = load_experts(device)
    datasets_list = ["mnist", "fmnist", "cifar10"]
    
    results = {}
    
    # Evaluate Experts alone (Oracles)
    print("\nEvaluating Individual Expert Oracles...")
    oracle_accs = {}
    for ds in datasets_list:
        acc = evaluate_merged(experts[ds], heads, device)[ds]
        oracle_accs[ds] = acc
    oracle_accs["average"] = sum(oracle_accs.values()) / len(datasets_list)
    results["Oracle"] = oracle_accs
    print(f"Oracle: MNIST: {oracle_accs['mnist']:.2f}%, FMNIST: {oracle_accs['fmnist']:.2f}%, CIFAR10: {oracle_accs['cifar10']:.2f}%, Avg: {oracle_accs['average']:.2f}%")

    # Evaluate Merging Configurations
    for merge_method in ["wa", "ta"]:
        lam = 0.3 if merge_method == "ta" else 0.0
        method_name = f"{merge_method.upper()}" + (f" (λ={lam})" if merge_method == "ta" else "")
        print(f"\n================= {method_name} Model Merging =================")
        
        # 1. Uncalibrated
        merged_model = merge_models(experts, base_model, method=merge_method, lam=lam)
        uncal_accs = evaluate_merged(merged_model, heads, device)
        results[f"{merge_method}_uncalibrated"] = uncal_accs
        print(f"Uncalibrated: MNIST: {uncal_accs['mnist']:.2f}%, FMNIST: {uncal_accs['fmnist']:.2f}%, CIFAR10: {uncal_accs['cifar10']:.2f}%, Avg: {uncal_accs['average']:.2f}%")
        
        # Uncalibrated + Head SFT (N=64)
        print("Evaluating Uncalibrated + Head SFT (N=64)...")
        uncal_model_sft = merge_models(experts, base_model, method=merge_method, lam=lam)
        adapted_heads_uncal = run_head_sft(uncal_model_sft, heads, datasets_list, 64, device)
        uncal_sft_accs = evaluate_merged(uncal_model_sft, adapted_heads_uncal, device)
        results[f"{merge_method}_uncalibrated_sft_N64"] = uncal_sft_accs
        print(f"Uncalibrated + SFT (N=64): MNIST: {uncal_sft_accs['mnist']:.2f}%, FMNIST: {uncal_sft_accs['fmnist']:.2f}%, CIFAR10: {uncal_sft_accs['cifar10']:.2f}%, Avg: {uncal_sft_accs['average']:.2f}%")

        # Sweep over all calibration methods and budgets N
        cal_methods = ["sp-taac", "tcac", "s-tcac", "l-fdsa", "c-fdsa", "l-fdsa-pr", "c-fdsa-pr"]
        for N in [16, 64, 128]:
            print(f"\n--- Calibration Budget N = {N} ---")
            
            for cal_method in cal_methods:
                print(f"Evaluating {cal_method.upper()} (N={N})...")
                cal_model = merge_models(experts, base_model, method=merge_method, lam=lam)
                hooks, hook_objs = run_calibration(cal_model, experts, heads, datasets_list, N, device, calibration_type=cal_method)
                accs = evaluate_merged(cal_model, heads, device)
                remove_hooks(hooks)
                
                results[f"{merge_method}_{cal_method}_N{N}"] = accs
                print(f"{cal_method.upper()} (N={N}): MNIST: {accs['mnist']:.2f}%, FMNIST: {accs['fmnist']:.2f}%, CIFAR10: {accs['cifar10']:.2f}%, Avg: {accs['average']:.2f}%")
                
                # Evaluation + SFT only for N=64 (our focal benchmark)
                if N == 64:
                    print(f"Evaluating {cal_method.upper()} + Head SFT (N=64)...")
                    cal_model_sft = merge_models(experts, base_model, method=merge_method, lam=lam)
                    hooks, hook_objs = run_calibration(cal_model_sft, experts, heads, datasets_list, N, device, calibration_type=cal_method)
                    adapted_heads = run_head_sft(cal_model_sft, heads, datasets_list, N, device)
                    sft_accs = evaluate_merged(cal_model_sft, adapted_heads, device)
                    remove_hooks(hooks)
                    
                    results[f"{merge_method}_{cal_method}_sft_N64"] = sft_accs
                    print(f"{cal_method.upper()} + SFT (N=64): MNIST: {sft_accs['mnist']:.2f}%, FMNIST: {sft_accs['fmnist']:.2f}%, CIFAR10: {sft_accs['cifar10']:.2f}%, Avg: {sft_accs['average']:.2f}%")

    # Write full detailed summary of experimental results to a text file
    with open("all_baselines_results.txt", "w") as f:
        f.write("Multi-Task Model Merging: Expanded Baselines & Methods Results Summary\n")
        f.write("====================================================================\n\n")
        f.write(f"Oracle Experts Average: {results['Oracle']['average']:.2f}%\n")
        f.write(f"MNIST Oracle: {results['Oracle']['mnist']:.2f}%\n")
        f.write(f"FMNIST Oracle: {results['Oracle']['fmnist']:.2f}%\n")
        f.write(f"CIFAR10 Oracle: {results['Oracle']['cifar10']:.2f}%\n\n")
        
        for m in ["wa", "ta"]:
            f.write(f"Merge Method: {m.upper()}\n")
            f.write(f"  Uncalibrated: {results[f'{m}_uncalibrated']['average']:.2f}%\n")
            f.write(f"  Uncalibrated + Head SFT (N=64): {results[f'{m}_uncalibrated_sft_N64']['average']:.2f}%\n")
            for N in [16, 64, 128]:
                f.write(f"  N = {N}:\n")
                for cal_method in cal_methods:
                    f.write(f"    {cal_method.upper()}: {results[f'{m}_{cal_method}_N{N}']['average']:.2f}%\n")
                    if N == 64:
                        f.write(f"    {cal_method.upper()} + Head SFT: {results[f'{m}_{cal_method}_sft_N64']['average']:.2f}%\n")
            f.write("\n")
    print("\nSaved summary of all experimental results to all_baselines_results.txt")
