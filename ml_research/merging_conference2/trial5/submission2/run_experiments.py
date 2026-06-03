import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# Create output directories
os.makedirs("results", exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load Datasets
print("Loading datasets for calibration and testing...")
datasets_dict = {
    "mnist": datasets.MNIST(root="data", train=True, download=True, transform=data_transform),
    "fmnist": datasets.FashionMNIST(root="data", train=True, download=True, transform=data_transform),
    "cifar10": datasets.CIFAR10(root="data", train=True, download=True, transform=data_transform)
}

test_datasets_dict = {
    "mnist": datasets.MNIST(root="data", train=False, download=True, transform=data_transform),
    "fmnist": datasets.FashionMNIST(root="data", train=False, download=True, transform=data_transform),
    "cifar10": datasets.CIFAR10(root="data", train=False, download=True, transform=data_transform)
}

# Create calibration and test subsets
subsets = {}
for name, dataset in datasets_dict.items():
    subsets[name] = {
        "calib": Subset(dataset, range(5000, 5128)),
        "test": test_datasets_dict[name]
    }

# Create Test Loaders
test_loaders = {
    name: DataLoader(subsets[name]["test"], batch_size=128, shuffle=False)
    for name in subsets
}

# Helper functions for model loading and merging
def load_expert_model(task_name):
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    path = f"models/resnet18_{task_name}.pth"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expert model path {path} does not exist. Ensure experts are trained first.")
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)

def get_backbone_state_dict(model_state_dict):
    return {k: v for k, v in model_state_dict.items() if "fc" not in k}

def get_head_state_dict(model_state_dict):
    return {k.replace("fc.", ""): v for k, v in model_state_dict.items() if "fc" in k}

# Weight Averaging (WA) merging
def merge_wa(experts):
    merged_backbone = {}
    keys = list(get_backbone_state_dict(experts[0].state_dict()).keys())
    for key in keys:
        tensors = [get_backbone_state_dict(m.state_dict())[key] for m in experts]
        if tensors[0].is_floating_point():
            merged_backbone[key] = torch.mean(torch.stack(tensors), dim=0)
        else:
            merged_backbone[key] = tensors[0].clone()
    return merged_backbone

# Task Arithmetic (TA) merging
def merge_ta(experts, base_model, lam=0.3):
    merged_backbone = {}
    base_backbone = get_backbone_state_dict(base_model.state_dict())
    keys = list(base_backbone.keys())
    for key in keys:
        tensors = [get_backbone_state_dict(m.state_dict())[key] for m in experts]
        if tensors[0].is_floating_point():
            task_vectors = [t - base_backbone[key] for t in tensors]
            merged_backbone[key] = base_backbone[key] + lam * torch.sum(torch.stack(task_vectors), dim=0)
        else:
            merged_backbone[key] = base_backbone[key].clone()
    return merged_backbone

# Activation Hook classes
class ActivationCollectHook:
    def __init__(self):
        self.activations = []
    def __call__(self, module, input, output):
        # Detach and CPU to save memory
        self.activations.append(output.detach().cpu())

class ActivationScaleShiftHook:
    def __init__(self, scale, shift=None):
        self.scale = scale  # shape: (1, C, 1, 1)
        self.shift = shift  # shape: (1, C, 1, 1) or None
    def __call__(self, module, input, output):
        s = self.scale.to(output.device)
        if self.shift is not None:
            b = self.shift.to(output.device)
            return output * s + b
        return output * s

class ActivationFDSAHook:
    def __init__(self, spectral_map):
        self.spectral_map = spectral_map  # shape: (H, W) or (C, H, W)
    def __call__(self, module, input, output):
        # Apply 2D FFT along spatial dimensions
        O_fft = torch.fft.fft2(output, dim=(-2, -1))
        
        Gamma = self.spectral_map.to(output.device)
        if len(Gamma.shape) == 2:
            # L-FDSA: (H, W) -> view as (1, 1, H, W)
            Gamma = Gamma.view(1, 1, Gamma.shape[0], Gamma.shape[1])
        else:
            # C-FDSA: (C, H, W) -> view as (1, C, H, W)
            Gamma = Gamma.view(1, Gamma.shape[0], Gamma.shape[1], Gamma.shape[2])
            
        O_fft_scaled = O_fft * Gamma
        # Inverse 2D FFT
        output_scaled = torch.real(torch.fft.ifft2(O_fft_scaled, dim=(-2, -1)))
        return output_scaled

class Layer2AnchorHook:
    def __init__(self):
        self.activation = None
    def __call__(self, module, input, output):
        # Global Average Pooling: average across H, W
        self.activation = torch.mean(output, dim=(-2, -1)) # shape: (B, C)

# Function to get BatchNorm modules in order
def get_bn_modules(model):
    bn_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_modules[name] = module
    return bn_modules

# Evaluate accuracy of a merged backbone with specific heads and optional hooks
def evaluate(backbone_weights, heads_dict, test_loaders_dict, hook_handles_setup_fn=None):
    # Initialize a clean ResNet-18
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    
    # Load backbone weights
    model.load_state_dict(backbone_weights, strict=False)
    model = model.to(device)
    model.eval()
    
    # Set up hooks if any
    handles = []
    if hook_handles_setup_fn is not None:
        handles = hook_handles_setup_fn(model)
        
    results = {}
    for task_name, loader in test_loaders_dict.items():
        # Set task-specific head
        model.fc.load_state_dict(heads_dict[task_name])
        model.fc = model.fc.to(device)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        results[task_name] = 100.0 * correct / total
        
    # Remove hooks
    for h in handles:
        h.remove()
        
    results["average"] = sum(results.values()) / len(results)
    return results

# Supervised Fine-Tuning (SFT) of classification heads
def sft_heads(backbone_weights, original_heads_dict, calib_subsets, num_samples=64, epochs=10, lr=1e-3, hook_handles_setup_fn=None):
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(backbone_weights, strict=False)
    model = model.to(device)
    
    # Register hooks if any
    handles = []
    if hook_handles_setup_fn is not None:
        handles = hook_handles_setup_fn(model)
        
    # Freeze backbone
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
            
    adapted_heads = {}
    for task_name, head_sd in original_heads_dict.items():
        # Load original head
        model.fc.load_state_dict(head_sd)
        model.fc = model.fc.to(device)
        model.fc.train()
        
        # Subsample calibration set
        calib_set = calib_subsets[task_name]
        indices = range(min(num_samples, len(calib_set)))
        subsample = Subset(calib_set, indices)
        loader = DataLoader(subsample, batch_size=16, shuffle=True)
        
        optimizer = optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        adapted_heads[task_name] = {k: v.cpu().clone() for k, v in model.fc.state_dict().items()}
        
    for h in handles:
        h.remove()
        
    return adapted_heads

# Measure inference latency of a model configuration
def profile_latency(backbone_weights, head_sd, hook_handles_setup_fn=None, iterations=50, batch_size=128):
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(backbone_weights, strict=False)
    model.fc.load_state_dict(head_sd)
    model = model.to(device)
    model.eval()
    
    handles = []
    if hook_handles_setup_fn is not None:
        handles = hook_handles_setup_fn(model)
        
    # Dummy input
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    
    # Warm up
    for _ in range(5):
        _ = model(dummy_input)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000.0
    avg_latency_ms = total_time_ms / iterations
    
    for h in handles:
        h.remove()
        
    return avg_latency_ms

# Run everything
def main():
    print("\nLoading experts and setting up configurations...")
    experts = [load_expert_model("mnist"), load_expert_model("fmnist"), load_expert_model("cifar10")]
    base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    
    heads_dict = {
        "mnist": get_head_state_dict(experts[0].state_dict()),
        "fmnist": get_head_state_dict(experts[1].state_dict()),
        "cifar10": get_head_state_dict(experts[2].state_dict())
    }
    
    # 1. Models merging
    print("Performing weight merging...")
    wa_backbone = merge_wa(experts)
    ta_backbone = merge_ta(experts, base_model, lam=0.3)
    
    # Define a helper to save hook parameters
    # We will run experts and merged model to collect activation stats
    calib_sets = {name: subsets[name]["calib"] for name in subsets}
    joint_calib_set = ConcatDataset(list(calib_sets.values()))
    
    print("\n--- Collecting Activation Calibration Statistics ---")
    
    # We collect activations for each expert and the merged model (WA) to compute calibration factors
    # We'll use WA backbone as the target for our deconstruction
    bn_stats = {}
    
    # Helper to collect activations for a model
    def collect_layer_activations(backbone_weights, head_sd, dataset):
        model = resnet18()
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(backbone_weights, strict=False)
        model.fc.load_state_dict(head_sd)
        model = model.to(device)
        model.eval()
        
        bn_modules = get_bn_modules(model)
        hooks = {name: ActivationCollectHook() for name in bn_modules}
        handles = [bn_modules[name].register_forward_hook(hooks[name]) for name in bn_modules]
        
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                _ = model(images)
                
        for h in handles:
            h.remove()
            
        # Concatenate collected activations along batch dimension
        results = {}
        for name, hook in hooks.items():
            results[name] = torch.cat(hook.activations, dim=0)
        return results

    print("Collecting activations of individual experts on their calibration sets...")
    expert_acts = {
        "mnist": collect_layer_activations(get_backbone_state_dict(experts[0].state_dict()), heads_dict["mnist"], calib_sets["mnist"]),
        "fmnist": collect_layer_activations(get_backbone_state_dict(experts[1].state_dict()), heads_dict["fmnist"], calib_sets["fmnist"]),
        "cifar10": collect_layer_activations(get_backbone_state_dict(experts[2].state_dict()), heads_dict["cifar10"], calib_sets["cifar10"])
    }
    
    # Initialize calibration parameter dictionaries (which will be populated dynamically per-backbone)
    sp_taac_params = {}
    taac_params = {}
    l_fdsa_params = {}
    c_fdsa_params = {}

    def get_channel_stats(act):
        # act shape: (B, C, H, W)
        C = act.shape[1]
        act_flat = act.transpose(0, 1).reshape(C, -1)
        mean = torch.mean(act_flat, dim=1)
        std = torch.std(act_flat, dim=1)
        return mean, std

    def get_fft_magnitudes(act):
        act_fft = torch.fft.fft2(act, dim=(-2, -1))
        return torch.abs(act_fft)

    # Define setup functions for hooks
    def setup_sp_taac_hooks(model):
        bn_modules = get_bn_modules(model)
        handles = []
        for name, module in bn_modules.items():
            gamma = sp_taac_params[name]
            # scale only, no shift
            hook = ActivationScaleShiftHook(scale=gamma.view(1, 1, 1, 1))
            handle = module.register_forward_hook(hook)
            handles.append(handle)
        return handles

    def setup_taac_hooks(model):
        bn_modules = get_bn_modules(model)
        handles = []
        for name, module in bn_modules.items():
            scale = taac_params[name]["scale"]
            shift = taac_params[name]["shift"]
            hook = ActivationScaleShiftHook(scale=scale, shift=shift)
            handle = module.register_forward_hook(hook)
            handles.append(handle)
        return handles

    def setup_l_fdsa_hooks(model):
        bn_modules = get_bn_modules(model)
        handles = []
        for name, module in bn_modules.items():
            spectral_map = l_fdsa_params[name]
            hook = ActivationFDSAHook(spectral_map=spectral_map)
            handle = module.register_forward_hook(hook)
            handles.append(handle)
        return handles

    def setup_c_fdsa_hooks(model):
        bn_modules = get_bn_modules(model)
        handles = []
        for name, module in bn_modules.items():
            spectral_map = c_fdsa_params[name]
            hook = ActivationFDSAHook(spectral_map=spectral_map)
            handle = module.register_forward_hook(hook)
            handles.append(handle)
        return handles

    # ZIO-CF (Zero-Inference-Overhead Calibration Fusion)
    # We directly fuse the TAAC parameters into the BatchNorm layers in-place
    # mathematically equivalent to TAAC but with absolutely zero inference overhead!
    def apply_zio_cf_fusion(backbone_weights):
        fused_backbone = {k: v.clone() for k, v in backbone_weights.items()}
        for bn_name in taac_params.keys():
            scale_c = taac_params[bn_name]["scale"].squeeze()  # shape (C,)
            shift_c = taac_params[bn_name]["shift"].squeeze()  # shape (C,)
            
            # BatchNorm parameters in state dict:
            # {bn_name}.weight, {bn_name}.bias
            w_key = f"{bn_name}.weight"
            b_key = f"{bn_name}.bias"
            
            fused_backbone[w_key] = fused_backbone[w_key] * scale_c.to(fused_backbone[w_key].device)
            fused_backbone[b_key] = fused_backbone[b_key] * scale_c.to(fused_backbone[b_key].device) + shift_c.to(fused_backbone[b_key].device)
        return fused_backbone

    zio_cf_backbone = apply_zio_cf_fusion(wa_backbone)

    # --- 4. SRAC (Self-Routing Activation Calibration) ---
    # Prototype Extraction
    print("Extracting SRAC prototypes from Layer 2...")
    def get_srac_prototypes(backbone_weights):
        model = resnet18()
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(backbone_weights, strict=False)
        model = model.to(device)
        model.eval()
        
        # We put the anchor hook on layer2
        anchor_hook = Layer2AnchorHook()
        handle = model.layer2.register_forward_hook(anchor_hook)
        
        prototypes = {}
        for task_name, cal_dataset in calib_sets.items():
            loader = DataLoader(cal_dataset, batch_size=128, shuffle=False)
            task_acts = []
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device)
                    _ = model(images)
                    task_acts.append(anchor_hook.activation.detach().cpu())
            # Average across batch to get task prototype
            proto = torch.cat(task_acts, dim=0).mean(dim=0)
            # L2 Normalize
            proto = proto / (torch.norm(proto, p=2) + 1e-5)
            prototypes[task_name] = proto
            
        handle.remove()
        return prototypes

    srac_prototypes = get_srac_prototypes(wa_backbone)

    # SRAC evaluation function
    def evaluate_srac(backbone_weights, original_heads_dict, test_loaders_dict, beta=30.0):
        model = resnet18()
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(backbone_weights, strict=False)
        model = model.to(device)
        model.eval()
        
        # Register anchor hook
        anchor_hook = Layer2AnchorHook()
        handle = model.layer2.register_forward_hook(anchor_hook)
        
        results = {}
        for task_name, loader in test_loaders_dict.items():
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    # Forward pass
                    logits_merged = model(images)
                    
                    # Extract Layer 2 features from hook
                    v = anchor_hook.activation  # shape (B, C)
                    # Normalize v
                    v_norm = v / (torch.norm(v, p=2, dim=1, keepdim=True) + 1e-5)
                    
                    # Compute cosine similarities with all prototypes
                    sims = []
                    for k in ["mnist", "fmnist", "cifar10"]:
                        proto = srac_prototypes[k].to(device).view(1, -1)
                        # Cosine similarity for each sample in batch
                        sim = torch.sum(v_norm * proto, dim=1) # shape (B,)
                        sims.append(sim)
                    sims = torch.stack(sims, dim=1) # shape (B, 3)
                    
                    # Softmax routing weights
                    weights = torch.softmax(beta * sims, dim=1) # shape (B, 3)
                    
                    # We compute dynamic classification head output
                    # For each task head, evaluate logits
                    task_logits = []
                    for i, k in enumerate(["mnist", "fmnist", "cifar10"]):
                        # Extract features right before fc (flatten the avgpool)
                        # In ResNet-18, we can just compute the fc for each task head
                        # Let's extract backbone features by running model up to fc
                        # We can construct the logits for each task manually
                        # fc(x) = x @ weight.T + bias
                        fc_weight = original_heads_dict[k]["weight"].to(device)
                        fc_bias = original_heads_dict[k]["bias"].to(device)
                        
                        # In order to get the output from the backbone features, we can reconstruct:
                        # logits_k = features @ weight.T + bias
                        # Let's get the features: we can run a partial forward from logits_merged using the inverse of the merged fc, 
                        # or we can simply hook the inputs to model.fc!
                        # The simplest way is to register another hook on model.fc to collect its input features!
                    
            # Instead of complex feature extraction, let's write a simplified, extremely robust test-time routing:
            # We can run the backbone once, extract the features (pooled activations before fc), 
            # and then compute dynamic head outputs!
            # Let's implement this cleanly!
            pass

    # A cleaner, unified SRAC model class for robust test-time routing
    class SRACModel(nn.Module):
        def __init__(self, backbone_weights, original_heads_dict, prototypes, beta=30.0):
            super().__init__()
            self.backbone = resnet18()
            self.backbone.load_state_dict(backbone_weights, strict=False)
            self.backbone.fc = nn.Identity()  # Outputs 512-dim features
            
            self.heads = nn.ModuleDict({
                k: nn.Linear(512, 10) for k in original_heads_dict
            })
            for k in original_heads_dict:
                self.heads[k].load_state_dict(original_heads_dict[k])
                
            self.prototypes = prototypes
            self.beta = beta
            
            # Setup anchor hook
            self.anchor_hook = Layer2AnchorHook()
            self.backbone.layer2.register_forward_hook(self.anchor_hook)
            
        def forward(self, x):
            features = self.backbone(x)  # shape (B, 512)
            v = self.anchor_hook.activation  # shape (B, C)
            v_norm = v / (torch.norm(v, p=2, dim=1, keepdim=True) + 1e-5)
            
            # Cosine similarity
            sims = []
            for k in ["mnist", "fmnist", "cifar10"]:
                proto = self.prototypes[k].to(x.device).view(1, -1)
                sim = torch.sum(v_norm * proto, dim=1)
                sims.append(sim)
            sims = torch.stack(sims, dim=1)  # shape (B, 3)
            weights = torch.softmax(self.beta * sims, dim=1)  # shape (B, 3)
            
            # Evaluate all heads
            logits_dict = {
                k: self.heads[k](features) for k in ["mnist", "fmnist", "cifar10"]
            }
            # Stack logits: shape (B, 3, 10)
            stacked_logits = torch.stack([logits_dict[k] for k in ["mnist", "fmnist", "cifar10"]], dim=1)
            
            # Weighted sum: shape (B, 10)
            final_logits = torch.sum(stacked_logits * weights.unsqueeze(2), dim=1)
            return final_logits

    def evaluate_srac_model(srac_model, test_loaders_dict):
        srac_model.eval()
        results = {}
        for task_name, loader in test_loaders_dict.items():
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = srac_model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            results[task_name] = 100.0 * correct / total
        results["average"] = sum(results.values()) / len(results)
        return results

    def run_evaluation_for_backbone(backbone, name):
        nonlocal sp_taac_params, taac_params, l_fdsa_params, c_fdsa_params, zio_cf_backbone, srac_prototypes
        print(f"\n=======================================================")
        print(f"RUNNING COMPREHENSIVE SUITE FOR BACKBONE: {name}")
        print(f"=======================================================")
        
        print(f"Collecting activations of uncalibrated merged model ({name}) on the joint calibration set...")
        # Use mnist head as placeholder for merged collection
        merged_acts = collect_layer_activations(backbone, heads_dict["mnist"], joint_calib_set)
        
        print("Computing calibration parameters...")
        sp_taac_params.clear()
        taac_params.clear()
        l_fdsa_params.clear()
        c_fdsa_params.clear()
        
        for bn_name in merged_acts.keys():
            # Expert activations
            mnist_act = expert_acts["mnist"][bn_name]
            fmnist_act = expert_acts["fmnist"][bn_name]
            cifar10_act = expert_acts["cifar10"][bn_name]
            
            # Merged activations
            merged_act = merged_acts[bn_name]
            
            # --- 1. SP-TAAC (Global Spatial Scaling) ---
            mnist_std = torch.std(mnist_act)
            fmnist_std = torch.std(fmnist_act)
            cifar10_std = torch.std(cifar10_act)
            target_std_global = (mnist_std + fmnist_std + cifar10_std) / 3.0
            
            merged_std_global = torch.std(merged_act)
            gamma = target_std_global / (merged_std_global + 1e-5)
            sp_taac_params[bn_name] = gamma
            
            # --- 2. TAAC (Channel-wise Affine Alignment) ---
            C = merged_act.shape[1]
            mnist_mean_c, mnist_std_c = get_channel_stats(mnist_act)
            fmnist_mean_c, fmnist_std_c = get_channel_stats(fmnist_act)
            cifar10_mean_c, cifar10_std_c = get_channel_stats(cifar10_act)
            
            target_mean_c = (mnist_mean_c + fmnist_mean_c + cifar10_mean_c) / 3.0
            target_std_c = (mnist_std_c + fmnist_std_c + cifar10_std_c) / 3.0
            
            merged_mean_c, merged_std_c = get_channel_stats(merged_act)
            
            s_c = target_std_c / (merged_std_c + 1e-5)
            s_c_clamped = torch.clamp(s_c, 0.05, 5.0)
            b_c = target_mean_c - s_c_clamped * merged_mean_c
            
            taac_params[bn_name] = {
                "scale": s_c_clamped.view(1, C, 1, 1),
                "shift": b_c.view(1, C, 1, 1)
            }
            
            # --- 3. L-FDSA and C-FDSA (Fourier Domain) ---
            mnist_fft_mag = get_fft_magnitudes(mnist_act)
            fmnist_fft_mag = get_fft_magnitudes(fmnist_act)
            cifar10_fft_mag = get_fft_magnitudes(cifar10_act)
            merged_fft_mag = get_fft_magnitudes(merged_act)
            
            # Layer-wise (L-FDSA)
            mnist_fft_mag_l = torch.mean(mnist_fft_mag, dim=(0, 1))
            fmnist_fft_mag_l = torch.mean(fmnist_fft_mag, dim=(0, 1))
            cifar10_fft_mag_l = torch.mean(cifar10_fft_mag, dim=(0, 1))
            target_fft_mag_l = (mnist_fft_mag_l + fmnist_fft_mag_l + cifar10_fft_mag_l) / 3.0
            
            merged_fft_mag_l = torch.mean(merged_fft_mag, dim=(0, 1))
            gamma_l_fdsa = target_fft_mag_l / (merged_fft_mag_l + 1e-5)
            gamma_l_fdsa_clamped = torch.clamp(gamma_l_fdsa, 1.0/5.0, 5.0)
            l_fdsa_params[bn_name] = gamma_l_fdsa_clamped
            
            # Channel-wise (C-FDSA)
            mnist_fft_mag_c = torch.mean(mnist_fft_mag, dim=0)
            fmnist_fft_mag_c = torch.mean(fmnist_fft_mag, dim=0)
            cifar10_fft_mag_c = torch.mean(cifar10_fft_mag, dim=0)
            target_fft_mag_c = (mnist_fft_mag_c + fmnist_fft_mag_c + cifar10_fft_mag_c) / 3.0
            
            merged_fft_mag_c = torch.mean(merged_fft_mag, dim=0)
            gamma_c_fdsa = target_fft_mag_c / (merged_fft_mag_c + 1e-5)
            gamma_c_fdsa_clamped = torch.clamp(gamma_c_fdsa, 1.0/5.0, 5.0)
            c_fdsa_params[bn_name] = gamma_c_fdsa_clamped

        print("Applying ZIO-CF fusion...")
        zio_cf_backbone = apply_zio_cf_fusion(backbone)
        
        print("Extracting SRAC prototypes...")
        srac_prototypes = get_srac_prototypes(backbone)
        
        # --- Standalone Calibration Evaluations ---
        print("\n--- STANDALONE CALIBRATION (FROZEN HEADS) ---")
        standalone_results = {}
        
        print(f"Evaluating Uncalibrated {name}...")
        standalone_results[f"Uncalibrated {name}"] = evaluate(backbone, heads_dict, test_loaders)
        
        print(f"Evaluating SP-TAAC (Spatial scaling) on {name}...")
        standalone_results[f"SP-TAAC on {name}"] = evaluate(backbone, heads_dict, test_loaders, setup_sp_taac_hooks)
        
        print(f"Evaluating TAAC (Channel-wise) on {name}...")
        standalone_results[f"TAAC on {name}"] = evaluate(backbone, heads_dict, test_loaders, setup_taac_hooks)
        
        print(f"Evaluating ZIO-CF (Fusing TAAC) on {name}...")
        standalone_results[f"ZIO-CF on {name}"] = evaluate(zio_cf_backbone, heads_dict, test_loaders)
        
        print(f"Evaluating L-FDSA (Fourier spatial) on {name}...")
        standalone_results[f"L-FDSA on {name}"] = evaluate(backbone, heads_dict, test_loaders, setup_l_fdsa_hooks)
        
        print(f"Evaluating C-FDSA (Fourier channel) on {name}...")
        standalone_results[f"C-FDSA on {name}"] = evaluate(backbone, heads_dict, test_loaders, setup_c_fdsa_hooks)
        
        print(f"Evaluating SRAC (Dynamic Routing) on {name}...")
        srac_model = SRACModel(backbone, heads_dict, srac_prototypes, beta=30.0).to(device)
        standalone_results[f"SRAC on {name}"] = evaluate_srac_model(srac_model, test_loaders)
        
        for config, res in standalone_results.items():
            print(f"{config} -> MNIST: {res['mnist']:.2f}%, F-MNIST: {res['fmnist']:.2f}%, CIFAR-10: {res['cifar10']:.2f}%, Avg: {res['average']:.2f}%")
            
        # --- Head SFT sweeps ---
        print("\n--- DECONFOUNDING STUDY WITH HEAD SFT ---")
        sft_sizes = [4, 16, 64, 128]
        sft_results = {}
        
        for N in sft_sizes:
            print(f"\n--- Running Head SFT with N = {N} calibration samples per task ---")
            sft_results[N] = {}
            
            print(f"Adapting heads on Uncalibrated {name}...")
            sft_heads_base = sft_heads(backbone, heads_dict, calib_sets, num_samples=N, epochs=10, lr=1e-3)
            sft_results[N][f"Uncalibrated {name} + SFT"] = evaluate(backbone, sft_heads_base, test_loaders)
            
            print(f"Adapting heads on SP-TAAC on {name}...")
            sft_heads_sp = sft_heads(backbone, heads_dict, calib_sets, num_samples=N, epochs=10, lr=1e-3, hook_handles_setup_fn=setup_sp_taac_hooks)
            sft_results[N][f"SP-TAAC + SFT"] = evaluate(backbone, sft_heads_sp, test_loaders, setup_sp_taac_hooks)
            
            print(f"Adapting heads on ZIO-CF on {name}...")
            sft_heads_zio = sft_heads(zio_cf_backbone, heads_dict, calib_sets, num_samples=N, epochs=10, lr=1e-3)
            sft_results[N][f"ZIO-CF + SFT"] = evaluate(zio_cf_backbone, sft_heads_zio, test_loaders)
            
            print(f"Adapting heads on L-FDSA on {name}...")
            sft_heads_lfdsa = sft_heads(backbone, heads_dict, calib_sets, num_samples=N, epochs=10, lr=1e-3, hook_handles_setup_fn=setup_l_fdsa_hooks)
            sft_results[N][f"L-FDSA + SFT"] = evaluate(backbone, sft_heads_lfdsa, test_loaders, setup_l_fdsa_hooks)
            
            for config, res in sft_results[N].items():
                print(f"N={N}: {config} -> MNIST: {res['mnist']:.2f}%, F-MNIST: {res['fmnist']:.2f}%, CIFAR-10: {res['cifar10']:.2f}%, Avg: {res['average']:.2f}%")
                
        return standalone_results, sft_results

    # Run comprehensive evaluations for both Weight Averaging (WA) and Task Arithmetic (TA)
    standalone_results_wa, sft_results_wa = run_evaluation_for_backbone(wa_backbone, "WA")
    standalone_results_ta, sft_results_ta = run_evaluation_for_backbone(ta_backbone, "TA")
    
    # Expose WA results as the default variables for latency profiling and compatibility
    standalone_results = standalone_results_wa
    sft_results = sft_results_wa

    print("\n=======================================================")
    print("EXPERIMENTAL EVALUATION: LATENCY AND COMPUTATIONAL PROFILING")
    print("=======================================================")
    
    latency_results = {}
    
    # Placeholder head for latency profiling
    dummy_head = heads_dict["mnist"]
    
    # 1. Base Uncalibrated WA Latency
    print("Profiling Uncalibrated WA...")
    latency_results["Uncalibrated WA"] = profile_latency(wa_backbone, dummy_head)
    
    # 2. SP-TAAC Latency
    print("Profiling SP-TAAC...")
    latency_results["SP-TAAC"] = profile_latency(wa_backbone, dummy_head, setup_sp_taac_hooks)
    
    # 3. TAAC Latency
    print("Profiling TAAC...")
    latency_results["TAAC"] = profile_latency(wa_backbone, dummy_head, setup_taac_hooks)
    
    # 4. ZIO-CF Latency
    print("Profiling ZIO-CF...")
    latency_results["ZIO-CF"] = profile_latency(zio_cf_backbone, dummy_head)
    
    # 5. L-FDSA Latency
    print("Profiling L-FDSA...")
    latency_results["L-FDSA"] = profile_latency(wa_backbone, dummy_head, setup_l_fdsa_hooks)
    
    # 6. C-FDSA Latency
    print("Profiling C-FDSA...")
    latency_results["C-FDSA"] = profile_latency(wa_backbone, dummy_head, setup_c_fdsa_hooks)
    
    # 7. SRAC Latency
    print("Profiling SRAC...")
    # Wrap in our SRACModel class
    srac_l_model = SRACModel(wa_backbone, heads_dict, srac_prototypes, beta=30.0).to(device)
    # Profile srac forward latency
    dummy_input = torch.randn(128, 3, 32, 32).to(device)
    for _ in range(5):
        _ = srac_l_model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = srac_l_model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    latency_results["SRAC"] = ((end_time - start_time) * 1000.0) / 50.0
    
    # Print Latency Results
    for config, lat in latency_results.items():
        overhead = ((lat - latency_results["Uncalibrated WA"]) / latency_results["Uncalibrated WA"]) * 100.0
        print(f"{config} Latency: {lat:.4f} ms per batch (Overhead: {overhead:+.2f}%)")

    # --- SAVE RESULTS ---
    output_data = {
        "standalone": standalone_results_wa,
        "sft": sft_results_wa,
        "standalone_wa": standalone_results_wa,
        "standalone_ta": standalone_results_ta,
        "sft_wa": sft_results_wa,
        "sft_ta": sft_results_ta,
        "latency": latency_results
    }
    with open("results/all_experiments_results.json", "w") as f:
        json.dump(output_data, f, indent=4)
    print("\nSaved all experimental results to results/all_experiments_results.json.")

if __name__ == "__main__":
    main()
