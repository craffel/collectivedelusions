import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import os
import copy
import json
import numpy as np

# Set device and disable cuDNN if it causes issues
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors.")

# -------------------------------------------------------------
# 1. Dataset Preparation
# -------------------------------------------------------------
print("Preparing datasets...")
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

# ImageNet normalization for ResNet-18
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # replicate to 3 channels
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_rgb = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_mnist = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_gray)
test_mnist = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_gray)

train_fmnist = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform_gray)
test_fmnist = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform_gray)

train_cifar = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_rgb)
test_cifar = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_rgb)

# We train on full training datasets for high-quality experts
# Create DataLoaders
batch_size = 256
loader_args = {"batch_size": batch_size, "num_workers": 2, "pin_memory": True}

train_loaders = {
    "mnist": DataLoader(train_mnist, shuffle=True, **loader_args),
    "fmnist": DataLoader(train_fmnist, shuffle=True, **loader_args),
    "cifar10": DataLoader(train_cifar, shuffle=True, **loader_args)
}

test_loaders = {
    "mnist": DataLoader(test_mnist, shuffle=False, **loader_args),
    "fmnist": DataLoader(test_fmnist, shuffle=False, **loader_args),
    "cifar10": DataLoader(test_cifar, shuffle=False, **loader_args)
}

# -------------------------------------------------------------
# 2. Expert Training or Loading
# -------------------------------------------------------------
experts_dir = "./experts"
os.makedirs(experts_dir, exist_ok=True)

# Shared pre-trained progenitor
print("Loading ImageNet-pretrained ResNet-18 progenitor...")
progenitor = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# Keep a copy of initial pre-trained weights for Task Arithmetic
progenitor_state = copy.deepcopy(progenitor.state_dict())

def train_expert(task_name, train_loader, epochs=5):
    expert_path = os.path.join(experts_dir, f"expert_{task_name}.pt")
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    if os.path.exists(expert_path):
        print(f"Loading pre-trained expert for {task_name} from {expert_path}...")
        model.load_state_dict(torch.load(expert_path, map_location=device))
        return model
    
    print(f"Training expert for {task_name} for {epochs} epochs...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
        
    torch.save(model.state_dict(), expert_path)
    print(f"Saved expert weights to {expert_path}.")
    return model

# Train the experts
expert_mnist = train_expert("mnist", train_loaders["mnist"])
expert_fmnist = train_expert("fmnist", train_loaders["fmnist"])
expert_cifar10 = train_expert("cifar10", train_loaders["cifar10"])

experts = {
    "mnist": expert_mnist,
    "fmnist": expert_fmnist,
    "cifar10": expert_cifar10
}

# -------------------------------------------------------------
# 3. Evaluation Function
# -------------------------------------------------------------
def evaluate_model(backbone, experts, test_loaders):
    """
    Evaluate a merged backbone using the task-specific fc heads of the experts.
    """
    backbone.eval()
    results = {}
    with torch.no_grad():
        for task_name, loader in test_loaders.items():
            # Create a model with the merged backbone but task-specific head
            eval_model = copy.deepcopy(backbone)
            # Load the task-specific head from the corresponding expert
            eval_model.fc = copy.deepcopy(experts[task_name].fc)
            eval_model = eval_model.to(device)
            eval_model.eval()
            
            correct = 0
            total = 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = eval_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            accuracy = 100.0 * correct / total
            results[task_name] = accuracy
            print(f"Task: {task_name:10s} | Accuracy: {accuracy:.2f}%")
            
    avg_accuracy = np.mean(list(results.values()))
    results["average"] = avg_accuracy
    print(f"Average Accuracy: {avg_accuracy:.2f}%")
    return results

print("\n--- Evaluating Oracle Experts (No Merging) ---")
oracle_results = {}
for task_name in ["mnist", "fmnist", "cifar10"]:
    print(f"Evaluating individual expert {task_name} on its test set...")
    oracle_results[task_name] = evaluate_model(experts[task_name], {task_name: experts[task_name]}, {task_name: test_loaders[task_name]})[task_name]
oracle_results["average"] = np.mean(list(oracle_results.values()))
print(f"Oracle Experts Average: {oracle_results['average']:.2f}%")

# -------------------------------------------------------------
# 4. Standard Merging Functions
# -------------------------------------------------------------
def get_standard_merge(models_dict, progenitor_state, merge_type="WA", lam=0.2):
    """
    Constructs standard merged state_dict for the backbone.
    """
    keys = [k for k in progenitor_state.keys() if "fc" not in k]
    merged_state = {}
    
    if merge_type == "WA":
        # Weight Averaging
        for key in keys:
            tensors = [m.state_dict()[key] for m in models_dict.values()]
            if torch.is_floating_point(tensors[0]):
                merged_state[key] = torch.mean(torch.stack(tensors), dim=0)
            else:
                merged_state[key] = tensors[0]
    elif merge_type == "TA":
        # Task Arithmetic: W_merged = W_init + lam * sum(W_k - W_init)
        for key in keys:
            if torch.is_floating_point(progenitor_state[key]):
                task_vectors = [m.state_dict()[key].cpu() - progenitor_state[key].cpu() for m in models_dict.values()]
                merged_state[key] = progenitor_state[key].cpu() + lam * torch.sum(torch.stack(task_vectors), dim=0)
            else:
                merged_state[key] = progenitor_state[key].cpu()
            
    # Load into a model structure
    merged_model = models.resnet18()
    merged_model.fc = nn.Linear(512, 10) # dummy head
    # Load backbone weights
    merged_model.load_state_dict(merged_state, strict=False)
    return merged_model

# -------------------------------------------------------------
# 5. Calibration Method Implementations
# -------------------------------------------------------------

# --- Corrected SP-TAAC Baseline (Real-data Calibration) ---
def apply_sp_taac(merged_model, experts_dict, train_loaders, num_samples=128):
    """
    Corrected SP-TAAC calibration using standard samples from training sets.
    """
    print(f"Applying Corrected SP-TAAC baseline with N={num_samples} samples per task...")
    merged_model = copy.deepcopy(merged_model).to(device)
    merged_model.eval()
    
    # Create calibration datasets
    calib_inputs = []
    for task_name, loader in train_loaders.items():
        count = 0
        for inputs, _ in loader:
            calib_inputs.append(inputs)
            count += inputs.size(0)
            if count >= num_samples:
                break
    calib_dataset = torch.cat(calib_inputs, dim=0)[:num_samples * len(train_loaders)].to(device)
    
    # We will register hooks on both experts and merged model to collect activations
    expert_acts = {name: [] for name in experts_dict.keys()}
    merged_acts = []
    
    # Temporary hooks
    def get_hook(container, name=None):
        def hook_fn(module, input, output):
            container.append(output.detach().cpu())
        return hook_fn
    
    # Run calibration pass and compute statistics
    with torch.no_grad():
        for name, module in merged_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Standard SP-TAAC targets BN layers (or preceding Conv, which scales the BN parameters)
                # In ResNet-18, BatchNorm weight and bias can be scaled directly
                
                # We collect activations before BN (which is the output of the preceding Conv layer)
                # To find preceding Conv, let's look at the parent or the name
                # In ResNet, a Conv is immediately followed by a BN. Let's find the activations *before* BN.
                pass
                
        # To make it robust and simple, let's implement the correction factors layer-by-layer.
        # For each BatchNorm layer, we get activations before BN.
        # Let's find all Conv-BN pairs
        conv_bn_pairs = []
        modules_dict = dict(merged_model.named_modules())
        
        # We can map each BN layer to its preceding Conv or just hook the BN layer's input!
        # This is incredibly elegant and completely robust to any architecture.
        # The input to a BatchNorm layer is exactly the output of the preceding Conv layer!
        for name, module in merged_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                conv_bn_pairs.append(name)
                
        # Register hooks for the inputs to all BatchNorm2d layers
        merged_hooks = []
        merged_container = {bn_name: [] for bn_name in conv_bn_pairs}
        for bn_name in conv_bn_pairs:
            m_bn = modules_dict[bn_name]
            h = m_bn.register_forward_hook(lambda mod, inp, out, name=bn_name: merged_container[name].append(inp[0].detach().cpu()))
            merged_hooks.append(h)
            
        # Hook experts
        expert_hooks = {task: [] for task in experts_dict.keys()}
        expert_containers = {task: {bn_name: [] for bn_name in conv_bn_pairs} for task in experts_dict.keys()}
        for task, exp_model in experts_dict.items():
            exp_model = exp_model.to(device)
            exp_model.eval()
            exp_modules = dict(exp_model.named_modules())
            for bn_name in conv_bn_pairs:
                e_bn = exp_modules[bn_name]
                h = e_bn.register_forward_hook(lambda mod, inp, out, t=task, name=bn_name: expert_containers[t][name].append(inp[0].detach().cpu()))
                expert_hooks[task].append(h)
                
        # Run forward passes on calibration data
        # Run merged model
        _ = merged_model(calib_dataset)
        # Run experts
        for task, exp_model in experts_dict.items():
            _ = exp_model(calib_dataset)
            
        # Remove hooks
        for h in merged_hooks:
            h.remove()
        for task in experts_dict.keys():
            for h in expert_hooks[task]:
                h.remove()
                
        # Now, compute scale correction factors and update BN parameters
        for bn_name in conv_bn_pairs:
            m_bn = modules_dict[bn_name]
            
            # Concatenate collected activations
            # In PyTorch forward hook, inp[0] has shape (B, C, H, W)
            v_merged = torch.cat(merged_container[bn_name], dim=0) # (N_tot, C, H, W)
            
            v_experts = []
            for task in experts_dict.keys():
                v_experts.append(torch.cat(expert_containers[task][bn_name], dim=0))
            v_target = torch.cat(v_experts, dim=0) # (N_tot * K, C, H, W)
            
            # Compute channel-wise standard deviations
            # Shape is (C,)
            std_merged = torch.std(v_merged, dim=(0, 2, 3))
            std_target = torch.std(v_target, dim=(0, 2, 3))
            
            # Scale factor
            gamma = std_target / (std_merged + 1e-8)
            gamma = torch.clamp(gamma, min=0.1, max=10.0).to(device)
            
            # Scale BN weights and biases in-place
            m_bn.weight.copy_(gamma * m_bn.weight)
            if m_bn.bias is not None:
                m_bn.bias.copy_(gamma * m_bn.bias)
                
    print("SP-TAAC calibration completed.")
    return merged_model


# --- Proposed Isotropic Parameter Resonance (IPR) - Weight-Level (W-IPR) ---
def apply_weight_level_ipr(merged_model, experts_dict):
    """
    Our proposed Weight-level IPR: scale Conv/Linear weights directly by 1/R.
    """
    print("Applying proposed Weight-Level Isotropic Parameter Resonance (W-IPR)...")
    merged_model = copy.deepcopy(merged_model)
    K = len(experts_dict)
    
    with torch.no_grad():
        for name, module in merged_model.named_modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "fc" not in name:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                
                norm_merged = torch.norm(module.weight, p="fro")
                norms_experts = torch.tensor([torch.norm(exp_mod.weight, p="fro") for exp_mod in expert_modules])
                avg_norm_experts = torch.mean(norms_experts)
                
                R = norm_merged / (avg_norm_experts + 1e-8)
                R = torch.clamp(R, min=0.1, max=10.0)
                
                # Scale weight and bias by 1 / R
                module.weight.copy_(module.weight / R)
                if module.bias is not None:
                    module.bias.copy_(module.bias / R)
                    
    print("W-IPR completed.")
    return merged_model


# --- Proposed Isotropic Parameter Resonance (IPR) - BN-Level (BN-IPR) ---
def apply_bn_level_ipr(merged_model, experts_dict):
    """
    Our proposed BN-level IPR: scale BN running stats directly by R and R^2.
    """
    print("Applying proposed BN-Level Isotropic Parameter Resonance (BN-IPR)...")
    merged_model = copy.deepcopy(merged_model)
    K = len(experts_dict)
    
    # We first map BN layers to their preceding Conv/Linear layers to find the correct scaling factor R
    # In ResNet-18, the structure is:
    # conv1 -> bn1 -> relu -> maxpool
    # layer1 -> BasicBlock -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> downsample (if exists, which is conv -> bn)
    # Let's write a general parser that traverses the named modules. When we see a Conv2d or Linear (not fc),
    # we record its name and its computed resonance ratio R.
    # When we see a BatchNorm2d immediately following it, we apply R to its running mean and running variance!
    
    # Let's compile a list of all layers in execution order to match them perfectly.
    # To do this robustly, we can use the named modules.
    resonance_ratios = {}
    
    with torch.no_grad():
        # First compute all resonance ratios for Conv and Linear layers
        for name, module in merged_model.named_modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "fc" not in name:
                expert_modules = [dict(exp.named_modules())[name] for exp in experts_dict.values()]
                norm_merged = torch.norm(module.weight, p="fro")
                norms_experts = torch.tensor([torch.norm(exp_mod.weight, p="fro") for exp_mod in expert_modules])
                avg_norm_experts = torch.mean(norms_experts)
                
                R = norm_merged / (avg_norm_experts + 1e-8)
                R = torch.clamp(R, min=0.1, max=10.0)
                resonance_ratios[name] = R.item()
                
        # Now, map each BatchNorm layer to its preceding Conv2d
        # To map them in a simple and robust way, let's find the last Conv2d seen before each BatchNorm2d
        # inside the named modules. Because named_modules() returns modules in pre-order depth-first traversal,
        # which matches sequential execution order for Conv-BN pairs!
        last_conv_r = 1.0
        last_conv_name = None
        
        for name, module in merged_model.named_modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and "fc" not in name:
                last_conv_name = name
                last_conv_r = resonance_ratios[name]
            elif isinstance(module, nn.BatchNorm2d):
                # Scale running mean and running variance of this BN layer
                if last_conv_name is not None:
                    module.running_mean.copy_(last_conv_r * module.running_mean)
                    module.running_var.copy_((last_conv_r ** 2) * module.running_var)
                    print(f"Mapped BN: {name:30s} to Conv: {last_conv_name:30s} | R: {last_conv_r:.4f}")
                else:
                    print(f"Warning: BN layer {name} has no preceding Conv layer found!")
                    
    print("BN-IPR completed.")
    return merged_model

# -------------------------------------------------------------
# 6. Running Merging and Calibration Sweeps
# -------------------------------------------------------------
results_table = []

# --- 6.1 Weight Averaging (WA) Experiments ---
print("\n=======================================================")
print("RUNNING WEIGHT AVERAGING (WA) EXPERIMENTS")
print("=======================================================")

# Get standard WA model
wa_base = get_standard_merge(experts, progenitor_state, merge_type="WA")

# Evaluation: Uncalibrated
print("\n--- Evaluating WA (Uncalibrated) ---")
res_wa_uncal = evaluate_model(wa_base, experts, test_loaders)
results_table.append({
    "merge_type": "WA",
    "calibration": "None (Uncalibrated)",
    "mnist": res_wa_uncal["mnist"],
    "fmnist": res_wa_uncal["fmnist"],
    "cifar10": res_wa_uncal["cifar10"],
    "average": res_wa_uncal["average"]
})

# Evaluation: Corrected SP-TAAC
wa_sptaac = apply_sp_taac(wa_base, experts, train_loaders, num_samples=128)
print("\n--- Evaluating WA + SP-TAAC (Real-data Calibration) ---")
res_wa_sptaac = evaluate_model(wa_sptaac, experts, test_loaders)
results_table.append({
    "merge_type": "WA",
    "calibration": "SP-TAAC (Real N=128)",
    "mnist": res_wa_sptaac["mnist"],
    "fmnist": res_wa_sptaac["fmnist"],
    "cifar10": res_wa_sptaac["cifar10"],
    "average": res_wa_sptaac["average"]
})

# Evaluation: Weight-level IPR (Ours, Data-Free)
wa_w_ipr = apply_weight_level_ipr(wa_base, experts)
print("\n--- Evaluating WA + W-IPR (Ours, Data-Free) ---")
res_wa_w_ipr = evaluate_model(wa_w_ipr, experts, test_loaders)
results_table.append({
    "merge_type": "WA",
    "calibration": "W-IPR (Ours, Data-Free)",
    "mnist": res_wa_w_ipr["mnist"],
    "fmnist": res_wa_w_ipr["fmnist"],
    "cifar10": res_wa_w_ipr["cifar10"],
    "average": res_wa_w_ipr["average"]
})

# Evaluation: BN-level IPR (Ours, Data-Free)
wa_bn_ipr = apply_bn_level_ipr(wa_base, experts)
print("\n--- Evaluating WA + BN-IPR (Ours, Data-Free) ---")
res_wa_bn_ipr = evaluate_model(wa_bn_ipr, experts, test_loaders)
results_table.append({
    "merge_type": "WA",
    "calibration": "BN-IPR (Ours, Data-Free)",
    "mnist": res_wa_bn_ipr["mnist"],
    "fmnist": res_wa_bn_ipr["fmnist"],
    "cifar10": res_wa_bn_ipr["cifar10"],
    "average": res_wa_bn_ipr["average"]
})


# --- 6.2 Task Arithmetic (TA) Experiments ---
print("\n=======================================================")
print("RUNNING TASK ARITHMETIC (TA) EXPERIMENTS")
print("=======================================================")

for lam in [0.2, 0.5]:
    print(f"\n--- TA with lambda = {lam} ---")
    ta_base = get_standard_merge(experts, progenitor_state, merge_type="TA", lam=lam)
    
    # Evaluation: Uncalibrated
    print(f"\n--- Evaluating TA (lambda={lam}, Uncalibrated) ---")
    res_ta_uncal = evaluate_model(ta_base, experts, test_loaders)
    results_table.append({
        "merge_type": f"TA (lam={lam})",
        "calibration": "None (Uncalibrated)",
        "mnist": res_ta_uncal["mnist"],
        "fmnist": res_ta_uncal["fmnist"],
        "cifar10": res_ta_uncal["cifar10"],
        "average": res_ta_uncal["average"]
    })
    
    # Evaluation: Corrected SP-TAAC
    ta_sptaac = apply_sp_taac(ta_base, experts, train_loaders, num_samples=128)
    print(f"\n--- Evaluating TA (lambda={lam}) + SP-TAAC ---")
    res_ta_sptaac = evaluate_model(ta_sptaac, experts, test_loaders)
    results_table.append({
        "merge_type": f"TA (lam={lam})",
        "calibration": "SP-TAAC (Real N=128)",
        "mnist": res_ta_sptaac["mnist"],
        "fmnist": res_ta_sptaac["fmnist"],
        "cifar10": res_ta_sptaac["cifar10"],
        "average": res_ta_sptaac["average"]
    })
    
    # Evaluation: Weight-level IPR (Ours, Data-Free)
    ta_w_ipr = apply_weight_level_ipr(ta_base, experts)
    print(f"\n--- Evaluating TA (lambda={lam}) + W-IPR (Ours, Data-Free) ---")
    res_ta_w_ipr = evaluate_model(ta_w_ipr, experts, test_loaders)
    results_table.append({
        "merge_type": f"TA (lam={lam})",
        "calibration": "W-IPR (Ours, Data-Free)",
        "mnist": res_ta_w_ipr["mnist"],
        "fmnist": res_ta_w_ipr["fmnist"],
        "cifar10": res_ta_w_ipr["cifar10"],
        "average": res_ta_w_ipr["average"]
    })
    
    # Evaluation: BN-level IPR (Ours, Data-Free)
    ta_bn_ipr = apply_bn_level_ipr(ta_base, experts)
    print(f"\n--- Evaluating TA (lambda={lam}) + BN-IPR (Ours, Data-Free) ---")
    res_ta_bn_ipr = evaluate_model(ta_bn_ipr, experts, test_loaders)
    results_table.append({
        "merge_type": f"TA (lam={lam})",
        "calibration": "BN-IPR (Ours, Data-Free)",
        "mnist": res_ta_bn_ipr["mnist"],
        "fmnist": res_ta_bn_ipr["fmnist"],
        "cifar10": res_ta_bn_ipr["cifar10"],
        "average": res_ta_bn_ipr["average"]
    })

# Add Oracle results for context
results_table.append({
    "merge_type": "Oracle Experts (Individual)",
    "calibration": "None",
    "mnist": oracle_results["mnist"],
    "fmnist": oracle_results["fmnist"],
    "cifar10": oracle_results["cifar10"],
    "average": oracle_results["average"]
})

# -------------------------------------------------------------
# 7. Print and Save Results
# -------------------------------------------------------------
print("\n=======================================================")
print("FINAL ACCURACY RESULTS SUMMARY TABLE")
print("=======================================================")
print(f"{'Merge Method':25s} | {'Calibration':25s} | {'MNIST':6s} | {'F-MNIST':6s} | {'CIFAR10':6s} | {'Average':6s}")
print("-" * 88)
for row in results_table:
    print(f"{row['merge_type']:25s} | {row['calibration']:25s} | {row['mnist']:5.2f}% | {row['fmnist']:5.2f}% | {row['cifar10']:5.2f}% | {row['average']:5.2f}%")
print("=======================================================")

# Save to JSON
with open("results.json", "w") as f:
    json.dump(results_table, f, indent=4)
print("Saved final results to results.json.")
