import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import os
import random
import numpy as np
import json
import matplotlib.pyplot as plt

# Global cache for expert target standard deviations to avoid redundant forward passes
expert_std_cache = {}

# Set seed helper
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  # Disable cuDNN to bypass CUDNN_STATUS_NOT_INITIALIZED

torch.backends.cudnn.enabled = False  # Disable cuDNN globally from start
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms
image_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

grayscale_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Helper to get deterministic subset (not overlapping with first 5000 used for training)
def get_calibration_subset(dataset, seed, size=128):
    indices = list(range(len(dataset)))
    # Shuffle with seed 42 to identify the training set indices
    rng = random.Random(42)
    rng.shuffle(indices)
    
    # Exclude the first 5000 indices (training set)
    remaining_indices = indices[5000:]
    
    # Now shuffle remaining indices with the specific seed for calibration set selection
    cal_rng = random.Random(seed)
    cal_rng.shuffle(remaining_indices)
    
    subset_indices = remaining_indices[:size]
    return Subset(dataset, subset_indices)

# Load test datasets and full train datasets (to extract calibration sets)
print("Loading datasets...")
train_mnist_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=grayscale_transforms)
test_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=grayscale_transforms)

train_fmnist_full = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=grayscale_transforms)
test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=grayscale_transforms)

train_cifar_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=image_transforms)
test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=image_transforms)

# Pre-load and pre-transform test datasets to GPU memory once for extreme evaluation speed
preloaded_test_data = {}

def preload_test_datasets():
    global preloaded_test_data
    print("Pre-loading and transforming test datasets on GPU...")
    tasks = ["mnist", "fmnist", "cifar10"]
    raw_datasets = {
        "mnist": test_mnist,
        "fmnist": test_fmnist,
        "cifar10": test_cifar
    }
    for t in tasks:
        loader = DataLoader(raw_datasets[t], batch_size=2048, shuffle=False, num_workers=0)
        all_x = []
        all_y = []
        for x, y in loader:
            all_x.append(x.to(device))
            all_y.append(y.to(device))
        preloaded_test_data[t] = (torch.cat(all_x, dim=0), torch.cat(all_y, dim=0))
    print("Pre-loading complete!")

# Keep test_loaders as a dummy mapping for compatibility with any references
test_loaders = {
    "mnist": None,
    "fmnist": None,
    "cifar10": None
}

# Load base model structure
def load_resnet18_structure():
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(512, 10)
    )
    return model

# Merge experts function
def merge_models(mode="WA", lmbda=0.3):
    """
    mode: "WA" (Weight Averaging) or "TA" (Task Arithmetic)
    lmbda: Task Arithmetic merging coefficient
    """
    tasks = ["mnist", "fmnist", "cifar10"]
    
    # Load expert models
    expert_state_dicts = {}
    for task in tasks:
        path = f"checkpoints/expert_{task}.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert checkpoint {path} not found. Please train experts first.")
        expert_state_dicts[task] = torch.load(path, map_location=device)
        
    # Load base pre-trained model (needed for Task Arithmetic)
    base_path = "checkpoints/base_model.pt"
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base model checkpoint {base_path} not found.")
    base_state_dict = torch.load(base_path, map_location=device)
    
    merged_state_dict = {}
    
    # Get all parameter keys
    keys = base_state_dict.keys()
    
    for key in keys:
        # Note: the classification head (fc) is task-specific and NOT merged.
        # We will keep task-specific fc weights from their respective experts during evaluation.
        if "fc" in key:
            # For classification head, we store each expert's head separately in the state dict 
            # as fc_mnist, fc_fmnist, fc_cifar10
            for task in tasks:
                merged_state_dict[f"{key}_{task}"] = expert_state_dicts[task][key]
            continue
            
        # Merge backbone layers
        if mode == "WA":
            # Simple Weight Averaging
            avg_param = (expert_state_dicts["mnist"][key] + 
                         expert_state_dicts["fmnist"][key] + 
                         expert_state_dicts["cifar10"][key]) / 3.0
            merged_state_dict[key] = avg_param
        elif mode == "TA":
            # Task Arithmetic: W_base + lmbda * \sum (W_expert - W_base)
            task_vector_sum = torch.zeros_like(base_state_dict[key])
            for task in tasks:
                task_vector_sum += (expert_state_dicts[task][key] - base_state_dict[key])
            merged_state_dict[key] = base_state_dict[key] + lmbda * task_vector_sum
            
    return merged_state_dict

# Model evaluator wrapper
class MergedEvaluator:
    def __init__(self, merged_state_dict):
        self.merged_state_dict = merged_state_dict
        self.model = load_resnet18_structure().to(device)
        
    def prepare_for_task(self, task_name):
        # Load backbone parameters + task-specific fc head/BN running stats from merged_state_dict
        state_dict_to_load = {}
        tasks = ["mnist", "fmnist", "cifar10"]
        for k, v in self.merged_state_dict.items():
            has_suffix = False
            for t in tasks:
                if k.endswith(f"_{t}"):
                    has_suffix = True
                    if t == task_name:
                        # Map back to clean key
                        clean_k = k[:-len(f"_{t}")]
                        state_dict_to_load[clean_k] = v
                    break
            if not has_suffix:
                state_dict_to_load[k] = v
        self.model.load_state_dict(state_dict_to_load)
        self.model.eval()

    def evaluate(self, task_name, loader=None):
        self.prepare_for_task(task_name)
        x_all, y_all = preloaded_test_data[task_name]
        correct = 0
        total = y_all.size(0)
        
        batch_size = 2048
        with torch.no_grad():
            for i in range(0, total, batch_size):
                x = x_all[i:i+batch_size]
                y = y_all[i:i+batch_size]
                outputs = self.model(x)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y).sum().item()
        return 100. * correct / total

# Calibration implementation
def get_activation_outputs(model, loader, module_name, device):
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach())
        
    module = None
    for name, mod in model.named_modules():
        if name == module_name:
            module = mod
            break
    assert module is not None, f"Module {module_name} not found"
    
    handle = module.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device))
    handle.remove()
    return torch.cat(activations, dim=0)

def run_calibration(merged_state_dict, calib_datasets, method="uncalibrated", alpha=0.05, seed=None, size=None):
    """
    method: "uncalibrated", "ntaac", "tcbna", "sptaac", "csc", "wrsc"
    alpha: regularization strength for WRSC (Wiener-Regularized Spatial Calibration)
    """
    if method == "uncalibrated":
        return merged_state_dict # No change
        
    tasks = ["mnist", "fmnist", "cifar10"]
    
    # 1. Prepare joint calibration loaders
    joint_images = []
    joint_labels = []
    task_images = {t: [] for t in tasks}
    
    for t in tasks:
        loader = DataLoader(calib_datasets[t], batch_size=128, shuffle=False)
        for x, y in loader:
            joint_images.append(x)
            joint_labels.append(y)
            task_images[t].append(x)
            
    joint_images_tensor = torch.cat(joint_images, dim=0)
    # Joint dataloader
    joint_loader = [(joint_images_tensor, None)] # Simple run once loader
    
    # Task-specific loaders
    task_loaders = {t: [(torch.cat(task_images[t], dim=0), None)] for t in tasks}
    
    # 2. If N-TAAC (Joint BN re-running), we simply run joint calibration in train mode for BN modules
    if method == "ntaac":
        evaluator = MergedEvaluator(merged_state_dict)
        evaluator.prepare_for_task("mnist")
        model = evaluator.model
        
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.train()
                m.momentum = None # Cumulative moving average
                
        with torch.no_grad():
            for x, _ in joint_loader:
                _ = model(x.to(device))
                    
        calibrated_state = model.state_dict()
        new_state_dict = merged_state_dict.copy()
        for k, v in calibrated_state.items():
            if "fc" not in k:
                new_state_dict[k] = v
        return new_state_dict

    # 2b. If TC-BNA (Task-Conditional BN re-running), we run task-specific calibration and save stats with task suffixes
    if method == "tcbna":
        new_state_dict = merged_state_dict.copy()
        for t in tasks:
            evaluator = MergedEvaluator(merged_state_dict)
            evaluator.prepare_for_task(t)
            model = evaluator.model
            
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()
                    m.train()
                    m.momentum = None # Cumulative moving average
                    
            with torch.no_grad():
                for x, _ in task_loaders[t]:
                    _ = model(x.to(device))
                        
            calibrated_state = model.state_dict()
            for k, v in calibrated_state.items():
                if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
                    new_state_dict[f"{k}_{t}"] = v
        return new_state_dict
        
    # 3. For SP-TAAC, CSC, WRSC, we calibrate layer-by-layer sequentially
    evaluator_merged = MergedEvaluator(merged_state_dict)
    
    # Find all BatchNorm layer names sequentially
    evaluator_merged.prepare_for_task("mnist") # dummy preparation to trace names
    bn_names = [name for name, mod in evaluator_merged.model.named_modules() if isinstance(mod, nn.BatchNorm2d)]
    
    # Copy state dict to modify
    calibrated_state_dict = merged_state_dict.copy()
    
    global expert_std_cache
    cache_available = (seed is not None and size is not None)
    
    # Expert models: only load/instantiate if there is a cache miss
    expert_models = {}
    expert_models_loaded = False
    
    # Calibrate sequentially
    for bn_name in bn_names:
        # Create a temporary evaluator with our currently calibrated state dict
        eval_temp = MergedEvaluator(calibrated_state_dict)
        eval_temp.prepare_for_task("mnist") # dummy load
        
        # Collect merged activations over joint calibration dataset
        merged_act = get_activation_outputs(eval_temp.model, joint_loader, bn_name, device)
        
        # Check cache for target stds
        cache_key = (seed, size, bn_name)
        if cache_available and cache_key in expert_std_cache:
            target_std_global, target_std_channel = expert_std_cache[cache_key]
        else:
            if not expert_models_loaded:
                for t in tasks:
                    model_exp = load_resnet18_structure().to(device)
                    model_exp.load_state_dict(torch.load(f"checkpoints/expert_{t}.pt", map_location=device))
                    model_exp.eval()
                    expert_models[t] = model_exp
                expert_models_loaded = True
                
            # Collect expert activations (each expert evaluated on its own calibration set)
            expert_acts = []
            for t in tasks:
                exp_act = get_activation_outputs(expert_models[t], task_loaders[t], bn_name, device)
                expert_acts.append(exp_act)
            target_act = torch.cat(expert_acts, dim=0)
            
            target_std_global = torch.std(target_act).cpu()
            target_std_channel = torch.std(target_act, dim=[0, 2, 3]).cpu()
            
            if cache_available:
                expert_std_cache[cache_key] = (target_std_global, target_std_channel)
                
        # Compute statistics
        # SP-TAAC: Global (layer-wise) scaling factor
        if method == "sptaac":
            merged_std = torch.std(merged_act)
            gamma = target_std_global / (merged_std.cpu() + 1e-5)
            # Clip scaling factor for stability
            gamma = torch.clamp(gamma, 0.1, 10.0)
            
            # Update weights and biases of this layer in calibrated_state_dict
            w_key = f"{bn_name}.weight"
            b_key = f"{bn_name}.bias"
            calibrated_state_dict[w_key] = calibrated_state_dict[w_key] * gamma.to(device)
            calibrated_state_dict[b_key] = calibrated_state_dict[b_key] * gamma.to(device)
            
        elif method == "csc":
            merged_std = torch.std(merged_act, dim=[0, 2, 3]) # shape: [C]
            gamma = target_std_channel / (merged_std.cpu() + 1e-5)
            # Clip for numerical stability
            gamma = torch.clamp(gamma, 0.1, 10.0)
            
            w_key = f"{bn_name}.weight"
            b_key = f"{bn_name}.bias"
            calibrated_state_dict[w_key] = calibrated_state_dict[w_key] * gamma.to(device)
            calibrated_state_dict[b_key] = calibrated_state_dict[b_key] * gamma.to(device)
            
        elif method == "wrsc":
            merged_std = torch.std(merged_act, dim=[0, 2, 3]) # shape: [C]
            
            # Wiener-like regularized spatial channel-wise scaling
            t_std = target_std_channel
            m_std = merged_std.cpu()
            gamma = (t_std * m_std) / (m_std ** 2 + alpha * (t_std ** 2) + 1e-8)
            
            # Clip for stability
            gamma = torch.clamp(gamma, 0.1, 10.0)
            
            w_key = f"{bn_name}.weight"
            b_key = f"{bn_name}.bias"
            calibrated_state_dict[w_key] = calibrated_state_dict[w_key] * gamma.to(device)
            calibrated_state_dict[b_key] = calibrated_state_dict[b_key] * gamma.to(device)
            
    return calibrated_state_dict

# Main evaluation sweep
def run_evaluation_sweep():
    # Preload the test datasets on GPU for extreme speed
    preload_test_datasets()
    
    # Sweep configurations
    merge_modes = ["WA", "TA"]
    calibration_sizes = [8, 16, 32, 64, 128]
    calibration_seeds = [42, 43, 44, 45, 46]
    methods = ["uncalibrated", "ntaac", "tcbna", "sptaac", "csc", "wrsc"]
    
    # Results dictionary
    results = {mode: {m: {size: [] for size in calibration_sizes} for m in methods} for mode in merge_modes}
    
    for mode in merge_modes:
        print(f"\n=========================================")
        print(f"      MERGE MODE: {mode}")
        print(f"=========================================")
        
        # Load merged state dict (without task-specific fc)
        lmbda = 0.3 if mode == "TA" else 0.5
        merged_state_dict = merge_models(mode=mode, lmbda=lmbda)
        
        for method in methods:
            print(f"\n--- Calibrating with {method.upper()} ---")
            
            # OPTIMIZATION: For uncalibrated baseline, we only need to evaluate once!
            if method == "uncalibrated":
                print("Evaluating uncalibrated baseline once...")
                evaluator = MergedEvaluator(merged_state_dict)
                mnist_acc = evaluator.evaluate("mnist", test_loaders["mnist"])
                fmnist_acc = evaluator.evaluate("fmnist", test_loaders["fmnist"])
                cifar_acc = evaluator.evaluate("cifar10", test_loaders["cifar10"])
                avg_acc = (mnist_acc + fmnist_acc + cifar_acc) / 3.0
                single_res = {
                    "seed": 42,
                    "mnist": mnist_acc,
                    "fmnist": fmnist_acc,
                    "cifar10": cifar_acc,
                    "average": avg_acc
                }
                for size in calibration_sizes:
                    results[mode][method][size] = [single_res.copy() for _ in calibration_seeds]
                    print(f"Size N={size:3d} | Mean Avg Acc: {avg_acc:.2f}% | Std Dev: 0.0000")
                continue
                
            for size in calibration_sizes:
                seed_accs = []
                for seed in calibration_seeds:
                    set_seed(seed)
                    
                    # Prepare calibration subsets for this seed and size
                    calib_datasets = {}
                    calib_datasets["mnist"] = get_calibration_subset(train_mnist_full, seed, size)
                    calib_datasets["fmnist"] = get_calibration_subset(train_fmnist_full, seed, size)
                    calib_datasets["cifar10"] = get_calibration_subset(train_cifar_full, seed, size)
                    
                    # Calibrate
                    calibrated_dict = run_calibration(merged_state_dict, calib_datasets, method=method, alpha=0.05, seed=seed, size=size)
                    
                    # Evaluate on all tasks
                    evaluator = MergedEvaluator(calibrated_dict)
                    mnist_acc = evaluator.evaluate("mnist", test_loaders["mnist"])
                    fmnist_acc = evaluator.evaluate("fmnist", test_loaders["fmnist"])
                    cifar_acc = evaluator.evaluate("cifar10", test_loaders["cifar10"])
                    
                    avg_acc = (mnist_acc + fmnist_acc + cifar_acc) / 3.0
                    seed_accs.append({
                        "seed": seed,
                        "mnist": mnist_acc,
                        "fmnist": fmnist_acc,
                        "cifar10": cifar_acc,
                        "average": avg_acc
                    })
                    
                # Aggregate results across seeds
                averages = [s["average"] for s in seed_accs]
                mean_acc = np.mean(averages)
                std_acc = np.std(averages)
                print(f"Size N={size:3d} | Mean Avg Acc: {mean_acc:.2f}% | Std Dev: {std_acc:.4f}", flush=True)
                
                results[mode][method][size] = seed_accs
                
    # Run WRSC Alpha Ablation Sweep
    ablation_results = run_ablation_sweep(calibration_seeds)
    
    # Save combined results to JSON
    combined_results = {
        "main_results": results,
        "ablation_results": ablation_results
    }
    with open("checkpoints/experimental_results.json", "w") as f:
        json.dump(combined_results, f, indent=4)
    print("\nResults saved to checkpoints/experimental_results.json", flush=True)
    
    # Generate summary tables and plots
    generate_summary_plots(results)
    generate_ablation_plots(ablation_results)

def run_ablation_sweep(calibration_seeds):
    print("\n=========================================")
    print("      WRSC ALPHA ABLATION SWEEP (N=16)")
    print("=========================================")
    
    merge_modes = ["WA", "TA"]
    alpha_vals = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    
    ablation_results = {mode: {str(alpha): [] for alpha in alpha_vals} for mode in merge_modes}
    
    for mode in merge_modes:
        print(f"\nAblation for mode: {mode}")
        lmbda = 0.3 if mode == "TA" else 0.5
        merged_state_dict = merge_models(mode=mode, lmbda=lmbda)
        
        for alpha in alpha_vals:
            seed_accs = []
            for seed in calibration_seeds:
                set_seed(seed)
                
                # Prepare calibration subsets for this seed and size N=16
                calib_datasets = {}
                calib_datasets["mnist"] = get_calibration_subset(train_mnist_full, seed, 16)
                calib_datasets["fmnist"] = get_calibration_subset(train_fmnist_full, seed, 16)
                calib_datasets["cifar10"] = get_calibration_subset(train_cifar_full, seed, 16)
                
                # Calibrate with specific alpha
                calibrated_dict = run_calibration(merged_state_dict, calib_datasets, method="wrsc", alpha=alpha, seed=seed, size=16)
                
                # Evaluate
                evaluator = MergedEvaluator(calibrated_dict)
                mnist_acc = evaluator.evaluate("mnist", test_loaders["mnist"])
                fmnist_acc = evaluator.evaluate("fmnist", test_loaders["fmnist"])
                cifar_acc = evaluator.evaluate("cifar10", test_loaders["cifar10"])
                
                avg_acc = (mnist_acc + fmnist_acc + cifar_acc) / 3.0
                seed_accs.append(avg_acc)
                
            mean_acc = np.mean(seed_accs)
            std_acc = np.std(seed_accs)
            print(f"Alpha: {alpha:.3f} | Mean Avg Acc: {mean_acc:.2f}% | Std Dev: {std_acc:.4f}", flush=True)
            ablation_results[mode][str(alpha)] = seed_accs
            
    return ablation_results

def generate_summary_plots(results):
    methods = ["uncalibrated", "ntaac", "tcbna", "sptaac", "csc", "wrsc"]
    calibration_sizes = [8, 16, 32, 64, 128]
    
    for mode in ["WA", "TA"]:
        plt.figure(figsize=(10, 6))
        for method in methods:
            means = []
            stds = []
            for size in calibration_sizes:
                seed_accs = results[mode][method][size]
                averages = [s["average"] for s in seed_accs]
                means.append(np.mean(averages))
                stds.append(np.std(averages))
                
            plt.errorbar(calibration_sizes, means, yerr=stds, marker='o', label=method.upper(), capsize=5, elinewidth=1.5, capthick=1.5)
            
        plt.title(f"Multi-Task Model Merging Accuracy ({mode} Backbone)\nunder Varying Calibration Budgets (N)")
        plt.xlabel("Calibration Dataset Size N (samples per task)")
        plt.ylabel("Average Test Accuracy (%) across Tasks")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plot_path = f"checkpoints/evaluation_{mode}.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Generated plot saved to {plot_path}", flush=True)
        plt.close()

def generate_ablation_plots(ablation_results):
    plt.figure(figsize=(8, 5))
    alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    
    for mode in ["WA", "TA"]:
        means = []
        stds = []
        for alpha in alphas:
            accs = ablation_results[mode][str(alpha)]
            means.append(np.mean(accs))
            stds.append(np.std(accs))
            
        plt.errorbar(alphas, means, yerr=stds, marker='s', linestyle='-', label=f"{mode} Backbone", capsize=4, elinewidth=1.2, capthick=1.2)
        
    plt.xscale('log')
    plt.title("WRSC Sensitivity to Regularization Strength ($\\alpha$)\nunder Calibration Size N=16")
    plt.xlabel("Scale-Invariant Regularization Parameter $\\alpha$ (Log Scale)")
    plt.ylabel("Average Test Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5, which="both")
    plt.legend()
    plt.tight_layout()
    plot_path = "checkpoints/wrsc_alpha_ablation.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Generated ablation plot saved to {plot_path}", flush=True)
    plt.close()

if __name__ == "__main__":
    run_evaluation_sweep()