import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on some cluster nodes
torch.backends.cudnn.enabled = False

# Define transforms
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_color = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class ExpertModel(nn.Module):
    def __init__(self, task_name):
        super().__init__()
        self.backbone = models.resnet18()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(in_features, 10)
        self.task_name = task_name
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# Load datasets
def get_datasets(task):
    if task == 'mnist':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_gray)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray)
    elif task == 'fashion':
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_gray)
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)
    elif task == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_color)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_color)
    else:
        raise ValueError(f"Unknown task {task}")
    return train_set, test_set

def get_calibration_subset(train_set, N=128, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(train_set), generator=g)[:N].tolist()
    return Subset(train_set, indices)

# Merge backbones function
def get_merged_backbone_state_dict(expert_state_dicts, pretrained_state_dict, method="WA", lambda_coeff=0.3):
    merged_state_dict = {}
    keys = list(expert_state_dicts[0].keys())
    
    for key in keys:
        if key.startswith('backbone.'):
            params = [state_dict[key] for state_dict in expert_state_dicts]
            is_running_stat = any(x in key for x in ['running_mean', 'running_var', 'num_batches_tracked'])
            
            if is_running_stat:
                # Average running statistics
                if 'num_batches_tracked' in key:
                    merged_state_dict[key] = torch.stack([p.float() for p in params]).mean(dim=0).long()
                else:
                    merged_state_dict[key] = torch.stack(params).mean(dim=0)
            else:
                # Learnable parameters (weights, biases, gamma, beta)
                if method == "WA":
                    merged_state_dict[key] = torch.stack(params).mean(dim=0)
                elif method == "TA":
                    pre_key = key.replace('backbone.', '')
                    pre_param = pretrained_state_dict[pre_key]
                    # Compute task vectors
                    task_vectors = [p - pre_param for p in params]
                    merged_state_dict[key] = pre_param + lambda_coeff * torch.stack(task_vectors).sum(dim=0)
                elif method == "DARE":
                    pre_key = key.replace('backbone.', '')
                    pre_param = pretrained_state_dict[pre_key]
                    # Compute task vectors
                    task_vectors = [p - pre_param for p in params]
                    # Apply DARE: random drop of updates with p_drop = 0.2, rescale remaining by 1 / (1 - p_drop)
                    p_drop = 0.2
                    dare_vectors = []
                    for tv in task_vectors:
                        # Generate random mask
                        mask = (torch.rand_like(tv) >= p_drop).float()
                        tv_dare = tv * mask / (1.0 - p_drop)
                        dare_vectors.append(tv_dare)
                    merged_state_dict[key] = pre_param + lambda_coeff * torch.stack(dare_vectors).sum(dim=0)
                elif method == "TIES":
                    pre_key = key.replace('backbone.', '')
                    pre_param = pretrained_state_dict[pre_key]
                    task_vectors = [p - pre_param for p in params]
                    # TIES steps:
                    # 1. Trim: Keep top r% of parameter updates by magnitude
                    r_keep = 0.2
                    trimmed_vectors = []
                    for tv in task_vectors:
                        flat_tv = tv.abs().view(-1)
                        k_val = int(len(flat_tv) * (1 - r_keep))
                        if k_val > 0 and k_val < len(flat_tv):
                            # Use kthvalue to find threshold
                            threshold = torch.kthvalue(flat_tv, k_val).values
                            mask = tv.abs() >= threshold
                            trimmed_vectors.append(tv * mask)
                        else:
                            trimmed_vectors.append(tv)
                    # 2. Elect sign & 3. Disjoint merge
                    stacked_trimmed = torch.stack(trimmed_vectors) # (K, *shape)
                    signs = torch.sign(stacked_trimmed)
                    sum_signs = signs.sum(dim=0)
                    dominant_sign = torch.sign(sum_signs)
                    
                    matching_mask = (signs == dominant_sign) & (dominant_sign != 0)
                    sum_val = (stacked_trimmed * matching_mask).sum(dim=0)
                    count = matching_mask.sum(dim=0)
                    average_val = torch.where(count > 0, sum_val / count, torch.zeros_like(sum_val))
                    
                    merged_state_dict[key] = pre_param + lambda_coeff * average_val
        else:
            # We do not merge heads, keep them as is
            pass
            
    return merged_state_dict

# Evaluation helper
def evaluate_model(model, test_loaders, device):
    model.eval()
    task_accuracies = {}
    
    for task_name, test_loader in test_loaders.items():
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # Ensure we use the correct head for this task
                features = model.backbone(inputs)
                outputs = model.heads[task_name](features)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        task_accuracies[task_name] = 100.0 * correct / total
        
    task_accuracies['average'] = sum(task_accuracies.values()) / len(task_accuracies)
    return task_accuracies

# Hook helper for capturing activations (needed for LSC)
class ActivationCapturer:
    def __init__(self):
        self.activations = {}
        self.hooks = []
        
    def register(self, name, module):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        self.hooks.append(module.register_forward_hook(hook))
        
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

# Class for multi-task model with single backbone and task-specific heads
class MultiTaskMergedModel(nn.Module):
    def __init__(self, tasks=['mnist', 'fashion', 'cifar10']):
        super().__init__()
        self.backbone = models.resnet18()
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleDict({
            task: nn.Linear(512, 10) for task in tasks
        })

# LSC implementation
def run_lsc(model, expert_models, cal_datasets, target_layers=None):
    device = next(model.parameters()).device
    scales = {}
    
    # 1. Capture expert statistics
    for task_name, expert_model in expert_models.items():
        expert_model = expert_model.to(device)
        expert_model.eval()
        
        # We capture standard deviations for BatchNorm layers
        capturer = ActivationCapturer()
        for name, module in expert_model.backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Filter by target layers if specified
                if target_layers is not None:
                    if not any(target in name for target in target_layers):
                        continue
                capturer.register(name, module)
                
        # Run calibration set
        cal_loader = DataLoader(cal_datasets[task_name], batch_size=len(cal_datasets[task_name]), shuffle=False)
        with torch.no_grad():
            for inputs, _ in cal_loader:
                inputs = inputs.to(device)
                _ = expert_model(inputs)
                break
                
        # Record expert stds
        for name, act in capturer.activations.items():
            # Global standard deviation across all channels/spatial dimensions (Layer-wise scaling)
            # Standard LSC computes standard deviation globally across all channels in a layer
            std_val = torch.std(act)
            if name not in scales:
                scales[name] = {}
            scales[name][task_name] = {'orig_std': std_val.item()}
            
        capturer.remove()
        expert_model.cpu() # free GPU memory
        
    # 2. Capture merged model statistics
    model.eval()
    capturer = ActivationCapturer()
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if target_layers is not None:
                if not any(target in name for target in target_layers):
                    continue
            capturer.register(name, module)
            
    for task_name in expert_models.keys():
        cal_loader = DataLoader(cal_datasets[task_name], batch_size=len(cal_datasets[task_name]), shuffle=False)
        with torch.no_grad():
            for inputs, _ in cal_loader:
                inputs = inputs.to(device)
                _ = model.backbone(inputs)
                break
                
        # Record merged stds
        for name, act in capturer.activations.items():
            std_val = torch.std(act)
            scales[name][task_name]['merged_std'] = std_val.item()
            
    capturer.remove()
    
    # 3. Compute scaling factors
    final_scales = {}
    for name, task_scales in scales.items():
        final_scales[name] = {}
        for task_name, vals in task_scales.items():
            orig_std = vals['orig_std']
            merged_std = vals['merged_std']
            # Compute scaling ratio
            final_scales[name][task_name] = orig_std / (merged_std + 1e-5)
            
    return final_scales

# Register scaling hooks for LSC evaluation
class LSCScaleHook:
    def __init__(self, scales, task_name):
        self.scales = scales
        self.task_name = task_name
        self.hooks = []
        
    def register(self, model):
        for name, module in model.backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in self.scales:
                scale_val = self.scales[name][self.task_name]
                def get_hook_fn(s_val):
                    def hook_fn(module, input, output):
                        return output * s_val
                    return hook_fn
                self.hooks.append(module.register_forward_hook(get_hook_fn(scale_val)))
                
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

# N-TAAC & T-NAC implementations
def run_native_calibration(model, joint_loader, target_layers=None, momentum=1.0):
    # Keep the model in eval mode by default
    model.eval()
    
    # Freeze all weights and biases
    for param in model.parameters():
        param.requires_grad = False
        
    # Configure BatchNorm modules
    calibrated_count = 0
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if target_layers is None:
                # N-TAAC: Calibrate all layers
                module.train()
                module.momentum = momentum
                calibrated_count += 1
            else:
                # T-NAC: Calibrate only targeted layers
                is_target = any(target in name for target in target_layers)
                if is_target:
                    module.train()
                    module.momentum = momentum
                    calibrated_count += 1
                else:
                    module.eval()
                    module.momentum = 0.0 # Do not modify
                    
    print(f"Running native calibration (momentum={momentum}) on {calibrated_count} layers...")
    
    # Run a single forward pass of the joint calibration set
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, _ in joint_loader:
            inputs = inputs.to(device)
            _ = model.backbone(inputs)
            break # Single batch
            
    # Return everything to eval mode
    model.eval()

# Main function
def main():
    parser = argparse.ArgumentParser(description="Merge models and run calibration/evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--N", type=int, default=128, help="Calibration samples per task")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running merging and evaluation on device: {device}")
    
    tasks = ['mnist', 'fashion', 'cifar10']
    
    # 1. Load expert checkpoints and datasets
    expert_state_dicts = []
    expert_models = {}
    test_loaders = {}
    cal_datasets = {}
    
    # Load pre-trained backbone
    pretrained_backbone_state_dict = torch.load("resnet18_pretrained.pth", map_location='cpu')
    
    for task in tasks:
        # Load datasets
        _, test_set = get_datasets(task)
        test_loaders[task] = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        
        # Load expert model checkpoint
        checkpoint_path = f"expert_{task}.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Expert checkpoint {checkpoint_path} not found! Please run train_expert.py first.")
            
        print(f"Loading expert {task} from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        expert_state_dicts.append(checkpoint['model_state_dict'])
        
        # Instantiate expert model for LSC
        exp_model = ExpertModel(task)
        exp_model.load_state_dict(checkpoint['model_state_dict'])
        expert_models[task] = exp_model
        
        # Get calibration subset
        train_set, _ = get_datasets(task)
        cal_datasets[task] = get_calibration_subset(train_set, N=args.N, seed=args.seed)
        print(f"Created calibration dataset for {task} with {len(cal_datasets[task])} samples.")
        
    # Create joint calibration dataset for N-TAAC & T-NAC
    joint_cal_dataset = ConcatDataset([cal_datasets[t] for t in tasks])
    joint_loader = DataLoader(joint_cal_dataset, batch_size=len(joint_cal_dataset), shuffle=False)
    print(f"Created joint calibration dataset with {len(joint_cal_dataset)} samples.")
    
    # Prepare results table
    results = {}
    
    # Helper to restore merged model base state dict (uncalibrated)
    def create_merged_model(method, lambda_coeff):
        merged_sd = get_merged_backbone_state_dict(expert_state_dicts, pretrained_backbone_state_dict, method, lambda_coeff)
        
        # Reconstruct MultiTaskMergedModel
        model = MultiTaskMergedModel(tasks)
        
        # Load merged backbone
        model.backbone.load_state_dict({k.replace('backbone.', ''): v for k, v in merged_sd.items() if k.startswith('backbone.')})
        
        # Load expert heads as-is
        for i, task in enumerate(tasks):
            head_state_dict = {
                'weight': expert_state_dicts[i]['head.weight'],
                'bias': expert_state_dicts[i]['head.bias']
            }
            model.heads[task].load_state_dict(head_state_dict)
            
        return model.to(device)
        
    # Evaluate a configured model
    def run_evaluation(model, name, lsc_scales=None):
        if lsc_scales is not None:
            # LSC requires registering hooks per task and evaluating sequentially
            task_accuracies = {}
            for task in tasks:
                hook = LSCScaleHook(lsc_scales, task)
                hook.register(model)
                accs = evaluate_model(model, {task: test_loaders[task]}, device)
                task_accuracies[task] = accs[task]
                hook.remove()
            task_accuracies['average'] = sum(task_accuracies.values()) / len(task_accuracies)
            accs = task_accuracies
        else:
            accs = evaluate_model(model, test_loaders, device)
            
        print(f"Results for {name}:")
        print(f"  MNIST: {accs['mnist']:.2f}% | Fashion-MNIST: {accs['fashion']:.2f}% | CIFAR-10: {accs['cifar10']:.2f}% | Avg: {accs['average']:.2f}%")
        return accs
        
    # List of merge setups
    setups = [
        ("WA", 0.333),
        ("TA", 0.2),
        ("TA", 0.3),
        ("TA", 0.4),
        ("DARE", 0.4),
        ("TIES", 0.4)
    ]
    
    for merge_method, lambda_coeff in setups:
        setup_name = f"{merge_method}_lambda_{lambda_coeff}"
        print("\n" + "="*50)
        print(f"Evaluating {merge_method} with lambda = {lambda_coeff}")
        print("="*50)
        
        results[setup_name] = {}
        
        # 1. Evaluate Uncalibrated
        model = create_merged_model(merge_method, lambda_coeff)
        results[setup_name]['NONE'] = run_evaluation(model, "Uncalibrated (NONE)")
        
        # 2. Evaluate LSC (Full layer-wise scaling)
        model = create_merged_model(merge_method, lambda_coeff)
        lsc_scales = run_lsc(model, expert_models, cal_datasets)
        results[setup_name]['LSC'] = run_evaluation(model, "LSC (Layer-wise Scaling)", lsc_scales)
        
        # 3. Evaluate N-TAAC (Full native calibration)
        model = create_merged_model(merge_method, lambda_coeff)
        run_native_calibration(model, joint_loader, target_layers=None)
        results[setup_name]['N_TAAC'] = run_evaluation(model, "N-TAAC")
        
        # 4. Evaluate Proposed T-NAC configurations (Targeted Native Calibration)
        # T-NAC: layer4 only
        model = create_merged_model(merge_method, lambda_coeff)
        run_native_calibration(model, joint_loader, target_layers=['layer4'])
        results[setup_name]['T_NAC_L4'] = run_evaluation(model, "T-NAC (layer4 only)")
        
        # T-NAC: layer3 and layer4 only
        model = create_merged_model(merge_method, lambda_coeff)
        run_native_calibration(model, joint_loader, target_layers=['layer3', 'layer4'])
        results[setup_name]['T_NAC_L34'] = run_evaluation(model, "T-NAC (layer3 + layer4)")
        
        # T-NAC: layer2, layer3, and layer4 only
        model = create_merged_model(merge_method, lambda_coeff)
        run_native_calibration(model, joint_loader, target_layers=['layer2', 'layer3', 'layer4'])
        results[setup_name]['T_NAC_L234'] = run_evaluation(model, "T-NAC (layer2 + layer3 + layer4)")
        
        # T-NAC: early layers only (bn1 and layer1)
        model = create_merged_model(merge_method, lambda_coeff)
        run_native_calibration(model, joint_loader, target_layers=['bn1', 'layer1'])
        results[setup_name]['T_NAC_Early'] = run_evaluation(model, "T-NAC (bn1 + layer1 early)")
        
    # Print comparison summary across all methods and setups
    print("\n" + "="*80)
    print("FINAL SUMMARY COMPARISON")
    print("="*80)
    for setup_name, setup_results in results.items():
        print(f"\nMerge Setup: {setup_name}")
        print("-"*40)
        for method_name, accs in setup_results.items():
            print(f"  {method_name:<15}: Avg Acc = {accs['average']:.2f}% | MNIST: {accs['mnist']:.2f}% | F-MNIST: {accs['fashion']:.2f}% | CIFAR-10: {accs['cifar10']:.2f}%")
            
    # Save results dictionary to disk for later use/plotting
    torch.save(results, "merging_results.pt")
    print("\nSaved merging results to merging_results.pt")

if __name__ == "__main__":
    main()
