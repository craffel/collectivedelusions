import os
import json
import argparse
import torch
# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED on the GPU partition
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataset import get_dataset, get_dataloaders
from models import MultiTaskResNet18

# A custom dataset that merges multiple datasets with task-ID tracking
class JointDataset(Dataset):
    def __init__(self, datasets_dict):
        self.samples = []
        for task, dataset in datasets_dict.items():
            for i in range(len(dataset)):
                img, label = dataset[i]
                self.samples.append((img, label, task))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def get_calibration_loaders(N=128, seed=42, imbalance_task=None, imbalance_ratio=1.0):
    """Generate task-specific and joint calibration loaders, optionally with task imbalance."""
    tasks = ['mnist', 'fashion', 'cifar']
    cal_datasets = {}
    
    for task in tasks:
        size = N
        if imbalance_task is not None:
            if task == imbalance_task:
                size = int(N * imbalance_ratio)
            else:
                # Scale down other tasks if imbalance_ratio > 1, or scale them up
                size = N
        # Ensure we have at least 1 sample
        size = max(4, size)
        
        # Use different seed for calibration set than training set (e.g. seed + 100) to avoid direct overlap
        cal_datasets[task] = get_dataset(task, train=True, subset_size=size, seed=seed + 100)
        
    # Task-specific loaders
    loaders = {task: DataLoader(cal_datasets[task], batch_size=len(cal_datasets[task]), shuffle=False) for task in tasks}
    
    # Joint loader (contains all samples from all tasks)
    joint_ds = JointDataset(cal_datasets)
    # Batch size is the total number of samples to process all joint calibration samples in a single forward pass
    total_size = len(joint_ds)
    joint_loader = DataLoader(joint_ds, batch_size=total_size, shuffle=False)
    
    return loaders, joint_loader

# ==========================================
# 1. Weight Merging Functions
# ==========================================
def assemble_merged_model(expert_paths, pretrained_path, merge_mode='wa', lambda_val=0.2):
    """Assemble the merged model by loading expert heads and merging backbones."""
    # Initialize the merged model
    merged_model = MultiTaskResNet18(pretrained=False)
    
    # Load expert state dicts
    expert_states = {}
    for task, path in expert_paths.items():
        expert_states[task] = torch.load(path, map_location='cpu')
        
    # Load the task-specific heads directly from the experts
    for task in ['mnist', 'fashion', 'cifar']:
        # Load the task head weights
        head_state = {k.replace(f'heads.{task}.', ''): v for k, v in expert_states[task].items() if k.startswith(f'heads.{task}')}
        merged_model.heads[task].load_state_dict(head_state)
        
    # Merge the backbones
    pretrained_state = torch.load(pretrained_path, map_location='cpu')
    merged_backbone_state = {}
    
    # Extract backbone state keys
    backbone_keys = [k for k in pretrained_state.keys() if k.startswith('backbone.')]
    
    if merge_mode == 'wa':
        # Weight Averaging: average of the 3 expert backbones
        for key in backbone_keys:
            merged_backbone_state[key.replace('backbone.', '')] = sum(
                expert_states[t][key] for t in ['mnist', 'fashion', 'cifar']
            ) / 3.0
    elif merge_mode == 'ta':
        # Task Arithmetic: base + lambda * sum(task_vectors)
        for key in backbone_keys:
            task_vectors = []
            for t in ['mnist', 'fashion', 'cifar']:
                v = expert_states[t][key] - pretrained_state[key]
                task_vectors.append(v)
            merged_backbone_state[key.replace('backbone.', '')] = pretrained_state[key] + lambda_val * sum(task_vectors)
    else:
        raise ValueError(f"Unknown merge mode: {merge_mode}")
        
    merged_model.backbone.load_state_dict(merged_backbone_state)
    return merged_model

# ==========================================
# 2. Calibration Helper Functions
# ==========================================
def record_expert_stds(expert_path, cal_loader, task, device):
    """Record standard deviations across all BatchNorm layers of an expert."""
    model = MultiTaskResNet18(pretrained=False).to(device)
    model.load_state_dict(torch.load(expert_path, map_location=device))
    model.eval()
    
    bn_stds = {}
    handles = []
    recorded_inputs = {}
    
    def make_hook(name):
        def hook_fn(module, input):
            recorded_inputs[name] = input[0].detach()
        return hook_fn
        
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            handle = module.register_forward_pre_hook(make_hook(name))
            handles.append(handle)
            
    # Run a single forward pass with calibration batch
    for inputs, targets in cal_loader:
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = model(inputs, task)
        break # Only use one batch
        
    for handle in handles:
        handle.remove()
        
    for name, act in recorded_inputs.items():
        # Compute layer-wise standard deviation globally over all dimensions
        bn_stds[name] = torch.sqrt(act.var() + 1e-5).item()
        
    return bn_stds

# ==========================================
# 3. Calibration Implementation
# ==========================================

# (A) TCAC (Task-Conditional Activation Calibration)
def apply_tcac(merged_model, expert_paths, device):
    """Apply TCAC by swapping expert BatchNorm stats and affine parameters."""
    expert_models = {}
    for task, path in expert_paths.items():
        m = MultiTaskResNet18(pretrained=False).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        expert_models[task] = m
        
    # TCAC needs to return a dictionary of BN states for each task
    # so we can swap them dynamically prior to evaluation of that task
    tcac_bn_states = {}
    for task in ['mnist', 'fashion', 'cifar']:
        bn_state = {}
        for name, module in expert_models[task].backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_state[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'weight': module.weight.clone() if module.weight is not None else None,
                    'bias': module.bias.clone() if module.bias is not None else None
                }
        tcac_bn_states[task] = bn_state
    return tcac_bn_states

def set_tcac_bn_state(model, tcac_bn_states, task):
    """Swap the BatchNorm layers of model with the pre-saved state of the target task."""
    with torch.no_grad():
        for name, module in model.backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                state = tcac_bn_states[task][name]
                module.running_mean.copy_(state['running_mean'])
                module.running_var.copy_(state['running_var'])
                if module.weight is not None and state['weight'] is not None:
                    module.weight.copy_(state['weight'])
                if module.bias is not None and state['bias'] is not None:
                    module.bias.copy_(state['bias'])

# (B) TAAC (Task-Agnostic Activation Calibration / N-TAAC)
def apply_taac(merged_model, joint_loader, device):
    """Apply native Task-Agnostic Activation Calibration (N-TAAC)."""
    merged_model.train() # Put in train mode to record running stats
    
    orig_momentums = {}
    # Freeze model params and set momentum to 1.0 to overwrite stats in a single pass
    for name, module in merged_model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            orig_momentums[name] = module.momentum
            module.momentum = 1.0
            module.track_running_stats = True
            
    with torch.no_grad():
        for inputs, targets, tasks in joint_loader:
            inputs = inputs.to(device)
            # Pass through backbone to compute statistics
            _ = merged_model.backbone(inputs)
            break # Single pass of joint calibration set!
            
    # Restore original momentums and set to eval mode
    for name, module in merged_model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = orig_momentums[name]
            
    merged_model.eval()

# (B-2) R-TAAC (Regularized Task-Agnostic Activation Calibration)
def apply_rtaac(merged_model, joint_loader, alpha, device):
    """Apply Regularized Task-Agnostic Activation Calibration (R-TAAC) with shrinkage parameter alpha."""
    # Save original (uncalibrated) running stats (which are the averaged expert stats)
    uncal_stats = {}
    for name, module in merged_model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            uncal_stats[name] = {
                'running_mean': module.running_mean.clone(),
                'running_var': module.running_var.clone()
            }
            
    # Apply standard TAAC to populate running stats on the joint calibration set
    apply_taac(merged_model, joint_loader, device)
    
    # Perform shrinkage between the joint calibration stats and the uncalibrated stats
    with torch.no_grad():
        for name, module in merged_model.backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                mu_joint = module.running_mean
                var_joint = module.running_var
                
                mu_uncal = uncal_stats[name]['running_mean'].to(device)
                var_uncal = uncal_stats[name]['running_var'].to(device)
                
                # Apply shrinkage
                module.running_mean.copy_(alpha * mu_joint + (1.0 - alpha) * mu_uncal)
                module.running_var.copy_(alpha * var_joint + (1.0 - alpha) * var_uncal)

# (C) LSC (Layer-wise Scaling-only Calibration)
def calibrate_lsc(merged_model, expert_stds_all, cal_loaders, device):
    """Perform sequential Layer-wise Scaling-only Calibration (LSC)."""
    merged_model.eval()
    
    bn_names = []
    for name, module in merged_model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_names.append(name)
            
    # Compute gammas sequentially for each task
    gammas_task = {}
    for task in ['mnist', 'fashion', 'cifar']:
        gammas = {}
        active_handles = []
        loader = cal_loaders[task]
        
        for target_name in bn_names:
            recorded_activations = []
            
            def record_hook(module, input):
                recorded_activations.append(input[0].detach())
                
            target_module = dict(merged_model.backbone.named_modules())[target_name]
            temp_handle = target_module.register_forward_pre_hook(record_hook)
            
            # Pass calibration batch
            for inputs, targets in loader:
                inputs = inputs.to(device)
                with torch.no_grad():
                    _ = merged_model(inputs, task)
                break
                
            temp_handle.remove()
            
            x = torch.cat(recorded_activations, dim=0)
            var = x.var()
            std_merged = torch.sqrt(var + 1e-5).item()
            
            # Get expert standard deviation
            std_orig = expert_stds_all[task][target_name]
            gamma = std_orig / std_merged
            gammas[target_name] = gamma
            
            # Register permanent scaling hook for this task calibration pass
            # to affect deeper layers sequentially
            def make_scaling_hook(g):
                def scaling_hook(module, input):
                    return (input[0] * g,)
                return scaling_hook
                
            h = target_module.register_forward_pre_hook(make_scaling_hook(gamma))
            active_handles.append(h)
            
        # Clean up calibration hooks
        for h in active_handles:
            h.remove()
            
        gammas_task[task] = gammas
        
    return gammas_task

def register_lsc_hooks(merged_model, gammas, task):
    """Register LSC scaling hooks on BN layers for evaluation of a target task."""
    handles = []
    
    def make_scaling_hook(g):
        def scaling_hook(module, input):
            return (input[0] * g,)
        return scaling_hook
        
    for name, module in merged_model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            g = gammas[task][name]
            h = module.register_forward_pre_hook(make_scaling_hook(g))
            handles.append(h)
    return handles

# (D) Our Proposed Method: SP-TAAC (Sparsity-Preserving Task-Agnostic Calibration)
def calibrate_sp_taac(merged_model, expert_stds_all, joint_loader, device):
    """Perform sequential Sparsity-Preserving Task-Agnostic Calibration (SP-TAAC)."""
    merged_model.eval()
    
    bn_names = []
    for name, module in merged_model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_names.append(name)
            
    # Target stds are the average of the experts' stds
    target_stds = {}
    for name in bn_names:
        target_stds[name] = sum(expert_stds_all[t][name] for t in ['mnist', 'fashion', 'cifar']) / 3.0
        
    gammas = {}
    active_handles = []
    
    for target_name in bn_names:
        recorded_activations = []
        
        def record_hook(module, input):
            recorded_activations.append(input[0].detach())
            
        target_module = dict(merged_model.backbone.named_modules())[target_name]
        temp_handle = target_module.register_forward_pre_hook(record_hook)
        
        # Pass joint calibration batch through backbone
        for inputs, targets, tasks in joint_loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                _ = merged_model.backbone(inputs)
            break
            
        temp_handle.remove()
        
        x = torch.cat(recorded_activations, dim=0)
        var = x.var()
        std_merged = torch.sqrt(var + 1e-5).item()
        
        # Compute task-agnostic global scaling factor
        gamma = target_stds[target_name] / std_merged
        gammas[target_name] = gamma
        
        # Register permanent scaling hook for the calibration pass
        # to stabilize representations for deeper layers sequentially
        def make_scaling_hook(g):
            def scaling_hook(module, input):
                return (input[0] * g,)
            return scaling_hook
            
        h = target_module.register_forward_pre_hook(make_scaling_hook(gamma))
        active_handles.append(h)
        
    # Clean up calibration hooks
    for h in active_handles:
        h.remove()
        
    return gammas

def register_sp_taac_hooks(merged_model, gammas):
    """Register SP-TAAC scaling hooks on BN layers for evaluation."""
    handles = []
    
    def make_scaling_hook(g):
        def scaling_hook(module, input):
            return (input[0] * g,)
        return scaling_hook
        
    for name, module in merged_model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            g = gammas[name]
            h = module.register_forward_pre_hook(make_scaling_hook(g))
            handles.append(h)
    return handles

# ==========================================
# 4. Evaluation Loop
# ==========================================
def evaluate_model(model, test_loaders, cal_loaders, joint_loader, expert_paths,
                   cal_method='none', lsc_gammas=None, sp_taac_gammas=None, tcac_bn_states=None,
                   rtaac_alpha=0.5, device='cuda'):
    model.eval()
    results = {}
    
    # Store original model state to avoid permanent modification across tasks
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Pre-apply task-agnostic calibrations that affect the running statistics directly (like TAAC)
    if cal_method == 'taac':
        apply_taac(model, joint_loader, device)
    elif cal_method == 'rtaac':
        apply_rtaac(model, joint_loader, rtaac_alpha, device)
        
    for task in ['mnist', 'fashion', 'cifar']:
        loader = test_loaders[task]
        correct = 0
        total = 0
        
        # Set up dynamic hooks or swaps based on calibration method
        eval_hooks = []
        
        if cal_method == 'tcac':
            # Swap in expert-specific BatchNorm statistics and parameters
            set_tcac_bn_state(model, tcac_bn_states, task)
        elif cal_method == 'lsc':
            # Register task-conditional scaling hooks
            eval_hooks = register_lsc_hooks(model, lsc_gammas, task)
        elif cal_method == 'sp_taac':
            # Register task-agnostic scaling hooks
            eval_hooks = register_sp_taac_hooks(model, sp_taac_gammas)
            
        # Run test evaluation
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, task)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        # Clean up hooks
        for h in eval_hooks:
            h.remove()
            
        # Restore original state if we did task-conditional parameter swaps (TCAC)
        if cal_method == 'tcac':
            model.load_state_dict({k: v.to(device) for k, v in original_state.items()})
            
        acc = 100.0 * correct / total
        results[task] = acc
        
    # If we modified running statistics (TAAC or RTAAC), restore the original state before finishing
    if cal_method in ['taac', 'rtaac']:
        model.load_state_dict({k: v.to(device) for k, v in original_state.items()})
        
    results['avg'] = sum(results[t] for t in ['mnist', 'fashion', 'cifar']) / 3.0
    return results

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate merged models and calibration methods.")
    parser.add_argument('--merge_mode', type=str, default='wa', choices=['wa', 'ta'],
                        help="Merging mode ('wa' for Weight Averaging, 'ta' for Task Arithmetic).")
    parser.add_argument('--lambda_val', type=float, default=0.2,
                        help="Lambda coefficient for Task Arithmetic.")
    parser.add_argument('--cal_method', type=str, default='all', choices=['none', 'tcac', 'taac', 'lsc', 'sp_taac', 'rtaac', 'all'],
                        help="Calibration method ('none', 'tcac', 'taac', 'lsc', 'sp_taac', 'rtaac', or 'all').")
    parser.add_argument('--rtaac_alpha', type=float, default=0.5,
                        help="Shrinkage alpha for R-TAAC (0.0=layer-wise, 1.0=channel-wise).")
    parser.add_argument('--cal_size', type=int, default=128,
                        help="Calibration set size N per task.")
    parser.add_argument('--imbalance_task', type=str, default=None, choices=['mnist', 'fashion', 'cifar'],
                        help="Task to make imbalanced in the calibration set.")
    parser.add_argument('--imbalance_ratio', type=float, default=1.0,
                        help="Ratio to scale the imbalanced task size.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for calibration splits.")
    parser.add_argument('--output_json', type=str, default=None,
                        help="Path to save results as JSON.")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    expert_paths = {
        'mnist': 'checkpoints/expert_mnist.pt',
        'fashion': 'checkpoints/expert_fashion.pt',
        'cifar': 'checkpoints/expert_cifar.pt'
    }
    pretrained_path = 'checkpoints/pretrained.pt'
    
    # 1. Verify checkpoints exist
    for task, path in expert_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert checkpoint not found for {task}: {path}. Please run train.py first.")
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained base checkpoint not found: {pretrained_path}.")
        
    # 2. Get data loaders
    _, test_loaders = get_dataloaders(batch_size=128)
    cal_loaders, joint_loader = get_calibration_loaders(
        N=args.cal_size, seed=args.seed,
        imbalance_task=args.imbalance_task, imbalance_ratio=args.imbalance_ratio
    )
    
    # 3. Assemble merged model
    print(f"\nMerging Model using mode={args.merge_mode.upper()} (lambda={args.lambda_val if args.merge_mode=='ta' else 'N/A'})...")
    model = assemble_merged_model(expert_paths, pretrained_path, args.merge_mode, args.lambda_val).to(device)
    
    # 4. Pre-compute expert stds for scaling calibrations (LSC, SP-TAAC)
    print("Pre-computing expert standard deviations on calibration sets...")
    expert_stds_all = {}
    for task in ['mnist', 'fashion', 'cifar']:
        expert_stds_all[task] = record_expert_stds(expert_paths[task], cal_loaders[task], task, device)
        
    # 5. Pre-compute calibration structures
    print("Pre-computing calibration statistics...")
    tcac_bn_states = apply_tcac(model, expert_paths, device)
    lsc_gammas = calibrate_lsc(model, expert_stds_all, cal_loaders, device)
    sp_taac_gammas = calibrate_sp_taac(model, expert_stds_all, joint_loader, device)
    
    # 6. Evaluate chosen calibration method(s)
    methods_to_eval = ['none', 'tcac', 'taac', 'lsc', 'sp_taac', 'rtaac'] if args.cal_method == 'all' else [args.cal_method]
    
    results = {}
    print(f"\n" + "="*70)
    print(f"{'Method':<20} | {'MNIST':<10} | {'F-MNIST':<10} | {'CIFAR-10':<10} | {'Average':<10}")
    print("="*70)
    
    for m in methods_to_eval:
        res = evaluate_model(
            model, test_loaders, cal_loaders, joint_loader, expert_paths,
            cal_method=m, lsc_gammas=lsc_gammas, sp_taac_gammas=sp_taac_gammas, tcac_bn_states=tcac_bn_states,
            rtaac_alpha=args.rtaac_alpha, device=device
        )
        results[m] = res
        print(f"{m.upper():<20} | {res['mnist']:<10.2f} | {res['fashion']:<10.2f} | {res['cifar']:<10.2f} | {res['avg']:<10.2f}")
        
    print("="*70)
    
    # 7. Save results
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved evaluation results to {args.output_json}")
