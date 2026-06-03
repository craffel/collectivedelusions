import os
import json
import argparse
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors.")

# Deterministic carrier generation for Holographic CDMA
def generate_bipolar_carrier(shape, task_id, layer_name, device):
    # Compute a deterministic integer seed from layer_name and task_id
    hash_object = hashlib.md5(layer_name.encode())
    layer_hash = int(hash_object.hexdigest(), 16) % 100000
    seed = (task_id + 1) * 100000 + layer_hash
    
    # Use CPU generator to ensure exact matching across platforms, then move to device
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    
    carrier = torch.empty(shape, device='cpu').bernoulli_(p=0.5, generator=gen)
    carrier = carrier * 2.0 - 1.0
    return carrier.to(device)

# TCAC Hook to calibrate and rescale activation statistics
class TCAC_Hook:
    def __init__(self):
        self.mean_orig = {}
        self.std_orig = {}
        self.mean_merged = {}
        self.std_merged = {}
        self.temp_outputs = []
        self.mode = 'idle'  # 'collect_orig', 'collect_merged', 'eval', 'idle'
        self.current_task = None

    def hook_fn(self, module, input, output):
        if self.mode == 'collect_orig' or self.mode == 'collect_merged':
            self.temp_outputs.append(output.detach())
            return output
        elif self.mode == 'eval':
            if self.current_task in self.mean_orig and self.current_task in self.mean_merged:
                mo = self.mean_orig[self.current_task]
                so = self.std_orig[self.current_task]
                mm = self.mean_merged[self.current_task]
                sm = self.std_merged[self.current_task]
                
                # Reshape statistics to match activation shapes
                if len(output.shape) == 4:  # Conv maps: [B, C, H, W]
                    mo = mo.view(1, -1, 1, 1)
                    so = so.view(1, -1, 1, 1)
                    mm = mm.view(1, -1, 1, 1)
                    sm = sm.view(1, -1, 1, 1)
                elif len(output.shape) == 2:  # Linear features: [B, C]
                    mo = mo.view(1, -1)
                    so = so.view(1, -1)
                    mm = mm.view(1, -1)
                    sm = sm.view(1, -1)
                
                # Clamp standard deviation to prevent division-by-zero or extreme amplification
                sm = torch.clamp(sm, min=1e-2)
                so = torch.clamp(so, min=1e-2)
                calibrated = ((output - mm) / sm) * so + mo
                return calibrated
            return output
        return output
            
    def compute_stats(self, task_id, is_orig=True):
        if len(self.temp_outputs) == 0:
            return
        all_outputs = torch.cat(self.temp_outputs, dim=0)
        self.temp_outputs = []
        
        if len(all_outputs.shape) == 4:
            mean = all_outputs.mean(dim=[0, 2, 3])
            std = all_outputs.std(dim=[0, 2, 3], unbiased=False)
        else:
            mean = all_outputs.mean(dim=0)
            std = all_outputs.std(dim=0, unbiased=False)
            
        if is_orig:
            self.mean_orig[task_id] = mean
            self.std_orig[task_id] = std
        else:
            self.mean_merged[task_id] = mean
            self.std_merged[task_id] = std

# Helper to register TCAC hooks to all batchnorm layers of a ResNet backbone
def register_tcac_hooks(model):
    hooks = {}
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hook = TCAC_Hook()
            handle = module.register_forward_hook(hook.hook_fn)
            hooks[name] = (hook, handle)
    return hooks

def remove_tcac_hooks(hooks_dict):
    for name, (hook, handle) in hooks_dict.items():
        handle.remove()

# Helper to load data subsets
def get_dataset_loaders(batch_size=128):
    os.makedirs('./data', exist_ok=True)
    
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 1. CIFAR-10
    cifar_train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_rgb)
    cifar_test_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb)
    
    # 2. SVHN
    svhn_train_full = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_rgb)
    svhn_test_full = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_rgb)
    
    # 3. MNIST
    mnist_train_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_gray)
    mnist_test_full = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray)
    
    # Subsetting helper
    def make_subsets(train_ds, test_ds, train_size=2000, test_size=500, cal_size=128, seed=42):
        np.random.seed(seed)
        train_indices = np.random.choice(len(train_ds), train_size, replace=False)
        test_indices = np.random.choice(len(test_ds), test_size, replace=False)
        cal_indices = np.random.choice(len(train_ds), cal_size, replace=False)
        
        return (
            Subset(train_ds, train_indices),
            Subset(test_ds, test_indices),
            Subset(train_ds, cal_indices)
        )
        
    cifar_train, cifar_test, cifar_cal = make_subsets(cifar_train_full, cifar_test_full)
    svhn_train, svhn_test, svhn_cal = make_subsets(svhn_train_full, svhn_test_full)
    mnist_train, mnist_test, mnist_cal = make_subsets(mnist_train_full, mnist_test_full)
    
    loaders = {
        'cifar': {
            'train': DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=2),
            'test': DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=2),
            'cal': DataLoader(cifar_cal, batch_size=batch_size, shuffle=False, num_workers=2)
        },
        'svhn': {
            'train': DataLoader(svhn_train, batch_size=batch_size, shuffle=True, num_workers=2),
            'test': DataLoader(svhn_test, batch_size=batch_size, shuffle=False, num_workers=2),
            'cal': DataLoader(svhn_cal, batch_size=batch_size, shuffle=False, num_workers=2)
        },
        'mnist': {
            'train': DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2),
            'test': DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2),
            'cal': DataLoader(mnist_cal, batch_size=batch_size, shuffle=False, num_workers=2)
        }
    }
    return loaders

# Task-specific classification wrapper
class ResNetExpert(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# Model training loop
def train_expert(model, train_loader, epochs=2, lr=1e-4, weight_decay=1e-2):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    return model

def evaluate_model(model, test_loader):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

# --- Model Merging Operators ---

# 1. Task Arithmetic
def merge_task_arithmetic(W0, Wi_list, lambda_val):
    W_merged = {}
    for name in W0.keys():
        if W0[name].dtype in [torch.float32, torch.float64, torch.float16]:
            tau_sum = sum(Wi[name] - W0[name] for Wi in Wi_list)
            W_merged[name] = W0[name] + lambda_val * tau_sum
        else:
            W_merged[name] = W0[name]  # Keep integer/long weights unchanged (e.g. buffers)
    return W_merged

# 2. TIES Merging
def merge_ties(W0, Wi_list, lambda_val, k=0.2):
    W_merged = {}
    for name in W0.keys():
        if W0[name].dtype not in [torch.float32, torch.float64, torch.float16]:
            W_merged[name] = W0[name]
            continue
            
        # Compute task vectors
        taus = [Wi[name] - W0[name] for Wi in Wi_list]
        
        # Trim: keep top-k% elements per task vector
        trimmed_taus = []
        for tau in taus:
            flat_tau = tau.flatten()
            num_keep = max(1, int(k * len(flat_tau)))
            threshold = flat_tau.abs().kthvalue(len(flat_tau) - num_keep + 1).values
            mask = flat_tau.abs() >= threshold
            trimmed_tau = tau * mask.view_as(tau)
            trimmed_taus.append(trimmed_tau)
            
        # Sign Consensus
        sign_matrix = torch.stack([t.sign() for t in trimmed_taus], dim=0)
        mag_matrix = torch.stack([t.abs() for t in trimmed_taus], dim=0)
        
        # Weighted sign vote
        vote = (sign_matrix * mag_matrix).sum(dim=0)
        consensus_sign = vote.sign()
        
        # Disagreement Resolution & Merging
        resolved_updates = []
        for t in trimmed_taus:
            # Mask out parameters that disagree with the consensus sign
            agree_mask = (t.sign() == consensus_sign) & (consensus_sign != 0)
            resolved_updates.append(t * agree_mask)
            
        # Average resolved updates and apply scaling
        if len(resolved_updates) > 0:
            merged_tau = torch.stack(resolved_updates, dim=0).sum(dim=0) / len(resolved_updates)
            W_merged[name] = W0[name] + lambda_val * merged_tau
        else:
            W_merged[name] = W0[name]
    return W_merged

# 3. DARE Merging
def merge_dare(W0, Wi_list, lambda_val, p=0.5):
    W_merged = {}
    for name in W0.keys():
        if W0[name].dtype not in [torch.float32, torch.float64, torch.float16]:
            W_merged[name] = W0[name]
            continue
            
        taus = []
        for Wi in Wi_list:
            tau = Wi[name] - W0[name]
            # Drop elements with probability p
            mask = (torch.rand_like(tau) > p).float()
            # Rescale remaining elements
            rescaled_tau = (tau * mask) / (1.0 - p)
            taus.append(rescaled_tau)
            
        merged_tau = sum(taus) / len(taus)
        W_merged[name] = W0[name] + lambda_val * merged_tau
    return W_merged

# 4. Holographic CDMA Merging (HCM) - Our Visionary Proposal
def merge_hcm(W0, Wi_list, lambda_val, sum_normalize=True):
    # Construct Holographic multiplexed state dict
    H_state_dict = {}
    N = len(Wi_list)
    for name in W0.keys():
        if W0[name].dtype not in [torch.float32, torch.float64, torch.float16] or len(W0[name].shape) < 2:
            # Skip 1D buffers/biases, we will average them normally in the final model loader
            continue
            
        shape = W0[name].shape
        H_layer = torch.zeros_like(W0[name])
        for i, Wi in enumerate(Wi_list):
            tau = Wi[name] - W0[name]
            carrier = generate_bipolar_carrier(shape, i, name, W0[name].device)
            H_layer += tau * carrier
            
        if sum_normalize:
            H_state_dict[name] = H_layer / N
        else:
            H_state_dict[name] = H_layer
            
    return H_state_dict

# Loader function to demodulate HCM weights for a specific task
def load_hcm_weights_for_task(model, W0, H_state_dict, Wi_list, task_id, lambda_val):
    # Demodulate HCM weights on-the-fly for task_id
    new_state_dict = {}
    N = len(Wi_list)
    for name in W0.keys():
        if name in H_state_dict:
            H = H_state_dict[name]
            carrier = generate_bipolar_carrier(H.shape, task_id, name, H.device)
            # Demodulate: W0 + lambda * H * carrier
            new_state_dict[name] = W0[name] + lambda_val * H * carrier
        else:
            # For 1D variables like biases, running stats, we linearly average the experts
            if W0[name].dtype in [torch.float32, torch.float64, torch.float16]:
                # Average expert offsets
                avg_update = sum(Wi[name] - W0[name] for Wi in Wi_list) / N
                new_state_dict[name] = W0[name] + lambda_val * avg_update
            else:
                new_state_dict[name] = W0[name]
                
    model.backbone.load_state_dict(new_state_dict, strict=False)

# --- Calibration Execution (TCAC) ---
def run_tcac_calibration(model, loaders, hooks_dict, is_hcm=False, W0=None, H_state_dict=None, W_merged=None, backbone_Wi_list=None, Wi_list=None, lambda_val=1.0):
    # Set all hooks to collect expert stats on original model weights
    for task_id, (task_name, task_loaders) in enumerate(loaders.items()):
        # 1. Load original expert weights (both backbone and heads)
        model.load_state_dict(Wi_list[task_id], strict=True)
        
        # Configure hooks to collect orig stats
        for name, (hook, handle) in hooks_dict.items():
            hook.mode = 'collect_orig'
            hook.current_task = task_id
            
        model.eval()
        with torch.no_grad():
            for inputs, _ in task_loaders['cal']:
                inputs = inputs.to(device)
                _ = model(inputs)
                
        # Compute stats for this task
        for name, (hook, handle) in hooks_dict.items():
            hook.compute_stats(task_id, is_orig=True)
            
    # 2. Configure model with merged weights, and collect merged statistics
    for task_id, (task_name, task_loaders) in enumerate(loaders.items()):
        if is_hcm:
            load_hcm_weights_for_task(model, W0, H_state_dict, backbone_Wi_list, task_id, lambda_val)
        else:
            model.backbone.load_state_dict(W_merged, strict=False)
            
        # Set task classification head
        model.backbone.fc.weight.data.copy_(Wi_list[task_id]['backbone.fc.weight'])
        model.backbone.fc.bias.data.copy_(Wi_list[task_id]['backbone.fc.bias'])
        
        # Configure hooks to collect merged stats
        for name, (hook, handle) in hooks_dict.items():
            hook.mode = 'collect_merged'
            hook.current_task = task_id
            
        model.eval()
        with torch.no_grad():
            for inputs, _ in task_loaders['cal']:
                inputs = inputs.to(device)
                _ = model(inputs)
                
        # Compute merged stats for this task
        for name, (hook, handle) in hooks_dict.items():
            hook.compute_stats(task_id, is_orig=False)
            
    # 3. Put hooks into evaluation mode
    for name, (hook, handle) in hooks_dict.items():
        hook.mode = 'eval'


# --- Main Orchestration ---
def main():
    parser = argparse.ArgumentParser(description="Holographic CDMA Merging Experiments")
    parser.add_argument('--retrain', action='store_true', help="Force retraining of experts")
    parser.add_argument('--epochs', type=int, default=2, help="Expert training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    
    print("Initializing Datasets...")
    loaders = get_dataset_loaders()
    
    tasks = ['cifar', 'svhn', 'mnist']
    expert_paths = [f"expert_{t}.pt" for t in tasks]
    
    Wi_list = []
    
    # 1. Base Pre-trained model initialization
    print("Loading base pre-trained model...")
    base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    # Save base model's backbone weights
    W0 = {k: v.cpu().clone() for k, v in base_model.state_dict().items() if not k.startswith('fc')}
    
    # 2. Train or Load Experts
    for i, (task_name, task_loaders) in enumerate(loaders.items()):
        model_path = expert_paths[i]
        expert_model = ResNetExpert()
        
        if os.path.exists(model_path) and not args.retrain:
            print(f"Loading trained expert for {task_name} from {model_path}...")
            expert_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            print(f"Fine-tuning expert for {task_name}...")
            expert_model = train_expert(expert_model, task_loaders['train'], epochs=args.epochs, lr=args.lr)
            torch.save(expert_model.state_dict(), model_path)
            print(f"Saved expert checkpoint to {model_path}")
            
        # Record expert validation performance
        acc = evaluate_model(expert_model, task_loaders['test'])
        print(f"Expert {task_name} Test Accuracy: {acc:.2f}%")
        
        # Save state dict
        Wi_list.append({k: v.cpu().clone() for k, v in expert_model.state_dict().items()})
        
    # Standard ResNet-18 model container to run merging evaluations
    eval_model = ResNetExpert()
    
    # Extract only backbone state dicts of experts for merging
    backbone_Wi_list = []
    for Wi in Wi_list:
        backbone_Wi = {k.replace('backbone.', ''): v for k, v in Wi.items() if k.startswith('backbone') and not k.startswith('backbone.fc')}
        backbone_Wi_list.append(backbone_Wi)
        
    # Baseline performances
    expert_accs = []
    for i, (task_name, task_loaders) in enumerate(loaders.items()):
        # Set exact expert weights
        eval_model.load_state_dict(Wi_list[i])
        acc = evaluate_model(eval_model, task_loaders['test'])
        expert_accs.append(acc)
    print(f"\nIndividual Expert upper bounds: CIFAR: {expert_accs[0]:.2f}%, SVHN: {expert_accs[1]:.2f}%, MNIST: {expert_accs[2]:.2f}%")
    print(f"Average Expert upper bound: {np.mean(expert_accs):.2f}%")
    
    # Setup sweeps over lambda scaling factor
    lambdas = np.arange(0.1, 1.6, 0.1)
    
    results = {
        'task_arithmetic': [],
        'ties': [],
        'dare': [],
        'tcac_arithmetic': [],
        'hcm': [],
        'hcm_tcac': []
    }
    
    # Sweep evaluations
    for l in lambdas:
        print(f"\nEvaluating lambda = {l:.1f}...")
        
        # --- Method 1: Task Arithmetic ---
        W_ta = merge_task_arithmetic(W0, backbone_Wi_list, l)
        ta_accs = []
        for i, (task_name, task_loaders) in enumerate(loaders.items()):
            # Set merged backbone and expert heads
            eval_model.backbone.load_state_dict(W_ta, strict=False)
            eval_model.backbone.fc.weight.data.copy_(Wi_list[i]['backbone.fc.weight'])
            eval_model.backbone.fc.bias.data.copy_(Wi_list[i]['backbone.fc.bias'])
            acc = evaluate_model(eval_model, task_loaders['test'])
            ta_accs.append(acc)
        results['task_arithmetic'].append(np.mean(ta_accs))
        print(f"  Task Arithmetic average accuracy: {np.mean(ta_accs):.2f}%")
        
        # --- Method 2: TIES ---
        W_ties = merge_ties(W0, backbone_Wi_list, l, k=0.2)
        ties_accs = []
        for i, (task_name, task_loaders) in enumerate(loaders.items()):
            eval_model.backbone.load_state_dict(W_ties, strict=False)
            eval_model.backbone.fc.weight.data.copy_(Wi_list[i]['backbone.fc.weight'])
            eval_model.backbone.fc.bias.data.copy_(Wi_list[i]['backbone.fc.bias'])
            acc = evaluate_model(eval_model, task_loaders['test'])
            ties_accs.append(acc)
        results['ties'].append(np.mean(ties_accs))
        print(f"  TIES average accuracy: {np.mean(ties_accs):.2f}%")
        
        # --- Method 3: DARE ---
        W_dare = merge_dare(W0, backbone_Wi_list, l, p=0.5)
        dare_accs = []
        for i, (task_name, task_loaders) in enumerate(loaders.items()):
            eval_model.backbone.load_state_dict(W_dare, strict=False)
            eval_model.backbone.fc.weight.data.copy_(Wi_list[i]['backbone.fc.weight'])
            eval_model.backbone.fc.bias.data.copy_(Wi_list[i]['backbone.fc.bias'])
            acc = evaluate_model(eval_model, task_loaders['test'])
            dare_accs.append(acc)
        results['dare'].append(np.mean(dare_accs))
        print(f"  DARE average accuracy: {np.mean(dare_accs):.2f}%")
        
        # --- Method 4: TCAC on Task Arithmetic ---
        tcac_hooks = register_tcac_hooks(eval_model)
        run_tcac_calibration(eval_model, loaders, tcac_hooks, is_hcm=False, W_merged=W_ta, Wi_list=Wi_list)
        
        tcac_ta_accs = []
        for i, (task_name, task_loaders) in enumerate(loaders.items()):
            eval_model.backbone.load_state_dict(W_ta, strict=False)
            eval_model.backbone.fc.weight.data.copy_(Wi_list[i]['backbone.fc.weight'])
            eval_model.backbone.fc.bias.data.copy_(Wi_list[i]['backbone.fc.bias'])
            # Set hook target task
            for name, (hook, handle) in tcac_hooks.items():
                hook.current_task = i
            acc = evaluate_model(eval_model, task_loaders['test'])
            tcac_ta_accs.append(acc)
        remove_tcac_hooks(tcac_hooks)
        results['tcac_arithmetic'].append(np.mean(tcac_ta_accs))
        print(f"  TCAC + Task Arithmetic average accuracy: {np.mean(tcac_ta_accs):.2f}%")
        
        # --- Method 5: Holographic CDMA Merging (HCM) ---
        H_hcm = merge_hcm(W0, backbone_Wi_list, l, sum_normalize=True)
        hcm_accs = []
        for i, (task_name, task_loaders) in enumerate(loaders.items()):
            # Demodulate weights for task i
            load_hcm_weights_for_task(eval_model, W0, H_hcm, backbone_Wi_list, i, l)
            eval_model.backbone.fc.weight.data.copy_(Wi_list[i]['backbone.fc.weight'])
            eval_model.backbone.fc.bias.data.copy_(Wi_list[i]['backbone.fc.bias'])
            acc = evaluate_model(eval_model, task_loaders['test'])
            hcm_accs.append(acc)
        results['hcm'].append(np.mean(hcm_accs))
        print(f"  Holographic CDMA (HCM) average accuracy: {np.mean(hcm_accs):.2f}%")
        
        # --- Method 6: HCM + TCAC ---
        tcac_hooks = register_tcac_hooks(eval_model)
        run_tcac_calibration(eval_model, loaders, tcac_hooks, is_hcm=True, W0=W0, H_state_dict=H_hcm, backbone_Wi_list=backbone_Wi_list, Wi_list=Wi_list, lambda_val=l)
        
        hcm_tcac_accs = []
        for i, (task_name, task_loaders) in enumerate(loaders.items()):
            load_hcm_weights_for_task(eval_model, W0, H_hcm, backbone_Wi_list, i, l)
            eval_model.backbone.fc.weight.data.copy_(Wi_list[i]['backbone.fc.weight'])
            eval_model.backbone.fc.bias.data.copy_(Wi_list[i]['backbone.fc.bias'])
            for name, (hook, handle) in tcac_hooks.items():
                hook.current_task = i
            acc = evaluate_model(eval_model, task_loaders['test'])
            hcm_tcac_accs.append(acc)
        remove_tcac_hooks(tcac_hooks)
        results['hcm_tcac'].append(np.mean(hcm_tcac_accs))
        print(f"  HCM + TCAC average accuracy: {np.mean(hcm_tcac_accs):.2f}%")
        
    # Log and print peak summary
    print("\n--- EXPERIMENT COMPLETE ---")
    print(f"Upper Bound Average: {np.mean(expert_accs):.2f}%")
    for method, accs in results.items():
        peak_idx = np.argmax(accs)
        print(f"Peak {method}: {accs[peak_idx]:.2f}% at lambda = {lambdas[peak_idx]:.1f}")
        
    # Save results to JSON
    with open('experiment_results.json', 'w') as f:
        json.dump({
            'lambdas': lambdas.tolist(),
            'expert_accs': expert_accs,
            'results': results
        }, f, indent=4)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.axhline(y=np.mean(expert_accs), color='r', linestyle='--', label='Expert Upper Bound')
    plt.plot(lambdas, results['task_arithmetic'], marker='o', label='Task Arithmetic')
    plt.plot(lambdas, results['ties'], marker='s', label='TIES Merging')
    plt.plot(lambdas, results['dare'], marker='^', label='DARE Merging')
    plt.plot(lambdas, results['tcac_arithmetic'], marker='d', label='Task Arithmetic + TCAC')
    plt.plot(lambdas, results['hcm'], marker='x', label='HCM (Ours)')
    plt.plot(lambdas, results['hcm_tcac'], marker='*', linewidth=2, label='HCM + TCAC (Ours)')
    plt.xlabel('Scaling Factor (lambda)')
    plt.ylabel('Average Test Accuracy (%)')
    plt.title('Comparison of Model Merging Methods across Scaling Sweeps')
    plt.grid(True)
    plt.legend()
    plt.savefig('merging_performance_sweep.png', dpi=300)
    print("Saved sweep performance plot to 'merging_performance_sweep.png'")

if __name__ == '__main__':
    main()
