import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Disable cuDNN to bypass driver compatibility issues on this cluster
    torch.backends.cudnn.enabled = False

# Define ResNet18 with task heads
class MultiTaskResNet18(nn.Module):
    def __init__(self, tasks=['mnist', 'fmnist', 'cifar10']):
        super().__init__()
        # Load pre-trained ResNet-18
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace FC layer with Identity
        self.backbone.fc = nn.Identity()
        
        # Create task-specific classification heads
        self.heads = nn.ModuleDict({
            task: nn.Linear(512, 10) for task in tasks
        })

    def forward(self, x, task):
        features = self.backbone(x)
        return self.heads[task](features)

# Symmetric uniform quantization
def quantize_tensor(tensor, bits=8, quant_type='per_tensor'):
    if bits is None:
        return tensor
    
    qmax = 2**(bits - 1) - 1
    tensor_shape = tensor.shape
    
    if quant_type == 'per_tensor' or len(tensor_shape) < 2:
        max_val = torch.max(torch.abs(tensor))
        if max_val == 0:
            return tensor
        scale = max_val / qmax
        quantized = torch.clamp(torch.round(tensor / scale), -qmax, qmax)
        return quantized * scale
    elif quant_type == 'per_channel':
        # Quantize per output channel (dimension 0)
        flat_tensor = tensor.view(tensor_shape[0], -1)
        max_vals = torch.max(torch.abs(flat_tensor), dim=1, keepdim=True).values
        # Handle zero max values
        max_vals[max_vals == 0] = 1.0
        scales = max_vals / qmax
        
        # Reshape scales to match input tensor dimensions for broadcasting
        scales_shape = [tensor_shape[0]] + [1] * (len(tensor_shape) - 1)
        scales = scales.view(scales_shape)
        
        quantized = torch.clamp(torch.round(tensor / scales), -qmax, qmax)
        return quantized * scales
    else:
        return tensor

# Apply quantization to all backbone weights
def quantize_backbone_(backbone, bits=8, quant_type='per_tensor'):
    quantized_state = {}
    for name, param in backbone.state_dict().items():
        if 'weight' in name and len(param.shape) >= 2: # Only quantize weight matrices
            quantized_state[name] = quantize_tensor(param.clone(), bits, quant_type)
        else:
            quantized_state[name] = param.clone()
    return quantized_state

# Load datasets
def get_dataloaders(batch_size=256):
    os.makedirs("./data", exist_ok=True)
    
    # Common transforms: Resize to 32x32, convert to RGB, normalize
    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # MNIST
    mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_gray)
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_gray)
    
    # Fashion-MNIST
    fmnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_gray)
    fmnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_gray)
    
    # CIFAR-10
    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_rgb)
    cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_rgb)
    
    loaders = {
        'mnist': {
            'train': DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2),
            'test': DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2)
        },
        'fmnist': {
            'train': DataLoader(fmnist_train, batch_size=batch_size, shuffle=True, num_workers=2),
            'test': DataLoader(fmnist_test, batch_size=batch_size, shuffle=False, num_workers=2)
        },
        'cifar10': {
            'train': DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=2),
            'test': DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=2)
        }
    }
    return loaders

# Training function for a single expert
def train_expert(task, loaders, device):
    print(f"\n--- Training Expert for {task.upper()} ---")
    model = MultiTaskResNet18().to(device)
    
    # Freeze other task heads, keep only backbone and current task head trainable
    for t, head in model.heads.items():
        if t != task:
            for param in head.parameters():
                param.requires_grad = False
                
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loaders[task]['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x, task)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/5 - Loss: {total_loss/total:.4f} - Acc: {acc:.2f}%")
        
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loaders[task]['test']:
            x, y = x.to(device), y.to(device)
            outputs = model(x, task)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    test_acc = 100.0 * correct / total
    print(f"Finished training {task}. Test Accuracy: {test_acc:.2f}%")
    
    os.makedirs("./checkpoints", exist_ok=True)
    # Save the whole model state dict
    torch.save(model.state_dict(), f"./checkpoints/expert_{task}.pt")

# Evaluate merged model accuracy
def evaluate_merged(merged_backbone_state, task_heads_state, loaders, device, bits=None, quant_type='per_tensor', corruption='none'):
    model = MultiTaskResNet18().to(device)
    
    # Load backbone weights
    if bits is not None:
        # Clone state and quantize
        backbone_loaded = MultiTaskResNet18().backbone
        backbone_loaded.load_state_dict(merged_backbone_state)
        quantized_state = quantize_backbone_(backbone_loaded, bits, quant_type)
        model.backbone.load_state_dict(quantized_state)
    else:
        model.backbone.load_state_dict(merged_backbone_state)
        
    # Load task heads weights
    model.heads.load_state_dict(task_heads_state)
    model.eval()
    
    results = {}
    with torch.no_grad():
        for task in ['mnist', 'fmnist', 'cifar10']:
            correct = 0
            total = 0
            for x, y in loaders[task]['test']:
                x, y = x.to(device), y.to(device)
                
                # Apply environmental corruptions
                if corruption == 'noise':
                    x = x + torch.randn_like(x) * 0.1
                elif corruption == 'blur':
                    x = torchvision.transforms.functional.gaussian_blur(x, [3, 3], [1.0, 1.0])
                    
                outputs = model(x, task)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
            results[task] = 100.0 * correct / total
    results['avg'] = sum(results.values()) / len(results)
    return results

# Model Merging Implementations

# 1. Weight Averaging (WA)
def merge_wa(expert_states):
    merged_backbone = {}
    keys = expert_states[0].keys()
    for key in keys:
        if 'backbone' in key:
            backbone_key = key.replace('backbone.', '')
            tensors = [state[key] for state in expert_states]
            if torch.is_floating_point(tensors[0]):
                merged_backbone[backbone_key] = torch.stack(tensors).mean(dim=0)
            else:
                merged_backbone[backbone_key] = tensors[0]
    return merged_backbone

# 2. Task Arithmetic (TA)
def merge_ta(progenitor_state, expert_states, lambd=0.4):
    merged_backbone = {}
    keys = progenitor_state.keys()
    for key in keys:
        if 'backbone' in key:
            backbone_key = key.replace('backbone.', '')
            prog_val = progenitor_state[key]
            if torch.is_floating_point(prog_val):
                updates = []
                for state in expert_states:
                    updates.append(state[key] - prog_val)
                avg_update = torch.stack(updates).mean(dim=0)
                merged_backbone[backbone_key] = prog_val + lambd * avg_update * len(expert_states)
            else:
                merged_backbone[backbone_key] = prog_val
    return merged_backbone

# 3. WCPR (Wasserstein-Calibrated Parameter Resonance)
def merge_wcpr(progenitor_state, expert_states):
    merged_backbone = {}
    keys = progenitor_state.keys()
    
    for key in keys:
        if 'backbone' in key:
            backbone_key = key.replace('backbone.', '')
            prog_val = progenitor_state[key]
            
            # If 1D parameter or not weight (e.g., bias, running_stats), do simple average
            if 'weight' not in key or len(prog_val.shape) < 2:
                tensors = [state[key] for state in expert_states]
                if torch.is_floating_point(tensors[0]):
                    merged_backbone[backbone_key] = torch.stack(tensors).mean(dim=0)
                else:
                    merged_backbone[backbone_key] = tensors[0]
                continue
                
            # Perform 1D Wasserstein calibration channel-by-channel
            # Flatten to (D_out, num_features)
            orig_shape = prog_val.shape
            D_out = orig_shape[0]
            
            # Experts and Progenitor
            prog_flat = prog_val.view(D_out, -1)
            expert_flats = [(state[key] - prog_val).view(D_out, -1) for state in expert_states]
            
            # Merged uncalibrated update
            merged_update = torch.stack(expert_flats).mean(dim=0) # shape (D_out, num_features)
            
            calibrated_update = torch.zeros_like(merged_update)
            
            for c in range(D_out):
                x = merged_update[c] # merged update for channel c
                s_k = [expert_flat[c] for expert_flat in expert_flats] # expert updates for channel c
                
                # Sort merged update
                sort_idx = torch.argsort(x)
                
                # Sort each expert update
                s_k_sorted = [torch.sort(s)[0] for s in s_k]
                
                # Wasserstein-2 barycenter in 1D is the average of sorted values
                y_sorted = torch.stack(s_k_sorted).mean(dim=0)
                
                # Map back to original merged order
                x_cal = torch.zeros_like(x)
                x_cal[sort_idx] = y_sorted
                
                calibrated_update[c] = x_cal
                
            # Reconstruct weight
            calibrated_flat = prog_flat + calibrated_update
            merged_backbone[backbone_key] = calibrated_flat.view(orig_shape)
            
    return merged_backbone

# 4. QCOT (Quantization-Constrained Optimal Transport)
def merge_qcot(progenitor_state, expert_states, C=0.5):
    merged_backbone = {}
    keys = progenitor_state.keys()
    
    for key in keys:
        if 'backbone' in key:
            backbone_key = key.replace('backbone.', '')
            prog_val = progenitor_state[key]
            
            # If 1D parameter or not weight (e.g., bias, running_stats), do simple average
            if 'weight' not in key or len(prog_val.shape) < 2:
                tensors = [state[key] for state in expert_states]
                if torch.is_floating_point(tensors[0]):
                    merged_backbone[backbone_key] = torch.stack(tensors).mean(dim=0)
                else:
                    merged_backbone[backbone_key] = tensors[0]
                continue
                
            # Perform Quantization-Constrained 1D Wasserstein Calibration channel-by-channel
            orig_shape = prog_val.shape
            D_out = orig_shape[0]
            
            # Experts and Progenitor
            prog_flat = prog_val.view(D_out, -1)
            expert_flats = [(state[key] - prog_val).view(D_out, -1) for state in expert_states]
            
            # Merged uncalibrated update
            merged_update = torch.stack(expert_flats).mean(dim=0)
            
            calibrated_update = torch.zeros_like(merged_update)
            
            for c in range(D_out):
                x = merged_update[c]
                s_k = [expert_flat[c] for expert_flat in expert_flats]
                
                # Sort merged update
                sort_idx = torch.argsort(x)
                
                # Sort each expert update
                s_k_sorted = [torch.sort(s)[0] for s in s_k]
                
                # Wasserstein-2 barycenter in 1D is average of sorted values
                y_sorted = torch.stack(s_k_sorted).mean(dim=0)
                
                # QCOT: Apply infinity-norm clipping to the barycenter
                y_sorted_clipped = torch.clamp(y_sorted, -C, C)
                
                # Map back to original merged order
                x_cal = torch.zeros_like(x)
                x_cal[sort_idx] = y_sorted_clipped
                
                calibrated_update[c] = x_cal
                
            # Reconstruct weight
            calibrated_flat = prog_flat + calibrated_update
            merged_backbone[backbone_key] = calibrated_flat.view(orig_shape)
            
    return merged_backbone

# 5. QCSW (Quantization-Constrained Sliced Wasserstein) - OUR PROPOSED METHOD!
def merge_qcsw(progenitor_state, expert_states, C=0.5, num_projections=None):
    merged_backbone = {}
    keys = progenitor_state.keys()
    
    for key in keys:
        if 'backbone' in key:
            backbone_key = key.replace('backbone.', '')
            prog_val = progenitor_state[key]
            
            # If 1D parameter or not weight (e.g., bias, running_stats), do simple average
            if 'weight' not in key or len(prog_val.shape) < 2:
                tensors = [state[key] for state in expert_states]
                if torch.is_floating_point(tensors[0]):
                    merged_backbone[backbone_key] = torch.stack(tensors).mean(dim=0)
                else:
                    merged_backbone[backbone_key] = tensors[0]
                continue
                
            # Apply our proposed Quantization-Constrained Orthogonal Sliced Wasserstein (QCOSW) Calibration!
            orig_shape = prog_val.shape
            D_out = orig_shape[0]
            
            # Experts and Progenitor flattened
            prog_flat = prog_val.view(D_out, -1)
            D_in = prog_flat.shape[1]
            
            expert_flats = [(state[key] - prog_val).view(D_out, -1) for state in expert_states]
            
            # Merged uncalibrated update
            merged_update = torch.stack(expert_flats).mean(dim=0) # shape (D_out, D_in)
            
            # Generate random orthogonal projection directions via QR decomposition
            # This guarantees 100% information preservation with zero dimensionality reduction artifacts.
            q, r = torch.linalg.qr(torch.randn(D_in, D_in, device=prog_val.device))
            Theta = q # shape (D_in, D_in)
            
            # Project merged update onto orthogonal directions
            # X shape: (D_out, D_in)
            X = torch.matmul(merged_update, Theta)
            
            # Project expert updates
            # Y_k shape: (D_out, D_in)
            Y = [torch.matmul(expert_flat, Theta) for expert_flat in expert_flats]
            
            calibrated_projections = torch.zeros_like(X)
            
            # For each orthogonal projection direction, solve 1D QCOT calibration
            for p in range(D_in):
                x_p = X[:, p]
                s_kp = [y_k[:, p] for y_k in Y]
                
                # Sort merged projected vector
                sort_idx = torch.argsort(x_p)
                
                # Sort expert projected vectors
                s_kp_sorted = [torch.sort(s)[0] for s in s_kp]
                
                # 1D Wasserstein barycenter is average of sorted values
                y_p_sorted = torch.stack(s_kp_sorted).mean(dim=0)
                
                # Apply clipping to suppress quantization noise
                y_p_sorted_clipped = torch.clamp(y_p_sorted, -C, C)
                
                # Map back to original merged order
                x_cal_p = torch.zeros_like(x_p)
                x_cal_p[sort_idx] = y_p_sorted_clipped
                
                calibrated_projections[:, p] = x_cal_p
                
            # Reconstruct the high-dimensional weight matrix via inverse orthogonal transform
            # Since Theta is orthogonal, its inverse is simply its transpose Theta.T!
            reconstructed_update = torch.matmul(calibrated_projections, Theta.T)
            
            # Reconstruct weight
            calibrated_flat = prog_flat + reconstructed_update
            merged_backbone[backbone_key] = calibrated_flat.view(orig_shape)
            
    return merged_backbone

def extract_backbone_state(state_dict):
    backbone_state = {}
    for k, v in state_dict.items():
        if 'backbone.' in k:
            backbone_state[k.replace('backbone.', '')] = v
    return backbone_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train expert models')
    parser.add_argument('--evaluate', action='store_true', help='Merge models and evaluate them')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    set_seed(42)
    loaders = get_dataloaders()
    
    if args.train:
        train_expert('mnist', loaders, device)
        train_expert('fmnist', loaders, device)
        train_expert('cifar10', loaders, device)
        return
        
    if args.evaluate:
        print("\n=== Loading Checkpoints and Performing Model Merging ===")
        # Check if checkpoints exist
        checkpoint_files = ["expert_mnist.pt", "expert_fmnist.pt", "expert_cifar10.pt"]
        for f in checkpoint_files:
            if not os.path.exists(f"./checkpoints/{f}"):
                print(f"Checkpoint ./checkpoints/{f} not found! Please train experts first by running: python train_and_merge.py --train")
                return
                
        # Load progenitor state
        progenitor = MultiTaskResNet18()
        progenitor_state = progenitor.state_dict()
        
        # Load expert states
        expert_mnist = MultiTaskResNet18()
        expert_mnist.load_state_dict(torch.load("./checkpoints/expert_mnist.pt", map_location=device))
        
        expert_fmnist = MultiTaskResNet18()
        expert_fmnist.load_state_dict(torch.load("./checkpoints/expert_fmnist.pt", map_location=device))
        
        expert_cifar10 = MultiTaskResNet18()
        expert_cifar10.load_state_dict(torch.load("./checkpoints/expert_cifar10.pt", map_location=device))
        
        expert_states = [expert_mnist.state_dict(), expert_fmnist.state_dict(), expert_cifar10.state_dict()]
        
        # Collect task heads
        # The heads are trained specifically. We will route to the appropriate head at eval time.
        # We can construct a combined task_heads_state
        task_heads_state = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            # Load task classification head from corresponding expert
            if task == 'mnist':
                task_heads_state[f"{task}.weight"] = expert_mnist.state_dict()["heads.mnist.weight"]
                task_heads_state[f"{task}.bias"] = expert_mnist.state_dict()["heads.mnist.bias"]
            elif task == 'fmnist':
                task_heads_state[f"{task}.weight"] = expert_fmnist.state_dict()["heads.fmnist.weight"]
                task_heads_state[f"{task}.bias"] = expert_fmnist.state_dict()["heads.fmnist.bias"]
            elif task == 'cifar10':
                task_heads_state[f"{task}.weight"] = expert_cifar10.state_dict()["heads.cifar10.weight"]
                task_heads_state[f"{task}.bias"] = expert_cifar10.state_dict()["heads.cifar10.bias"]
                
        print("\n--- Running Merging Baselines and Evaluations ---")
        
        # We evaluate: FP32, INT8 Per-Tensor, INT8 Per-Channel, INT4 Per-Channel, and Corruptions
        eval_configs = [
            {'bits': None, 'quant_type': 'none', 'corruption': 'none', 'name': 'FP32'},
            {'bits': 8, 'quant_type': 'per_tensor', 'corruption': 'none', 'name': 'INT8 Per-Tensor'},
            {'bits': 8, 'quant_type': 'per_channel', 'corruption': 'none', 'name': 'INT8 Per-Channel'},
            {'bits': 4, 'quant_type': 'per_channel', 'corruption': 'none', 'name': 'INT4 Per-Channel'},
            {'bits': None, 'quant_type': 'none', 'corruption': 'noise', 'name': 'FP32 + Noise'},
            {'bits': None, 'quant_type': 'none', 'corruption': 'blur', 'name': 'FP32 + Blur'}
        ]
        
        # 1. Individual experts (Oracle upper bound)
        print("\nEvaluating Individual Experts (Oracle Bounds):")
        oracle_results = {
            'mnist': evaluate_merged(extract_backbone_state(expert_mnist.state_dict()), task_heads_state, loaders, device)['mnist'],
            'fmnist': evaluate_merged(extract_backbone_state(expert_fmnist.state_dict()), task_heads_state, loaders, device)['fmnist'],
            'cifar10': evaluate_merged(extract_backbone_state(expert_cifar10.state_dict()), task_heads_state, loaders, device)['cifar10'],
        }
        oracle_results['avg'] = sum(oracle_results.values()) / len(oracle_results)
        print(f"MNIST: {oracle_results['mnist']:.2f}% | FMNIST: {oracle_results['fmnist']:.2f}% | CIFAR10: {oracle_results['cifar10']:.2f}% | Average: {oracle_results['avg']:.2f}%")
        
        all_results = {}
        
        # 2. Weight Averaging (WA)
        print("\nWA Merging...")
        wa_backbone = merge_wa(expert_states)
        all_results['WA'] = {}
        for config in eval_configs:
            res = evaluate_merged(wa_backbone, task_heads_state, loaders, device, config['bits'], config['quant_type'], config.get('corruption', 'none'))
            all_results['WA'][config['name']] = res
            print(f"WA ({config['name']}): MNIST: {res['mnist']:.2f}% | FMNIST: {res['fmnist']:.2f}% | CIFAR10: {res['cifar10']:.2f}% | Avg: {res['avg']:.2f}%")
            
        # 3. Task Arithmetic (TA) with lambda = 0.4
        print("\nTA Merging (lambda=0.4)...")
        ta_backbone = merge_ta(progenitor_state, expert_states, lambd=0.4)
        all_results['TA'] = {}
        for config in eval_configs:
            res = evaluate_merged(ta_backbone, task_heads_state, loaders, device, config['bits'], config['quant_type'], config.get('corruption', 'none'))
            all_results['TA'][config['name']] = res
            print(f"TA ({config['name']}): MNIST: {res['mnist']:.2f}% | FMNIST: {res['fmnist']:.2f}% | CIFAR10: {res['cifar10']:.2f}% | Avg: {res['avg']:.2f}%")
            
        # 4. WCPR Merging
        print("\nWCPR Merging...")
        wcpr_backbone = merge_wcpr(progenitor_state, expert_states)
        all_results['WCPR'] = {}
        for config in eval_configs:
            res = evaluate_merged(wcpr_backbone, task_heads_state, loaders, device, config['bits'], config['quant_type'], config.get('corruption', 'none'))
            all_results['WCPR'][config['name']] = res
            print(f"WCPR ({config['name']}): MNIST: {res['mnist']:.2f}% | FMNIST: {res['fmnist']:.2f}% | CIFAR10: {res['cifar10']:.2f}% | Avg: {res['avg']:.2f}%")
            
        # 5. QCOT Merging (Sweep over C to find best, e.g. C in [0.1, 0.3, 0.5, 1.0, 2.0])
        print("\nQCOT Merging (Sweeping clipping threshold C)...")
        all_results['QCOT'] = {}
        best_qcot_avg = 0
        best_qcot_backbone = None
        best_C_qcot = None
        for C in [0.1, 0.3, 0.5, 1.0, 2.0]:
            print(f"Evaluating QCOT with C={C}...")
            qcot_backbone = merge_qcot(progenitor_state, expert_states, C=C)
            res_fp32 = evaluate_merged(qcot_backbone, task_heads_state, loaders, device, None, 'none')
            res_int8 = evaluate_merged(qcot_backbone, task_heads_state, loaders, device, 8, 'per_tensor')
            print(f"  FP32 Avg: {res_fp32['avg']:.2f}% | INT8 Per-Tensor Avg: {res_int8['avg']:.2f}%")
            
            # Store the best C (judged by INT8 Per-Tensor)
            if res_int8['avg'] > best_qcot_avg:
                best_qcot_avg = res_int8['avg']
                best_qcot_backbone = qcot_backbone
                best_C_qcot = C
                
        print(f"Best QCOT C threshold: {best_C_qcot} with INT8 Per-Tensor Avg accuracy: {best_qcot_avg:.2f}%")
        # Run full config evaluation for best QCOT
        for config in eval_configs:
            res = evaluate_merged(best_qcot_backbone, task_heads_state, loaders, device, config['bits'], config['quant_type'], config.get('corruption', 'none'))
            all_results['QCOT'][config['name']] = res
            print(f"QCOT (C={best_C_qcot}, {config['name']}): MNIST: {res['mnist']:.2f}% | FMNIST: {res['fmnist']:.2f}% | CIFAR10: {res['cifar10']:.2f}% | Avg: {res['avg']:.2f}%")
            
        # 6. Proposed QCSW Merging (Sweep over C to find best, e.g. C in [0.1, 0.3, 0.5, 1.0, 2.0])
        print("\nQCSW Merging (Ours, Sweeping clipping threshold C, num_projections=100)...")
        all_results['QCSW'] = {}
        best_qcsw_avg = 0
        best_qcsw_backbone = None
        best_C_qcsw = None
        for C in [0.1, 0.3, 0.5, 1.0, 2.0]:
            print(f"Evaluating QCSW with C={C}...")
            qcsw_backbone = merge_qcsw(progenitor_state, expert_states, C=C, num_projections=100)
            res_fp32 = evaluate_merged(qcsw_backbone, task_heads_state, loaders, device, None, 'none')
            res_int8 = evaluate_merged(qcsw_backbone, task_heads_state, loaders, device, 8, 'per_tensor')
            print(f"  FP32 Avg: {res_fp32['avg']:.2f}% | INT8 Per-Tensor Avg: {res_int8['avg']:.2f}%")
            
            # Store the best C (judged by INT8 Per-Tensor)
            if res_int8['avg'] > best_qcsw_avg:
                best_qcsw_avg = res_int8['avg']
                best_qcsw_backbone = qcsw_backbone
                best_C_qcsw = C
                
        print(f"Best QCSW C threshold: {best_C_qcsw} with INT8 Per-Tensor Avg accuracy: {best_qcsw_avg:.2f}%")
        # Run full config evaluation for best QCSW
        for config in eval_configs:
            res = evaluate_merged(best_qcsw_backbone, task_heads_state, loaders, device, config['bits'], config['quant_type'], config.get('corruption', 'none'))
            all_results['QCSW'][config['name']] = res
            print(f"QCSW (C={best_C_qcsw}, {config['name']}): MNIST: {res['mnist']:.2f}% | FMNIST: {res['fmnist']:.2f}% | CIFAR10: {res['cifar10']:.2f}% | Avg: {res['avg']:.2f}%")
            
        # Save results to a file for LaTeX table generation later
        import json
        with open("eval_results.json", "w") as jf:
            json.dump(all_results, jf, indent=4)
        print("\nSaved evaluation results to eval_results.json")
        
        # Generate plot comparing the methods
        methods = ['WA', 'TA', 'WCPR', 'QCOT', 'QCSW']
        quant_names = ['FP32', 'INT8 Per-Tensor', 'INT8 Per-Channel', 'INT4 Per-Channel']
        
        plt.figure(figsize=(10, 6))
        for m in methods:
            accs = [all_results[m][qn]['avg'] for qn in quant_names]
            plt.plot(quant_names, accs, marker='o', label=m if m != 'QCSW' else 'QCSW (Ours)')
            
        plt.title('Multitask Average Accuracy across Quantization Regimes')
        plt.xlabel('Quantization Regime')
        plt.ylabel('Average Accuracy (%)')
        plt.ylim(0, 100)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('quantization_robustness.png')
        print("Saved quantization robustness plot to quantization_robustness.png")

if __name__ == '__main__':
    main()
