import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# 1. Dataset Loading Utility
def get_dataloaders(batch_size=256, num_workers=4, cal_size=64):
    """
    Downloads and prepares MNIST, Fashion-MNIST, and CIFAR-10 datasets.
    Resizes MNIST and Fashion-MNIST to 32x32 and replicates to 3 channels.
    """
    # Grayscale datasets transform (MNIST and Fashion-MNIST)
    gray_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # RGB dataset transform (CIFAR-10)
    rgb_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Download datasets
    datasets = {
        'mnist': {
            'train': torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=gray_transform),
            'test': torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=gray_transform)
        },
        'fmnist': {
            'train': torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=gray_transform),
            'test': torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=gray_transform)
        },
        'cifar10': {
            'train': torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=rgb_transform),
            'test': torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=rgb_transform)
        }
    }
    
    loaders = {}
    for task, data in datasets.items():
        # Create standard train and test loaders
        train_loader = DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(data['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        # Create a tiny calibration loader (deterministic subset of train data)
        indices = list(range(cal_size))
        cal_subset = Subset(data['train'], indices)
        cal_loader = DataLoader(cal_subset, batch_size=cal_size, shuffle=False, num_workers=1)
        
        loaders[task] = {
            'train': train_loader,
            'test': test_loader,
            'cal': cal_loader
        }
        
    return loaders

# 2. Fake Quantization Helpers
def quantize_tensor_symmetric(tensor, num_bits, per_channel=False):
    """
    Symmetric uniform quantization for weights.
    Supports per-tensor and per-channel quantization.
    """
    if num_bits is None or num_bits >= 32:
        return tensor
        
    qmin = -(2**(num_bits - 1) - 1)
    qmax = 2**(num_bits - 1) - 1
    
    if per_channel and len(tensor.shape) >= 2:
        # Per-channel along dimension 0 (out_channels)
        shape = tensor.shape
        out_channels = shape[0]
        flat_tensor = tensor.view(out_channels, -1)
        
        # Compute max abs per channel
        max_vals = torch.max(torch.abs(flat_tensor), dim=1, keepdim=True)[0]
        max_vals = torch.clamp(max_vals, min=1e-8)
        
        scale = max_vals / qmax
        quant = torch.clamp(torch.round(flat_tensor / scale), qmin, qmax)
        dequant = quant * scale
        return dequant.view(shape)
    else:
        # Per-tensor quantization
        max_val = torch.max(torch.abs(tensor))
        if max_val == 0:
            return tensor
            
        scale = max_val / qmax
        quant = torch.clamp(torch.round(tensor / scale), qmin, qmax)
        dequant = quant * scale
        return dequant

def apply_weight_quantization(model, num_bits, per_channel=False):
    """
    Applies fake weight quantization to all Conv2d and Linear layers in the model.
    """
    if num_bits is None or num_bits >= 32:
        return
        
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.copy_(quantize_tensor_symmetric(module.weight, num_bits, per_channel=per_channel))

# 3. Data-Efficient BatchNorm Calibration (DE-BN)
def calibrate_bn(model, calibration_loader, device):
    """
    Calibrates BatchNorm running statistics task-specifically using a tiny calibration set.
    Uses momentum=1.0 to set running statistics exactly to the batch statistics.
    """
    model.train()
    original_momentums = {}
    
    # Store original momentums and set to 1.0
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            original_momentums[name] = module.momentum
            module.momentum = 1.0
            module.reset_running_stats()
            
    # Run a single batch of calibration data to update running statistics
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            break  # Just one batch is sufficient for DE-BN
            
    # Restore original momentums and set back to eval
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.momentum = original_momentums[name]
    model.eval()

# 4. Activation-Aware Structured Pruning (ACP-QMM) Helper
def collect_activation_scales(model, cal_loader, device, target_layers, metric='l1'):
    """
    Collects activation scales (mean absolute values) for target convolutional layers.
    Supports metrics: 'l1', 'l2', and 'variance'.
    """
    model.eval()
    activation_scales = {layer_name: None for layer_name in target_layers}
    hooks = []
    
    def get_hook(name):
        def hook(module, input, output):
            # output shape: (batch_size, out_channels, H, W)
            dims = (0, 2, 3)
            if metric == 'l1':
                # Mean absolute value
                scale = torch.mean(torch.abs(output), dim=dims).detach().cpu()
            elif metric == 'l2':
                # Root mean square (RMS)
                scale = torch.sqrt(torch.mean(output ** 2, dim=dims)).detach().cpu()
            elif metric == 'variance':
                # Standard deviation
                mean = torch.mean(output, dim=dims, keepdim=True)
                var = torch.mean((output - mean) ** 2, dim=dims).detach().cpu()
                scale = torch.sqrt(torch.clamp(var, min=1e-8))
            else:
                raise ValueError(f"Unknown metric {metric}")
            activation_scales[name] = scale
        return hook
        
    # Register hooks
    for name, module in model.named_modules():
        if name in target_layers:
            hooks.append(module.register_forward_hook(get_hook(name)))
            
    # Run a single forward pass
    with torch.no_grad():
        for inputs, _ in cal_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            break
            
    # Remove hooks
    for hook in hooks:
        hook.remove()
        
    return activation_scales

def get_target_layers_mapping():
    """
    Returns the target layers for pruning and their immediately following BatchNorm
    and Conv layers, ensuring we do not violate residual connection matching constraints.
    """
    # We prune L1.conv1, which is followed by L1.bn1, and its output feeds into L1.conv2 input.
    # This is 100% safe from residual additions!
    mapping = {
        'conv1': {'bn': 'bn1', 'next_conv': 'layer1.0.conv1'},
        'layer1.0.conv1': {'bn': 'layer1.0.bn1', 'next_conv': 'layer1.0.conv2'},
        'layer1.1.conv1': {'bn': 'layer1.1.bn1', 'next_conv': 'layer1.1.conv2'},
        'layer2.0.conv1': {'bn': 'layer2.0.bn1', 'next_conv': 'layer2.0.conv2'},
        'layer2.1.conv1': {'bn': 'layer2.1.bn1', 'next_conv': 'layer2.1.conv2'},
        'layer3.0.conv1': {'bn': 'layer3.0.bn1', 'next_conv': 'layer3.0.conv2'},
        'layer3.1.conv1': {'bn': 'layer3.1.bn1', 'next_conv': 'layer3.1.conv2'},
        'layer4.0.conv1': {'bn': 'layer4.0.bn1', 'next_conv': 'layer4.0.conv2'},
        'layer4.1.conv1': {'bn': 'layer4.1.bn1', 'next_conv': 'layer4.1.conv2'}
    }
    return mapping

def apply_structured_pruning_mask(model, layer_name, mask):
    """
    Applies a binary channel mask to a target Conv layer and its dependent BN and succeeding Conv layer.
    """
    mapping = get_target_layers_mapping()
    if layer_name not in mapping:
        return
        
    m_info = mapping[layer_name]
    device = mask.device
    
    # Get modules
    conv_module = dict(model.named_modules())[layer_name]
    bn_module = dict(model.named_modules())[m_info['bn']]
    next_conv_module = dict(model.named_modules())[m_info['next_conv']]
    
    with torch.no_grad():
        # 1. Zero out conv output channels
        # Weight shape: (out_channels, in_channels, K, K)
        conv_module.weight.mul_(mask.view(-1, 1, 1, 1))
        if conv_module.bias is not None:
            conv_module.bias.mul_(mask)
            
        # 2. Zero out BN channels
        bn_module.weight.mul_(mask)
        bn_module.bias.mul_(mask)
        bn_module.running_mean.mul_(mask)
        # For variance, we set the variance to 1.0 for pruned channels to avoid division by zero
        bn_module.running_var.copy_(bn_module.running_var * mask + (1.0 - mask))
        
        # 3. Zero out next conv input channels
        # Weight shape: (out_channels, in_channels, K, K)
        next_conv_module.weight.mul_(mask.view(1, -1, 1, 1))

# 5. Evaluation Utility
def evaluate_model(model, test_loader, device):
    """
    Evaluates the model on a test loader and returns top-1 accuracy.
    """
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
