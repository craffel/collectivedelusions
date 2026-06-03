import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy

def get_datasets(data_dir='./data', batch_size=128):
    """
    Downloads and returns DataLoaders for MNIST, FashionMNIST, and CIFAR-10.
    All datasets are normalized, resized to 32x32, and formatted to 3 channels.
    """
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    mnist_train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_gray)
    mnist_test = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_gray)
    
    fashion_train = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform_gray)
    fashion_test = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform_gray)
    
    cifar_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_color)
    cifar_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_color)
    
    loaders = {
        'mnist': {
            'train': DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            'test': DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        },
        'fashion': {
            'train': DataLoader(fashion_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            'test': DataLoader(fashion_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        },
        'cifar10': {
            'train': DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            'test': DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        }
    }
    return loaders

def get_resnet18_progenitor(pretrained=True):
    """
    Instantiates a ResNet-18 model from torchvision.
    """
    if pretrained:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
    else:
        model = torchvision.models.resnet18()
    return model

def estimate_fisher_bn(model, dataloader, device, num_batches=15, use_synthetic=False):
    """
    Estimates the empirical diagonal Fisher Information for each BatchNorm layer's
    weight (gamma) and bias (beta) parameters.
    Can use either real dataloader batches or synthetic Gaussian noise.
    """
    model.eval()
    
    # Identify BatchNorm layers with affine parameters
    bn_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) or module.__class__.__name__ == 'TestTimeBatchNorm2d':
            if module.affine:
                module.weight.requires_grad = True
                module.bias.requires_grad = True
                bn_modules[name] = module
                
    if not bn_modules:
        return {}
        
    fisher_dict = {
        name: {
            'weight': torch.zeros_like(mod.weight.data),
            'bias': torch.zeros_like(mod.bias.data)
        } for name, mod in bn_modules.items()
    }
    
    criterion = nn.CrossEntropyLoss()
    count = 0
    
    if use_synthetic:
        # Generate synthetic noise and use mock forward-backward
        for _ in range(num_batches):
            inputs = torch.randn(128, 3, 32, 32, device=device)
            # Use mock targets or model's own predictions
            with torch.no_grad():
                outputs_noop = model(inputs)
                targets = outputs_noop.argmax(dim=-1)
                
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, mod in bn_modules.items():
                if mod.weight.grad is not None:
                    fisher_dict[name]['weight'] += mod.weight.grad.data.clone() ** 2
                if mod.bias.grad is not None:
                    fisher_dict[name]['bias'] += mod.bias.grad.data.clone() ** 2
            count += 1
    else:
        # Use real data
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, mod in bn_modules.items():
                if mod.weight.grad is not None:
                    fisher_dict[name]['weight'] += mod.weight.grad.data.clone() ** 2
                if mod.bias.grad is not None:
                    fisher_dict[name]['bias'] += mod.bias.grad.data.clone() ** 2
            count += 1
            
    # Normalize
    if count > 0:
        for name in fisher_dict:
            fisher_dict[name]['weight'] /= count
            fisher_dict[name]['bias'] /= count
            
    # Reset requires_grad to False
    for mod in bn_modules.values():
        mod.weight.requires_grad = False
        mod.bias.requires_grad = False
        
    return fisher_dict

def estimate_activation_variance_bn(model, dataloader, device, num_batches=15):
    """
    Estimates the activation variance for each channel of BatchNorm layers.
    """
    model.eval()
    var_dict = {}
    
    # We can use forward hooks to capture the variance of activations before each BN
    hooks = []
    bn_activations = {}
    
    def get_hook(name):
        def hook_fn(module, input, output):
            # input[0] has shape [B, C, H, W]
            x = input[0]
            # Compute channel-wise variance
            v = x.var(dim=(0, 2, 3), unbiased=False)
            if name not in bn_activations:
                bn_activations[name] = []
            bn_activations[name].append(v.detach().cpu())
        return hook_fn
        
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) or module.__class__.__name__ == 'TestTimeBatchNorm2d':
            hooks.append(module.register_forward_hook(get_hook(name)))
            
    # Run some batches
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
            
    # Remove hooks
    for hook in hooks:
        hook.remove()
        
    # Aggregate
    for name, val_list in bn_activations.items():
        if val_list:
            stacked = torch.stack(val_list, dim=0) # [num_batches, C]
            var_dict[name] = stacked.mean(dim=0).to(device)
            
    return var_dict

def estimate_grad_norm_bn(model, dataloader, device, num_batches=15):
    """
    Estimates the standard L2 gradient norm of BatchNorm parameters.
    """
    model.eval()
    bn_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) or module.__class__.__name__ == 'TestTimeBatchNorm2d':
            if module.affine:
                module.weight.requires_grad = True
                module.bias.requires_grad = True
                bn_modules[name] = module
                
    grad_norm_dict = {
        name: torch.zeros_like(mod.weight.data) for name, mod in bn_modules.items()
    }
    
    criterion = nn.CrossEntropyLoss()
    count = 0
    
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for name, mod in bn_modules.items():
            if mod.weight.grad is not None:
                # Store the channel-wise gradient norm (using magnitude of gradients)
                grad_norm_dict[name] += torch.abs(mod.weight.grad.data.clone())
        count += 1
        
    if count > 0:
        for name in grad_norm_dict:
            grad_norm_dict[name] /= count
            
    for mod in bn_modules.values():
        mod.weight.requires_grad = False
        mod.bias.requires_grad = False
        
    return grad_norm_dict

def evaluate_model(model, dataloader, device):
    """
    Evaluates accuracy of a model on a given dataloader.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total
