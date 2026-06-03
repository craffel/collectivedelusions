import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from datasets_utils import get_dataloaders
from calibration_methods import merge_models, calibrate_sequential, get_bn_modules

def collect_layer_variances(model, loaders, device, cal_size=128):
    model.eval()
    bn_modules = get_bn_modules(model)
    bn_names = [name for name, _ in bn_modules]
    
    # Register forward hooks to capture the outputs of each BatchNorm layer
    captured_outputs = {name: [] for name in bn_names}
    
    def make_hook(name):
        def hook_fn(module, input, output):
            captured_outputs[name].append(output.detach().cpu())
        return hook_fn
        
    handles = []
    for name, module in bn_modules:
        handles.append(module.register_forward_hook(make_hook(name)))
        
    # Run forward pass on joint calibration dataset
    tasks = ['mnist', 'fashion', 'cifar']
    with torch.no_grad():
        for task in tasks:
            cal_loader = loaders[task]['cal']
            samples_collected = 0
            for images, _ in cal_loader:
                if cal_size is not None and samples_collected >= cal_size:
                    break
                needed = cal_size - samples_collected if cal_size is not None else images.size(0)
                if images.size(0) > needed:
                    images = images[:needed]
                images = images.to(device)
                _ = model(images)
                samples_collected += images.size(0)
                
    for h in handles:
        h.remove()
        
    # Compute average variance per layer
    layer_vars = []
    for name in bn_names:
        outputs_tensor = torch.cat(captured_outputs[name], dim=0) # (N_joint, C, H, W)
        # Compute variance across batch, height, and width (dim=(0, 2, 3))
        channel_vars = outputs_tensor.var(dim=(0, 2, 3), unbiased=False)
        # Average across all channels
        mean_var = channel_vars.mean().item()
        layer_vars.append(mean_var)
        
    return layer_vars

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    loaders = get_dataloaders(batch_size=128)
    
    # Load Experts
    mnist_expert = models.resnet18(weights=None)
    mnist_expert.fc = nn.Linear(512, 10)
    mnist_expert.load_state_dict(torch.load('mnist_expert.pt', map_location='cpu'))
    mnist_expert = mnist_expert.to(device)
    
    fashion_expert = models.resnet18(weights=None)
    fashion_expert.fc = nn.Linear(512, 10)
    fashion_expert.load_state_dict(torch.load('fashion_expert.pt', map_location='cpu'))
    fashion_expert = fashion_expert.to(device)
    
    cifar_expert = models.resnet18(weights=None)
    cifar_expert.fc = nn.Linear(512, 10)
    cifar_expert.load_state_dict(torch.load('cifar_expert.pt', map_location='cpu'))
    cifar_expert = cifar_expert.to(device)
    
    experts_list = [mnist_expert, fashion_expert, cifar_expert]
    
    # 1. Oracle variances
    print("Collecting Oracle variances...")
    oracle_vars_mnist = collect_layer_variances(mnist_expert, loaders, device)
    oracle_vars_fashion = collect_layer_variances(fashion_expert, loaders, device)
    oracle_vars_cifar = collect_layer_variances(cifar_expert, loaders, device)
    oracle_vars = [(v1 + v2 + v3)/3.0 for v1, v2, v3 in zip(oracle_vars_mnist, oracle_vars_fashion, oracle_vars_cifar)]
    
    # 2. Uncalibrated WA variances
    print("Collecting Uncalibrated WA variances...")
    model_uncal = merge_models(merge_mode='wa')
    model_uncal = model_uncal.to(device)
    uncal_vars = collect_layer_variances(model_uncal, loaders, device)
    
    # 3. Parallel Calibrated WA (PBR) variances
    print("Collecting PBR variances...")
    model_pbr = merge_models(merge_mode='wa')
    model_pbr = model_pbr.to(device)
    model_pbr, _ = calibrate_sequential(model_pbr, experts_list, loaders, 'pbr', device)
    pbr_vars = collect_layer_variances(model_pbr, loaders, device)
    
    # 4. Sequential Calibrated WA (SBR) variances
    print("Collecting SBR variances...")
    model_sbr = merge_models(merge_mode='wa')
    model_sbr = model_sbr.to(device)
    model_sbr, _ = calibrate_sequential(model_sbr, experts_list, loaders, 'sbr', device)
    sbr_vars = collect_layer_variances(model_sbr, loaders, device)
    
    # Plotting
    plt.figure(figsize=(8, 5))
    x = np.arange(1, len(oracle_vars) + 1)
    
    plt.plot(x, oracle_vars, label='Oracle (Average Specialists)', color='black', linestyle='--', marker='o')
    plt.plot(x, uncal_vars, label='WA (Uncalibrated)', color='red', marker='x')
    plt.plot(x, pbr_vars, label='WA + PBR (Parallel)', color='orange', marker='s')
    plt.plot(x, sbr_vars, label='WA + SBR (Ours)', color='green', marker='^')
    
    plt.xlabel('BatchNorm Layer Index (Forward Order)', fontsize=12)
    plt.ylabel('Average Representation Variance', fontsize=12)
    plt.title('Empirical Validation of Post-Merge Representation Collapse', fontsize=13, fontweight='bold')
    plt.xticks(x)
    plt.yscale('log') # Log scale to clearly show exponential decay
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plt.savefig('variance_collapse.pdf')
    plt.savefig('variance_collapse.png', dpi=300)
    print("Variance collapse plots saved to variance_collapse.pdf and variance_collapse.png")

if __name__ == '__main__':
    main()
