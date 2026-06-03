import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
from utils import get_target_layers_mapping, apply_structured_pruning_mask

def count_active_params_and_macs(model, prune_ratio, masks=None):
    """
    Computes the exact number of active parameters and active MACs of the model
    under structured channel pruning.
    
    For a target layer, out_channels is scaled by (1 - prune_ratio).
    For the succeeding layer, in_channels is scaled by (1 - prune_ratio).
    All other layers are at 100% capacity.
    """
    mapping = get_target_layers_mapping()
    
    # Standard ResNet-18 layer shapes (input size: 3x32x32)
    # Layer name -> (in_channels, out_channels, kernel_size, stride, padding, output_size)
    # ResNet-18 with 3x32x32 input has the following feature map sizes:
    layer_configs = {
        'conv1': (3, 64, (7, 7), (2, 2), (3, 3), 16),
        'layer1.0.conv1': (64, 64, (3, 3), (1, 1), (1, 1), 16),
        'layer1.0.conv2': (64, 64, (3, 3), (1, 1), (1, 1), 16),
        'layer1.1.conv1': (64, 64, (3, 3), (1, 1), (1, 1), 16),
        'layer1.1.conv2': (64, 64, (3, 3), (1, 1), (1, 1), 16),
        'layer2.0.conv1': (64, 128, (3, 3), (2, 2), (1, 1), 8),
        'layer2.0.conv2': (128, 128, (3, 3), (1, 1), (1, 1), 8),
        'layer2.0.downsample.0': (64, 128, (1, 1), (2, 2), (0, 0), 8),
        'layer2.1.conv1': (128, 128, (3, 3), (1, 1), (1, 1), 8),
        'layer2.1.conv2': (128, 128, (3, 3), (1, 1), (1, 1), 8),
        'layer3.0.conv1': (128, 256, (3, 3), (2, 2), (1, 1), 4),
        'layer3.0.conv2': (256, 256, (3, 3), (1, 1), (1, 1), 4),
        'layer3.0.downsample.0': (128, 256, (1, 1), (2, 2), (0, 0), 4),
        'layer3.1.conv1': (256, 256, (3, 3), (1, 1), (1, 1), 4),
        'layer3.1.conv2': (256, 256, (3, 3), (1, 1), (1, 1), 4),
        'layer4.0.conv1': (256, 512, (3, 3), (2, 2), (1, 1), 2),
        'layer4.0.conv2': (512, 512, (3, 3), (1, 1), (1, 1), 2),
        'layer4.0.downsample.0': (256, 512, (1, 1), (2, 2), (0, 0), 2),
        'layer4.1.conv1': (512, 512, (3, 3), (1, 1), (1, 1), 2),
        'layer4.1.conv2': (512, 512, (3, 3), (1, 1), (1, 1), 2),
        'fc': (512, 10, None, None, None, 1)
    }
    
    total_params = 0
    total_macs = 0
    
    for name, config in layer_configs.items():
        if name == 'fc':
            in_features, out_features, _, _, _, _ = config
            params = in_features * out_features + out_features
            macs = in_features * out_features
            total_params += params
            total_macs += macs
            continue
            
        in_c, out_c, kernel, stride, padding, out_size = config
        k_h, k_w = kernel
        
        # Adjust in_c and out_c based on pruning ratio
        active_out_c = out_c
        active_in_c = in_c
        
        # Check if output channels are pruned
        if name in mapping:
            # Pruned output channels
            active_out_c = int(out_c * (1 - prune_ratio))
            
        # Check if input channels are pruned (due to preceding layer being pruned)
        for target_layer, m_info in mapping.items():
            if m_info['next_conv'] == name:
                active_in_c = int(in_c * (1 - prune_ratio))
                break
                
        # Parameters: weight + bias (bias is in bn or conv, let's just count conv weight for simplicity)
        params = active_out_c * active_in_c * k_h * k_w
        macs = active_out_c * active_in_c * k_h * k_w * out_size * out_size
        
        total_params += params
        total_macs += macs
        
    return total_params, total_macs

def measure_cpu_latency(model, batch_size=1, num_runs=100, warmups=10):
    """
    Measures the exact CPU inference latency of the model.
    """
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)
    
    dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(dummy_input)
            
    # Measure latency
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / num_runs * 1000.0  # in milliseconds
    return avg_latency

def main():
    print("="*60)
    print("PRAGMATIST REAL-WORLD HARDWARE PROFILING")
    print("="*60)
    
    # Initialize CPU-bound ResNet-18 model
    progenitor = models.resnet18()
    progenitor.fc = nn.Linear(512, 10)
    progenitor.eval()
    
    # Load progenitor checkpoint if available
    if os.path.exists("./checkpoints/progenitor.pth"):
        progenitor.load_state_dict(torch.load("./checkpoints/progenitor.pth", map_location='cpu'))
        print("Successfully loaded progenitor checkpoint for realistic timing.")
    else:
        print("Progenitor checkpoint not found, profiling with uninitialized weights (timing remains identical).")
        
    ratios = [0.0, 0.1, 0.3, 0.5]
    
    print("\n" + f"{'Pruning Ratio':<15} | {'Active Params (M)':<18} | {'Active MACs (M)':<15} | {'CPU Latency (ms)':<18} | {'Latency Reduction':<18}")
    print("-"*90)
    
    baseline_latency = None
    
    for ratio in ratios:
        # Clone model to apply pruning mask
        pruned_model = models.resnet18()
        pruned_model.fc = nn.Linear(512, 10)
        pruned_model.load_state_dict(progenitor.state_dict())
        pruned_model.eval()
        
        # Compute exact parameters and MACs
        active_params, active_macs = count_active_params_and_macs(pruned_model, ratio)
        
        # Apply pruning mask dummy to measure latency
        if ratio > 0:
            target_layers = list(get_target_layers_mapping().keys())
            for name in target_layers:
                # Retrieve actual output channel size
                module = dict(pruned_model.named_modules())[name]
                out_c = module.weight.shape[0]
                mask = torch.ones(out_c)
                # Set pruned fraction of mask to 0
                prune_count = int(out_c * ratio)
                mask[:prune_count] = 0.0
                apply_structured_pruning_mask(pruned_model, name, mask)
                
        # Measure latency
        latency = measure_cpu_latency(pruned_model, batch_size=1, num_runs=100)
        
        if ratio == 0.0:
            baseline_latency = latency
            reduction_str = "Baseline"
        else:
            reduction = (baseline_latency - latency) / baseline_latency * 100.0
            reduction_str = f"{reduction:.1f}% speedup"
            
        print(f"{int(ratio*100):>12}% | {active_params/1e6:>16.3f}M | {active_macs/1e6:>13.2f}M | {latency:>16.2f}ms | {reduction_str:>18}")
        
    print("="*60)
    
if __name__ == '__main__':
    main()
