import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
from utils import (
    get_dataloaders,
    calibrate_bn,
    collect_activation_scales,
    apply_structured_pruning_mask,
    get_target_layers_mapping
)

def profile_calibration_overhead():
    print("="*70)
    print("PROFILING ON-DEVICE CALIBRATION & ADAPTATION OVERHEAD")
    print("="*70)
    
    device = torch.device('cpu')
    print(f"Running profiling on: {device}")
    
    # 1. Initialize ResNet-18 progenitor model
    model = models.resnet18()
    model.fc = nn.Linear(512, 10)
    model.eval().to(device)
    
    # 2. Sizes of N to sweep
    N_sizes = [16, 32, 64, 128]
    metrics = ['l1', 'l2', 'variance']
    target_layers = list(get_target_layers_mapping().keys())
    
    # We will measure the time for each component over 10 trials to get stable averages
    num_trials = 10
    
    results = {}
    
    for N in N_sizes:
        # Load dataloaders for specific calibration size N
        loaders = get_dataloaders(cal_size=N)
        cal_loader = loaders['mnist']['cal']  # Use MNIST as representative task
        
        results[N] = {
            'bn_cal_time_ms': 0.0,
            'scale_collect_time_ms': {m: 0.0 for m in metrics},
            'mask_apply_time_ms': 0.0
        }
        
        # A. Measure BN Calibration Time
        bn_times = []
        for _ in range(num_trials):
            # Reset running stats before calibrating
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.reset_running_stats()
            
            start_time = time.time()
            calibrate_bn(model, cal_loader, device)
            bn_times.append((time.time() - start_time) * 1000.0) # in ms
        results[N]['bn_cal_time_ms'] = sum(bn_times) / num_trials
        
        # B. Measure Activation Scale Collection Time per metric
        for metric in metrics:
            metric_times = []
            for _ in range(num_trials):
                start_time = time.time()
                _ = collect_activation_scales(model, cal_loader, device, target_layers, metric=metric)
                metric_times.append((time.time() - start_time) * 1000.0) # in ms
            results[N]['scale_collect_time_ms'][metric] = sum(metric_times) / num_trials
            
        # C. Measure Mask Computation & Application Time
        # Let's generate a dummy mask and measure the time to apply it across all target layers
        mask_times = []
        for _ in range(num_trials):
            start_time = time.time()
            for layer_name in target_layers:
                # Get the actual output channels
                module = dict(model.named_modules())[layer_name]
                out_c = module.weight.shape[0]
                # Dummy binary mask (prune 30%)
                mask = torch.ones(out_c, device=device)
                prune_count = int(out_c * 0.3)
                mask[:prune_count] = 0.0
                apply_structured_pruning_mask(model, layer_name, mask)
            mask_times.append((time.time() - start_time) * 1000.0)
        results[N]['mask_apply_time_ms'] = sum(mask_times) / num_trials

    # 3. Print the results in a clean table
    print("\n" + f"{'Cal Size N':<12} | {'BN Cal (ms)':<15} | {'Scale Coll (ms, Var)':<23} | {'Mask Apply (ms)':<18} | {'Total Adaptation (ms)':<22}")
    print("-"*98)
    for N in N_sizes:
        bn_time = results[N]['bn_cal_time_ms']
        scale_time = results[N]['scale_collect_time_ms']['variance']
        mask_time = results[N]['mask_apply_time_ms']
        total_time = bn_time + scale_time + mask_time
        print(f"{N:>10}   | {bn_time:>13.2f}   | {scale_time:>21.2f}   | {mask_time:>16.2f}   | {total_time:>20.2f}")
        
    print("\n" + f"{'Cal Size N':<12} | {'Scale Coll L1 (ms)':<20} | {'Scale Coll L2 (ms)':<20} | {'Scale Coll Var (ms)':<20}")
    print("-"*80)
    for N in N_sizes:
        t_l1 = results[N]['scale_collect_time_ms']['l1']
        t_l2 = results[N]['scale_collect_time_ms']['l2']
        t_var = results[N]['scale_collect_time_ms']['variance']
        print(f"{N:>10}   | {t_l1:>18.2f}   | {t_l2:>18.2f}   | {t_var:>18.2f}")
        
    print("="*70)
    
if __name__ == '__main__':
    profile_calibration_overhead()
