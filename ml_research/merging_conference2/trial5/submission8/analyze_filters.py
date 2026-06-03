import os
import torch
import numpy as np
from run_experiments import (
    MultiTaskModel,
    get_dataloaders,
    merge_models,
    collect_expert_statistics,
    calibrate_tc_merged_model,
    get_bn_layers
)

def analyze_frequency_bands(gamma):
    """
    Analyzes the average scaling factors in low vs high frequency bands.
    Assumes gamma is of shape (H, W).
    """
    H, W = gamma.shape
    # Center the FFT representation
    gamma_shifted = torch.fft.fftshift(gamma)
    
    # Create a grid of distances from the center
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_y, center_x = H // 2, W // 2
    distances = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Set radius adaptively
    min_dim = min(H, W)
    radius = max(1.0, min_dim / 4.0)
    
    # Low frequency mask
    low_mask = distances <= radius
    # High frequency mask
    high_mask = distances > radius
    
    low_mean = gamma_shifted[low_mask].mean().item() if low_mask.any() else 1.0
    high_mean = gamma_shifted[high_mask].mean().item() if high_mask.any() else 1.0
    
    return low_mean, high_mean, radius

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    _, cal_loaders, _ = get_dataloaders()
    
    # Load experts
    tasks = ['mnist', 'fmnist', 'cifar10']
    expert_models = []
    for task in tasks:
        ckpt_path = f"{task}_expert.pth"
        model = MultiTaskModel().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        expert_models.append(model)
        
    # Create Weight Averaged merged model
    merged_model = merge_models(expert_models, method='WA').to(device)
    
    # Collect expert statistics
    expert_profiles = collect_expert_statistics(expert_models, cal_loaders, device)
    
    # Compute TC-ZOSF filters
    print("Computing task-conditional filters...")
    cal_filters_tc = calibrate_tc_merged_model(merged_model, expert_profiles, cal_loaders, device, gamma_max=5.0)
    
    bn_layers = get_bn_layers(merged_model)
    first_bn_name = bn_layers[0][0]
    middle_bn_name = bn_layers[len(bn_layers)//2][0]
    
    print("\n=== Spectral Scaling Map Analysis (Low vs High Frequencies) ===")
    for layer_name in [first_bn_name, middle_bn_name]:
        print(f"\nLayer: {layer_name}")
        for task in tasks:
            gamma = cal_filters_tc[task][layer_name]['gamma']
            low, high, radius = analyze_frequency_bands(gamma)
            print(f"  Task: {task:<8} | Shape: {str(list(gamma.shape)):<8} | Radius: {radius:<4.1f} | Low-Freq Scaling: {low:.4f} | High-Freq Scaling: {high:.4f} | High/Low Ratio: {high/low:.4f}")

if __name__ == '__main__':
    main()
