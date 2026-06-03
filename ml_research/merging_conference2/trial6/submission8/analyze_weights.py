import torch
import torch.nn as nn
import numpy as np
from run_experiments import (
    get_dataloaders,
    extract_hybrid,
    train_micro_linear_router
)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load data loaders with seed 42
    loaders = get_dataloaders(seed=42)
    
    # 2. Extract standard calibration sets (N=128)
    N_eval = 128
    calib_images = {}
    for name, task_loader in loaders.items():
        inputs_list = []
        count = 0
        for inputs, targets in task_loader['train_expert']:
            inputs_list.append(inputs)
            count += inputs.size(0)
            if count >= N_eval:
                break
        task_inputs = torch.cat(inputs_list)[:N_eval]
        calib_images[name] = task_inputs

    # 3. Define the hybrid feature function and dimension
    hlbr_feat_fn = lambda x: extract_hybrid(x, bins=16, size=6)
    hlbr_dim = 52
    
    # 4. Train the micro-linear router classifier
    print("Training micro-linear router classifier...")
    classifier = train_micro_linear_router(calib_images, hlbr_feat_fn, hlbr_dim, device)
    
    # 5. Extract weights and biases
    # The weight shape is (3, 52), bias shape is (3)
    weights = classifier.weight.detach().cpu().numpy()
    biases = classifier.bias.detach().cpu().numpy()
    
    tasks = ['mnist', 'fmnist', 'cifar']
    
    print("\n================ HLBR Learned Weights Analysis ================")
    for idx, task in enumerate(tasks):
        task_w = weights[idx]
        task_b = biases[idx]
        
        # Split into spectral (first 16 bins) and spatial (remaining 36 coordinates)
        spectral_w = task_w[:16]
        spatial_w = task_w[16:]
        
        print(f"\nTask: {task.upper()} (Bias: {task_b:.4f})")
        
        # Spectral stats
        print(f"  Spectral Weights (1D Hist):")
        print(f"    Min: {spectral_w.min():.4f}, Max: {spectral_w.max():.4f}, Mean: {spectral_w.mean():.4f}")
        # Top positive/negative bins
        top_pos_spec = np.argsort(spectral_w)[::-1][:3]
        top_neg_spec = np.argsort(spectral_w)[:3]
        print(f"    Top Positive Bins (indices 0-15): {top_pos_spec} (values: {spectral_w[top_pos_spec]})")
        print(f"    Top Negative Bins (indices 0-15): {top_neg_spec} (values: {spectral_w[top_neg_spec]})")
        
        # Spatial stats
        print(f"  Spatial Weights (6x6 Grid):")
        print(f"    Min: {spatial_w.min():.4f}, Max: {spatial_w.max():.4f}, Mean: {spatial_w.mean():.4f}")
        # Top positive/negative regions
        top_pos_spat = np.argsort(spatial_w)[::-1][:3]
        top_neg_spat = np.argsort(spatial_w)[:3]
        print(f"    Top Positive Grid Positions (indices 0-35): {top_pos_spat} (values: {spatial_w[top_pos_spat]})")
        print(f"    Top Negative Grid Positions (indices 0-35): {top_neg_spat} (values: {spatial_w[top_neg_spat]})")
        
if __name__ == '__main__':
    main()
