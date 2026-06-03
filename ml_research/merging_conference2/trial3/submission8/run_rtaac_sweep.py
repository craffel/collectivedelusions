import os
import json
import numpy as np
import torch
from dataset import get_dataloaders
from models import MultiTaskResNet18
from eval import (
    get_calibration_loaders,
    assemble_merged_model,
    evaluate_model
)

def main():
    os.makedirs('results', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting R-TAAC Sweep on device: {device.upper()}")
    
    # Load test dataloaders ONCE
    print("Loading test dataloaders...")
    _, test_loaders = get_dataloaders(batch_size=128)
    
    expert_paths = {
        'mnist': 'checkpoints/expert_mnist.pt',
        'fashion': 'checkpoints/expert_fashion.pt',
        'cifar': 'checkpoints/expert_cifar.pt'
    }
    pretrained_path = 'checkpoints/pretrained.pt'
    
    cal_sizes = [4, 8, 16, 32, 64, 128, 256]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = {
        'wa': {},
        'ta': {}
    }
    
    for N in cal_sizes:
        print(f"\nEvaluating N = {N}...")
        cal_loaders, joint_loader = get_calibration_loaders(N=N, seed=42)
        
        results['wa'][str(N)] = {}
        results['ta'][str(N)] = {}
        
        # Assemble models once per N
        model_wa = assemble_merged_model(expert_paths, pretrained_path, 'wa', 0.2).to(device)
        model_ta = assemble_merged_model(expert_paths, pretrained_path, 'ta', 0.2).to(device)
        
        for alpha in alphas:
            print(f"  Evaluating alpha = {alpha}...")
            # WA
            res_wa = evaluate_model(
                model_wa, test_loaders, cal_loaders, joint_loader, expert_paths,
                cal_method='rtaac', rtaac_alpha=alpha, device=device
            )
            results['wa'][str(N)][f"{alpha:.1f}"] = res_wa
            
            # TA
            res_ta = evaluate_model(
                model_ta, test_loaders, cal_loaders, joint_loader, expert_paths,
                cal_method='rtaac', rtaac_alpha=alpha, device=device
            )
            results['ta'][str(N)][f"{alpha:.1f}"] = res_ta
            
            print(f"    WA Avg: {res_wa['avg']:.2f}% | TA Avg: {res_ta['avg']:.2f}%")
            
    # Save the sweep results
    sweep_path = 'results/rtaac_sweep.json'
    with open(sweep_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nR-TAAC Sweep complete! Results saved to {sweep_path}")
    
    # Also print a beautiful summary table
    print("\n" + "="*60)
    print("R-TAAC WEIGHT AVERAGING SWEEP SUMMARY (Avg Acc %):")
    print("="*60)
    alpha_strs = [f"{a:.1f}" for a in alphas]
    header = f"{'N Samples':<10} | " + " | ".join(f"a={a}" for a in alpha_strs)
    print(header)
    print("-" * len(header))
    for N in cal_sizes:
        row = f"N = {N:<6} | " + " | ".join(f"{results['wa'][str(N)][a]['avg']:.2f}" for a in alpha_strs)
        print(row)
    print("="*60)

if __name__ == '__main__':
    main()
