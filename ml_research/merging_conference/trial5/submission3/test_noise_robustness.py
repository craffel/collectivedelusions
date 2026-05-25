import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import copy
import os

from models import ResNetEncoder, ClassificationHead
from merge_eval import TestTimeModelMerger, get_test_streams
from final_evaluation import FinalEvaluator, get_noisy_stream

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Noise Robustness Sensitivity Study on device: {device}")
    
    torch.backends.cudnn.enabled = False
    alt_stream, _ = get_test_streams(batch_size=64)
    
    evaluator = FinalEvaluator(device)
    
    # Compute and normalize Fisher Information for LFWA
    layer_fisher = evaluator.compute_fisher_information(num_samples=256)
    fisher_vals = np.array(list(layer_fisher.values()))
    mean_fisher = fisher_vals.mean()
    for name in evaluator.parameter_names:
        evaluator.layer_fisher[name] /= mean_fisher
        
    configs = {
        'Static (Task Arithmetic)': {
            'method': 'Static',
            'lr': 0.0
        },
        'Uniform TTA (SGD)': {
            'method': 'Uniform',
            'lr': 0.10
        },
        'LFWA TTA (SGD)': {
            'method': 'LFWA',
            'lr': 0.01
        },
        'AdaSNR TTA (SGD-Standard)': {
            'method': 'AdaSNR_Standard',
            'lr': 1.00
        },
        'AdaSNR-Adam-TC (Ours)': {
            'method': 'AdaSNR_Adam_TC',
            'lr': 0.02
        },
        'AdaSNR-Adam-TC-CG (Ours)': {
            'method': 'AdaSNR_Adam_TC_CG',
            'lr': 0.02
        }
    }
    
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {name: [] for name in configs.keys()}
    
    print("\nSweeping noise levels...")
    for sigma in noise_levels:
        print(f"Evaluating noise std={sigma:.2f}...")
        noisy_stream = get_noisy_stream(alt_stream, noise_std=sigma)
        
        for name, cfg in configs.items():
            avg_acc, _, _ = evaluator.run_eval(noisy_stream, cfg['method'], cfg['lr'])
            results[name].append(avg_acc)
            
    print("\n" + "="*110)
    print(f"{'Method':<30} | " + " | ".join([f"std={sigma:.2f}" for sigma in noise_levels]))
    print("="*110)
    for name in configs.keys():
        row_str = f"{name:<30} | " + " | ".join([f"{acc:.2f}%" for acc in results[name]])
        print(row_str)
    print("="*110)
    
    with open("results_noise_robustness.txt", "w") as f:
        f.write("Noise Robustness Sensitivity Study Results:\n\n")
        f.write("="*110 + "\n")
        f.write(f"{'Method':<30} | " + " | ".join([f"std={sigma:.2f}" for sigma in noise_levels]) + "\n")
        f.write("="*110 + "\n")
        for name in configs.keys():
            row_str = f"{name:<30} | " + " | ".join([f"{acc:.2f}%" for acc in results[name]]) + "\n"
            f.write(row_str)
        f.write("="*110 + "\n")
        
    print("\nResults saved to results_noise_robustness.txt!")

if __name__ == "__main__":
    main()
