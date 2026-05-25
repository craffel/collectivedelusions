import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
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
    print(f"Running Momentum & Scaling Sensitivity Sweep on device: {device}")
    
    torch.backends.cudnn.enabled = False
    alt_stream, seq_stream = get_test_streams(batch_size=64)
    noisy_stream = get_noisy_stream(alt_stream, noise_std=0.15)
    
    evaluator = FinalEvaluator(device)
    
    # Compute and normalize Fisher Information for LFWA (required to initialize evaluator)
    layer_fisher = evaluator.compute_fisher_information(num_samples=256)
    fisher_vals = np.array(list(layer_fisher.values()))
    mean_fisher = fisher_vals.mean()
    for name in evaluator.parameter_names:
        evaluator.layer_fisher[name] /= mean_fisher
        
    beta1_list = [0.0, 0.5, 0.9]
    beta2_list = [0.9, 0.99, 0.999]
    
    print("\nSweep over beta1 and beta2 for AdaSNR-Adam-TC-CG (Ours)...")
    print(f"{'beta1':<6} | {'beta2':<6} | {'Alternating Stream':<26} | {'Block-Sequential Stream':<26} | {'Noisy OOD Stream':<26}")
    print(f"{'':<6} | {'':<6} | {'Avg Acc':<8} | {'CIFAR/SVHN':<15} | {'Avg Acc':<8} | {'CIFAR/SVHN':<15} | {'Avg Acc':<8} | {'CIFAR/SVHN':<15}")
    print("-" * 116)
    
    results = []
    for b1 in beta1_list:
        for b2 in beta2_list:
            avg_alt, c_alt, s_alt = evaluator.run_eval(alt_stream, 'AdaSNR_Adam_TC_CG', lr_base=0.02, beta1=b1, beta2=b2)
            avg_seq, c_seq, s_seq = evaluator.run_eval(seq_stream, 'AdaSNR_Adam_TC_CG', lr_base=0.02, beta1=b1, beta2=b2)
            avg_noise, c_noise, s_noise = evaluator.run_eval(noisy_stream, 'AdaSNR_Adam_TC_CG', lr_base=0.02, beta1=b1, beta2=b2)
            
            print(f"{b1:<6.2f} | {b2:<6.3f} | {avg_alt:.2f}%   | {c_alt:.1f}%/{s_alt:.1f}% | {avg_seq:.2f}%   | {c_seq:.1f}%/{s_seq:.1f}% | {avg_noise:.2f}%   | {c_noise:.1f}%/{s_noise:.1f}%")
            results.append((b1, b2, avg_alt, avg_seq, avg_noise))
            
    print("-" * 116)
    
    # Let's save the results to a text file for our paper's LaTeX integration
    with open("results_sensitivity.txt", "w") as f:
        f.write("Momentum and Scaling Sensitivity Sweep (beta1 vs beta2):\n\n")
        f.write("-" * 116 + "\n")
        f.write(f"{'beta1':<6} | {'beta2':<6} | {'Alternating Stream':<26} | {'Block-Sequential Stream':<26} | {'Noisy OOD Stream':<26}\n")
        f.write(f"{'':<6} | {'':<6} | {'Avg Acc':<8} | {'CIFAR/SVHN':<15} | {'Avg Acc':<8} | {'CIFAR/SVHN':<15} | {'Avg Acc':<8} | {'CIFAR/SVHN':<15}\n")
        f.write("-" * 116 + "\n")
        for b1, b2, avg_alt, avg_seq, avg_noise in results:
            f.write(f"{b1:<6.2f} | {b2:<6.3f} | {avg_alt:.2f}%   |                 | {avg_seq:.2f}%   |                 | {avg_noise:.2f}%\n")
        f.write("-" * 116 + "\n")
        
    print("Sensitivity sweep finished and saved!")

if __name__ == '__main__':
    main()
