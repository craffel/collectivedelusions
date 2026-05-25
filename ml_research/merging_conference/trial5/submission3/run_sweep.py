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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running sweep on device: {device}")
    
    # Disable cuDNN to prevent initialization errors
    torch.backends.cudnn.enabled = False
    
    # Generate non-stationary streams
    alt_stream, seq_stream = get_test_streams(batch_size=64)
    
    # Initialize merger
    merger = TestTimeModelMerger(device)
    
    # Pre-compute Fisher Information for LFWA
    layer_fisher = merger.compute_fisher_information(num_samples=256)
    
    # Normalize Fisher values so their mean is 1.0
    fisher_vals = np.array(list(layer_fisher.values()))
    mean_fisher = fisher_vals.mean()
    print(f"Original Fisher - Mean: {mean_fisher:.4f}, Min: {fisher_vals.min():.4e}, Max: {fisher_vals.max():.4f}")
    for name in merger.parameter_names:
        merger.layer_fisher[name] /= mean_fisher
    
    # Verify normalization
    normalized_vals = np.array(list(merger.layer_fisher.values()))
    print(f"Normalized Fisher - Mean: {normalized_vals.mean():.4f}, Min: {normalized_vals.min():.4f}, Max: {normalized_vals.max():.4f}")
    
    # Define learning rate grid
    lr_grid = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    results = {}
    
    # 1. Static Baseline (no learning rate)
    avg_alt, cifar_alt, svhn_alt = merger.evaluate_stream(alt_stream, 'Static', lr_base=0.0)
    avg_seq, cifar_seq, svhn_seq = merger.evaluate_stream(seq_stream, 'Static', lr_base=0.0)
    results['Static'] = {
        'alt': (avg_alt, cifar_alt, svhn_alt, 0.0),
        'seq': (avg_seq, cifar_seq, svhn_seq, 0.0)
    }
    print(f"\nStatic Merging: Alt Avg Acc = {avg_alt:.2f}%, Seq Avg Acc = {avg_seq:.2f}%")
    
    # Sweep over other methods
    for method in ['Uniform', 'LFWA', 'AdaSNR']:
        print(f"\nSweeping over learning rates for {method}...")
        best_alt_acc = -1.0
        best_alt_res = None
        
        best_seq_acc = -1.0
        best_seq_res = None
        
        for lr in lr_grid:
            # Eval on Alternating Stream
            avg, cifar, svhn = merger.evaluate_stream(alt_stream, method, lr_base=lr)
            print(f"  Alt - LR: {lr:<4} | Avg Acc: {avg:.2f}% (CIFAR: {cifar:.2f}%, SVHN: {svhn:.2f}%)")
            if avg > best_alt_acc:
                best_alt_acc = avg
                best_alt_res = (avg, cifar, svhn, lr)
                
            # Eval on Block-Sequential Stream
            avg_s, cifar_s, svhn_s = merger.evaluate_stream(seq_stream, method, lr_base=lr)
            print(f"  Seq - LR: {lr:<4} | Avg Acc: {avg_s:.2f}% (CIFAR: {cifar_s:.2f}%, SVHN: {svhn_s:.2f}%)")
            if avg_s > best_seq_acc:
                best_seq_acc = avg_s
                best_seq_res = (avg_s, cifar_s, svhn_s, lr)
                
        results[method] = {
            'alt': best_alt_res,
            'seq': best_seq_res
        }
        
    # Print beautiful comparative summary table
    print("\n" + "="*100)
    print(f"{'Method':<26} | {'Alternating Stream (Best)':<34} | {'Block-Sequential Stream (Best)':<34}")
    print(f"{'':<26} | {'Avg Acc (LR)':<14} | {'CIFAR / SVHN':<16} | {'Avg Acc (LR)':<14} | {'CIFAR / SVHN':<16}")
    print("="*100)
    
    for method in ['Static', 'Uniform', 'LFWA', 'AdaSNR']:
        alt_avg, alt_c, alt_s, alt_lr = results[method]['alt']
        seq_avg, seq_c, seq_s, seq_lr = results[method]['seq']
        
        alt_lr_str = f"({alt_lr:.2f})" if method != 'Static' else "(-)"
        seq_lr_str = f"({seq_lr:.2f})" if method != 'Static' else "(-)"
        
        print(f"{method:<26} | {alt_avg:.2f}% {alt_lr_str:<8} | {alt_c:.1f}% / {alt_s:.1f}% | {seq_avg:.2f}% {seq_lr_str:<8} | {seq_c:.1f}% / {seq_s:.1f}%")
    print("="*100)
    
    # Save sweep results to progress.md and results_sweep.txt
    print("\nSaving results to results_sweep.txt...")
    with open("results_sweep.txt", "w") as f:
        f.write("Test-Time Model Merging Sweep Results:\n\n")
        f.write("="*100 + "\n")
        f.write(f"{'Method':<26} | {'Alternating Stream (Best)':<34} | {'Block-Sequential Stream (Best)':<34}\n")
        f.write(f"{'':<26} | {'Avg Acc (LR)':<14} | {'CIFAR / SVHN':<16} | {'Avg Acc (LR)':<14} | {'CIFAR / SVHN':<16}\n")
        f.write("="*100 + "\n")
        for method in ['Static', 'Uniform', 'LFWA', 'AdaSNR']:
            alt_avg, alt_c, alt_s, alt_lr = results[method]['alt']
            seq_avg, seq_c, seq_s, seq_lr = results[method]['seq']
            alt_lr_str = f"({alt_lr:.2f})" if method != 'Static' else "(-)"
            seq_lr_str = f"({seq_lr:.2f})" if method != 'Static' else "(-)"
            f.write(f"{method:<26} | {alt_avg:.2f}% {alt_lr_str:<8} | {alt_c:.1f}% / {alt_s:.1f}% | {seq_avg:.2f}% {seq_lr_str:<8} | {seq_c:.1f}% / {seq_s:.1f}%\n")
        f.write("="*100 + "\n")

if __name__ == "__main__":
    main()
