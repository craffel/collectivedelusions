import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from dataset import get_datasets, get_dataloaders
from models import create_expert_model, get_base_model, load_checkpoint
from calibrate import collect_target_stats, calibrate_sp_taac, apply_spectral_calibration
from evaluate import evaluate_on_task, merge_backbones

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors on cluster GPUs
torch.backends.cudnn.enabled = False

def run_sample_size_ablation(device='cuda'):
    print("\n" + "="*50)
    print("RUNNING SAMPLE SIZE ABLATION STUDY")
    print("="*50)
    
    # Load Experts
    print("=== Loading Expert Models ===")
    expert_models = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        model = create_expert_model(num_classes=10)
        model = load_checkpoint(model, f"checkpoints/{task}_expert.pt", device=device)
        expert_models[task] = model.to(device)
        
    expert_heads = {task: model.fc for task, model in expert_models.items()}
    
    # We will evaluate Weight Averaging (WA)
    merged_model_base = merge_backbones(expert_models, merge_mode='wa', lambda_coeff=0.3, device=device).to(device)
    
    sizes = [16, 64, 128]
    results = {}
    
    for N in sizes:
        print(f"\n--- Running Ablation for Calibration Size N = {N} per task (Total = {3 * N}) ---")
        splits = get_datasets(calib_size=N)
        loaders = get_dataloaders(splits, batch_size=128)
        
        # Create Joint Calibration Dataset loader
        joint_calib_dataset = ConcatDataset([
            splits['mnist']['calib'],
            splits['fmnist']['calib'],
            splits['cifar10']['calib']
        ])
        joint_calib_loader = DataLoader(joint_calib_dataset, batch_size=128, shuffle=False, num_workers=2)
        
        # Collect Target Statistics
        target_stats = collect_target_stats(expert_models, {task: loaders[task]['calib'] for task in ['mnist', 'fmnist', 'cifar10']}, device=device)
        
        # 1. Uncalibrated (Doesn't depend on N, but let's include for completeness)
        uncal_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            acc = evaluate_on_task(merged_model_base, expert_heads[task], loaders[task]['test'], device=device)
            uncal_accs[task] = acc
        uncal_avg = sum(uncal_accs.values()) / 3
        results[(N, 'uncalibrated')] = {**uncal_accs, 'average': uncal_avg}
        
        # 2. SP-TAAC
        model_sp = copy.deepcopy(merged_model_base)
        model_sp = calibrate_sp_taac(model_sp, target_stats, joint_calib_loader, device=device)
        sp_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            acc = evaluate_on_task(model_sp, expert_heads[task], loaders[task]['test'], device=device)
            sp_accs[task] = acc
        sp_avg = sum(sp_accs.values()) / 3
        results[(N, 'sp-taac')] = {**sp_accs, 'average': sp_avg}
        
        # 3. FDSA
        model_fdsa = copy.deepcopy(merged_model_base)
        model_fdsa, fdsa_hooks = apply_spectral_calibration(model_fdsa, target_stats, joint_calib_loader, method='fdsa', device=device)
        fdsa_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            acc = evaluate_on_task(model_fdsa, expert_heads[task], loaders[task]['test'], device=device)
            fdsa_accs[task] = acc
        fdsa_avg = sum(fdsa_accs.values()) / 3
        results[(N, 'fdsa')] = {**fdsa_accs, 'average': fdsa_avg}
        for h in fdsa_hooks:
            h.remove()
            
        # 4. WRSA (c=0.30)
        model_wrsa = copy.deepcopy(merged_model_base)
        model_wrsa, wrsa_hooks = apply_spectral_calibration(model_wrsa, target_stats, joint_calib_loader, method='wrsa', c_val=0.30, device=device)
        wrsa_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            acc = evaluate_on_task(model_wrsa, expert_heads[task], loaders[task]['test'], device=device)
            wrsa_accs[task] = acc
        wrsa_avg = sum(wrsa_accs.values()) / 3
        results[(N, 'wrsa_c0.3')] = {**wrsa_accs, 'average': wrsa_avg}
        for h in wrsa_hooks:
            h.remove()
            
    print("\n=== SAMPLE SIZE ABLATION SUMMARY (WA Merging) ===")
    print(f"{'N (per task)':<12} | {'Method':<15} | {'MNIST (%)':<10} | {'F-MNIST (%)':<12} | {'CIFAR-10 (%)':<13} | {'Average (%)':<12}")
    print("-" * 80)
    for N in sizes:
        for calib in ['uncalibrated', 'sp-taac', 'fdsa', 'wrsa_c0.3']:
            r = results[(N, calib)]
            print(f"{N:<12} | {calib:<15} | {r['mnist']:<10.2f} | {r['fmnist']:<12.2f} | {r['cifar10']:<13.2f} | {r['average']:<12.2f}")
    print("="*80)
    
    return results

def run_multi_seed_stability(device='cuda'):
    print("\n" + "="*50)
    print("RUNNING MULTI-SEED STABILITY AND VARIANCE ANALYSIS")
    print("="*50)
    
    # Load Experts
    print("=== Loading Expert Models ===")
    expert_models = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        model = create_expert_model(num_classes=10)
        model = load_checkpoint(model, f"checkpoints/{task}_expert.pt", device=device)
        expert_models[task] = model.to(device)
        
    expert_heads = {task: model.fc for task, model in expert_models.items()}
    
    # Weight Averaging (WA)
    merged_model_base = merge_backbones(expert_models, merge_mode='wa', lambda_coeff=0.3, device=device).to(device)
    
    seeds = [42, 43, 44, 45, 46]
    N = 128
    
    fdsa_seed_averages = []
    wrsa_seed_averages = []
    
    fdsa_results = []
    wrsa_results = []
    
    for seed in seeds:
        print(f"\n--- Running Evaluation with Random Seed = {seed} (N = {N}) ---")
        splits = get_datasets(calib_size=N, calib_seed=seed)
        loaders = get_dataloaders(splits, batch_size=128)
        
        # Create Joint Calibration Dataset loader
        joint_calib_dataset = ConcatDataset([
            splits['mnist']['calib'],
            splits['fmnist']['calib'],
            splits['cifar10']['calib']
        ])
        joint_calib_loader = DataLoader(joint_calib_dataset, batch_size=128, shuffle=False, num_workers=2)
        
        # Collect Target Statistics
        target_stats = collect_target_stats(expert_models, {task: loaders[task]['calib'] for task in ['mnist', 'fmnist', 'cifar10']}, device=device)
        
        # FDSA
        model_fdsa = copy.deepcopy(merged_model_base)
        model_fdsa, fdsa_hooks = apply_spectral_calibration(model_fdsa, target_stats, joint_calib_loader, method='fdsa', device=device)
        fdsa_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            acc = evaluate_on_task(model_fdsa, expert_heads[task], loaders[task]['test'], device=device)
            fdsa_accs[task] = acc
        fdsa_avg = sum(fdsa_accs.values()) / 3
        fdsa_seed_averages.append(fdsa_avg)
        fdsa_results.append(fdsa_accs)
        for h in fdsa_hooks:
            h.remove()
            
        # WRSA (c=0.30)
        model_wrsa = copy.deepcopy(merged_model_base)
        model_wrsa, wrsa_hooks = apply_spectral_calibration(model_wrsa, target_stats, joint_calib_loader, method='wrsa', c_val=0.30, device=device)
        wrsa_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            acc = evaluate_on_task(model_wrsa, expert_heads[task], loaders[task]['test'], device=device)
            wrsa_accs[task] = acc
        wrsa_avg = sum(wrsa_accs.values()) / 3
        wrsa_seed_averages.append(wrsa_avg)
        wrsa_results.append(wrsa_accs)
        for h in wrsa_hooks:
            h.remove()
            
        print(f"Seed {seed}: FDSA Average = {fdsa_avg:.2f}%, WRSA Average = {wrsa_avg:.2f}%")
        
    fdsa_mean = np.mean(fdsa_seed_averages)
    fdsa_std = np.std(fdsa_seed_averages)
    wrsa_mean = np.mean(wrsa_seed_averages)
    wrsa_std = np.std(wrsa_seed_averages)
    
    print("\n=== MULTI-SEED STABILITY SUMMARY ===")
    print(f"Method | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean Acc (%) | Std Dev")
    print("-" * 85)
    print(f"FDSA   | {fdsa_seed_averages[0]:.2f}% | {fdsa_seed_averages[1]:.2f}% | {fdsa_seed_averages[2]:.2f}% | {fdsa_seed_averages[3]:.2f}% | {fdsa_seed_averages[4]:.2f}% | {fdsa_mean:.2f}%       | {fdsa_std:.4f}")
    print(f"WRSA   | {wrsa_seed_averages[0]:.2f}% | {wrsa_seed_averages[1]:.2f}% | {wrsa_seed_averages[2]:.2f}% | {wrsa_seed_averages[3]:.2f}% | {wrsa_seed_averages[4]:.2f}% | {wrsa_mean:.2f}%       | {wrsa_std:.4f}")
    print("="*85)
    
    # Detailed task-wise standard deviations
    for task in ['mnist', 'fmnist', 'cifar10']:
        fdsa_task = [r[task] for r in fdsa_results]
        wrsa_task = [r[task] for r in wrsa_results]
        print(f"{task.upper()} - FDSA: {np.mean(fdsa_task):.2f} +/- {np.std(fdsa_task):.4f}% | WRSA: {np.mean(wrsa_task):.2f} +/- {np.std(wrsa_task):.4f}%")
        
    return fdsa_seed_averages, wrsa_seed_averages

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    run_sample_size_ablation(device=device)
    run_multi_seed_stability(device=device)
