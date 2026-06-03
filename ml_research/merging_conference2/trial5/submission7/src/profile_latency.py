import torch
import torch.nn as nn
import time
import copy
from dataset import get_datasets, get_dataloaders
from models import create_expert_model, load_checkpoint
from calibrate import collect_target_stats, calibrate_sp_taac, apply_spectral_calibration
from evaluate import merge_backbones
from torch.utils.data import DataLoader, ConcatDataset

# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED errors on cluster GPUs
torch.backends.cudnn.enabled = False

def run_latency_profile(device='cpu'):
    print(f"\n" + "="*50)
    print(f"PROFILING INFERENCE LATENCY ON {device.upper()}")
    print("="*50)
    
    # Load Datasets to get proper dataloaders
    splits = get_datasets(calib_size=128)
    loaders = get_dataloaders(splits, batch_size=128)
    
    # Create Joint Calibration Dataset loader
    joint_calib_dataset = ConcatDataset([
        splits['mnist']['calib'],
        splits['fmnist']['calib'],
        splits['cifar10']['calib']
    ])
    joint_calib_loader = DataLoader(joint_calib_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Load Experts
    expert_models = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        model = create_expert_model(num_classes=10)
        model = load_checkpoint(model, f"checkpoints/{task}_expert.pt", device=device)
        expert_models[task] = model.to(device)
        
    # Get target stats
    target_stats = collect_target_stats(expert_models, {task: loaders[task]['calib'] for task in ['mnist', 'fmnist', 'cifar10']}, device=device)
    
    # Weight Averaging
    merged_model_base = merge_backbones(expert_models, merge_mode='wa', lambda_coeff=0.3, device=device).to(device)
    
    # Prepare dummy input of batch size 128 (matching standard evaluation batch size)
    dummy_input = torch.randn(128, 3, 32, 32, device=device)
    
    # Define models
    models_dict = {}
    hooks_dict = {}
    
    # 1. Uncalibrated / SP-TAAC (fused, zero overhead)
    # Since SP-TAAC scales weights in-place, its latency is mathematically identical to the uncalibrated model.
    # We will profile the baseline model.
    model_uncal = copy.deepcopy(merged_model_base)
    model_uncal.eval()
    models_dict['Uncalibrated / SP-TAAC (Fused)'] = model_uncal
    
    # 2. FDSA (Active Hooks)
    model_fdsa = copy.deepcopy(merged_model_base)
    model_fdsa, fdsa_hooks = apply_spectral_calibration(model_fdsa, target_stats, joint_calib_loader, method='fdsa', device=device)
    model_fdsa.eval()
    models_dict['FDSA (Active Hooks)'] = model_fdsa
    hooks_dict['FDSA (Active Hooks)'] = fdsa_hooks
    
    # 3. WRSA (Active Hooks)
    model_wrsa = copy.deepcopy(merged_model_base)
    model_wrsa, wrsa_hooks = apply_spectral_calibration(model_wrsa, target_stats, joint_calib_loader, method='wrsa', c_val=0.30, device=device)
    model_wrsa.eval()
    models_dict['WRSA (Active Hooks, Ours)'] = model_wrsa
    hooks_dict['WRSA (Active Hooks, Ours)'] = wrsa_hooks
    
    warmups = 50
    iterations = 100
    
    print("\n=== Latency Benchmark ===")
    for name, model in models_dict.items():
        # Warmup
        print(f"Warming up {name}...")
        with torch.no_grad():
            for _ in range(warmups):
                _ = model(dummy_input)
                
        # Timing
        print(f"Timing {name} for {iterations} iterations...")
        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(dummy_input)
                
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000.0
        avg_time_ms = total_time_ms / iterations
        print(f"Result for {name}: {avg_time_ms:.4f} ms per batch (batch_size=128)")
        
    # Clean up hooks
    for name, hooks in hooks_dict.items():
        for h in hooks:
            h.remove()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()
    run_latency_profile(device=args.device)
