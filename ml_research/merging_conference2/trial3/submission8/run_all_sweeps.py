import os
import json
import numpy as np
import torch
import torch.nn as nn
from dataset import get_dataloaders
from models import MultiTaskResNet18
from eval import (
    get_calibration_loaders,
    assemble_merged_model,
    record_expert_stds,
    apply_tcac,
    calibrate_lsc,
    calibrate_sp_taac,
    evaluate_model
)

def run_single_eval(model_mode, lambda_val, cal_size, seed, test_loaders,
                    imbalance_task=None, imbalance_ratio=1.0, device='cuda'):
    # Helper to run a single configuration and return the results
    expert_paths = {
        'mnist': 'checkpoints/expert_mnist.pt',
        'fashion': 'checkpoints/expert_fashion.pt',
        'cifar': 'checkpoints/expert_cifar.pt'
    }
    pretrained_path = 'checkpoints/pretrained.pt'
    
    # Get calibration loaders with potential imbalance
    cal_loaders, joint_loader = get_calibration_loaders(
        N=cal_size, seed=seed,
        imbalance_task=imbalance_task, imbalance_ratio=imbalance_ratio
    )
    
    # Assemble merged model
    model = assemble_merged_model(expert_paths, pretrained_path, model_mode, lambda_val).to(device)
    
    # Pre-compute expert stds
    expert_stds_all = {}
    for task in ['mnist', 'fashion', 'cifar']:
        expert_stds_all[task] = record_expert_stds(expert_paths[task], cal_loaders[task], task, device)
        
    # Pre-compute calibrations
    tcac_bn_states = apply_tcac(model, expert_paths, device)
    lsc_gammas = calibrate_lsc(model, expert_stds_all, cal_loaders, device)
    sp_taac_gammas = calibrate_sp_taac(model, expert_stds_all, joint_loader, device)
    
    # Evaluate all methods
    results = {}
    for m in ['none', 'tcac', 'taac', 'lsc', 'sp_taac']:
        res = evaluate_model(
            model, test_loaders, cal_loaders, joint_loader, expert_paths,
            cal_method=m, lsc_gammas=lsc_gammas, sp_taac_gammas=sp_taac_gammas, tcac_bn_states=tcac_bn_states,
            device=device
        )
        results[m] = res
        
    return results

def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/sample_efficiency', exist_ok=True)
    os.makedirs('results/multi_seed', exist_ok=True)
    os.makedirs('results/imbalance', exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting sweeps in a single Python process on device: {device.upper()}")
    
    # Load test dataloaders ONCE
    print("Loading test dataloaders...")
    _, test_loaders = get_dataloaders(batch_size=128)
    
    # ==========================================
    # Sweep 1: Main Benchmark
    # ==========================================
    print("\n--- Sweep 1: Main Benchmark ---")
    # Weight Averaging
    print("Evaluating Weight Averaging...")
    wa_out = "results/main_wa.json"
    if not os.path.exists(wa_out):
        res = run_single_eval('wa', 0.2, 128, 42, test_loaders, device=device)
        with open(wa_out, "w") as f:
            json.dump(res, f, indent=4)
    else:
        print("Skipping (already exists)")
        
    # Task Arithmetic
    for lam in [0.1, 0.2, 0.3, 0.4]:
        print(f"Evaluating Task Arithmetic (λ={lam})...")
        ta_out = f"results/main_ta_lam{lam}.json"
        if not os.path.exists(ta_out):
            res = run_single_eval('ta', lam, 128, 42, test_loaders, device=device)
            with open(ta_out, "w") as f:
                json.dump(res, f, indent=4)
        else:
            print("Skipping (already exists)")
            
    # ==========================================
    # Sweep 2: Sample Efficiency N-Sweep
    # ==========================================
    print("\n--- Sweep 2: Sample Efficiency N-Sweep ---")
    cal_sizes = [4, 8, 16, 32, 64, 128, 256]
    for N in cal_sizes:
        print(f"Evaluating WA/TA with N={N}...")
        # WA
        wa_out = f"results/sample_efficiency/wa_N{N}.json"
        if not os.path.exists(wa_out):
            res = run_single_eval('wa', 0.2, N, 42, test_loaders, device=device)
            with open(wa_out, "w") as f:
                json.dump(res, f, indent=4)
        else:
            print("Skipping WA (already exists)")
            
        # TA
        ta_out = f"results/sample_efficiency/ta_N{N}.json"
        if not os.path.exists(ta_out):
            res = run_single_eval('ta', 0.2, N, 42, test_loaders, device=device)
            with open(ta_out, "w") as f:
                json.dump(res, f, indent=4)
        else:
            print("Skipping TA (already exists)")
            
    # ==========================================
    # Sweep 3: Multi-Seed Robustness Sweep
    # ==========================================
    print("\n--- Sweep 3: Multi-Seed Robustness Sweep ---")
    seeds = [42, 43, 44, 45, 46]
    for seed in seeds:
        print(f"Evaluating with Seed={seed}...")
        # WA
        wa_out = f"results/multi_seed/wa_seed{seed}.json"
        if not os.path.exists(wa_out):
            res = run_single_eval('wa', 0.2, 128, seed, test_loaders, device=device)
            with open(wa_out, "w") as f:
                json.dump(res, f, indent=4)
        else:
            print("Skipping WA (already exists)")
            
        # TA
        ta_out = f"results/multi_seed/ta_seed{seed}.json"
        if not os.path.exists(ta_out):
            res = run_single_eval('ta', 0.2, 128, seed, test_loaders, device=device)
            with open(ta_out, "w") as f:
                json.dump(res, f, indent=4)
        else:
            print("Skipping TA (already exists)")
            
    # ==========================================
    # Sweep 4: Task Imbalance Sweep
    # ==========================================
    print("\n--- Sweep 4: Task Imbalance Sweep ---")
    for task in ['mnist', 'fashion', 'cifar']:
        for ratio in [0.25, 4.0]:
            print(f"Evaluating Imbalance task={task} ratio={ratio}...")
            # WA
            wa_out = f"results/imbalance/wa_task_{task}_ratio{ratio}.json"
            if not os.path.exists(wa_out):
                res = run_single_eval('wa', 0.2, 64, 42, test_loaders,
                                      imbalance_task=task, imbalance_ratio=ratio, device=device)
                with open(wa_out, "w") as f:
                    json.dump(res, f, indent=4)
            else:
                print("Skipping WA (already exists)")
                
            # TA
            ta_out = f"results/imbalance/ta_task_{task}_ratio{ratio}.json"
            if not os.path.exists(ta_out):
                res = run_single_eval('ta', 0.2, 64, 42, test_loaders,
                                      imbalance_task=task, imbalance_ratio=ratio, device=device)
                with open(ta_out, "w") as f:
                    json.dump(res, f, indent=4)
            else:
                print("Skipping TA (already exists)")
                
    print("\n" + "="*50)
    print("ALL SWEEPS COMPLETED! COMPILING RESULTS...")
    print("="*50)
    
    compile_and_summarize()

def compile_and_summarize():
    summary_data = {}
    
    # 1. Compile Main Benchmark
    print("\n=== COMPILING MAIN BENCHMARK RESULTS ===")
    main_results = {}
    
    # Weight Averaging
    try:
        with open("results/main_wa.json", "r") as f:
            main_results["WA"] = json.load(f)
    except Exception as e:
        print("WA results missing:", e)
        
    # Task Arithmetic
    for lam in [0.1, 0.2, 0.3, 0.4]:
        try:
            with open(f"results/main_ta_lam{lam}.json", "r") as f:
                main_results[f"TA (λ={lam})"] = json.load(f)
        except Exception as e:
            print(f"TA λ={lam} results missing:", e)
            
    summary_data["main_benchmark"] = main_results
    
    # Print table
    print("\nMerged Model + Calibration Method Average Accuracies (%):")
    print("-"*90)
    print(f"{'Method/Setting':<20} | {'Uncalibrated':<12} | {'TCAC':<10} | {'TAAC':<10} | {'LSC':<10} | {'SP-TAAC (Ours)':<15}")
    print("-"*90)
    for setting, methods in main_results.items():
        none_acc = methods.get('none', {}).get('avg', 0.0)
        tcac_acc = methods.get('tcac', {}).get('avg', 0.0)
        taac_acc = methods.get('taac', {}).get('avg', 0.0)
        lsc_acc = methods.get('lsc', {}).get('avg', 0.0)
        sp_taac_acc = methods.get('sp_taac', {}).get('avg', 0.0)
        print(f"{setting:<20} | {none_acc:<12.2f} | {tcac_acc:<10.2f} | {taac_acc:<10.2f} | {lsc_acc:<10.2f} | {sp_taac_acc:<15.2f}")
    print("-"*90)
    
    # 2. Compile Sample Efficiency Sweep
    print("\n=== COMPILING SAMPLE EFFICIENCY N-SWEEP ===")
    se_results = {"wa": {}, "ta": {}}
    cal_sizes = [4, 8, 16, 32, 64, 128, 256]
    for N in cal_sizes:
        for mode in ['wa', 'ta']:
            try:
                with open(f"results/sample_efficiency/{mode}_N{N}.json", "r") as f:
                    se_results[mode][N] = json.load(f)
            except Exception:
                pass
                
    summary_data["sample_efficiency"] = se_results
    
    # Print WA N-Sweep Table
    print("\nWeight Averaging Sample Efficiency (N-Sweep) Average Accuracies (%):")
    print("-"*80)
    print(f"{'N Samples/Task':<15} | {'Uncalibrated':<12} | {'TCAC':<10} | {'TAAC':<10} | {'LSC':<10} | {'SP-TAAC (Ours)':<15}")
    print("-"*80)
    for N in cal_sizes:
        methods = se_results["wa"].get(N, {})
        none_acc = methods.get('none', {}).get('avg', 0.0)
        tcac_acc = methods.get('tcac', {}).get('avg', 0.0)
        taac_acc = methods.get('taac', {}).get('avg', 0.0)
        lsc_acc = methods.get('lsc', {}).get('avg', 0.0)
        sp_taac_acc = methods.get('sp_taac', {}).get('avg', 0.0)
        print(f"N = {N:<11} | {none_acc:<12.2f} | {tcac_acc:<10.2f} | {taac_acc:<10.2f} | {lsc_acc:<10.2f} | {sp_taac_acc:<15.2f}")
    print("-"*80)
    
    # 3. Compile Multi-Seed Sweep
    print("\n=== COMPILING MULTI-SEED ROBUSTNESS SWEEP ===")
    seeds = [42, 43, 44, 45, 46]
    seed_data = {"wa": {}, "ta": {}}
    for mode in ['wa', 'ta']:
        for m in ['none', 'tcac', 'taac', 'lsc', 'sp_taac']:
            seed_data[mode][m] = []
            
    for seed in seeds:
        for mode in ['wa', 'ta']:
            try:
                with open(f"results/multi_seed/{mode}_seed{seed}.json", "r") as f:
                    res = json.load(f)
                    for m in res:
                        seed_data[mode][m].append(res[m]['avg'])
            except Exception:
                pass
                
    summary_data["multi_seed"] = seed_data
    
    print("\nWeight Averaging Robustness Across 5 Seeds (Mean ± Std %):")
    print("-"*80)
    for m in ['none', 'tcac', 'taac', 'lsc', 'sp_taac']:
        vals = seed_data["wa"][m]
        if vals:
            print(f"{m.upper():<20} : {np.mean(vals):.2f}% ± {np.std(vals):.2f}%")
        else:
            print(f"{m.upper():<20} : N/A")
    print("-"*80)
    
    with open("results/compiled_summary.json", "w") as f:
        json.dump(summary_data, f, indent=4)
    print("\nSaved compiled summary to results/compiled_summary.json")

if __name__ == '__main__':
    main()
