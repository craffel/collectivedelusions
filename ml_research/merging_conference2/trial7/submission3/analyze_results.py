import os
import json
import numpy as np

def analyze_file(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist yet.")
        return None
        
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    sweep = data['sweep']
    baselines = data['baselines']
    strategy = data['strategy']
    
    # 1. Find overall best config (no noise)
    no_noise_sweep = [entry for entry in sweep if entry['noise_std'] == 0.0]
    best_entry = max(no_noise_sweep, key=lambda x: x['avg_accuracy'])
    
    # 2. Analyze performance vs batch size (no noise, best alpha per batch size)
    batch_sizes = sorted(list(set(entry['test_batch_size'] for entry in no_noise_sweep)))
    best_by_bs = {}
    for bs in batch_sizes:
        bs_sweep = [entry for entry in no_noise_sweep if entry['test_batch_size'] == bs]
        best_bs_entry = max(bs_sweep, key=lambda x: x['avg_accuracy'])
        best_by_bs[bs] = {
            'accuracy': best_bs_entry['avg_accuracy'],
            'alpha': best_bs_entry['alpha'],
            'paradigm': best_bs_entry['paradigm'],
            'lambda': best_bs_entry['lambda']
        }
        
    # 3. Analyze noise robustness (best overall config under noise)
    noise_levels = sorted(list(set(entry['noise_std'] for entry in sweep)))
    best_by_noise = {}
    for noise in noise_levels:
        noise_sweep = [entry for entry in sweep if entry['noise_std'] == noise]
        # We look at the best overall configuration from no_noise (best_entry) and see how it scales under noise
        matching_entries = [
            entry for entry in noise_sweep 
            if entry['paradigm'] == best_entry['paradigm'] 
            and entry['lambda'] == best_entry['lambda']
            and entry['test_batch_size'] == best_entry['test_batch_size']
            and entry['alpha'] == best_entry['alpha']
        ]
        if matching_entries:
            best_by_noise[noise] = matching_entries[0]['avg_accuracy']
        else:
            # Fallback to absolute best for this noise level
            best_noise_entry = max(noise_sweep, key=lambda x: x['avg_accuracy'])
            best_by_noise[noise] = best_noise_entry['avg_accuracy']
            
    return {
        'strategy': strategy,
        'baselines': baselines,
        'best_config': best_entry,
        'best_by_bs': best_by_bs,
        'best_by_noise': best_by_noise
    }

def main():
    strategies = ['uniform', 'variance', 'fisher_syn', 'fisher_real', 'grad_norm']
    results = {}
    
    print("=== Model Merging BN Calibration Sweep Analysis ===\n")
    
    for s in strategies:
        file_path = f"results/{s}.json"
        res = analyze_file(file_path)
        if res:
            results[s] = res
            
    if not results:
        print("No results found yet.")
        return
        
    print("\n" + "="*50)
    print("1. OVERALL BEST CONFIGURATIONS (NO NOISE)")
    print("="*50)
    for s, res in results.items():
        best = res['best_config']
        pad_str = f"{best['paradigm'].upper()}" + (f" (lam={best['lambda']:.2f})" if best['lambda'] else "")
        print(f"Strategy: {s.upper():<12} | Avg Acc: {best['avg_accuracy']:.2f}% | Paradigm: {pad_str:<18} | B: {best['test_batch_size']:3d} | Alpha: {best['alpha']:.1f}")
        print(f"  Task Accuracies: MNIST={best['task_accuracies']['mnist']:.2f}%, Fashion={best['task_accuracies']['fashion']:.2f}%, CIFAR={best['task_accuracies']['cifar10']:.2f}%")
        print("-" * 50)
        
    print("\n" + "="*50)
    print("2. ACCURACY BY TEST BATCH SIZE (NO NOISE)")
    print("="*50)
    header = f"{'Strategy':<12} | " + " | ".join(f"B={bs:<3d}" for bs in [1, 4, 16, 64, 256])
    print(header)
    print("-" * len(header))
    for s, res in results.items():
        row_vals = []
        for bs in [1, 4, 16, 64, 256]:
            val = res['best_by_bs'].get(bs, {}).get('accuracy', 0.0)
            row_vals.append(f"{val:.2f}%")
        print(f"{s.upper():<12} | " + " | ".join(row_vals))
        
    print("\n" + "="*50)
    print("3. TEST-TIME NOISE ROBUSTNESS (BEST CONFIG SCHEDULE)")
    print("="*50)
    header = f"{'Strategy':<12} | " + " | ".join(f"Noise={n:.1f}" for n in [0.0, 0.1, 0.2])
    print(header)
    print("-" * len(header))
    for s, res in results.items():
        row_vals = []
        for n in [0.0, 0.1, 0.2]:
            val = res['best_by_noise'].get(n, 0.0)
            row_vals.append(f"{val:.2f}%")
        print(f"{s.upper():<12} | " + " | ".join(row_vals))

if __name__ == '__main__':
    main()
