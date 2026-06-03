import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_sweep_data():
    strategies = ['uniform', 'variance', 'fisher_syn', 'fisher_real', 'grad_norm']
    results = {}
    
    for s in strategies:
        file_path = f"results/{s}.json"
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        sweep = data['sweep']
        no_noise_sweep = [entry for entry in sweep if entry['noise_std'] == 0.0]
        
        # Analyze performance vs batch size (best alpha per batch size)
        batch_sizes = sorted(list(set(entry['test_batch_size'] for entry in no_noise_sweep)))
        best_by_bs = {}
        for bs in batch_sizes:
            bs_sweep = [entry for entry in no_noise_sweep if entry['test_batch_size'] == bs]
            best_bs_entry = max(bs_sweep, key=lambda x: x['avg_accuracy'])
            best_by_bs[bs] = best_bs_entry['avg_accuracy']
            
        # Analyze noise robustness (best overall config under noise)
        best_entry = max(no_noise_sweep, key=lambda x: x['avg_accuracy'])
        noise_levels = sorted(list(set(entry['noise_std'] for entry in sweep)))
        best_by_noise = {}
        for noise in noise_levels:
            noise_sweep = [entry for entry in sweep if entry['noise_std'] == noise]
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
                best_noise_entry = max(noise_sweep, key=lambda x: x['avg_accuracy'])
                best_by_noise[noise] = best_noise_entry['avg_accuracy']
                
        results[s] = {
            'best_by_bs': best_by_bs,
            'best_by_noise': best_by_noise
        }
    return results

def load_temp_data():
    json_path = 'results/temperature_sweep.json'
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    results = {}
    for entry in data:
        strat = entry['strategy']
        temp = entry['temperature']
        bs = entry['test_batch_size']
        alpha = entry['alpha']
        avg_acc = entry['avg_accuracy']
        
        if strat not in results:
            results[strat] = {}
        if temp not in results[strat]:
            results[strat][temp] = {}
        if bs not in results[strat][temp]:
            results[strat][temp][bs] = {}
        results[strat][temp][bs][alpha] = avg_acc
    return results

def plot_batch_size_sweep(sweep_data):
    plt.figure(figsize=(7, 4.5))
    
    strategies = {
        'uniform': ('Uniform', 'o', '-', 'C0'),
        'variance': ('Activation Variance', 's', '--', 'C1'),
        'fisher_real': ('Fisher (Real)', '^', '-.', 'C2'),
        'fisher_syn': ('Fisher (Synthetic)', 'x', ':', 'C3'),
        'grad_norm': ('Gradient Norm', 'd', '-', 'C4')
    }
    
    batch_sizes = [1, 4, 16, 64, 256]
    
    for s, (label, marker, linestyle, color) in strategies.items():
        if s not in sweep_data:
            continue
        accs = [sweep_data[s]['best_by_bs'].get(bs, 0.0) for bs in batch_sizes]
        plt.plot(batch_sizes, accs, label=label, marker=marker, linestyle=linestyle, color=color, linewidth=2, markersize=7)
        
    plt.xscale('log')
    plt.xticks(batch_sizes, [str(bs) for bs in batch_sizes])
    plt.xlabel('Test Batch Size (B)', fontsize=11, fontweight='bold')
    plt.ylabel('Average Accuracy (%)', fontsize=11, fontweight='bold')
    plt.title('Post-Merge Calibration Accuracy vs. Test Batch Size', fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.ylim(10, 90)
    plt.tight_layout()
    plt.savefig('fig_batch_size.png', dpi=300)
    plt.savefig('fig_batch_size.pdf', dpi=300)
    plt.close()
    print("Saved fig_batch_size.png and fig_batch_size.pdf")

def plot_temperature_sweep(temp_data):
    if not temp_data:
        print("No temperature sweep data found to plot.")
        return
        
    plt.figure(figsize=(7, 4.5))
    
    strats = {
        'fisher_syn': ('Fisher-Syn-Soft', 'o', '-', 'C3'),
        'fisher_real': ('Fisher-Real-Soft', '^', '--', 'C2')
    }
    
    temperatures = sorted(list(temp_data['fisher_syn'].keys()))
    
    for strat, (label, marker, linestyle, color) in strats.items():
        if strat not in temp_data:
            continue
        # Get static accuracy at B=1, alpha=0.0
        accs = []
        for temp in temperatures:
            acc = temp_data[strat][temp][1].get(0.0, 0.0)
            accs.append(acc)
        plt.plot(temperatures, accs, label=label, marker=marker, linestyle=linestyle, color=color, linewidth=2, markersize=7)
        
    # Draw uniform reference line at 65.10%
    plt.axhline(65.10, color='C0', linestyle=':', linewidth=2, label='Uniform Baseline (65.10%)')
    
    plt.xlabel('Temperature (T)', fontsize=11, fontweight='bold')
    plt.ylabel('Average Accuracy (%) at B=1', fontsize=11, fontweight='bold')
    plt.title('Fisher-Soft Temperature Sweeps at Batch Size B=1', fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', frameon=True, fontsize=10)
    plt.ylim(30, 70)
    plt.tight_layout()
    plt.savefig('fig_temperature.png', dpi=300)
    plt.savefig('fig_temperature.pdf', dpi=300)
    plt.close()
    print("Saved fig_temperature.png and fig_temperature.pdf")

def plot_noise_robustness(sweep_data):
    plt.figure(figsize=(7, 4.5))
    
    strategies = {
        'uniform': ('Uniform', 'C0'),
        'variance': ('Activation Variance', 'C1'),
        'fisher_real': ('Fisher (Real)', 'C2'),
        'fisher_syn': ('Fisher (Synthetic)', 'C3'),
        'grad_norm': ('Gradient Norm', 'C4')
    }
    
    noise_levels = [0.0, 0.1, 0.2]
    x = np.arange(len(noise_levels))
    width = 0.15
    
    for idx, (s, (label, color)) in enumerate(strategies.items()):
        if s not in sweep_data:
            continue
        accs = [sweep_data[s]['best_by_noise'].get(n, 0.0) for n in noise_levels]
        plt.bar(x + (idx - 2) * width, accs, width, label=label, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        
    plt.xticks(x, [f"clean ($\sigma=0.0$)", f"noise ($\sigma=0.1$)", f"noise ($\sigma=0.2$)"], fontsize=10)
    plt.xlabel('Test-Time Covariate Noise Level ($\sigma$)', fontsize=11, fontweight='bold')
    plt.ylabel('Average Accuracy (%)', fontsize=11, fontweight='bold')
    plt.title('Covariate Shift Robustness under Test-Time Noise', fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', frameon=True, fontsize=10)
    plt.ylim(50, 90)
    plt.tight_layout()
    plt.savefig('fig_noise.png', dpi=300)
    plt.savefig('fig_noise.pdf', dpi=300)
    plt.close()
    print("Saved fig_noise.png and fig_noise.pdf")

def main():
    sweep_data = load_sweep_data()
    temp_data = load_temp_data()
    
    plot_batch_size_sweep(sweep_data)
    plot_temperature_sweep(temp_data)
    plot_noise_robustness(sweep_data)

if __name__ == '__main__':
    main()
