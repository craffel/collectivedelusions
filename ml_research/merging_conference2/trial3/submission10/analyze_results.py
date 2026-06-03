import json
import numpy as np

def analyze():
    with open("results.json", "r") as f:
        results = json.load(f)
        
    # Get all methods evaluated (keys of the first seed)
    first_seed = list(results.keys())[0]
    methods = list(results[first_seed].keys())
    
    print("=========================================================================")
    print("                      SMACS EXPERIMENTAL RESULTS                         ")
    print("=========================================================================")
    print(f"{'Method/Config':<35} | {'MNIST Mean (±Std)':<20} | {'F-MNIST Mean (±Std)':<20} | {'CIFAR10 Mean (±Std)':<20} | {'Average Mean (±Std)':<20}")
    print("-" * 122)
    
    for method in methods:
        mnist_vals = []
        fmnist_vals = []
        cifar_vals = []
        avg_vals = []
        
        for seed in results.keys():
            res = results[seed][method]
            # Some methods might not have all task fields
            if 'mnist' in res:
                mnist_vals.append(res['mnist'])
            if 'fmnist' in res:
                fmnist_vals.append(res['fmnist'])
            if 'cifar10' in res:
                cifar_vals.append(res['cifar10'])
            if 'average' in res:
                avg_vals.append(res['average'])
                
        # Compute mean and std
        def format_stat(vals):
            if len(vals) == 0:
                return "N/A"
            mean = np.mean(vals)
            std = np.std(vals)
            return f"{mean:.2f}% (±{std:.2f})"
            
        print(f"{method:<35} | {format_stat(mnist_vals):<20} | {format_stat(fmnist_vals):<20} | {format_stat(cifar_vals):<20} | {format_stat(avg_vals):<20}")
        
if __name__ == "__main__":
    analyze()
