import os
import subprocess
import json

# Define the hyperparameter sweeps
methods = {
    'arithmetic': {
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    },
    'ties': {
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'ties_fraction': [0.2]
    },
    'dare': {
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'dare_drop_rate': [0.9]
    },
    'orthomerge': {
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    },
    'saim': {
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'gamma': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    },
    'dor_saim': {
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'gamma': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    }
}

all_results = []

print("Starting model merging sweep...")

for method, params in methods.items():
    if method in ['arithmetic', 'orthomerge']:
        for alpha in params['alpha']:
            cmd = f"python merge_models.py --method {method} --alpha {alpha}"
            print(f"Running: {cmd}")
            subprocess.run(cmd, shell=True)
            
            # Read results
            res_path = f"merged_models/resnet18_{method}_results.txt"
            if os.path.exists(res_path):
                with open(res_path, 'r') as f:
                    lines = f.readlines()
                res_dict = {'method': method, 'alpha': alpha, 'gamma': 'N/A'}
                for line in lines[3:]: # Skip first 3 metadata lines
                    if ':' in line:
                        k, v = line.split(':')
                        res_dict[k.strip()] = float(v.replace('%', '').strip())
                all_results.append(res_dict)
                
    elif method == 'ties':
        for alpha in params['alpha']:
            for fraction in params['ties_fraction']:
                cmd = f"python merge_models.py --method {method} --alpha {alpha} --ties_fraction {fraction}"
                print(f"Running: {cmd}")
                subprocess.run(cmd, shell=True)
                
                res_path = f"merged_models/resnet18_{method}_results.txt"
                if os.path.exists(res_path):
                    with open(res_path, 'r') as f:
                        lines = f.readlines()
                    res_dict = {'method': f"ties (p={fraction})", 'alpha': alpha, 'gamma': 'N/A'}
                    for line in lines[3:]:
                        if ':' in line:
                            k, v = line.split(':')
                            res_dict[k.strip()] = float(v.replace('%', '').strip())
                    all_results.append(res_dict)
                    
    elif method == 'dare':
        for alpha in params['alpha']:
            for drop in params['dare_drop_rate']:
                cmd = f"python merge_models.py --method {method} --alpha {alpha} --dare_drop_rate {drop}"
                print(f"Running: {cmd}")
                subprocess.run(cmd, shell=True)
                
                res_path = f"merged_models/resnet18_{method}_results.txt"
                if os.path.exists(res_path):
                    with open(res_path, 'r') as f:
                        lines = f.readlines()
                    res_dict = {'method': f"dare (d={drop})", 'alpha': alpha, 'gamma': 'N/A'}
                    for line in lines[3:]:
                        if ':' in line:
                            k, v = line.split(':')
                            res_dict[k.strip()] = float(v.replace('%', '').strip())
                    all_results.append(res_dict)
                    
    elif method in ['saim', 'dor_saim']:
        for alpha in params['alpha']:
            for gamma in params['gamma']:
                cmd = f"python merge_models.py --method {method} --alpha {alpha} --gamma {gamma}"
                print(f"Running: {cmd}")
                subprocess.run(cmd, shell=True)
                
                res_path = f"merged_models/resnet18_{method}_results.txt"
                if os.path.exists(res_path):
                    with open(res_path, 'r') as f:
                        lines = f.readlines()
                    res_dict = {'method': method, 'alpha': alpha, 'gamma': gamma}
                    for line in lines[3:]:
                        if ':' in line:
                            k, v = line.split(':')
                            res_dict[k.strip()] = float(v.replace('%', '').strip())
                    all_results.append(res_dict)

# Save all results to a JSON file
with open("sweep_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

# Compile results into a markdown table
markdown_lines = [
    "# Model Merging Hyperparameter Sweep Results",
    "",
    "| Method | Alpha | Gamma | CIFAR-10 Acc | SVHN Acc | FMNIST Acc | Average Acc |",
    "|---|---|---|---|---|---|---||",
]

for res in all_results:
    m = res.get('method', 'N/A')
    a = res.get('alpha', 'N/A')
    g = res.get('gamma', 'N/A')
    c10 = res.get('cifar10', 0.0)
    svhn = res.get('svhn', 0.0)
    fmn = res.get('fmnist', 0.0)
    avg = res.get('Average', 0.0)
    markdown_lines.append(f"| {m} | {a} | {g} | {c10:.2f}% | {svhn:.2f}% | {fmn:.2f}% | {avg:.2f}% |")

with open("sweep_summary.md", "w") as f:
    f.write("\n".join(markdown_lines))

print("Sweeps completed and results compiled!")
