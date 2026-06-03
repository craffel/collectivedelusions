import os
import argparse
import subprocess
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--alphas', type=float, nargs='+', default=None)
    parser.add_argument('--suffix', type=str, default=None)
    args = parser.parse_args()
    
    # Define hyperparameter sweeps
    params_dict = {
        'arithmetic': {'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]},
        'ties': {'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 'ties_fraction': [0.2]},
        'dare': {'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 'dare_drop_rate': [0.9]},
        'orthomerge': {'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]},
        'saim': {'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 'gamma': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]},
        'dor_saim': {'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 'gamma': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}
    }
    
    method = args.method
    arch = args.arch
    params = params_dict[method]
    
    if args.alphas is not None:
        params['alpha'] = args.alphas
    
    results = []
    
    if method in ['arithmetic', 'orthomerge']:
        for alpha in params['alpha']:
            cmd = f"python merge_models.py --arch {arch} --method {method} --alpha {alpha}"
            print(f"Running: {cmd}")
            subprocess.run(cmd, shell=True)
            
            res_path = f"merged_models/{arch}_{method}_results.txt"
            if os.path.exists(res_path):
                with open(res_path, 'r') as f:
                    lines = f.readlines()
                res_dict = {'arch': arch, 'method': method, 'alpha': alpha, 'gamma': 'N/A'}
                for line in lines[3:]:
                    if ':' in line:
                        k, v = line.split(':')
                        res_dict[k.strip()] = float(v.replace('%', '').strip())
                results.append(res_dict)
                
    elif method == 'ties':
        for alpha in params['alpha']:
            for fraction in params['ties_fraction']:
                cmd = f"python merge_models.py --arch {arch} --method {method} --alpha {alpha} --ties_fraction {fraction}"
                print(f"Running: {cmd}")
                subprocess.run(cmd, shell=True)
                
                res_path = f"merged_models/{arch}_{method}_results.txt"
                if os.path.exists(res_path):
                    with open(res_path, 'r') as f:
                        lines = f.readlines()
                    res_dict = {'arch': arch, 'method': f"ties (p={fraction})", 'alpha': alpha, 'gamma': 'N/A'}
                    for line in lines[3:]:
                        if ':' in line:
                            k, v = line.split(':')
                            res_dict[k.strip()] = float(v.replace('%', '').strip())
                    results.append(res_dict)
                    
    elif method == 'dare':
        for alpha in params['alpha']:
            for drop in params['dare_drop_rate']:
                cmd = f"python merge_models.py --arch {arch} --method {method} --alpha {alpha} --dare_drop_rate {drop}"
                print(f"Running: {cmd}")
                subprocess.run(cmd, shell=True)
                
                res_path = f"merged_models/{arch}_{method}_results.txt"
                if os.path.exists(res_path):
                    with open(res_path, 'r') as f:
                        lines = f.readlines()
                    res_dict = {'arch': arch, 'method': f"dare (d={drop})", 'alpha': alpha, 'gamma': 'N/A'}
                    for line in lines[3:]:
                        if ':' in line:
                            k, v = line.split(':')
                            res_dict[k.strip()] = float(v.replace('%', '').strip())
                    results.append(res_dict)
                    
    elif method in ['saim', 'dor_saim']:
        for alpha in params['alpha']:
            for gamma in params['gamma']:
                cmd = f"python merge_models.py --arch {arch} --method {method} --alpha {alpha} --gamma {gamma}"
                print(f"Running: {cmd}")
                subprocess.run(cmd, shell=True)
                
                res_path = f"merged_models/{arch}_{method}_results.txt"
                if os.path.exists(res_path):
                    with open(res_path, 'r') as f:
                        lines = f.readlines()
                    res_dict = {'arch': arch, 'method': method, 'alpha': alpha, 'gamma': gamma}
                    for line in lines[3:]:
                        if ':' in line:
                            k, v = line.split(':')
                            res_dict[k.strip()] = float(v.replace('%', '').strip())
                    results.append(res_dict)
                    
    # Save partial results
    os.makedirs("sweep_results", exist_ok=True)
    if args.suffix:
        out_path = f"sweep_results/{arch}_{method}_results_{args.suffix}.json"
    else:
        out_path = f"sweep_results/{arch}_{method}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved partial results to {out_path}")

if __name__ == "__main__":
    main()
