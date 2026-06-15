import os
import sys

# Prevent OpenBLAS/MKL/OMP multi-threading deadlocks on large multi-core HPC nodes
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import torch
import copy
import gc
from tqdm import tqdm

# Add task_vectors and local_packages to path
sys.path.append(os.path.abspath('task_vectors'))
sys.path.append(os.path.abspath('local_packages'))

from src.eval import eval_single_dataset
from src.args import parse_arguments
from orim_engine import merge_orim_state_dicts, merge_ties_state_dicts

def main():
    # 1. Config and arguments
    model_name = 'ViT-B-32'
    save_dir = f'task_vectors_checkpoints/{model_name}'
    pretrained_path = f'{save_dir}/zeroshot.pt'
    
    datasets = ['MNIST', 'SVHN', 'GTSRB']
    
    task_paths_dict = {
        dataset: f'{save_dir}/{dataset}/finetuned.pt'
        for dataset in datasets
    }
    
    # Parse standard arguments using src.args
    sys.argv = [
        'run_orim_sweeps.py',
        '--data-location', 'data',
        '--model', model_name,
        '--save', save_dir,
    ]
    args = parse_arguments()
    args.batch_size = 256 # Higher batch size for GPU speedup
    args.num_workers = 0 # Prevent OOM and speed up data loading safely by avoiding worker process spawning
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(f"Using device: {device}")
    
    # Load pretrained model and state dict once
    pretrained_model = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    pretrained_sd = pretrained_model.state_dict()
    
    # Create a single evaluation model copy on device to avoid memory-heavy deepcopies inside loop
    evaluation_model = copy.deepcopy(pretrained_model).to(device)
    evaluation_model.eval()
    
    # Load task state dicts once
    task_sds = []
    for dataset in datasets:
        path = task_paths_dict[dataset]
        print(f"Loading task checkpoint for {dataset} from {path}...")
        task_model = torch.load(path, map_location='cpu', weights_only=False)
        task_sds.append(task_model.state_dict())
        
    results = []
    
    # Check if there are existing results to resume from
    if os.path.exists('sweep_results.json'):
        try:
            with open('sweep_results.json', 'r') as f:
                results = json.load(f)
            print(f"Resuming with {len(results)} existing sweep results.")
        except Exception:
            pass
            
    # Helper to check if a configuration has already been run
    def is_run(method_name, s, extra_keys={}):
        for r in results:
            if r['method'] == method_name and abs(r['scaling'] - s) < 1e-5:
                match = True
                for k, v in extra_keys.items():
                    if r.get(k) != v:
                        match = False
                        break
                if match:
                    return True
        return False
    
    # We define a function to scale and evaluate a merged model state dict
    def evaluate_merged_sd(merged_sd, s, method_name, config_info):
        # Check if already computed
        if is_run(method_name, s, config_info):
            print(f"Config {method_name} s={s} {config_info} already evaluated. Skipping.")
            return
            
        print(f"\nEvaluating {method_name} with scaling s={s}...")
        
        # Scale the merged state dict: W_scaled = W0 + s * (W_merged - W0)
        scaled_sd = {}
        for key in pretrained_sd.keys():
            W0 = pretrained_sd[key].to(device)
            W_merged = merged_sd[key].to(device)
            W_scaled = W0 + s * (W_merged - W0)
            scaled_sd[key] = W_scaled.to(dtype=pretrained_sd[key].dtype)
            
        # Apply scaled state dict to image encoder in-place
        evaluation_model.load_state_dict(scaled_sd, strict=False)
        
        # Evaluate on all datasets
        dataset_accuracies = {}
        total_acc = 0.0
        for dataset in datasets:
            try:
                metrics = eval_single_dataset(evaluation_model, dataset, args)
                acc = metrics['top1']
                dataset_accuracies[dataset] = acc
                total_acc += acc
            except Exception as e:
                print(f"Error evaluating on {dataset}: {e}")
                dataset_accuracies[dataset] = 0.0
                
        avg_acc = total_acc / len(datasets)
        print(f"Average Accuracy on 3 datasets: {100*avg_acc:.2f}%")
        
        res = {
            'method': method_name,
            'scaling': s,
            'dataset_accuracies': dataset_accuracies,
            'average_accuracy': avg_acc,
            **config_info
        }
        results.append(res)
        
        # Append immediately to a json file to avoid losing results
        with open('sweep_results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        # Clear CUDA memory
        del scaled_sd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
            
    # 2. RUN EXPERIMENTS AND SWEEPS
    
    # ================= BASELINE 1: Task Arithmetic (Standard Average) =================
    print("\n--- Running Baseline: Task Arithmetic ---")
    ta_sd = merge_orim_state_dicts(pretrained_sd, task_sds, gamma=1.0, decoupling_mode='global', use_decoupling=False)
    
    # Sweep scaling factor s for Task Arithmetic
    for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        evaluate_merged_sd(ta_sd, s, "Task Arithmetic", {'gamma': 1.0, 'decoupling': 'none', 'use_decoupling': False})
    del ta_sd
    gc.collect()
        
    # ================= BASELINE 2: Pure Isotropic Merging (SAIM) =================
    print("\n--- Running Baseline: Pure Isotropic Merging (SAIM) ---")
    for gamma in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        iso_sd = merge_orim_state_dicts(pretrained_sd, task_sds, gamma=gamma, decoupling_mode='global', use_decoupling=False)
        for s in [0.3, 0.4, 0.5, 0.6, 0.7]:
            evaluate_merged_sd(iso_sd, s, "Pure Isotropic Merging", {'gamma': gamma, 'decoupling': 'none', 'use_decoupling': False})
        del iso_sd
        gc.collect()

    # ================= BASELINE 3: OrthoMerge (Yang et al., 2026 - gamma=1.0) =================
    print("\n--- Running Baseline: OrthoMerge ---")
    for mode in ['global', 'conflict_aware']:
        om_sd = merge_orim_state_dicts(pretrained_sd, task_sds, gamma=1.0, decoupling_mode=mode, use_decoupling=True)
        for s in [0.3, 0.4, 0.5, 0.6, 0.7]:
            evaluate_merged_sd(om_sd, s, f"OrthoMerge ({mode})", {'gamma': 1.0, 'decoupling': mode, 'use_decoupling': True})
        del om_sd
        gc.collect()

    # ================= BASELINE 4: TIES-Merging =================
    print("\n--- Running Baseline: TIES-Merging ---")
    for k_val in [20]:
        ties_sd = merge_ties_state_dicts(pretrained_sd, task_sds, k=k_val)
        for s in [0.3, 0.4, 0.5, 0.6, 0.7]:
            evaluate_merged_sd(ties_sd, s, f"TIES-Merging (k={k_val})", {'gamma': 1.0, 'decoupling': 'ties', 'use_decoupling': False, 'k': k_val})
        del ties_sd
        gc.collect()

    # ================= OUR METHOD: ORIM =================
    print("\n--- Running Our Method: ORIM ---")
    for mode in ['global', 'conflict_aware']:
        for gamma in [0.1, 0.5, 0.9]:
            orim_sd = merge_orim_state_dicts(pretrained_sd, task_sds, gamma=gamma, decoupling_mode=mode, use_decoupling=True)
            for s in [0.3, 0.4, 0.5, 0.6, 0.7]:
                evaluate_merged_sd(orim_sd, s, f"ORIM ({mode})", {'gamma': gamma, 'decoupling': mode, 'use_decoupling': True})
            del orim_sd
            gc.collect()

    print("\nAll sweeps completed! Results saved to sweep_results.json")

if __name__ == '__main__':
    main()
