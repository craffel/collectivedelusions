import os
import sys
import torch
import torch.nn as nn

# Add src to python path
sys.path.append(os.path.abspath("src"))

# Disable cuDNN due to local cluster compatibility issues
torch.backends.cudnn.enabled = False

from data import get_tta_streams
from models import MultiTaskResNet18, get_base_state_dict
from tta import run_tta
from main import load_or_train_experts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 1. Load experts
    expert_encoders, expert_heads = load_or_train_experts(device)
    
    # 2. Extract base parameters and compute task vectors
    base_model = MultiTaskResNet18(num_tasks=3, num_classes=10).to(device)
    base_params = get_base_state_dict(base_model)
    
    task_vectors = []
    for k in range(3):
        tv = {}
        for name, base_param in base_params.items():
            tv[name] = (expert_encoders[k][name] - base_param).clone().detach()
        task_vectors.append(tv)
        
    environments = ['clean', 'noise', 'blur', 'contrast', 'rotation']
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
    
    # Pre-load all streams to save time and reduce I/O
    print("Pre-loading all TTA streams and evaluation loaders to RAM...")
    loaded_streams = {}
    for env in environments:
        loaded_streams[env] = get_tta_streams(img_size=32, corruption=env, num_samples_per_task=512, batch_size=32, stream_type='sequential')
    print("Pre-loading complete!\n")
    
    sweep_results = {}
    
    print("Starting alpha_sym sweep for S-CASS (FG-CASS_sym_ALR)...")
    
    for alpha in alphas:
        print(f"\n--- Running sweep for alpha_sym = {alpha} ---")
        os.environ['ALPHA_SYM'] = str(alpha)
        
        env_accs = {}
        for env in environments:
            tta_stream, eval_loaders = loaded_streams[env]
            # Use 1.0 for head anchoring
            lambdas, task_accs, avg_acc = run_tta(
                'FG-CASS_sym_ALR', base_params, task_vectors, expert_encoders, expert_heads,
                tta_stream, eval_loaders, device, mu=1.0
            )
            env_accs[env] = avg_acc
            print(f"  [{env.upper()}] Avg Acc: {avg_acc:.2f}%")
            
        ood_mean = sum([env_accs[env] for env in ['noise', 'blur', 'contrast', 'rotation']]) / 4.0
        sweep_results[alpha] = {
            'accs': env_accs,
            'ood_mean': ood_mean
        }
        print(f"  -> CLEAN: {env_accs['clean']:.2f}%, OOD MEAN: {ood_mean:.2f}%")
        
    # Print results summary table
    print("\n\n=== SWEEP RESULTS SUMMARY ===")
    print("| alpha_sym | Clean | Noise | Blur | Contrast | Rotation | OOD Mean |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for alpha in alphas:
        res = sweep_results[alpha]
        accs = res['accs']
        print(f"| {alpha} | {accs['clean']:.2f} | {accs['noise']:.2f} | {accs['blur']:.2f} | {accs['contrast']:.2f} | {accs['rotation']:.2f} | **{res['ood_mean']:.2f}** |")

if __name__ == "__main__":
    main()
