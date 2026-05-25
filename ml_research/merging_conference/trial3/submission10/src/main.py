import os
import torch
import torch.nn as nn

# Disable cuDNN due to local cluster compatibility issues
torch.backends.cudnn.enabled = False

from data import get_expert_dataloaders, get_tta_streams
from models import MultiTaskResNet18, get_base_state_dict
from tta import run_tta

def load_or_train_experts(device):
    # Check if expert checkpoints exist
    experts_exist = True
    for i in range(3):
        if not (os.path.exists(f"checkpoints/expert_{i}_encoder.pt") and os.path.exists(f"checkpoints/expert_{i}_head.pt")):
            experts_exist = False
            break
            
    if not experts_exist:
        print("Expert checkpoints not found. Training experts from scratch...")
        # Run training
        from train_experts import train_expert
        mnist_loader, fmnist_loader, kmnist_loader = get_expert_dataloaders(img_size=32, batch_size=128, num_train_samples=10000)
        train_expert(0, mnist_loader, device, num_epochs=3, lr=1e-4)
        train_expert(1, fmnist_loader, device, num_epochs=3, lr=1e-4)
        train_expert(2, kmnist_loader, device, num_epochs=3, lr=1e-4)
    else:
        print("Expert checkpoints found. Loading checkpoints...")
        
    # Load expert weights
    expert_encoders = []
    expert_heads = []
    
    # We use a dummy model to instantiate the correct head layer shapes
    dummy_model = MultiTaskResNet18(num_tasks=3, num_classes=10)
    
    for i in range(3):
        enc_sd = torch.load(f"checkpoints/expert_{i}_encoder.pt", map_location=device)
        head_sd = torch.load(f"checkpoints/expert_{i}_head.pt", map_location=device)
        
        expert_encoders.append(enc_sd)
        
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(head_sd)
        expert_heads.append(head)
        
    return expert_encoders, expert_heads

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 1. Load or train experts
    expert_encoders, expert_heads = load_or_train_experts(device)
    
    # 2. Extract base pre-trained encoder parameters and compute task vectors
    base_model = MultiTaskResNet18(num_tasks=3, num_classes=10).to(device)
    base_params = get_base_state_dict(base_model)
    
    task_vectors = []
    for k in range(3):
        tv = {}
        for name, base_param in base_params.items():
            tv[name] = (expert_encoders[k][name] - base_param).clone().detach()
        task_vectors.append(tv)
        
    # List of environments and methods
    environments = ['clean', 'noise', 'blur', 'contrast', 'rotation']
    methods = [
        'TaskArithmetic',
        'AdaMerging',
        'SyMerge',
        'SAT-SyMerge',
        'ASAM-SyMerge',
        'SBF-SAT-SyMerge',
        'FG-CASS',
        'FG-CASS_sym_ALR',
        'FG-CASS_sc_ALR',
        'FG-CASS_BACR',
        'FG-CASS_BACR_sym_ALR',
        'FG-CASS_no_ALR',
        'FG-CASS_no_APR',
        'FG-CASS_no_EHA',
        'FG-CASS_no_TDP'
    ]
    
    results = {}
    
    for env in environments:
        print(f"\n==================================================")
        print(f"Running TTA Stream Environment: {env.upper()}")
        print(f"==================================================")
        
        # Get sequential TTA stream (severe non-stationarity) and full test evaluation loaders
        tta_stream, eval_loaders = get_tta_streams(img_size=32, corruption=env, num_samples_per_task=512, batch_size=32, stream_type='sequential')
        
        results[env] = {}
        for method in methods:
            print(f"Running adaptation method: {method} ...")
            mu_val = 0.1 if (method.startswith('FG-CASS') and 'no_EHA' not in method) else 0.0
            lambdas, task_accs, avg_acc = run_tta(
                method, base_params, task_vectors, expert_encoders, expert_heads,
                tta_stream, eval_loaders, device, mu=mu_val
            )
            results[env][method] = {
                'lambdas': lambdas,
                'task_accs': task_accs,
                'avg_acc': avg_acc
            }
            print(f"[{method}] final lambdas: {list(map(lambda x: round(x, 3), lambdas))}, task accs: {list(map(lambda x: round(x, 2), task_accs))}, Avg Acc: {avg_acc:.2f}%")
            
    # 3. Print a beautiful summary markdown table
    print("\n\n==================================================")
    print("FINAL EXPERIMENTAL RESULTS SUMMARY")
    print("==================================================")
    
    # We output a table for each environment and a summary table of Average Acc across all environments
    summary_lines = []
    summary_lines.append("# Experimental Results Comparison\n")
    
    for env in environments:
        summary_lines.append(f"### Environment: {env.upper()}\n")
        summary_lines.append("| Method | MNIST Acc (%) | F-MNIST Acc (%) | K-MNIST Acc (%) | Avg Acc (%) | Final Lambdas |")
        summary_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for method in methods:
            r = results[env][method]
            lambdas_str = f"[{r['lambdas'][0]:.2f}, {r['lambdas'][1]:.2f}, {r['lambdas'][2]:.2f}]"
            summary_lines.append(f"| {method} | {r['task_accs'][0]:.2f} | {r['task_accs'][1]:.2f} | {r['task_accs'][2]:.2f} | **{r['avg_acc']:.2f}** | {lambdas_str} |")
        summary_lines.append("\n")
        
    # Overall summary table
    summary_lines.append("### Overall Comparison across Environments (Avg Acc %)\n")
    summary_lines.append("| Method | Clean | Noise | Blur | Contrast | Rotation | OOD Mean |")
    summary_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for method in methods:
        clean_acc = results['clean'][method]['avg_acc']
        noise_acc = results['noise'][method]['avg_acc']
        blur_acc = results['blur'][method]['avg_acc']
        contrast_acc = results['contrast'][method]['avg_acc']
        rotation_acc = results['rotation'][method]['avg_acc']
        
        ood_accs = [noise_acc, blur_acc, contrast_acc, rotation_acc]
        ood_mean = sum(ood_accs) / len(ood_accs)
        
        summary_lines.append(f"| {method} | {clean_acc:.2f} | {noise_acc:.2f} | {blur_acc:.2f} | {contrast_acc:.2f} | {rotation_acc:.2f} | **{ood_mean:.2f}** |")
        
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    # Save the results to progress.md and a separate file
    with open("results.md", "w") as f:
        f.write(summary_text)
        
    with open("progress.md", "a") as f:
        f.write("\n\n## Phase 2: Experimental Results\n\n")
        f.write(summary_text)
        
    print("\nResults successfully saved to results.md and appended to progress.md.")

if __name__ == "__main__":
    main()
