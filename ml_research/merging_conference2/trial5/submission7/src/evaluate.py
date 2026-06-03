import argparse
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from dataset import get_datasets, get_dataloaders
from models import create_expert_model, get_base_model, load_checkpoint
from calibrate import collect_target_stats, calibrate_sp_taac, apply_spectral_calibration

def evaluate_on_task(backbone_model, expert_head, test_loader, device='cuda'):
    # Combine merged backbone with task-specific expert head
    model = copy.deepcopy(backbone_model)
    model.fc = copy.deepcopy(expert_head)
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100. * correct / total

def merge_backbones(expert_models, merge_mode='wa', lambda_coeff=0.3, device='cuda'):
    # Load base pre-trained ImageNet model
    base_model = get_base_model()
    base_state = base_model.state_dict()
    
    merged_model = create_expert_model(num_classes=10)
    merged_state = merged_model.state_dict()
    
    # Identify backbone keys
    backbone_keys = [k for k in merged_state.keys() if 'fc' not in k]
    
    expert_states = [model.state_dict() for model in expert_models.values()]
    
    if merge_mode == 'wa':
        # Weight Averaging
        for k in backbone_keys:
            if expert_states[0][k].is_floating_point():
                merged_state[k] = torch.mean(torch.stack([es[k] for es in expert_states]), dim=0)
            else:
                merged_state[k] = expert_states[0][k]
    elif merge_mode == 'ta':
        # Task Arithmetic: W_merged = W_0 + lambda * sum(W_t - W_0)
        for k in backbone_keys:
            if expert_states[0][k].is_floating_point():
                task_vectors = [es[k].cpu() - base_state[k].cpu() for es in expert_states]
                sum_task_vectors = torch.sum(torch.stack(task_vectors), dim=0).to(device)
                merged_state[k] = base_state[k].to(device) + lambda_coeff * sum_task_vectors
            else:
                merged_state[k] = expert_states[0][k]
                
    merged_model.load_state_dict(merged_state)
    return merged_model

def run_evaluation(device='cuda'):
    print("=== Loading Datasets ===")
    splits = get_datasets()
    loaders = get_dataloaders(splits, batch_size=128)
    
    # Create Joint Calibration Dataset loader (384 samples)
    joint_calib_dataset = ConcatDataset([
        splits['mnist']['calib'],
        splits['fmnist']['calib'],
        splits['cifar10']['calib']
    ])
    joint_calib_loader = DataLoader(joint_calib_dataset, batch_size=128, shuffle=False, num_workers=2)
    print(f"Loaded {len(joint_calib_dataset)} samples for joint calibration.")
    
    # Load Experts
    print("=== Loading Expert Models ===")
    expert_models = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        model = create_expert_model(num_classes=10)
        model = load_checkpoint(model, f"checkpoints/{task}_expert.pt", device=device)
        expert_models[task] = model.to(device)
        
    # Get heads
    expert_heads = {task: model.fc for task, model in expert_models.items()}
    
    # Evaluate Experts (Oracle Baseline)
    print("=== Evaluating Oracle (Experts) ===")
    oracle_accs = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        acc = evaluate_on_task(expert_models[task], expert_heads[task], loaders[task]['test'], device=device)
        oracle_accs[task] = acc
        print(f"Oracle Expert Accuracy on {task}: {acc:.2f}%")
    oracle_avg = sum(oracle_accs.values()) / 3
    print(f"Oracle Average: {oracle_avg:.2f}%\n")
    
    # Collect Target Statistics for Calibration
    print("=== Collecting Target Statistics ===")
    target_stats = collect_target_stats(expert_models, {task: loaders[task]['calib'] for task in ['mnist', 'fmnist', 'cifar10']}, device=device)
    
    # Merge Modes to evaluate
    merge_modes = ['wa', 'ta']
    results = {}
    
    for mode in merge_modes:
        print(f"\n======================================")
        print(f"Evaluating Merge Mode: {mode.upper()}")
        print(f"======================================")
        
        # Merge the models
        merged_model_base = merge_backbones(expert_models, merge_mode=mode, lambda_coeff=0.3, device=device)
        
        # 1. Uncalibrated Baseline
        print("\n--- 1. Uncalibrated Baseline ---")
        uncal_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            acc = evaluate_on_task(merged_model_base, expert_heads[task], loaders[task]['test'], device=device)
            uncal_accs[task] = acc
            print(f"Uncalibrated Baseline on {task}: {acc:.2f}%")
        uncal_avg = sum(uncal_accs.values()) / 3
        print(f"Average Uncalibrated: {uncal_avg:.2f}%")
        results[(mode, 'uncalibrated')] = {**uncal_accs, 'average': uncal_avg}
        
        # 2. SP-TAAC Calibration (In-place Weight/Bias Fusion)
        print("\n--- 2. SP-TAAC (In-place Fusion) ---")
        model_sp = copy.deepcopy(merged_model_base)
        model_sp = calibrate_sp_taac(model_sp, target_stats, joint_calib_loader, device=device)
        sp_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            acc = evaluate_on_task(model_sp, expert_heads[task], loaders[task]['test'], device=device)
            sp_accs[task] = acc
            print(f"SP-TAAC on {task}: {acc:.2f}%")
        sp_avg = sum(sp_accs.values()) / 3
        print(f"Average SP-TAAC: {sp_avg:.2f}%")
        results[(mode, 'sp-taac')] = {**sp_accs, 'average': sp_avg}
        
        # 3. FDSA Calibration (Active Hook)
        print("\n--- 3. FDSA (Pointwise Clamped Spectral Hook) ---")
        model_fdsa = copy.deepcopy(merged_model_base)
        model_fdsa, fdsa_hooks = apply_spectral_calibration(model_fdsa, target_stats, joint_calib_loader, method='fdsa', device=device)
        fdsa_accs = {}
        for task in ['mnist', 'fmnist', 'cifar10']:
            acc = evaluate_on_task(model_fdsa, expert_heads[task], loaders[task]['test'], device=device)
            fdsa_accs[task] = acc
            print(f"FDSA on {task}: {acc:.2f}%")
        fdsa_avg = sum(fdsa_accs.values()) / 3
        print(f"Average FDSA: {fdsa_avg:.2f}%")
        results[(mode, 'fdsa')] = {**fdsa_accs, 'average': fdsa_avg}
        # Clean up hooks
        for h in fdsa_hooks:
            h.remove()
            
        # 4. WRSA Calibration (Sweep c parameter)
        for c in [0.01, 0.05, 0.1, 0.2]:
            print(f"\n--- 4. WRSA (Wiener Spectral Hook, c={c}) ---")
            model_wrsa = copy.deepcopy(merged_model_base)
            model_wrsa, wrsa_hooks = apply_spectral_calibration(model_wrsa, target_stats, joint_calib_loader, method='wrsa', c_val=c, device=device)
            wrsa_accs = {}
            for task in ['mnist', 'fmnist', 'cifar10']:
                acc = evaluate_on_task(model_wrsa, expert_heads[task], loaders[task]['test'], device=device)
                wrsa_accs[task] = acc
                print(f"WRSA (c={c}) on {task}: {acc:.2f}%")
            wrsa_avg = sum(wrsa_accs.values()) / 3
            print(f"Average WRSA (c={c}): {wrsa_avg:.2f}%")
            results[(mode, f'wrsa_c{c}')] = {**wrsa_accs, 'average': wrsa_avg}
            # Clean up hooks
            for h in wrsa_hooks:
                h.remove()
                
    # Final Summary Table
    print("\n\n=========================================================================")
    print("                           FINAL PERFORMANCE SUMMARY")
    print("=========================================================================")
    print(f"{'Merge Mode':<12} | {'Calibration':<15} | {'MNIST':<8} | {'F-MNIST':<8} | {'CIFAR-10':<8} | {'Average':<8}")
    print("-" * 73)
    
    for mode in merge_modes:
        for calib in ['uncalibrated', 'sp-taac', 'fdsa', 'wrsa_c0.01', 'wrsa_c0.05', 'wrsa_c0.1', 'wrsa_c0.2']:
            r = results[(mode, calib)]
            print(f"{mode.upper():<12} | {calib:<15} | {r['mnist']:<8.2f} | {r['fmnist']:<8.2f} | {r['cifar10']:<8.2f} | {r['average']:<8.2f}")
    print("=========================================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Model Merging and Calibration')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    args = parser.parse_args()
    
    run_evaluation(device=args.device)
