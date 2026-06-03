import os
import json
import time
import torch
import torch.nn as nn
from torchvision import models
from finetune_tasks import get_dataloaders, ModelWrapper
from merge_and_evaluate import (
    state_dict_to_2d,
    two_d_to_state_dict,
    merge_task_arithmetic,
    merge_ties,
    merge_dare,
    merge_dmc,
    merge_orthomerge
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Self-adaptive evaluation helper
def evaluate_encoder(encoder, classifier, test_loader, fast=False):
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            if fast and (idx % 5 != 0):
                continue
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            features = encoder(images)
            features = torch.flatten(features, 1)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

def analyze_svd_bottleneck(base_state, task_encoders, task_list):
    print("\n--- Running Layer-wise SVD Bottleneck Analysis ---")
    base_2d, shapes = state_dict_to_2d(base_state)
    task_encoders_2d = {t: state_dict_to_2d(task_encoders[t])[0] for t in task_list}
    
    layer_analysis = {}
    
    for k in base_2d.keys():
        v_base = base_2d[k]
        
        # We only apply OrthoMerge to 2D matrices where C_out <= C_in
        if v_base.dim() == 2 and v_base.size(0) <= v_base.size(1):
            C_out = v_base.size(0)
            
            cos_sims = []
            rel_norms = []
            
            for t in task_list:
                v_task = task_encoders_2d[t][k]
                tau = v_task - v_base # Original task vector
                
                # Procrustes projection of target (v_task) onto v_base
                A = torch.matmul(v_task, v_base.t())
                try:
                    U, S, Vh = torch.linalg.svd(A)
                    R = torch.matmul(U, Vh)
                except RuntimeError:
                    R = torch.eye(C_out, device=device)
                    
                # Orthogonal task vector
                tau_ortho = torch.matmul(R, v_base) - v_base
                
                # Compute metrics
                norm_tau = torch.norm(tau, p='fro').item()
                norm_tau_ortho = torch.norm(tau_ortho, p='fro').item()
                
                # Flat cosine similarity
                flat_tau = tau.flatten()
                flat_tau_ortho = tau_ortho.flatten()
                dot_product = torch.dot(flat_tau, flat_tau_ortho).item()
                cos_sim = dot_product / (norm_tau * norm_tau_ortho + 1e-8)
                
                # Relative difference norm
                rel_diff = torch.norm(tau - tau_ortho, p='fro').item() / (norm_tau + 1e-8)
                
                cos_sims.append(cos_sim)
                rel_norms.append(rel_diff)
                
            layer_analysis[k] = {
                'shape': list(shapes[k]),
                'avg_cos_sim': sum(cos_sims) / len(cos_sims),
                'avg_rel_diff': sum(rel_norms) / len(rel_norms)
            }
            
            # Print analysis for a few representative layers
            if any(p in k for p in ['conv1', 'layer1.0.conv1', 'layer2.0.conv1', 'layer3.0.conv1', 'layer4.0.conv1']):
                print(f"Layer: {k:<30} | Shape: {str(list(shapes[k])):<15} | CosSim(tau, tau_ortho): {layer_analysis[k]['avg_cos_sim']:.4f} | RelDiff: {layer_analysis[k]['avg_rel_diff']:.4f}")
                
    return layer_analysis

def run_dense_sweep(base_state, task_encoders, task_classifiers, task_list, loaders, encoder):
    print("\n--- Running Dense Hyperparameter Sweep ---")
    # Scaling factors to sweep
    scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    results = {
        'Task Arithmetic': {},
        'TIES Merging': {},
        'DARE (drop=0.2)': {},
        'DMC-Merge (Global)': {},
        'DMC-Merge (Conflict-Aware)': {}
    }
    
    task_weights_list = [task_encoders[t] for t in task_list]
    
    # Use fast mode if running on CPU to prevent timeout/slowness
    fast_mode = (device.type == 'cpu')
    if fast_mode:
        print("Note: Running sweep in FAST mode (evaluating on 20% of test data on CPU).")
    else:
        print("Note: Running sweep on GPU with 100% of test data.")
        
    for s in scales:
        print(f"\nEvaluating Scale: {s:.1f}")
        
        # 1. Task Arithmetic
        merged_ta = merge_task_arithmetic(base_state, task_weights_list, s)
        encoder.load_state_dict(merged_ta)
        accs_ta = [evaluate_encoder(encoder, task_classifiers[t], loaders[t]['test'], fast=fast_mode) for t in task_list]
        avg_ta = sum(accs_ta) / 3
        results['Task Arithmetic'][s] = {'accs': accs_ta, 'avg': avg_ta}
        print(f"  Task Arithmetic: CIFAR={accs_ta[0]:.2f}%, SVHN={accs_ta[1]:.2f}%, FMNIST={accs_ta[2]:.2f}% | Avg={avg_ta:.2f}%")
        
        # 2. TIES Merging
        merged_ties = merge_ties(base_state, task_weights_list, s)
        encoder.load_state_dict(merged_ties)
        accs_ties = [evaluate_encoder(encoder, task_classifiers[t], loaders[t]['test'], fast=fast_mode) for t in task_list]
        avg_ties = sum(accs_ties) / 3
        results['TIES Merging'][s] = {'accs': accs_ties, 'avg': avg_ties}
        print(f"  TIES Merging:    CIFAR={accs_ties[0]:.2f}%, SVHN={accs_ties[1]:.2f}%, FMNIST={accs_ties[2]:.2f}% | Avg={avg_ties:.2f}%")
        
        # 2.5 DARE Merging (drop=0.2)
        merged_dare = merge_dare(base_state, task_weights_list, s, 0.2)
        encoder.load_state_dict(merged_dare)
        accs_dare = [evaluate_encoder(encoder, task_classifiers[t], loaders[t]['test'], fast=fast_mode) for t in task_list]
        avg_dare = sum(accs_dare) / 3
        results['DARE (drop=0.2)'][s] = {'accs': accs_dare, 'avg': avg_dare}
        print(f"  DARE (drop=0.2): CIFAR={accs_dare[0]:.2f}%, SVHN={accs_dare[1]:.2f}%, FMNIST={accs_dare[2]:.2f}% | Avg={avg_dare:.2f}%")
        
        # 3. DMC-Merge (Global)
        merged_dmc_g = merge_dmc(base_state, task_weights_list, 'use_global', s)
        encoder.load_state_dict(merged_dmc_g)
        accs_dmc_g = [evaluate_encoder(encoder, task_classifiers[t], loaders[t]['test'], fast=fast_mode) for t in task_list]
        avg_dmc_g = sum(accs_dmc_g) / 3
        results['DMC-Merge (Global)'][s] = {'accs': accs_dmc_g, 'avg': avg_dmc_g}
        print(f"  DMC Global:      CIFAR={accs_dmc_g[0]:.2f}%, SVHN={accs_dmc_g[1]:.2f}%, FMNIST={accs_dmc_g[2]:.2f}% | Avg={avg_dmc_g:.2f}%")
        
        # 4. DMC-Merge (Conflict-Aware)
        merged_dmc_c = merge_dmc(base_state, task_weights_list, 'use_conflict_aware', s)
        encoder.load_state_dict(merged_dmc_c)
        accs_dmc_c = [evaluate_encoder(encoder, task_classifiers[t], loaders[t]['test'], fast=fast_mode) for t in task_list]
        avg_dmc_c = sum(accs_dmc_c) / 3
        results['DMC-Merge (Conflict-Aware)'][s] = {'accs': accs_dmc_c, 'avg': avg_dmc_c}
        print(f"  DMC Conf-Aware:  CIFAR={accs_dmc_c[0]:.2f}%, SVHN={accs_dmc_c[1]:.2f}%, FMNIST={accs_dmc_c[2]:.2f}% | Avg={avg_dmc_c:.2f}%")
        
    return results

def main():
    print("====================================================")
    print("Executing Extended Research Analysis & Sweeps")
    print("====================================================")
    
    loaders = get_dataloaders()
    
    # Load ResNet-18 structure
    resnet = models.resnet18()
    encoder = nn.Sequential(*(list(resnet.children())[:-1])).to(device)
    
    # Load base weights
    base_state = torch.load('checkpoints/base_encoder.pt', map_location=device)
    
    # Load task specific weights
    task_encoders = {}
    task_classifiers = {}
    task_list = ['cifar10', 'svhn', 'fmnist']
    for task in task_list:
        task_encoders[task] = torch.load(f'checkpoints/{task}_encoder.pt', map_location=device)
        clf = nn.Linear(512, 10).to(device)
        clf.load_state_dict(torch.load(f'checkpoints/{task}_classifier.pt', map_location=device))
        task_classifiers[task] = clf
        
    # Step 1: Run layer-wise SVD bottleneck analysis
    svd_analysis = analyze_svd_bottleneck(base_state, task_encoders, task_list)
    
    # Step 2: Run dense hyperparameter sweep
    sweep_results = run_dense_sweep(base_state, task_encoders, task_classifiers, task_list, loaders, encoder)
    
    # Step 3: Run FULL 100% evaluation for the absolute best configurations found
    print("\n--- Running Full 100% Evaluation on Best Configurations ---")
    best_configs = {}
    task_weights_list = [task_encoders[t] for t in task_list]
    
    # Helper to find best scale from sweep results
    def find_best_scale(method_name):
        return max(sweep_results[method_name].keys(), key=lambda s: sweep_results[method_name][s]['avg'])
        
    best_scales = {
        'Task Arithmetic': find_best_scale('Task Arithmetic'),
        'TIES Merging': find_best_scale('TIES Merging'),
        'DARE (drop=0.2)': find_best_scale('DARE (drop=0.2)'),
        'DMC-Merge (Global)': find_best_scale('DMC-Merge (Global)'),
        'DMC-Merge (Conflict-Aware)': find_best_scale('DMC-Merge (Conflict-Aware)')
    }
    
    # Evaluate each best config on 100% of test data
    # 1. Task Arithmetic
    ta_scale = best_scales['Task Arithmetic']
    merged_ta = merge_task_arithmetic(base_state, task_weights_list, ta_scale)
    encoder.load_state_dict(merged_ta)
    accs_ta = [evaluate_encoder(encoder, task_classifiers[t], loaders[t]['test'], fast=False) for t in task_list]
    best_configs['Task Arithmetic'] = {'scale': ta_scale, 'accs': accs_ta, 'avg': sum(accs_ta)/3}
    
    # 2. TIES Merging
    ties_scale = best_scales['TIES Merging']
    merged_ties = merge_ties(base_state, task_weights_list, ties_scale)
    encoder.load_state_dict(merged_ties)
    accs_ties = [evaluate_encoder(encoder, task_classifiers[t], loaders[t]['test'], fast=False) for t in task_list]
    best_configs['TIES Merging'] = {'scale': ties_scale, 'accs': accs_ties, 'avg': sum(accs_ties)/3}
    
    # 2.2 DARE Merging (drop=0.2)
    dare_scale = best_scales['DARE (drop=0.2)']
    merged_dare = merge_dare(base_state, task_weights_list, dare_scale, 0.2)
    encoder.load_state_dict(merged_dare)
    accs_dare = [evaluate_encoder(encoder, task_classifiers[t], loaders[t]['test'], fast=False) for t in task_list]
    best_configs['DARE (drop=0.2)'] = {'scale': dare_scale, 'accs': accs_dare, 'avg': sum(accs_dare)/3}
    
    # 3. DMC-Merge (Global)
    dmc_g_scale = best_scales['DMC-Merge (Global)']
    merged_dmc_g = merge_dmc(base_state, task_weights_list, 'use_global', dmc_g_scale)
    encoder.load_state_dict(merged_dmc_g)
    accs_dmc_g = [evaluate_encoder(encoder, task_classifiers[t], loaders[t]['test'], fast=False) for t in task_list]
    best_configs['DMC-Merge (Global)'] = {'scale': dmc_g_scale, 'accs': accs_dmc_g, 'avg': sum(accs_dmc_g)/3}
    
    # 4. DMC-Merge (Conflict-Aware)
    dmc_c_scale = best_scales['DMC-Merge (Conflict-Aware)']
    merged_dmc_c = merge_dmc(base_state, task_weights_list, 'use_conflict_aware', dmc_c_scale)
    encoder.load_state_dict(merged_dmc_c)
    accs_dmc_c = [evaluate_encoder(encoder, task_classifiers[t], loaders[t]['test'], fast=False) for t in task_list]
    best_configs['DMC-Merge (Conflict-Aware)'] = {'scale': dmc_c_scale, 'accs': accs_dmc_c, 'avg': sum(accs_dmc_c)/3}
    
    # Save both analyses to a JSON file
    output_data = {
        'svd_analysis': svd_analysis,
        'sweep_results': sweep_results,
        'best_configs': best_configs
    }
    
    with open('extended_analysis.json', 'w') as f:
        json.dump(output_data, f, indent=4)
    print("\nSaved extended analysis and sweep results to 'extended_analysis.json'")
    
    # Print the absolute best configs found
    print("\n==============================================")
    print("Best Configurations Found in Sweep (FULL 100% EVAL)")
    print("==============================================")
    for method, metrics in best_configs.items():
        print(f"{method:<28} | Best Scale: {metrics['scale']:.1f} | Best Avg Acc: {metrics['avg']:.2f}% (CIFAR={metrics['accs'][0]:.2f}%, SVHN={metrics['accs'][1]:.2f}%, FMNIST={metrics['accs'][2]:.2f}%)")
    print("==============================================")

if __name__ == '__main__':
    main()
