import os
import torch
import torch.nn as nn
import numpy as np
import timm
from run_evaluation import (
    get_val_and_test_loaders,
    compute_val_accs,
    tune_adamerging,
    tune_zipmerge,
    tlc_tune,
    merge_adamerging,
    merge_zipmerge,
    merge_epm,
    evaluate_model
)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tasks = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    
    # Load base model
    print("Loading base pre-trained model...")
    base_checkpoint = torch.load('checkpoints/base_model.pt', map_location=device)
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    base_model.head = nn.Linear(base_model.num_features, 10)
    base_model.load_state_dict(base_checkpoint['state_dict'])
    base_model = base_model.to(device)
    base_state_dict = {k: v.to(device) for k, v in base_checkpoint['state_dict'].items()}
    
    # Load expert models
    expert_models = []
    expert_state_dicts = []
    for task in tasks:
        print(f"Loading expert model for {task}...")
        expert_checkpoint = torch.load(f'checkpoints/{task}_expert.pt', map_location=device)
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        model.head = nn.Linear(model.num_features, 10)
        model.load_state_dict(expert_checkpoint['state_dict'])
        model = model.to(device)
        expert_models.append(model)
        
        sd = {k: v.to(device) for k, v in expert_checkpoint['state_dict'].items()}
        expert_state_dicts.append(sd)
        
    K = len(tasks)
    # Compute global stds once for EPM
    global_stds = []
    for k in range(K):
        all_vals = []
        for key in base_state_dict.keys():
            if 'head' in key or not base_state_dict[key].is_floating_point():
                continue
            all_vals.append((expert_state_dicts[k][key] - base_state_dict[key]).view(-1))
        all_vals_flat = torch.cat(all_vals)
        std = torch.std(all_vals_flat).item()
        global_stds.append(std)
        
    val_sizes = [128, 256, 512, 1024]
    results = {
        size: {
            'AdaMerging (Dense)': [],
            'ZipMerge (p=0.5)': [],
            'EPM (TLC-Tune, Dense)': [],
            'EPM (TLC-Tune, p=0.5)': []
        } for size in val_sizes
    }
    
    for val_size in val_sizes:
        print(f"\n================ Running Sweep for Validation Size = {val_size} ================")
        
        # Load validation split of specific size
        val_batches = []
        test_loaders = []
        for task in tasks:
            val_loader, test_loader = get_val_and_test_loaders(task, val_size=val_size)
            for images, labels in val_loader:
                val_batches.append((images.to(device), labels.to(device)))
                break
            test_loaders.append(test_loader)
            
        # 1. AdaMerging (Dense)
        print("Optimizing AdaMerging...")
        ada_best_coeffs = tune_adamerging(base_state_dict, expert_state_dicts, base_model, expert_models, val_batches, device, steps=40)
        ada_sd = merge_adamerging(base_state_dict, expert_state_dicts, ada_best_coeffs)
        ada_accs = evaluate_model(ada_sd, base_model, expert_models, test_loaders, device)
        results[val_size]['AdaMerging (Dense)'] = ada_accs
        print(f"AdaMerging Test Mean: {np.mean(ada_accs):.4f}")
        
        # 2. ZipMerge (p=0.5)
        print("Optimizing ZipMerge...")
        zm_best_coeffs, zm_best_sparsities = tune_zipmerge(base_state_dict, expert_state_dicts, base_model, expert_models, val_batches, sparsity=0.5, device=device, steps=40)
        zm_sd = merge_zipmerge(base_state_dict, expert_state_dicts, zm_best_coeffs, zm_best_sparsities)
        zm_accs = evaluate_model(zm_sd, base_model, expert_models, test_loaders, device)
        results[val_size]['ZipMerge (p=0.5)'] = zm_accs
        print(f"ZipMerge Test Mean: {np.mean(zm_accs):.4f}")
        
        # 3. EPM (TLC-Tune, Dense)
        print("Optimizing EPM (Dense)...")
        epm_dense_lambdas = tlc_tune(base_state_dict, expert_state_dicts, base_model, expert_models, val_batches, sparsity=0.0, global_stds=global_stds, device=device, steps=40)
        epm_dense_sd = merge_epm(base_state_dict, expert_state_dicts, epm_dense_lambdas, sparsity=0.0, global_stds=global_stds)
        epm_dense_accs = evaluate_model(epm_dense_sd, base_model, expert_models, test_loaders, device)
        results[val_size]['EPM (TLC-Tune, Dense)'] = epm_dense_accs
        print(f"EPM Dense Test Mean: {np.mean(epm_dense_accs):.4f}")
        
        # 4. EPM (TLC-Tune, p=0.5)
        print("Optimizing EPM (p=0.5)...")
        epm_sparse_lambdas = tlc_tune(base_state_dict, expert_state_dicts, base_model, expert_models, val_batches, sparsity=0.5, global_stds=global_stds, device=device, steps=40)
        epm_sparse_sd = merge_epm(base_state_dict, expert_state_dicts, epm_sparse_lambdas, sparsity=0.5, global_stds=global_stds)
        epm_sparse_accs = evaluate_model(epm_sparse_sd, base_model, expert_models, test_loaders, device)
        results[val_size]['EPM (TLC-Tune, p=0.5)'] = epm_sparse_accs
        print(f"EPM Sparse Test Mean: {np.mean(epm_sparse_accs):.4f}")
        
    # Print Markdown table of results
    print("\n\n### Validation Size Sweep Results")
    markdown_out = "# Validation Split Size Sensitivity Sweep\n\n"
    markdown_out += "| Validation Size (per task) | AdaMerging (Dense) Mean Acc | ZipMerge (p=0.5) Mean Acc | EPM (TLC Dense) Mean Acc | EPM (TLC p=0.5) Mean Acc |\n"
    markdown_out += "|---|---|---|---|---|\n"
    for size in val_sizes:
        ada_mean = np.mean(results[size]['AdaMerging (Dense)'])
        zm_mean = np.mean(results[size]['ZipMerge (p=0.5)'])
        epm_d_mean = np.mean(results[size]['EPM (TLC-Tune, Dense)'])
        epm_s_mean = np.mean(results[size]['EPM (TLC-Tune, p=0.5)'])
        markdown_out += f"| {size} | {ada_mean:.4f} | {zm_mean:.4f} | **{epm_d_mean:.4f}** | **{epm_s_mean:.4f}** |\n"
        
    print(markdown_out)
    with open("val_size_sweep_results.md", "w") as f:
        f.write(markdown_out)
        
if __name__ == '__main__':
    main()
