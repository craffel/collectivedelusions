import os
import torch
import torch.nn as nn
import numpy as np
import timm
from run_evaluation import (
    get_val_and_test_loaders,
    compute_val_accs,
    tlc_tune,
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
    individual_accs = []
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
        individual_accs.append(expert_checkpoint['test_acc'])
        
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
        
    val_size = 128
    val_batches = []
    test_loaders = []
    for task in tasks:
        val_loader, test_loader = get_val_and_test_loaders(task, val_size=val_size)
        for images, labels in val_loader:
            val_batches.append((images.to(device), labels.to(device)))
            break
        test_loaders.append(test_loader)
        
    sparsities = [0.0, 0.5, 0.8]
    
    results = {}
    for p in sparsities:
        results[p] = {
            'Global_Untuned': [],
            'Global_Tuned': [],
            'Layerwise_Untuned': [],
            'Layerwise_Tuned': []
        }
        
    for p in sparsities:
        print(f"\n================ Evaluating Comparison for Sparsity p = {p} ================")
        gamma_val = 0.2 + (1.0 - 0.2) * (p ** 2)
        print(f"Using Coherence Factor (DCS): {gamma_val:.4f}")
        
        # 1. Global Untuned
        print("Running Global EPM Untuned...")
        glob_un_sd = merge_epm(base_state_dict, expert_state_dicts, [1.0]*K, sparsity=p, global_stds=global_stds, gamma=gamma_val, use_layerwise_std=False)
        glob_un_accs = evaluate_model(glob_un_sd, base_model, expert_models, test_loaders, device)
        results[p]['Global_Untuned'] = glob_un_accs
        print(f"Global Untuned Accs: {glob_un_accs} (Mean: {np.mean(glob_un_accs):.4f})")
        
        # 2. Global Tuned
        print("Running Global EPM Tuned...")
        glob_tuned_lambdas = tlc_tune(base_state_dict, expert_state_dicts, base_model, expert_models, val_batches, sparsity=p, global_stds=global_stds, device=device, steps=40, gamma=gamma_val, use_layerwise_std=False)
        glob_tuned_sd = merge_epm(base_state_dict, expert_state_dicts, glob_tuned_lambdas, sparsity=p, global_stds=global_stds, gamma=gamma_val, use_layerwise_std=False)
        glob_tuned_accs = evaluate_model(glob_tuned_sd, base_model, expert_models, test_loaders, device)
        results[p]['Global_Tuned'] = glob_tuned_accs
        print(f"Global Tuned Accs: {glob_tuned_accs} (Mean: {np.mean(glob_tuned_accs):.4f})")
        
        # 3. Layer-wise Untuned
        print("Running Layerwise EPM Untuned...")
        lay_un_sd = merge_epm(base_state_dict, expert_state_dicts, [1.0]*K, sparsity=p, global_stds=None, gamma=gamma_val, use_layerwise_std=True)
        lay_un_accs = evaluate_model(lay_un_sd, base_model, expert_models, test_loaders, device)
        results[p]['Layerwise_Untuned'] = lay_un_accs
        print(f"Layerwise Untuned Accs: {lay_un_accs} (Mean: {np.mean(lay_un_accs):.4f})")
        
        # 4. Layer-wise Tuned
        print("Running Layerwise EPM Tuned...")
        lay_tuned_lambdas = tlc_tune(base_state_dict, expert_state_dicts, base_model, expert_models, val_batches, sparsity=p, global_stds=None, device=device, steps=40, gamma=gamma_val, use_layerwise_std=True)
        lay_tuned_sd = merge_epm(base_state_dict, expert_state_dicts, lay_tuned_lambdas, sparsity=p, global_stds=None, gamma=gamma_val, use_layerwise_std=True)
        lay_tuned_accs = evaluate_model(lay_tuned_sd, base_model, expert_models, test_loaders, device)
        results[p]['Layerwise_Tuned'] = lay_tuned_accs
        print(f"Layerwise Tuned Accs: {lay_tuned_accs} (Mean: {np.mean(lay_tuned_accs):.4f})")
        
    # Print out summary Markdown table
    markdown_out = "# Global vs. Layer-wise Task Vector Standardization Comparison\n\n"
    markdown_out += "We systematically evaluate and compare EPM under **Global Task Vector Standardization** (default) vs. **Layer-wise Task Vector Standardization** across dense ($p=0.0$), moderate sparsity ($p=0.5$), and extreme sparsity ($p=0.8$) on a shared ViT-Tiny backbone.\n\n"
    
    for p in sparsities:
        gamma_val = 0.2 + (1.0 - 0.2) * (p ** 2)
        markdown_out += f"### Target Sparsity $p = {p}$ (DCS $\\gamma = {gamma_val:.2f}$)\n\n"
        markdown_out += "| Standardization Scheme | MNIST Acc | FashionMNIST Acc | CIFAR-10 Acc | SVHN Acc | Joint Mean Acc |\n"
        markdown_out += "|---|---|---|---|---|---|\n"
        
        g_un = results[p]['Global_Untuned']
        g_tu = results[p]['Global_Tuned']
        l_un = results[p]['Layerwise_Untuned']
        l_tu = results[p]['Layerwise_Tuned']
        
        markdown_out += f"| Global Standardization (Untuned, $\\Lambda = \\mathbf{{1.0}}$) | {g_un[0]:.4f} | {g_un[1]:.4f} | {g_un[2]:.4f} | {g_un[3]:.4f} | **{np.mean(g_un):.4f}** |\n"
        markdown_out += f"| Global Standardization (TLC-Tuned) | {g_tu[0]:.4f} | {g_tu[1]:.4f} | {g_tu[2]:.4f} | {g_tu[3]:.4f} | **{np.mean(g_tu):.4f}** |\n"
        markdown_out += f"| Layer-wise Standardization (Untuned, $\\Lambda = \\mathbf{{1.0}}$) | {l_un[0]:.4f} | {l_un[1]:.4f} | {l_un[2]:.4f} | {l_un[3]:.4f} | **{np.mean(l_un):.4f}** |\n"
        markdown_out += f"| Layer-wise Standardization (TLC-Tuned) | {l_tu[0]:.4f} | {l_tu[1]:.4f} | {l_tu[2]:.4f} | {l_tu[3]:.4f} | **{np.mean(l_tu):.4f}** |\n\n"
        
    print("\n\n=== FINAL COMPARISON RESULTS ===")
    print(markdown_out)
    
    with open("standardization_study_results.md", "w") as f:
        f.write(markdown_out)
        
if __name__ == '__main__':
    main()
