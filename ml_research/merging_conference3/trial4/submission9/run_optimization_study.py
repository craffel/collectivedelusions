import os
import torch
import torch.nn as nn
import numpy as np
import timm
from run_evaluation import (
    get_val_and_test_loaders,
    compute_val_accs,
    evaluate_model,
    merge_adamerging,
    merge_zipmerge,
    merge_epm
)

def tune_adamerging_study(base_state_dict, expert_state_dicts, base_model, expert_models, val_batches, test_loaders, device, max_steps=500, log_steps=[40, 100, 250, 500]):
    print("\n--- Running AdaMerging Optimization Study (Dense) ---")
    K = len(expert_state_dicts)
    G = 14
    coefficients = [[0.3] * K for _ in range(G)]
    
    sigma = 0.05
    alpha_up = 1.15
    beta_down = 0.85
    
    current_sd = merge_adamerging(base_state_dict, expert_state_dicts, coefficients)
    current_accs = compute_val_accs(current_sd, base_model, expert_models, val_batches, device)
    current_score = min(current_accs) + 0.1 * np.mean(current_accs)
    
    best_coeffs = [list(row) for row in coefficients]
    best_score = current_score
    
    history = {}
    
    for step in range(1, max_steps + 1):
        candidate_coeffs = []
        for g in range(G):
            perturbation = torch.normal(0, sigma, size=(K,)).tolist()
            row = [max(0.0, coefficients[g][k] + perturbation[k]) for k in range(K)]
            candidate_coeffs.append(row)
            
        candidate_sd = merge_adamerging(base_state_dict, expert_state_dicts, candidate_coeffs)
        candidate_accs = compute_val_accs(candidate_sd, base_model, expert_models, val_batches, device)
        candidate_score = min(candidate_accs) + 0.1 * np.mean(candidate_accs)
        
        if candidate_score > current_score:
            coefficients = candidate_coeffs
            current_score = candidate_score
            sigma = sigma * alpha_up
        else:
            sigma = sigma * beta_down
            
        if current_score > best_score:
            best_score = current_score
            best_coeffs = [list(row) for row in coefficients]
            
        if step in log_steps:
            # Evaluate on test set
            eval_sd = merge_adamerging(base_state_dict, expert_state_dicts, best_coeffs)
            test_accs = evaluate_model(eval_sd, base_model, expert_models, test_loaders, device)
            test_mean = np.mean(test_accs)
            history[step] = {
                'best_val_score': best_score,
                'test_mean': test_mean,
                'test_accs': test_accs
            }
            print(f"Step {step}: Best Val Score = {best_score:.4f}, Test Mean = {test_mean:.4f}, Test Accs = {[f'{acc:.4f}' for acc in test_accs]}")
            
    return history

def tune_zipmerge_study(base_state_dict, expert_state_dicts, base_model, expert_models, val_batches, test_loaders, sparsity, device, max_steps=500, log_steps=[40, 100, 250, 500]):
    print(f"\n--- Running ZipMerge Optimization Study (Sparsity: {sparsity}) ---")
    K = len(expert_state_dicts)
    G = 14
    coefficients = [[0.3] * K for _ in range(G)]
    sparsities = [sparsity] * G
    
    sigma_c = 0.05
    sigma_s = 0.05
    alpha_up = 1.15
    beta_down = 0.85
    
    current_sd = merge_zipmerge(base_state_dict, expert_state_dicts, coefficients, sparsities)
    current_accs = compute_val_accs(current_sd, base_model, expert_models, val_batches, device)
    current_score = min(current_accs) + 0.1 * np.mean(current_accs)
    
    best_coeffs = [list(row) for row in coefficients]
    best_sparsities = list(sparsities)
    best_score = current_score
    
    history = {}
    
    for step in range(1, max_steps + 1):
        candidate_coeffs = []
        for g in range(G):
            perturbation = torch.normal(0, sigma_c, size=(K,)).tolist()
            row = [max(0.0, coefficients[g][k] + perturbation[k]) for k in range(K)]
            candidate_coeffs.append(row)
            
        perturbation_s = torch.normal(0, sigma_s, size=(G,)).tolist()
        candidate_sparsities = [min(0.99, max(0.0, sparsities[g] + perturbation_s[g])) for g in range(G)]
        
        candidate_sd = merge_zipmerge(base_state_dict, expert_state_dicts, candidate_coeffs, candidate_sparsities)
        candidate_accs = compute_val_accs(candidate_sd, base_model, expert_models, val_batches, device)
        candidate_score = min(candidate_accs) + 0.1 * np.mean(candidate_accs)
        
        if candidate_score > current_score:
            coefficients = candidate_coeffs
            sparsities = candidate_sparsities
            current_score = candidate_score
            sigma_c = sigma_c * alpha_up
            sigma_s = sigma_s * alpha_up
        else:
            sigma_c = sigma_c * beta_down
            sigma_s = sigma_s * beta_down
            
        if current_score > best_score:
            best_score = current_score
            best_coeffs = [list(row) for row in coefficients]
            best_sparsities = list(sparsities)
            
        if step in log_steps:
            # Evaluate on test set
            eval_sd = merge_zipmerge(base_state_dict, expert_state_dicts, best_coeffs, best_sparsities)
            test_accs = evaluate_model(eval_sd, base_model, expert_models, test_loaders, device)
            test_mean = np.mean(test_accs)
            history[step] = {
                'best_val_score': best_score,
                'test_mean': test_mean,
                'test_accs': test_accs
            }
            print(f"Step {step}: Best Val Score = {best_score:.4f}, Test Mean = {test_mean:.4f}, Test Accs = {[f'{acc:.4f}' for acc in test_accs]}")
            
    return history

def tune_tlc_study(base_state_dict, expert_state_dicts, base_model, expert_models, val_batches, test_loaders, sparsity, global_stds, device, max_steps=500, log_steps=[40, 100, 250, 500]):
    print(f"\n--- Running TLC-Tune Optimization Study (Sparsity: {sparsity}) ---")
    K = len(expert_state_dicts)
    lambdas = [1.0] * K
    
    sigma = 0.1
    alpha_up = 1.22
    beta_down = 0.82
    
    current_sd = merge_epm(base_state_dict, expert_state_dicts, lambdas, sparsity, global_stds)
    current_accs = compute_val_accs(current_sd, base_model, expert_models, val_batches, device)
    current_score = min(current_accs) + 0.1 * np.mean(current_accs)
    
    best_lambdas = list(lambdas)
    best_score = current_score
    
    history = {}
    
    for step in range(1, max_steps + 1):
        perturbation = torch.normal(0, sigma, size=(K,)).tolist()
        candidate_lambdas = [lambdas[i] + perturbation[i] for i in range(K)]
        candidate_lambdas = [max(0.0, l) for l in candidate_lambdas]
        
        candidate_sd = merge_epm(base_state_dict, expert_state_dicts, candidate_lambdas, sparsity, global_stds)
        candidate_accs = compute_val_accs(candidate_sd, base_model, expert_models, val_batches, device)
        candidate_score = min(candidate_accs) + 0.1 * np.mean(candidate_accs)
        
        if candidate_score > current_score:
            lambdas = candidate_lambdas
            current_score = candidate_score
            sigma = sigma * alpha_up
        else:
            sigma = sigma * beta_down
            
        if current_score > best_score:
            best_score = current_score
            best_lambdas = list(lambdas)
            
        if step in log_steps:
            # Evaluate on test set
            eval_sd = merge_epm(base_state_dict, expert_state_dicts, best_lambdas, sparsity, global_stds)
            test_accs = evaluate_model(eval_sd, base_model, expert_models, test_loaders, device)
            test_mean = np.mean(test_accs)
            history[step] = {
                'best_val_score': best_score,
                'test_mean': test_mean,
                'test_accs': test_accs
            }
            print(f"Step {step}: Best Val Score = {best_score:.4f}, Test Mean = {test_mean:.4f}, Test Accs = {[f'{acc:.4f}' for acc in test_accs]}")
            
    return history

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
        
    val_batches = []
    test_loaders = []
    for task in tasks:
        val_loader, test_loader = get_val_and_test_loaders(task, val_size=128)
        for images, labels in val_loader:
            val_batches.append((images.to(device), labels.to(device)))
            break
        test_loaders.append(test_loader)
        
    K = len(tasks)
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
        
    # Run studies
    ada_history = tune_adamerging_study(base_state_dict, expert_state_dicts, base_model, expert_models, val_batches, test_loaders, device)
    zip_history = tune_zipmerge_study(base_state_dict, expert_state_dicts, base_model, expert_models, val_batches, test_loaders, sparsity=0.5, device=device)
    tlc_dense_history = tune_tlc_study(base_state_dict, expert_state_dicts, base_model, expert_models, val_batches, test_loaders, sparsity=0.0, global_stds=global_stds, device=device)
    tlc_sparse_history = tune_tlc_study(base_state_dict, expert_state_dicts, base_model, expert_models, val_batches, test_loaders, sparsity=0.5, global_stds=global_stds, device=device)
    
    # Save results
    study_results_path = 'optimization_study_results.md'
    with open(study_results_path, 'w') as f:
        f.write("# Optimization Steps and Generalization Study (500 Steps)\n\n")
        f.write("We systematically sweep the number of optimization steps $T \\in \\{40, 100, 250, 500\\}$ on the 128-sample-per-task validation split to evaluate the convergence and generalization behavior of high-dimensional lyer-group-wise tuning (AdaMerging, 56 continuous parameters; ZipMerge, 70 parameters) against our low-dimensional TLC-Tune global coefficient scaling (4 parameters).\n\n")
        
        f.write("### 1. AdaMerging (Dense, 56 parameters)\n")
        f.write("| Optimization Steps | Best Validation Score | Test Mean Accuracy | Test Accs (MNIST/F-MNIST/CIFAR-10/SVHN) |\n")
        f.write("|---|---|---|---|\n")
        for step in [40, 100, 250, 500]:
            h = ada_history[step]
            f.write(f"| {step} | {h['best_val_score']:.4f} | {h['test_mean']:.4f} | {[f'{x:.4f}' for x in h['test_accs']]} |\n")
        f.write("\n")
        
        f.write("### 2. ZipMerge (p=0.5, 70 parameters)\n")
        f.write("| Optimization Steps | Best Validation Score | Test Mean Accuracy | Test Accs (MNIST/F-MNIST/CIFAR-10/SVHN) |\n")
        f.write("|---|---|---|---|\n")
        for step in [40, 100, 250, 500]:
            h = zip_history[step]
            f.write(f"| {step} | {h['best_val_score']:.4f} | {h['test_mean']:.4f} | {[f'{x:.4f}' for x in h['test_accs']]} |\n")
        f.write("\n")
        
        f.write("### 3. TLC-Tune EPM (Dense, 4 parameters)\n")
        f.write("| Optimization Steps | Best Validation Score | Test Mean Accuracy | Test Accs (MNIST/F-MNIST/CIFAR-10/SVHN) |\n")
        f.write("|---|---|---|---|\n")
        for step in [40, 100, 250, 500]:
            h = tlc_dense_history[step]
            f.write(f"| {step} | {h['best_val_score']:.4f} | {h['test_mean']:.4f} | {[f'{x:.4f}' for x in h['test_accs']]} |\n")
        f.write("\n")
        
        f.write("### 4. TLC-Tune EPM (p=0.5, 4 parameters)\n")
        f.write("| Optimization Steps | Best Validation Score | Test Mean Accuracy | Test Accs (MNIST/F-MNIST/CIFAR-10/SVHN) |\n")
        f.write("|---|---|---|---|\n")
        for step in [40, 100, 250, 500]:
            h = tlc_sparse_history[step]
            f.write(f"| {step} | {h['best_val_score']:.4f} | {h['test_mean']:.4f} | {[f'{x:.4f}' for x in h['test_accs']]} |\n")
        f.write("\n")
        
    print(f"Results saved to {study_results_path}")

if __name__ == '__main__':
    main()
