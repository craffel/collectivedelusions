import os
import random
import numpy as np
import torch
import timm

import sys
sys.path.insert(0, './local_packages_310')

TASKS = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
DEVICE = 'cpu'

def get_layer_group(param_name):
    if 'patch_embed' in param_name or 'cls_token' in param_name or 'pos_embed' in param_name or 'norm_pre' in param_name:
        return 0
    elif 'blocks' in param_name:
        parts = param_name.split('.')
        block_idx = int(parts[1])
        return block_idx + 1
    elif 'norm.' in param_name:
        return 13
    else:
        return -1

def main():
    # Load base model
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(DEVICE)
    base_dict = {n: p.clone().detach().to(DEVICE) for n, p in base_model.named_parameters()}
    
    seed = 42
    radii = [0.0, 0.01, 0.05, 0.1, 0.2]
    
    print("=== GEOMETRIC ANALYSIS OF TASK VECTORS ACROSS SAM RADII ===")
    
    for rho in radii:
        print(f"\n--- SAM Radius rho = {rho} ---")
        expert_dicts = []
        for task in TASKS:
            expert_file = f"checkpoints/expert_{task}_seed{seed}_rho{rho}.pt"
            sd = torch.load(expert_file, map_location=DEVICE)
            expert_dicts.append(sd)
            
        # Compute global task vectors
        task_vectors_global = []
        for k in range(4):
            # Flatten and concatenate all parameter task vectors
            flat_vector = []
            for n in base_dict:
                if get_layer_group(n) >= 0:
                    diff = expert_dicts[k][n] - base_dict[n]
                    flat_vector.append(diff.view(-1))
            task_vectors_global.append(torch.cat(flat_vector))
            
        # 1. Compute L2 Norms
        norms = [v.norm(2).item() for v in task_vectors_global]
        mean_norm = np.mean(norms)
        print(f"  Task Vector L2 Norms:")
        for k, task in enumerate(TASKS):
            print(f"    {task}: {norms[k]:.3f}")
        print(f"    Mean L2 Norm: {mean_norm:.3f}")
        
        # 2. Compute Pairwise Cosine Similarities
        cos_sims = []
        for i in range(4):
            for j in range(i+1, 4):
                sim = torch.cosine_similarity(task_vectors_global[i], task_vectors_global[j], dim=0).item()
                cos_sims.append(sim)
        mean_sim = np.mean(cos_sims)
        print(f"  Pairwise Cosine Similarities (Mean: {mean_sim:.3f}):")
        idx = 0
        for i in range(4):
            for j in range(i+1, 4):
                print(f"    {TASKS[i]} <-> {TASKS[j]}: {cos_sims[idx]:.3f}")
                idx += 1

if __name__ == '__main__':
    main()
