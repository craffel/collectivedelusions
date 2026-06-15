import os
import copy
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import CLIPModel
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from run_experiments import merge_full_network, load_datasets

def evaluate_model(merged_model, loaders, heads, tasks):
    merged_model.eval()
    accuracies = {}
    with torch.no_grad():
        for task in tasks:
            correct = 0
            total = 0
            loader = loaders[task['name']]
            head = heads[task['name']]
            
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                image_features = merged_model.get_image_features(pixel_values=images).pooler_output
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = head(image_features)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            acc = 100. * correct / total
            accuracies[task['name']] = acc
    accuracies['Average'] = np.mean([accuracies[t['name']] for t in tasks])
    return accuracies

def main():
    print("Loading base model...")
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    
    tasks = [
        {'name': 'MNIST', 'num_classes': 10, 'is_gray': True},
        {'name': 'FashionMNIST', 'num_classes': 10, 'is_gray': True},
        {'name': 'CIFAR10', 'num_classes': 10, 'is_gray': False},
        {'name': 'SVHN', 'num_classes': 10, 'is_gray': False}
    ]
    
    expert_dicts = []
    heads = {}
    loaders = {}
    
    for task in tasks:
        ckpt_path = f"checkpoints/{task['name']}_expert.pt"
        print(f"Loading checkpoint for {task['name']}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        
        head = nn.Linear(512, task['num_classes'])
        head.weight.data.copy_(ckpt['head_weight'])
        head.bias.data.copy_(ckpt['head_bias'])
        heads[task['name']] = head.to(device).eval()
        
        expert_dicts.append(ckpt['visual_state_dict'])
        
        # Load standard test dataset of 1,000 samples per dataset (consistent with standard evaluation)
        _, test_loader = load_datasets(task['name'], task['is_gray'], num_train=1, num_test=1000, batch_size=128)
        loaders[task['name']] = test_loader

    # Shared SVD cache to optimize sweep performance and demonstrate cached timing
    svd_cache = {}

    # 1. Merge with Standard SVS (Rank 128, uniform, uncached SVD)
    print("\n--- Merging with Standard SVS (Rank 128, Uncached) ---")
    lambdas = [0.5, 0.5, 0.5, 0.5]
    start_time = time.time()
    merged_svs = merge_full_network(
        base_model, 
        expert_dicts, 
        lambdas, 
        method="svs", 
        rank=128, 
        apply_bwn=True, 
        apply_entropy=False,
        svd_cache=svd_cache
    )
    uncached_time = time.time() - start_time
    print(f"Standard SVS merged (Uncached SVD) in {uncached_time:.2f} seconds.")
    
    accs_svs = evaluate_model(merged_svs, loaders, heads, tasks)
    for task in tasks:
        print(f"Standard SVS - {task['name']}: {accs_svs[task['name']]:.2f}%")
    print(f"Standard SVS Average Accuracy: {accs_svs['Average']:.2f}%")

    # 2. Sweep over Entropy Multipliers (demonstrating Cached SVD speedup)
    print("\n--- Sweeping Entropy Multipliers for Entropy-SVS (Cached) ---")
    entropy_mults = [1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    sweep_results = []
    
    for mult in entropy_mults:
        print(f"\nEvaluating entropy_mult = {mult}...")
        rank_list_out = []
        start_time = time.time()
        merged_entropy = merge_full_network(
            base_model, 
            expert_dicts, 
            lambdas, 
            method="svs", 
            rank=128, 
            apply_bwn=True, 
            apply_entropy=True, 
            entropy_mult=mult,
            rank_list_out=rank_list_out,
            svd_cache=svd_cache
        )
        cached_time = time.time() - start_time
        print(f"Entropy-SVS merged (Cached SVD) in {cached_time:.4f} seconds!")
        
        avg_rank = np.mean(rank_list_out)
        compression_rate = (1.0 - avg_rank / 128.0) * 100.0
        
        accs = evaluate_model(merged_entropy, loaders, heads, tasks)
        print(f"Entropy-SVS (mult={mult}) - Avg Rank: {avg_rank:.2f} (Compression: {compression_rate:.2f}%)")
        print(f"Average Accuracy: {accs['Average']:.2f}%")
        
        sweep_results.append({
            'entropy_mult': mult,
            'avg_rank': avg_rank,
            'compression_rate': compression_rate,
            'accuracies': accs,
            'uncached_merge_time': uncached_time,
            'cached_merge_time': cached_time
        })
        
    os.makedirs('results', exist_ok=True)
    with open('results/entropy_svs_sweep.json', 'w') as f:
        json.dump(sweep_results, f, indent=2)
        
    # Plot Pareto curve: Accuracy vs. Average Rank
    avg_ranks = [res['avg_rank'] for res in sweep_results]
    accs_avg = [res['accuracies']['Average'] for res in sweep_results]
    
    plt.figure(figsize=(8, 5))
    plt.plot(avg_ranks, accs_avg, marker='o', linestyle='-', linewidth=2, color='#1f77b4', label='Entropy-SVS Sweep')
    plt.axhline(y=accs_svs['Average'], color='r', linestyle='--', label='Standard SVS (Rank 128)')
    
    # Add annotations for multipliers
    for res in sweep_results:
        plt.annotate(
            f"m={res['entropy_mult']}",
            (res['avg_rank'], res['accuracies']['Average']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8
        )
        
    plt.title('Accuracy vs. Average Rank (Entropy-SVS Pareto Frontier)', fontsize=12, fontweight='bold')
    plt.xlabel('Average Slicing Rank across Layers', fontsize=10)
    plt.ylabel('Multi-Task Average Accuracy (%)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('results/fig5_entropy_svs_pareto.png', dpi=300)
    plt.close()
    print("\nSweep completed! Results saved to results/entropy_svs_sweep.json and plot saved to results/fig5_entropy_svs_pareto.png")

if __name__ == '__main__':
    main()
