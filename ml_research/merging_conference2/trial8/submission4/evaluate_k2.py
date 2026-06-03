import torch
import torch.nn as nn
import torchvision.models as models
import copy
import os
import json
import matplotlib.pyplot as plt
from train_and_merge import get_model, get_dataset, evaluate_model, merge_experts

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def evaluate_k2():
    # 1. Prepare test dataloaders
    test_loaders = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        _, test_loaders[task] = get_dataset(task)
        
    # 2. Instantiate progenitor (pretrained ResNet-18)
    print("Loading ImageNet pre-trained progenitor...")
    progenitor = get_model()
    progenitor_state = copy.deepcopy(progenitor.state_dict())
    
    # 3. Load expert models
    experts_states = {}
    for task in ['mnist', 'fmnist', 'cifar10']:
        ckpt_path = f"checkpoints/resnet18_{task}.pt"
        if not os.path.exists(ckpt_path):
            print(f"Error: checkpoint {ckpt_path} not found.")
            return
        expert_data = torch.load(ckpt_path, map_location=device)
        experts_states[task] = expert_data['state_dict']
        
    # Define pairs
    pairs = [
        ('mnist_fmnist', ['mnist', 'fmnist']),
        ('mnist_cifar10', ['mnist', 'cifar10']),
        ('fmnist_cifar10', ['fmnist', 'cifar10'])
    ]
    
    c_vals = [0.1, 0.5, 0.8, 1.0, 1.2, 1.414, 1.6, 1.8, 2.0, 2.2, 2.5]
    
    results = {}
    
    model = get_model().to(device)
    
    for pair_name, tasks in pairs:
        print(f"\nEvaluating Pair: {pair_name}")
        pair_results = []
        
        # Subset of experts
        pair_experts_states = {t: experts_states[t] for t in tasks}
        
        for c_val in c_vals:
            # Merge K=2 experts
            merged_state = merge_experts(progenitor_state, pair_experts_states, 'cpr', c_val=c_val)
            
            accuracies = {}
            for task in tasks:
                task_state = copy.deepcopy(merged_state)
                task_state['fc.weight'] = copy.deepcopy(experts_states[task]['fc.weight'])
                task_state['fc.bias'] = copy.deepcopy(experts_states[task]['fc.bias'])
                
                model.load_state_dict(task_state)
                acc = evaluate_model(model, test_loaders[task])
                accuracies[task] = acc
                
            avg_acc = sum(accuracies.values()) / len(accuracies)
            pair_results.append({
                'c': c_val,
                'avg': avg_acc,
                'mnist': accuracies.get('mnist'),
                'fmnist': accuracies.get('fmnist'),
                'cifar10': accuracies.get('cifar10')
            })
            print(f"  c={c_val:.3f}: Avg Acc={avg_acc:.2f}% ({', '.join([f'{t}:{accuracies[t]:.2f}%' for t in tasks])})")
            
        results[pair_name] = pair_results
        
    # Find best c for each pair
    best_results = {}
    for pair_name, pair_results in results.items():
        best_r = max(pair_results, key=lambda x: x['avg'])
        best_results[pair_name] = best_r
        print(f"Best for {pair_name}: c={best_r['c']:.3f} with Avg Acc: {best_r['avg']:.2f}%")
        
    # Save results to a json
    out_data = {
        'results': results,
        'best_results': best_results
    }
    with open('results_k2.json', 'w') as f:
        json.dump(out_data, f, indent=4)
    print("\nSuccessfully saved K=2 results to results_k2.json!")
    
    # Let's plot the results
    plt.figure(figsize=(10, 6))
    
    colors = {'mnist_fmnist': 'blue', 'mnist_cifar10': 'orange', 'fmnist_cifar10': 'red'}
    labels = {
        'mnist_fmnist': 'MNIST + FMNIST (K=2)',
        'mnist_cifar10': 'MNIST + CIFAR-10 (K=2)',
        'fmnist_cifar10': 'FMNIST + CIFAR-10 (K=2)'
    }
    
    for pair_name, pair_results in results.items():
        x = [r['c'] for r in pair_results]
        y = [r['avg'] for r in pair_results]
        plt.plot(x, y, marker='o', linestyle='-', color=colors[pair_name], label=labels[pair_name])
        
    # Draw vertical line at theoretical c = sqrt(2)
    plt.axvline(x=1.414, color='gray', linestyle=':', label=r'Theoretical Attractor $\sqrt{2} \approx 1.414$')
    
    plt.title(r'CPR Performance on $K=2$ Expert Merging (Varying $c$)')
    plt.xlabel(r'Scaling Factor $c$')
    plt.ylabel('Average Accuracy of Merged Tasks (%)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.savefig('cpr_k2_results.png', dpi=300, bbox_inches='tight')
    print("Saved plot to cpr_k2_results.png")

if __name__ == '__main__':
    evaluate_k2()
