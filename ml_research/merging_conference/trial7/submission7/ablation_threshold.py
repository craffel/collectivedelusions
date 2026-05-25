import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model import get_resnet18_model, merge_weights_and_buffers
from dataset import get_test_streams
from eval_all import load_experts, precompute_prototypes, extract_features, project_onto_simplex

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_hat_merge_metrics(stream, experts, fisher_sens, static_model, mu_static, prototypes, threshold):
    # Modified HAT-Merge evaluation to compute detailed metrics: Overall, Task Accs, NDR, FPR
    lambdas_novel = {}
    merged_model = get_resnet18_model().to(device)
    for name, param in merged_model.named_parameters():
        lambdas_novel[name] = torch.tensor([1/3, 1/3, 1/3], device=device)
        
    correct = 0
    total = 0
    task_correct = [0, 0, 0]
    task_total = [0, 0, 0]
    
    # Novelty detection metrics
    novel_total = 0       # Total actual novel samples (FashionMNIST)
    novel_detected = 0    # Actual novel samples classified as novel (True Positives)
    known_total = 0       # Total actual known samples (MNIST + KMNIST)
    known_detected = 0    # Actual known samples classified as novel (False Positives)
    
    lr = 0.05
    
    for inputs, targets, task_ids in stream:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # 1. Feature Extraction in Unified Static Space
        with torch.no_grad():
            feats = extract_features(static_model, inputs) - mu_static
            feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
            
        # 2. Sample-level routing and novelty detection
        sample_cohesions = torch.zeros(batch_size, 2, device=device)
        for k in [0, 1]:
            sim_matrix = torch.matmul(feats, prototypes[k].t()) # (B, 10)
            max_sims, _ = sim_matrix.max(dim=1)
            sample_cohesions[:, k] = max_sims
            
        max_cohesions, routed_experts = sample_cohesions.max(dim=1)
        is_novel = (max_cohesions < threshold)
        
        # Track detection stats
        for i in range(batch_size):
            actual_task = task_ids[i].item()
            if actual_task == 2:
                novel_total += 1
                if is_novel[i]:
                    novel_detected += 1
            else:
                known_total += 1
                if is_novel[i]:
                    known_detected += 1
                    
        # Partition the batch into sub-batches
        known_indices = [[] for _ in range(2)]
        novel_indices = []
        
        for i in range(batch_size):
            if is_novel[i]:
                novel_indices.append(i)
            else:
                k_idx = routed_experts[i].item()
                known_indices[k_idx].append(i)
                
        # Final prediction tensor for the batch
        final_preds = torch.zeros_like(targets)
        
        # 3. Process known sub-batches: Run directly through expert models
        for k in range(2):
            if len(known_indices[k]) > 0:
                idxs = torch.tensor(known_indices[k], device=device)
                with torch.no_grad():
                    out = experts[k](inputs[idxs])
                    final_preds[idxs] = out.argmax(dim=1)
                    
        # 4. Process novel sub-batch (if non-empty)
        if len(novel_indices) > 0:
            idxs = torch.tensor(novel_indices, device=device)
            novel_inputs = inputs[idxs]
            
            # Evaluate individual expert predictive entropy on this novel sub-batch
            entropies = []
            for expert in experts:
                expert.eval()
                with torch.no_grad():
                    outputs = expert(novel_inputs)
                    prob = torch.softmax(outputs, dim=1)
                    entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1).mean().item()
                    entropies.append(entropy)
            # Choose lowest entropy expert
            routed_idx = np.argmin(entropies)
            target_l = torch.zeros(3, device=device)
            target_l[routed_idx] = 1.0
            
            # Fisher-Preconditioned Riemannian update on lambdas_novel
            for name in lambdas_novel:
                gw_inv = fisher_sens.get(name, 1.0)
                lambdas_novel[name] = project_onto_simplex(lambdas_novel[name] - lr * gw_inv * (lambdas_novel[name] - target_l))
                
            # Merge experts using adapted lambdas_novel
            merge_weights_and_buffers(merged_model, experts, lambdas_novel)
            merged_model.eval()
            
            # Inference on novel sub-batch
            with torch.no_grad():
                out = merged_model(novel_inputs)
                final_preds[idxs] = out.argmax(dim=1)
                
        # Evaluate performance on the entire batch
        for i in range(batch_size):
            actual_task = task_ids[i].item()
            task_total[actual_task] += 1
            if final_preds[i] == targets[i]:
                task_correct[actual_task] += 1
                correct += 1
            total += 1
            
    overall_acc = correct / total
    task_accs = [task_correct[k] / max(1, task_total[k]) for k in range(3)]
    ndr = novel_detected / max(1, novel_total)
    fpr = known_detected / max(1, known_total)
    
    return overall_acc, task_accs, ndr, fpr

def run_ablation():
    print("Loading experts and dataset...")
    experts = load_experts()
    from eval_all import compute_source_fisher
    s_fisher = compute_source_fisher(experts, device)
    static_model, mu_static, prototypes = precompute_prototypes(experts, device)
    
    # Get heterogeneous stream
    _, _, het_stream = get_test_streams(batch_size=32, corruption="clean")
    
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    
    results = []
    print("\nStarting novelty detection threshold sweep (tau)...")
    print(f"{'Threshold (tau)':<16} | {'Overall Acc':<12} | {'MNIST Acc':<10} | {'KMNIST Acc':<10} | {'Novel Acc':<10} | {'NDR':<8} | {'FPR':<8}")
    print("-" * 90)
    
    for tau in thresholds:
        overall, task_accs, ndr, fpr = evaluate_hat_merge_metrics(
            het_stream, experts, s_fisher, static_model, mu_static, prototypes, threshold=tau
        )
        results.append({
            'threshold': tau,
            'overall': overall,
            'mnist': task_accs[0],
            'kmnist': task_accs[1],
            'novel': task_accs[2],
            'ndr': ndr,
            'fpr': fpr
        })
        print(f"{tau:<16.2f} | {overall*100:<11.2f}% | {task_accs[0]*100:<9.2f}% | {task_accs[1]*100:<9.2f}% | {task_accs[2]*100:<9.2f}% | {ndr*100:<7.2f}% | {fpr*100:<7.2f}%")
        
    # Plot results
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.0

    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)

    # Convert results for plotting
    taus = [r['threshold'] for r in results]
    overall_accs = [r['overall'] * 100 for r in results]
    ndrs = [r['ndr'] * 100 for r in results]
    fprs = [r['fpr'] * 100 for r in results]
    
    color_acc = '#1f77b4'
    ax1.set_xlabel('Novelty Threshold ($\\tau$)', fontweight='bold')
    ax1.set_ylabel('Overall Accuracy (%)', color=color_acc, fontweight='bold')
    line1 = ax1.plot(taus, overall_accs, color=color_acc, marker='o', linewidth=2.0, label='Overall Accuracy')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color_ndr = '#d62728'
    color_fpr = '#2ca02c'
    ax2.set_ylabel('Detection Rates (%)', color='black', fontweight='bold')
    line2 = ax2.plot(taus, ndrs, color=color_ndr, marker='s', linestyle='--', linewidth=1.5, label='Novelty Detection Rate (NDR)')
    line3 = ax2.plot(taus, fprs, color=color_fpr, marker='^', linestyle=':', linewidth=1.5, label='False Positive Rate (FPR)')
    ax2.tick_params(axis='y', labelcolor='black')

    # added these lines
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', frameon=True)

    plt.title('Ablation Study: Novelty Detection Threshold ($\\tau$)', fontweight='bold', fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig('threshold_ablation.png', bbox_inches='tight')
    print("\nSaved ablation plot as threshold_ablation.png")

if __name__ == "__main__":
    run_ablation()
