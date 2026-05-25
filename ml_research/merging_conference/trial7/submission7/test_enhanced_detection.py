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

def evaluate_enhanced_hat_merge(stream, experts, fisher_sens, static_model, mu_static, prototypes, threshold=0.35, ent_threshold=1.0):
    # Modified HAT-Merge with predictive entropy in novelty detection
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
            
        # 2. Sample-level routing and novelty detection (Cohesion)
        sample_cohesions = torch.zeros(batch_size, 2, device=device)
        for k in [0, 1]:
            sim_matrix = torch.matmul(feats, prototypes[k].t()) # (B, 10)
            max_sims, _ = sim_matrix.max(dim=1)
            sample_cohesions[:, k] = max_sims
            
        max_cohesions, routed_experts = sample_cohesions.max(dim=1)
        
        # 3. Sample-level predictive entropy
        with torch.no_grad():
            sample_entropies = torch.zeros(batch_size, 2, device=device)
            for k in range(2):
                outputs = experts[k](inputs)
                probs = torch.softmax(outputs, dim=1)
                sample_entropies[:, k] = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                
        min_entropies, _ = sample_entropies.min(dim=1)
        
        # Combined novelty detection criteria
        # A sample is novel if its cohesion is below threshold OR its minimum entropy is above entropy threshold
        is_novel = (max_cohesions < threshold) | (min_entropies > ent_threshold)
        
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

def run_tests():
    print("Loading experts and dataset...")
    experts = load_experts()
    from eval_all import compute_source_fisher
    s_fisher = compute_source_fisher(experts, device)
    static_model, mu_static, prototypes = precompute_prototypes(experts, device)
    
    # Get heterogeneous stream
    _, _, het_stream = get_test_streams(batch_size=32, corruption="clean")
    
    print("\nEvaluating Baseline (Cohesion-only tau=0.35)...")
    base_overall, base_tasks, base_ndr, base_fpr = evaluate_enhanced_hat_merge(
        het_stream, experts, s_fisher, static_model, mu_static, prototypes, threshold=0.35, ent_threshold=999.0
    )
    print(f"Baseline | Overall Acc: {base_overall*100:.2f}% | MNIST: {base_tasks[0]*100:.2f}%, KMNIST: {base_tasks[1]*100:.2f}%, Novel: {base_tasks[2]*100:.2f}% | NDR: {base_ndr*100:.2f}%, FPR: {base_fpr*100:.2f}%")
    
    print("\nEvaluating Enhanced HAT-Merge (Cohesion tau=0.35 + Entropy H=1.2)...")
    ent_overall, ent_tasks, ent_ndr, ent_fpr = evaluate_enhanced_hat_merge(
        het_stream, experts, s_fisher, static_model, mu_static, prototypes, threshold=0.35, ent_threshold=1.2
    )
    print(f"H=1.2    | Overall Acc: {ent_overall*100:.2f}% | MNIST: {ent_tasks[0]*100:.2f}%, KMNIST: {ent_tasks[1]*100:.2f}%, Novel: {ent_tasks[2]*100:.2f}% | NDR: {ent_ndr*100:.2f}%, FPR: {ent_fpr*100:.2f}%")

    print("\nEvaluating Enhanced HAT-Merge (Cohesion tau=0.35 + Entropy H=1.0)...")
    ent_overall_2, ent_tasks_2, ent_ndr_2, ent_fpr_2 = evaluate_enhanced_hat_merge(
        het_stream, experts, s_fisher, static_model, mu_static, prototypes, threshold=0.35, ent_threshold=1.0
    )
    print(f"H=1.0    | Overall Acc: {ent_overall_2*100:.2f}% | MNIST: {ent_tasks_2[0]*100:.2f}%, KMNIST: {ent_tasks_2[1]*100:.2f}%, Novel: {ent_tasks_2[2]*100:.2f}% | NDR: {ent_ndr_2*100:.2f}%, FPR: {ent_fpr_2*100:.2f}%")

    print("\nEvaluating Enhanced HAT-Merge (Cohesion tau=0.35 + Entropy H=0.8)...")
    ent_overall_3, ent_tasks_3, ent_ndr_3, ent_fpr_3 = evaluate_enhanced_hat_merge(
        het_stream, experts, s_fisher, static_model, mu_static, prototypes, threshold=0.35, ent_threshold=0.8
    )
    print(f"H=0.8    | Overall Acc: {ent_overall_3*100:.2f}% | MNIST: {ent_tasks_3[0]*100:.2f}%, KMNIST: {ent_tasks_3[1]*100:.2f}%, Novel: {ent_tasks_3[2]*100:.2f}% | NDR: {ent_ndr_3*100:.2f}%, FPR: {ent_fpr_3*100:.2f}%")

if __name__ == "__main__":
    run_tests()
