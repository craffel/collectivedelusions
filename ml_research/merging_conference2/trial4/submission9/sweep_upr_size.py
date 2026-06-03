import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from run_experiments import get_datasets, create_base_resnet, ExpertModel, merge_backbones, apply_n_taac

device = torch.device("cpu")

def evaluate_upr(subsets, expert_paths, m_size, beta=20.0):
    sorted_tasks = ['mnist', 'fmnist', 'cifar']
    
    # Load experts and separate heads
    experts = {}
    for name in sorted_tasks:
        ckpt = torch.load(f"expert_{name}.pth", map_location=device)
        backbone = create_base_resnet().to(device)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        head = nn.Linear(512, 10).to(device)
        head.load_state_dict(ckpt['head_state_dict'])
        experts[name] = ExpertModel(backbone, head).to(device)
        
    heads = {name: experts[name].head for name in sorted_tasks}
    
    # Reconstruct N-TAAC backbone (using the same N-TAAC backbone for fair routing comparison)
    merged_backbone = merge_backbones(expert_paths)
    n_taac_backbone = apply_n_taac(merged_backbone, subsets)
    n_taac_backbone.eval()
    
    # Register forward hook on layer2
    anchor_act = None
    def anchor_hook(module, input, output):
        nonlocal anchor_act
        anchor_act = output.detach()
        
    hook_handle = n_taac_backbone.layer2.register_forward_hook(anchor_hook)
    
    all_activations = []
    true_labels = []
    
    # Extract only first m_size samples for prototype generation
    for label_idx, name in enumerate(sorted_tasks):
        cal_sub = Subset(subsets[name]['cal'].dataset, subsets[name]['cal'].indices[:m_size])
        cal_loader = DataLoader(cal_sub, batch_size=1, shuffle=False)
        with torch.no_grad():
            for x, _ in cal_loader:
                n_taac_backbone(x)
                pooled = anchor_act.mean(dim=[2, 3]) # [1, 128]
                pooled_norm = pooled / (pooled.norm(p=2) + 1e-8)
                all_activations.append(pooled_norm.squeeze(0).numpy())
                true_labels.append(label_idx)
                
    hook_handle.remove()
    
    all_activations = np.array(all_activations)
    true_labels = np.array(true_labels)
    
    # Run K-Means with K=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(all_activations)
    
    # Analyze alignment and purity
    cluster_to_task_mapping = {}
    purities = []
    for cluster_id in range(3):
        indices_in_cluster = np.where(cluster_labels == cluster_id)[0]
        if len(indices_in_cluster) == 0:
            continue
        true_labels_in_cluster = true_labels[indices_in_cluster]
        counts = Counter(true_labels_in_cluster)
        
        majority_task_idx = counts.most_common(1)[0][0]
        majority_count = counts.most_common(1)[0][1]
        purity = majority_count / len(indices_in_cluster)
        purities.append(purity)
        
        majority_task_name = sorted_tasks[majority_task_idx]
        cluster_to_task_mapping[cluster_id] = majority_task_name
        
    mapped_tasks = list(cluster_to_task_mapping.values())
    is_bijective = len(set(mapped_tasks)) == 3
    avg_purity = np.mean(purities) * 100.0
    
    # If not bijective, we cannot map routing correctly
    if not is_bijective:
        print(f"  M={m_size}: Clustering is NOT bijective! Cannot route cleanly.")
        return False, avg_purity, {}
        
    cluster_centroids = kmeans.cluster_centers_
    cluster_centroids_norm = cluster_centroids / (np.linalg.norm(cluster_centroids, axis=1, keepdims=True) + 1e-8)
    prototypes = torch.tensor(cluster_centroids_norm, dtype=torch.float32)
    
    # Evaluate Routing
    routing_weights = None
    def test_anchor_hook(module, input, output):
        nonlocal routing_weights
        pooled = output.mean(dim=[2, 3])
        pooled_norm = pooled / (pooled.norm(p=2, dim=1, keepdim=True) + 1e-8)
        
        sims = []
        for c in range(3):
            proto = prototypes[c]
            sim = torch.sum(pooled_norm * proto.unsqueeze(0), dim=1)
            sims.append(sim)
        sims = torch.stack(sims, dim=1)
        
        weights_over_clusters = torch.softmax(beta * sims, dim=1)
        
        weights_over_tasks = torch.zeros(pooled.shape[0], 3, device=pooled.device)
        for cluster_id, task_name in cluster_to_task_mapping.items():
            task_idx = sorted_tasks.index(task_name)
            weights_over_tasks[:, task_idx] += weights_over_clusters[:, cluster_id]
            
        routing_weights = weights_over_tasks
        
    h_test = n_taac_backbone.layer2.register_forward_hook(test_anchor_hook)
    
    accuracies = {}
    for name in sorted_tasks:
        test_loader = DataLoader(subsets[name]['test'], batch_size=256, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                features = n_taac_backbone(x)
                
                logits_mnist = heads['mnist'](features)
                logits_fmnist = heads['fmnist'](features)
                logits_cifar = heads['cifar'](features)
                logits_all = torch.stack([logits_mnist, logits_fmnist, logits_cifar], dim=1)
                
                logits = torch.sum(routing_weights.unsqueeze(-1) * logits_all, dim=1)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        accuracies[name] = (correct / total) * 100.0
        
    h_test.remove()
    return True, avg_purity, accuracies

def main():
    print("--- Running UPR Calibration Size Sweep ---")
    subsets = get_datasets()
    sorted_tasks = ['mnist', 'fmnist', 'cifar']
    expert_paths = [f"expert_{name}.pth" for name in sorted_tasks]
    
    sizes = [16, 32, 64, 128]
    results = {}
    
    for m in sizes:
        print(f"\nEvaluating calibration size M = {m} per task...")
        is_bij, purity, accs = evaluate_upr(subsets, expert_paths, m_size=m)
        if is_bij:
            avg_acc = np.mean([accs[k] for k in sorted_tasks])
            print(f"  Bijective: Yes, Avg Purity: {purity:.2f}%, Avg Acc: {avg_acc:.2f}%")
            results[m] = {
                "bijective": True,
                "purity": purity,
                "mnist": accs['mnist'],
                "fmnist": accs['fmnist'],
                "cifar": accs['cifar'],
                "average": avg_acc
            }
        else:
            print(f"  Bijective: No, Avg Purity: {purity:.2f}%")
            results[m] = {
                "bijective": False,
                "purity": purity,
                "average": 0.0
            }
            
    print("\n--- Final Sweep Summary ---")
    print(f"{'M (per task)':<15}{'Bijective?':<12}{'Purity (%)':<15}{'Average Accuracy (%)'}")
    for m in sizes:
        res = results[m]
        bij_str = "Yes" if res['bijective'] else "No"
        acc_str = f"{res['average']:.2f}%" if res['bijective'] else "N/A"
        print(f"{m:<15}{bij_str:<12}{res['purity']:.2f}%{acc_str:>18}")

if __name__ == "__main__":
    main()
