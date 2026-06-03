import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from run_experiments import get_datasets, create_base_resnet, ExpertModel, merge_backbones, apply_n_taac

device = torch.device("cpu")

def main():
    print("--- Running Unsupervised Prototype Routing Experiment ---")
    subsets = get_datasets()
    sorted_tasks = ['mnist', 'fmnist', 'cifar']
    expert_paths = [f"expert_{name}.pth" for name in sorted_tasks]
    
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
    
    # Reconstruct N-TAAC backbone
    merged_backbone = merge_backbones(expert_paths)
    n_taac_backbone = apply_n_taac(merged_backbone, subsets)
    n_taac_backbone.eval()
    
    # 1. Collect Layer 2 activations from the entire joint calibration set WITHOUT task labels
    # Joint calibration set contains 128 MNIST + 128 F-MNIST + 128 CIFAR = 384 samples
    anchor_act = None
    def anchor_hook(module, input, output):
        nonlocal anchor_act
        anchor_act = output.detach()
        
    hook_handle = n_taac_backbone.layer2.register_forward_hook(anchor_hook)
    
    all_activations = []
    true_labels = [] # To verify cluster alignment (0: MNIST, 1: F-MNIST, 2: CIFAR)
    
    for label_idx, name in enumerate(sorted_tasks):
        cal_loader = DataLoader(subsets[name]['cal'], batch_size=1, shuffle=False) # batch_size=1 for precision
        with torch.no_grad():
            for x, _ in cal_loader:
                n_taac_backbone(x)
                pooled = anchor_act.mean(dim=[2, 3]) # [1, 128]
                pooled_norm = pooled / (pooled.norm(p=2) + 1e-8)
                all_activations.append(pooled_norm.squeeze(0).numpy())
                true_labels.append(label_idx)
                
    hook_handle.remove()
    
    all_activations = np.array(all_activations) # [384, 128]
    true_labels = np.array(true_labels) # [384]
    
    # 2. Run K-Means with K=3 on unlabeled activations
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(all_activations)
    
    # 3. Analyze Cluster Content (Alignment with true tasks)
    print("\nCluster-to-Task Alignment Analysis:")
    cluster_to_task_mapping = {}
    for cluster_id in range(3):
        indices_in_cluster = np.where(cluster_labels == cluster_id)[0]
        true_labels_in_cluster = true_labels[indices_in_cluster]
        counts = Counter(true_labels_in_cluster)
        
        # Pretty print cluster content
        task_counts_str = ", ".join([f"{name.upper()}: {counts[idx]}" for idx, name in enumerate(sorted_tasks)])
        print(f"  Cluster {cluster_id}: {task_counts_str} (Total: {len(indices_in_cluster)})")
        
        # Map cluster to the majority true task
        majority_task_idx = counts.most_common(1)[0][0]
        majority_task_name = sorted_tasks[majority_task_idx]
        cluster_to_task_mapping[cluster_id] = majority_task_name
        print(f"    Mapped Cluster {cluster_id} -> {majority_task_name.upper()}")
        
    # Check if we have a bijection (each cluster mapped to a unique task)
    mapped_tasks = list(cluster_to_task_mapping.values())
    is_bijective = len(set(mapped_tasks)) == 3
    print(f"  Unsupervised clustering is BIJECTIVE? {is_bijective}")
    
    # 4. Use cluster centroids as our task prototypes for test-time routing
    cluster_centroids = kmeans.cluster_centers_ # [3, 128]
    # Re-normalize centroids to unit L2-norm
    cluster_centroids_norm = cluster_centroids / (np.linalg.norm(cluster_centroids, axis=1, keepdims=True) + 1e-8)
    prototypes = torch.tensor(cluster_centroids_norm, dtype=torch.float32) # [3, 128]
    
    # 5. Evaluate Unsupervised Prototype Routing
    beta = 20.0
    print(f"\nEvaluating SRAC with Unsupervised Prototypes (Beta={beta}):")
    
    routing_weights = None
    def test_anchor_hook(module, input, output):
        nonlocal routing_weights
        pooled = output.mean(dim=[2, 3])
        pooled_norm = pooled / (pooled.norm(p=2, dim=1, keepdim=True) + 1e-8) # [B, 128]
        
        sims = []
        for c in range(3):
            proto = prototypes[c] # [128]
            sim = torch.sum(pooled_norm * proto.unsqueeze(0), dim=1) # [B]
            sims.append(sim)
        sims = torch.stack(sims, dim=1) # [B, 3]
        
        # Softmax over clusters
        weights_over_clusters = torch.softmax(beta * sims, dim=1) # [B, 3]
        
        # Map cluster weights back to task weights
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
    avg_acc = np.mean([accuracies[k] for k in sorted_tasks])
    
    print(f"\nResults of Unsupervised Prototype Routing:")
    print(f"  Average Accuracy: {avg_acc:.2f}% (MNIST: {accuracies['mnist']:.2f}%, F-MNIST: {accuracies['fmnist']:.2f}%, CIFAR: {accuracies['cifar']:.2f}%)")
    print(f"  (For comparison, supervised routing at Beta=20.0 got 52.36%)")

if __name__ == "__main__":
    main()
