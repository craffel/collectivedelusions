import torch
import numpy as np
from run_experiments import get_dataloaders, extract_hybrid

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load loaders
    loaders = get_dataloaders(seed=42)
    tasks = ['mnist', 'fmnist', 'cifar']
    
    # Collect hybrid features for 1000 test images per task
    num_samples = 1000
    task_features = {}
    
    for task in tasks:
        feats_list = []
        count = 0
        loader = loaders[task]['test']
        for inputs, _ in loader:
            inputs = inputs.to(device)
            feats = extract_hybrid(inputs, bins=16, size=6)
            feats_list.append(feats.cpu())
            count += inputs.size(0)
            if count >= num_samples:
                break
        task_features[task] = torch.cat(feats_list)[:num_samples].numpy()
        print(f"Collected {task_features[task].shape[0]} samples for {task}")

    # Compute means and covariances
    means = {}
    covs = {}
    for task in tasks:
        feats = task_features[task]
        means[task] = np.mean(feats, axis=0)
        covs[task] = np.cov(feats, rowvar=False)

    print("\n================ Pairwise Mean Cosine Similarities ================")
    # Pairwise mean cosine similarity
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            t1, t2 = tasks[i], tasks[j]
            m1, m2 = means[t1], means[t2]
            sim = np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2))
            print(f"Cosine Similarity ({t1.upper()} Mean, {t2.upper()} Mean): {sim:.4f}")

    # Also check spectral and spatial components separately
    print("\n================ Pairwise Spectral (Bins 0-15) Mean Cosine Similarities ================")
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            t1, t2 = tasks[i], tasks[j]
            m1, m2 = means[t1][:16], means[t2][:16]
            sim = np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2))
            print(f"Spectral Cosine Similarity ({t1.upper()} Mean, {t2.upper()} Mean): {sim:.4f}")

    print("\n================ Pairwise Spatial (Bins 16-51) Mean Cosine Similarities ================")
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            t1, t2 = tasks[i], tasks[j]
            m1, m2 = means[t1][16:], means[t2][16:]
            sim = np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2))
            print(f"Spatial Cosine Similarity ({t1.upper()} Mean, {t2.upper()} Mean): {sim:.4f}")

    # Compute between-class scatter Sb and within-class scatter Sw
    for feat_name, feat_slice in [('Spectral (16D)', slice(0, 16)), ('Spatial (36D)', slice(16, 52)), ('Hybrid (52D)', slice(0, 52))]:
        D_slice = feat_slice.stop - feat_slice.start
        global_mean = np.mean([means[task][feat_slice] for task in tasks], axis=0)
        
        Sb = np.zeros((D_slice, D_slice))
        for task in tasks:
            diff = (means[task][feat_slice] - global_mean).reshape(-1, 1)
            Sb += diff @ diff.T
            
        Sw = np.zeros((D_slice, D_slice))
        for task in tasks:
            Sw += covs[task][feat_slice, feat_slice]
            
        # Standardize Sw with a small diagonal term for numerical stability
        Sw_reg = Sw + 1e-4 * np.eye(D_slice)
        
        # Compute trace ratio
        Sw_inv = np.linalg.inv(Sw_reg)
        trace_ratio = np.trace(Sw_inv @ Sb)
        print(f"\nFeature Type: {feat_name}")
        print(f"  Within-class Scatter Sw eigenvalues min: {np.linalg.eigvalsh(Sw).min():.4e}, max: {np.linalg.eigvalsh(Sw).max():.4f}")
        print(f"  Between-class Scatter Sb eigenvalues min: {np.linalg.eigvalsh(Sb).min():.4e}, max: {np.linalg.eigvalsh(Sb).max():.4f}")
        print(f"  Separability Trace Ratio J = Tr(Sw^-1 Sb): {trace_ratio:.4f}")

if __name__ == '__main__':
    main()
