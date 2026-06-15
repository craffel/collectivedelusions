import torch
import numpy as np
from simulate_sandbox import load_digits, set_seed, run_gravimerge

def run_ood_study():
    D = 192
    K = 4
    num_seeds = 5
    
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    id_labels = [0, 1, 2, 3]
    ood_labels = [5, 6, 7, 8]  # Out of distribution
    
    print("=" * 60)
    print("Testing Out-of-Distribution (OOD) Sentinel Attractor Dynamics (SAD)")
    print("=" * 60)
    
    # We will measure the average standard deviation of ensembling weights alpha.
    # If the weights are uniform (alpha = 1/K = 0.25), the standard deviation is 0.0.
    # If the weights are highly peaked (e.g., [1, 0, 0, 0]), the standard deviation is ~0.433.
    # For OOD inputs, we want the ensembling weights to be highly uniform (low standard deviation),
    # meaning the system gracefully falls back to uniform ensembling rather than drifting to a random expert.
    
    avg_id_std = []
    avg_id_sad = []
    avg_ood_std = []
    avg_ood_sad = []
    
    for seed in range(num_seeds):
        curr_seed = 42 + seed
        set_seed(curr_seed)
        
        projection_matrix = torch.randn(64, D)
        projection_matrix = projection_matrix / torch.norm(projection_matrix, dim=0, keepdim=True)
        
        # Build centroids for ID tasks (0, 1, 2, 3)
        id_samples = []
        for k in range(K):
            indices = np.where(y == id_labels[k])[0]
            indices = indices[:200]
            data_k = torch.tensor(X[indices], dtype=torch.float32)
            projected_k = data_k @ projection_matrix
            id_samples.append(projected_k)
            
        centroids = torch.zeros(K, D)
        for k in range(K):
            mean_h3 = torch.mean(id_samples[k][:64], dim=0)
            centroids[k] = mean_h3 / torch.norm(mean_h3)
            
        # Get ID test samples (data not used for centroids)
        test_samples_id_list = []
        for k in range(K):
            test_samples_id_list.append(id_samples[k][64:])
        test_samples_id = torch.cat(test_samples_id_list, dim=0)
        
        # Get OOD test samples
        test_samples_ood_list = []
        for ood_label in ood_labels:
            indices = np.where(y == ood_label)[0]
            indices = indices[:136]  # match size
            data_k = torch.tensor(X[indices], dtype=torch.float32)
            projected_k = data_k @ projection_matrix
            test_samples_ood_list.append(projected_k)
        test_samples_ood = torch.cat(test_samples_ood_list, dim=0)
        
        # We normalize test samples like standard initialization
        h3_id = test_samples_id.clone()
        h3_ood = test_samples_ood.clone()
        
        # Standard vs. Safeguarded GraviMerge
        with torch.no_grad():
            h3_id_norm = h3_id / torch.norm(h3_id, dim=1, keepdim=True)
            cos_id = h3_id_norm @ centroids.t()
            max_sim_id = torch.mean(torch.max(cos_id, dim=1)[0]).item()
            
            h3_ood_norm = h3_ood / torch.norm(h3_ood, dim=1, keepdim=True)
            cos_ood = h3_ood_norm @ centroids.t()
            max_sim_ood = torch.mean(torch.max(cos_ood, dim=1)[0]).item()
            
        # Use delta_ood as threshold
        delta_ood = (max_sim_id + max_sim_ood) / 2.0
        
        # Run standard GraviMerge on ID and OOD
        _, alpha_id_std = run_gravimerge(h3_id, centroids, ood_safe=False)
        _, alpha_ood_std = run_gravimerge(h3_ood, centroids, ood_safe=False)
        
        # Run OOD-Safeguarded GraviMerge (SAD) on ID and OOD
        _, alpha_id_sad = run_gravimerge(h3_id, centroids, ood_safe=True, delta_ood=delta_ood, tau_ood=0.05)
        _, alpha_ood_sad = run_gravimerge(h3_ood, centroids, ood_safe=True, delta_ood=delta_ood, tau_ood=0.05)
        
        # Compute standard deviations of ensembling weights across categories (mean over layers and batches)
        std_id_std = torch.mean(torch.std(alpha_id_std, dim=2)).item()
        std_ood_std = torch.mean(torch.std(alpha_ood_std, dim=2)).item()
        
        std_id_sad = torch.mean(torch.std(alpha_id_sad, dim=2)).item()
        std_ood_sad = torch.mean(torch.std(alpha_ood_sad, dim=2)).item()
        
        avg_id_std.append(std_id_std)
        avg_id_sad.append(std_id_sad)
        avg_ood_std.append(std_ood_std)
        avg_ood_sad.append(std_ood_sad)
        
        if seed == 0:
            print(f"Seed {curr_seed}:")
            print(f"  Mean Max Similarity - ID: {max_sim_id:.4f} | OOD: {max_sim_ood:.4f}")
            print(f"  OOD Threshold (delta_ood): {delta_ood:.4f}")
            print(f"  Ensembling Weight Standard Deviation (Lower = More Uniform/Safeguarded):")
            print(f"    - ID (Standard)     : {std_id_std:.4f}")
            print(f"    - ID (Safeguarded)  : {std_id_sad:.4f}")
            print(f"    - OOD (Standard)    : {std_ood_std:.4f}")
            print(f"    - OOD (Safeguarded) : {std_ood_sad:.4f}")
            print("-" * 60)
            
    print(f"Summary over {num_seeds} seeds:")
    print(f"  ID (Standard)     : {np.mean(avg_id_std):.4f} ± {np.std(avg_id_std):.4f}")
    print(f"  ID (Safeguarded)  : {np.mean(avg_id_sad):.4f} ± {np.std(avg_id_sad):.4f}")
    print(f"  OOD (Standard)    : {np.mean(avg_ood_std):.4f} ± {np.std(avg_ood_std):.4f}")
    print(f"  OOD (Safeguarded) : {np.mean(avg_ood_sad):.4f} ± {np.std(avg_ood_sad):.4f}")
    print("=" * 60)
    print("OOD Resilience Study Completed Successfully.")

if __name__ == '__main__':
    run_ood_study()
