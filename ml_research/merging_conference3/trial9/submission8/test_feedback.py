import torch
import numpy as np
from simulate_sandbox import load_digits, set_seed, compute_accuracy, compute_jitter, run_gravimerge

def test_feedback():
    D = 192
    K = 4
    num_seeds = 5
    
    digits = load_digits()
    X = digits.data
    y = digits.target
    task_labels = [0, 1, 2, 3]
    
    etas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    print("Testing Closed-Loop Feedback Coupling Variant of GraviMerge...")
    print(f"{'Eta':<10} | {'Accuracy (%)':<15} {'Jitter (MAD)':<15}")
    print("-" * 45)
    
    for eta in etas:
        accs = []
        jitters = []
        for seed in range(num_seeds):
            curr_seed = 42 + seed
            set_seed(curr_seed)
            
            projection_matrix = torch.randn(64, D)
            projection_matrix = projection_matrix / torch.norm(projection_matrix, dim=0, keepdim=True)
            
            task_samples = []
            true_tasks_list = []
            for k in range(K):
                indices = np.where(y == task_labels[k])[0]
                indices = indices[:200]
                data_k = torch.tensor(X[indices], dtype=torch.float32)
                projected_k = data_k @ projection_matrix
                task_samples.append(projected_k)
                true_tasks_list.append(torch.full((len(indices),), k, dtype=torch.long))
                
            centroids = torch.zeros(K, D)
            for k in range(K):
                mean_h3 = torch.mean(task_samples[k][:64], dim=0)
                centroids[k] = mean_h3 / torch.norm(mean_h3)
                
            test_samples_homog_list = []
            true_tasks_homog_list = []
            for k in range(K):
                test_samples_homog_list.append(task_samples[k][64:])
                true_tasks_homog_list.append(true_tasks_list[k][64:])
                
            test_samples_homog = torch.cat(test_samples_homog_list, dim=0)
            true_tasks_homog = torch.cat(true_tasks_homog_list, dim=0)
            
            shuffled_idx = torch.randperm(test_samples_homog.shape[0])
            test_samples_heterog = test_samples_homog[shuffled_idx]
            true_tasks_heterog = true_tasks_homog[shuffled_idx]
            
            h3_heterog = test_samples_heterog.clone()
            
            h_grav, alphas = run_gravimerge(
                h3_heterog, centroids, tau=0.05, G=0.05, epsilon=0.8, drag=0.9, dt=1.0, feedback_eta=eta
            )
            
            accs.append(compute_accuracy(h_grav, centroids, true_tasks_heterog))
            jitters.append(compute_jitter(alphas))
            
        print(f"{eta:<10.3f} | {np.mean(accs)*100:<15.2f} {np.mean(jitters):<15.6f}")

if __name__ == '__main__':
    test_feedback()
