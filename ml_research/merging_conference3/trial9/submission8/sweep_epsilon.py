import torch
import numpy as np
from simulate_sandbox import load_digits, set_seed, run_gravimerge, compute_accuracy, compute_jitter
from sweep import run_gravimerge_track

def run_sweep():
    D = 192
    K = 4
    num_seeds = 5
    
    digits = load_digits()
    X = digits.data
    y = digits.target
    task_labels = [0, 1, 2, 3]
    
    G_vals = [0.01, 0.02, 0.03, 0.05]
    epsilon_vals = [0.1, 0.3, 0.5, 0.8, 1.0]
    drag_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"{'G':<6} {'eps':<5} {'drag':<5} | {'Accuracy (%)':<15} {'Jitter (MAD)':<15} {'Movement':<15}")
    print("-" * 70)
    
    for G in G_vals:
        for eps in epsilon_vals:
            for drag in drag_vals:
                acc_list = []
                jitter_list = []
                move_list = []
                
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
                    
                    h_grav, alphas, total_movement = run_gravimerge_track(h3_heterog, centroids, tau=0.05, G=G, epsilon=eps, drag=drag, dt=1.0)
                    
                    acc = compute_accuracy(h_grav, centroids, true_tasks_heterog)
                    jitter = compute_jitter(alphas)
                    
                    acc_list.append(acc)
                    jitter_list.append(jitter)
                    move_list.append(total_movement)
                    
                mean_acc = np.mean(acc_list) * 100
                mean_jitter = np.mean(jitter_list)
                mean_move = np.mean(move_list)
                
                print(f"{G:<6.3f} {eps:<5.1f} {drag:<5.1f} | {mean_acc:<15.2f} {mean_jitter:<15.6f} {mean_move:<15.4f}")

if __name__ == '__main__':
    run_sweep()
