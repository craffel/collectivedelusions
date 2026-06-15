import torch
import numpy as np
from simulate_sandbox import load_digits, set_seed, run_gravimerge, compute_accuracy, compute_jitter

def run_sweep():
    D = 192
    K = 4
    num_seeds = 5 # use 5 seeds for speed in sweep
    
    digits = load_digits()
    X = digits.data
    y = digits.target
    task_labels = [0, 1, 2, 3]
    
    G_vals = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    drag_vals = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    
    print(f"{'G':<6} {'drag':<6} | {'Accuracy (%)':<15} {'Jitter (MAD)':<15} {'Movement':<15}")
    print("-" * 60)
    
    best_acc = 0.0
    best_jitter = 1.0
    best_G = 0.0
    best_drag = 0.0
    
    for G in G_vals:
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
                
                # In order to measure movement, we'll modify run_gravimerge to also return total_movement
                # But since run_gravimerge doesn't return movement directly, let's write a custom run_gravimerge_track inside sweep
                h_grav, alphas, total_movement = run_gravimerge_track(h3_heterog, centroids, tau=0.05, G=G, epsilon=0.1, drag=drag, dt=1.0)
                
                acc = compute_accuracy(h_grav, centroids, true_tasks_heterog)
                jitter = compute_jitter(alphas)
                
                acc_list.append(acc)
                jitter_list.append(jitter)
                move_list.append(total_movement)
                
            mean_acc = np.mean(acc_list) * 100
            mean_jitter = np.mean(jitter_list)
            mean_move = np.mean(move_list)
            
            print(f"{G:<6.3f} {drag:<6.2f} | {mean_acc:<15.2f} {mean_jitter:<15.6f} {mean_move:<15.4f}")

def run_gravimerge_track(h3, centroids, gamma=0.3, tau=0.05, G=0.002, epsilon=0.1, drag=0.5, dt=1.0, num_layers=14):
    N, D = h3.shape
    K = centroids.shape[0]
    h = h3.clone()
    h_sc = h3 / torch.norm(h3, dim=1, keepdim=True)
    v = torch.zeros((N, D))
    cos_sim3 = h_sc @ centroids.t()
    sim_max, _ = torch.max(cos_sim3, dim=1, keepdim=True)
    M = torch.exp((cos_sim3 - sim_max) / tau)
    alpha_history = []
    total_movement = 0.0
    for l in range(4, num_layers + 1):
        cos_sim_sc = h_sc @ centroids.t()
        r = torch.sqrt(torch.clamp(2.0 * (1.0 - cos_sim_sc), min=1e-8))
        force_mag = G * M / (r**2 + epsilon**2)
        alpha = force_mag / torch.sum(force_mag, dim=1, keepdim=True)
        alpha_history.append(alpha.clone())
        diff = centroids.unsqueeze(0) - h_sc.unsqueeze(1)
        diff_norm = torch.norm(diff, dim=2, keepdim=True)
        u_hat = diff / torch.clamp(diff_norm, min=1e-8)
        force_vecs = force_mag.unsqueeze(2) * u_hat
        a = torch.sum(force_vecs, dim=1)
        a_tangent = a - torch.sum(a * h_sc, dim=1, keepdim=True) * h_sc
        v_tentative = drag * v + a_tangent * dt
        v_tangent = v_tentative - torch.sum(v_tentative * h_sc, dim=1, keepdim=True) * h_sc
        v_norm = torch.norm(v_tangent, dim=1, keepdim=True)
        v_norm_clamp = torch.clamp(v_norm, min=1e-8)
        h_sc_new = torch.cos(v_norm * dt) * h_sc + torch.sin(v_norm * dt) * (v_tangent / v_norm_clamp)
        h_sc_new = h_sc_new / torch.norm(h_sc_new, dim=1, keepdim=True)
        total_movement += torch.mean(v_norm).item()
        cos_theta = torch.sum(h_sc * h_sc_new, dim=1, keepdim=True)
        proj_coeff = torch.sum(v_tangent * h_sc_new, dim=1, keepdim=True) / (1.0 + cos_theta)
        v = v_tangent - (h_sc + h_sc_new) * proj_coeff
        h_sc = h_sc_new
        blended_centroid = alpha @ centroids
        h_new = h + gamma * (blended_centroid - h)
        h = h_new / torch.norm(h_new, dim=1, keepdim=True)
    return h, torch.stack(alpha_history, dim=1), total_movement

if __name__ == '__main__':
    run_sweep()
