import torch
import numpy as np
from simulate_sandbox import load_digits, set_seed, run_ema, run_chemmerge, run_gravimerge, compute_accuracy, compute_jitter

def main():
    D = 192
    K = 4
    num_seeds = 5 # use 5 seeds for speed
    
    digits = load_digits()
    X = digits.data
    y = digits.target
    task_labels = [0, 1, 2, 3]
    
    print("Sweeping EMA smoothing parameter beta...")
    betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for beta in betas:
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
            
            h_ema, alphas = run_ema(h3_heterog, centroids, gamma=0.3, tau=0.05, beta=beta)
            accs.append(compute_accuracy(h_ema, centroids, true_tasks_heterog))
            jitters.append(compute_jitter(alphas))
            
        print(f"Beta: {beta:<5.1f} | Accuracy: {np.mean(accs)*100:<6.2f}% | Jitter (MAD): {np.mean(jitters):<8.6f}")
        
    print("\nSweeping ChemMerge dt parameter...")
    dts = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    for dt in dts:
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
            
            h_chem, alphas = run_chemmerge(h3_heterog, centroids, gamma=0.3, tau=0.05, k_decay=0.3, dt=dt)
            accs.append(compute_accuracy(h_chem, centroids, true_tasks_heterog))
            jitters.append(compute_jitter(alphas))
            
        print(f"dt: {dt:<5.1f} | Accuracy: {np.mean(accs)*100:<6.2f}% | Jitter (MAD): {np.mean(jitters):<8.6f}")

    print("\nSweeping GraviMerge G parameter (with epsilon=0.8, drag=0.9)...")
    Gs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    for G in Gs:
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
            
            h_grav, alphas = run_gravimerge(h3_heterog, centroids, tau=0.05, G=G, epsilon=0.8, drag=0.9, dt=1.0)
            accs.append(compute_accuracy(h_grav, centroids, true_tasks_heterog))
            jitters.append(compute_jitter(alphas))
            
        print(f"G: {G:<5.3f} | Accuracy: {np.mean(accs)*100:<6.2f}% | Jitter (MAD): {np.mean(jitters):<8.6f}")

if __name__ == '__main__':
    main()
