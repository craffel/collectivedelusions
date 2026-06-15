import torch
import numpy as np
from simulate_sandbox import load_digits, set_seed, run_uniform, run_sps_zca, run_sable, run_chemmerge, run_ema, run_gravimerge, compute_accuracy, compute_jitter

# Second-order Weight Momentum Filter
def run_weight_momentum(h3, centroids, gamma=0.3, tau=0.05, beta1=0.9, beta2=0.9, num_layers=14):
    N, D = h3.shape
    K = centroids.shape[0]
    h = h3.clone()
    
    alpha_history = []
    alpha_smooth = torch.full((N, K), 1.0 / K)
    v_alpha = torch.zeros((N, K))
    
    for l in range(4, num_layers + 1):
        h_norm = h / torch.norm(h, dim=1, keepdim=True)
        cos_sim = h_norm @ centroids.t()
        alpha_instant = torch.softmax(cos_sim / tau, dim=1)
        
        # Second-order momentum update
        v_alpha = beta1 * v_alpha + (1.0 - beta1) * (alpha_instant - alpha_smooth)
        alpha_smooth = alpha_smooth + beta2 * v_alpha
        # Normalize to prevent drift and ensure valid ensembling weights
        alpha_smooth = torch.clamp(alpha_smooth, min=1e-8)
        alpha = alpha_smooth / torch.sum(alpha_smooth, dim=1, keepdim=True)
        alpha_history.append(alpha.clone())
        
        blended_centroid = alpha @ centroids
        h_new = h + gamma * (blended_centroid - h)
        h = h_new / torch.norm(h_new, dim=1, keepdim=True)
        
    return h, torch.stack(alpha_history, dim=1)

def run_comparison():
    D = 192
    K = 4
    num_seeds = 10
    
    digits = load_digits()
    X = digits.data
    y = digits.target
    task_labels = [0, 1, 2, 3]
    
    accs = {'SPS-ZCA': [], 'SABLE': [], 'EMA': [], 'WMomentum': [], 'GraviMerge': []}
    jitters = {'SPS-ZCA': [], 'SABLE': [], 'EMA': [], 'WMomentum': [], 'GraviMerge': []}
    
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
        
        h3 = test_samples_heterog.clone()
        
        # Run SPS-ZCA
        h, alphas = run_sps_zca(h3, centroids)
        accs['SPS-ZCA'].append(compute_accuracy(h, centroids, true_tasks_heterog))
        jitters['SPS-ZCA'].append(compute_jitter(alphas))
        
        # Run SABLE
        h, alphas = run_sable(h3, centroids)
        accs['SABLE'].append(compute_accuracy(h, centroids, true_tasks_heterog))
        jitters['SABLE'].append(compute_jitter(alphas))
        
        # Run EMA
        h, alphas = run_ema(h3, centroids)
        accs['EMA'].append(compute_accuracy(h, centroids, true_tasks_heterog))
        jitters['EMA'].append(compute_jitter(alphas))
        
        # Run WMomentum (second-order)
        h, alphas = run_weight_momentum(h3, centroids, beta1=0.8, beta2=0.5)
        accs['WMomentum'].append(compute_accuracy(h, centroids, true_tasks_heterog))
        jitters['WMomentum'].append(compute_jitter(alphas))
        
        # Run GraviMerge (ours)
        h, alphas = run_gravimerge(h3, centroids, tau=0.05, G=0.05, epsilon=0.8, drag=0.9)
        accs['GraviMerge'].append(compute_accuracy(h, centroids, true_tasks_heterog))
        jitters['GraviMerge'].append(compute_jitter(alphas))
        
    print("\n" + "="*50)
    print("COMPARISON RESULTS (Mean ± Std %)")
    print("="*50)
    for m in accs:
        print(f"{m:<12} | Accuracy: {np.mean(accs[m])*100:.2f}% ± {np.std(accs[m])*100:.2f}% | Jitter: {np.mean(jitters[m]):.5f}")

if __name__ == '__main__':
    run_comparison()
