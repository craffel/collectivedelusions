import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from simulate_sandbox import load_digits, set_seed, run_ema, run_chemmerge, run_gravimerge, run_uniform, run_sps_zca, run_sable, run_weight_momentum, compute_accuracy, compute_jitter

def main():
    D = 192
    K = 4
    num_seeds = 5
    
    digits = load_digits()
    X = digits.data
    y = digits.target
    task_labels = [0, 1, 2, 3]
    
    # 1. Sweep EMA
    print("Sweeping EMA smoothing parameter beta...")
    betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ema_accs = []
    ema_jitters = []
    for beta in betas:
        acc_list = []
        jitter_list = []
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
            
            h_ema, alphas = run_ema(test_samples_heterog.clone(), centroids, gamma=0.3, tau=0.05, beta=beta)
            acc_list.append(compute_accuracy(h_ema, centroids, true_tasks_heterog))
            jitter_list.append(compute_jitter(alphas))
        
        ema_accs.append(np.mean(acc_list) * 100)
        ema_jitters.append(np.mean(jitter_list))
        print(f"Beta: {beta:<5.1f} | Acc: {ema_accs[-1]:.2f}% | Jitter: {ema_jitters[-1]:.6f}")
        
    # 2. Sweep ChemMerge
    print("\nSweeping ChemMerge dt parameter...")
    dts = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    chem_accs = []
    chem_jitters = []
    for dt in dts:
        acc_list = []
        jitter_list = []
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
            
            h_chem, alphas = run_chemmerge(test_samples_heterog.clone(), centroids, gamma=0.3, tau=0.05, k_decay=0.3, dt=dt)
            acc_list.append(compute_accuracy(h_chem, centroids, true_tasks_heterog))
            jitter_list.append(compute_jitter(alphas))
            
        chem_accs.append(np.mean(acc_list) * 100)
        chem_jitters.append(np.mean(jitter_list))
        print(f"dt: {dt:<5.1f} | Acc: {chem_accs[-1]:.2f}% | Jitter: {chem_jitters[-1]:.6f}")

    # 3. Sweep GraviMerge
    print("\nSweeping GraviMerge G parameter...")
    Gs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    grav_accs = []
    grav_jitters = []
    for G in Gs:
        acc_list = []
        jitter_list = []
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
            
            h_grav, alphas = run_gravimerge(test_samples_heterog.clone(), centroids, tau=0.05, G=G, epsilon=0.8, drag=0.9, dt=1.0)
            acc_list.append(compute_accuracy(h_grav, centroids, true_tasks_heterog))
            jitter_list.append(compute_jitter(alphas))
            
        grav_accs.append(np.mean(acc_list) * 100)
        grav_jitters.append(np.mean(jitter_list))
        print(f"G: {G:<5.3f} | Acc: {grav_accs[-1]:.2f}% | Jitter: {grav_jitters[-1]:.6f}")

    # 4. Single baselines for comparison
    print("\nComputing single baselines...")
    baselines = {}
    for name, run_fn in [
        ('Uniform', lambda h, c: run_uniform(h, c, gamma=0.3)),
        ('SPS-ZCA', lambda h, c: run_sps_zca(h, c, gamma=0.3, tau=0.05)),
        ('SABLE', lambda h, c: run_sable(h, c, gamma=0.3, tau=0.05)),
        ('WMomentum', lambda h, c: run_weight_momentum(h, c, gamma=0.3, tau=0.05, beta1=0.8, beta2=0.5))
    ]:
        acc_list = []
        jitter_list = []
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
            
            h_out, alphas = run_fn(test_samples_heterog.clone(), centroids)
            acc_list.append(compute_accuracy(h_out, centroids, true_tasks_heterog))
            if alphas is not None:
                jitter_list.append(compute_jitter(alphas))
            else:
                jitter_list.append(0.0)
                
        baselines[name] = (np.mean(acc_list) * 100, np.mean(jitter_list))
        print(f"Baseline {name:<10} | Acc: {baselines[name][0]:.2f}% | Jitter: {baselines[name][1]:.6f}")

    # Plot Pareto Frontier
    plt.figure(figsize=(8, 6))
    
    # Plot EMA curve
    plt.plot(ema_jitters, ema_accs, 'o--', color='tab:blue', linewidth=2, label=r'EMA (sweeping $\beta \in [0.0, 0.9]$)')
    # Annotate EMA beta=0.9
    plt.annotate(r'$\beta=0.9$', xy=(ema_jitters[-1], ema_accs[-1]), xytext=(ema_jitters[-1]+0.001, ema_accs[-1]-0.5),
                 arrowprops=dict(arrowstyle="->", color='tab:blue'), color='tab:blue', fontsize=9)
                 
    # Plot ChemMerge curve
    plt.plot(chem_jitters, chem_accs, 's--', color='tab:green', linewidth=2, label=r'ChemMerge (sweeping $dt \in [0.1, 5.0]$)')
    # Annotate ChemMerge dt=0.5
    idx_dt_05 = dts.index(0.5)
    plt.annotate(r'$dt=0.5$', xy=(chem_jitters[idx_dt_05], chem_accs[idx_dt_05]), xytext=(chem_jitters[idx_dt_05]-0.002, chem_accs[idx_dt_05]+1.0),
                 arrowprops=dict(arrowstyle="->", color='tab:green'), color='tab:green', fontsize=9)
                 
    # Plot GraviMerge curve
    plt.plot(grav_jitters, grav_accs, '^-', color='tab:red', linewidth=3, label=r'GraviMerge (Ours, sweeping $G \in [0.001, 0.2]$)')
    # Annotate GraviMerge G=0.05
    idx_G_05 = Gs.index(0.05)
    plt.annotate(r'$G=0.05$ (Default)', xy=(grav_jitters[idx_G_05], grav_accs[idx_G_05]), xytext=(grav_jitters[idx_G_05]+0.0005, grav_accs[idx_G_05]-1.5),
                 arrowprops=dict(arrowstyle="->", color='tab:red', lw=1.5), color='tab:red', weight='bold', fontsize=10)
    plt.annotate(r'$G=0.001$', xy=(grav_jitters[0], grav_accs[0]), xytext=(grav_jitters[0]+0.0005, grav_accs[0]+0.5),
                 arrowprops=dict(arrowstyle="->", color='tab:red'), color='tab:red', fontsize=9)

    # Plot single baselines
    # SABLE (which is stateless SABLE, equivalent to EMA beta=0.0)
    plt.scatter([baselines['SABLE'][1]], [baselines['SABLE'][0]], marker='*', color='gold', s=200, zorder=5, label='SABLE SOTA')
    
    # SPS-ZCA
    plt.scatter([baselines['SPS-ZCA'][1]], [baselines['SPS-ZCA'][0]], marker='D', color='purple', s=100, zorder=5, label='SPS-ZCA SOTA')
    
    # WMomentum
    plt.scatter([baselines['WMomentum'][1]], [baselines['WMomentum'][0]], marker='X', color='magenta', s=120, zorder=5, label='WMomentum (Simplex Clamped)')
    
    # Uniform
    plt.scatter([baselines['Uniform'][1]], [baselines['Uniform'][0]], marker='p', color='black', s=120, zorder=5, label='Uniform Merging')

    plt.xlabel('Routing Jitter (MAD, lower is more stable)', fontsize=12)
    plt.ylabel('Model Serving Accuracy (%, higher is better)', fontsize=12)
    plt.title('Accuracy-Stability Pareto Frontier in Dynamic Model Merging', fontsize=13, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Focus x-axis on relevant parts
    plt.xlim(-0.0005, 0.018)
    plt.ylim(50, 93)
    
    plt.legend(loc='lower left', fontsize=9.5)
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/fig1.png', dpi=300)
    plt.savefig('submission/fig1.png', dpi=300)
    plt.close()
    print("Pareto frontier figure saved to results/fig1.png and submission/fig1.png successfully!")

if __name__ == '__main__':
    main()
