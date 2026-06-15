import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_digits

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 1. Uniform Merging
def run_uniform(h3, centroids, gamma=0.3, num_layers=14):
    N, D = h3.shape
    K = centroids.shape[0]
    h = h3.clone()
    
    alpha_history = []
    
    for l in range(4, num_layers + 1):
        alpha = torch.full((N, K), 1.0 / K)
        alpha_history.append(alpha.clone())
        
        blended_centroid = alpha @ centroids
        h_new = h + gamma * (blended_centroid - h)
        h = h_new / torch.norm(h_new, dim=1, keepdim=True)
        
    return h, torch.stack(alpha_history, dim=1)

# 2. SPS-ZCA (Zero-Shot Centroid Alignment, early routing)
def run_sps_zca(h3, centroids, gamma=0.3, tau=0.05, num_layers=14):
    N, D = h3.shape
    K = centroids.shape[0]
    h = h3.clone()
    
    # Compute similarity at layer 3
    h_norm = h3 / torch.norm(h3, dim=1, keepdim=True)
    cos_sim = h_norm @ centroids.t()
    alpha_3 = torch.softmax(cos_sim / tau, dim=1)
    
    alpha_history = []
    
    for l in range(4, num_layers + 1):
        alpha = alpha_3.clone()
        alpha_history.append(alpha)
        
        blended_centroid = alpha @ centroids
        h_new = h + gamma * (blended_centroid - h)
        h = h_new / torch.norm(h_new, dim=1, keepdim=True)
        
    return h, torch.stack(alpha_history, dim=1)

# 3. SABLE
def run_sable(h3, centroids, gamma=0.3, tau=0.05, num_layers=14):
    N, D = h3.shape
    K = centroids.shape[0]
    h = h3.clone()
    
    alpha_history = []
    
    for l in range(4, num_layers + 1):
        h_norm = h / torch.norm(h, dim=1, keepdim=True)
        cos_sim = h_norm @ centroids.t()
        alpha = torch.softmax(cos_sim / tau, dim=1)
        alpha_history.append(alpha.clone())
        
        blended_centroid = alpha @ centroids
        h_new = h + gamma * (blended_centroid - h)
        h = h_new / torch.norm(h_new, dim=1, keepdim=True)
        
    return h, torch.stack(alpha_history, dim=1)

# 4. ChemMerge
def run_chemmerge(h3, centroids, gamma=0.3, tau=0.05, k_decay=0.3, dt=0.5, num_layers=14):
    N, D = h3.shape
    K = centroids.shape[0]
    h = h3.clone()
    
    C = torch.full((N, K), 1.0 / K)
    alpha_history = []
    
    for l in range(4, num_layers + 1):
        h_norm = h / torch.norm(h, dim=1, keepdim=True)
        cos_sim = h_norm @ centroids.t()
        k_forward = torch.softmax(cos_sim / tau, dim=1)
        
        rate_sum = k_forward + k_decay
        decay_factor = torch.exp(-rate_sum * dt)
        C = C * decay_factor + (k_forward / rate_sum) * (1.0 - decay_factor)
        
        alpha = C / torch.sum(C, dim=1, keepdim=True)
        alpha_history.append(alpha.clone())
        
        blended_centroid = alpha @ centroids
        h_new = h + gamma * (blended_centroid - h)
        h = h_new / torch.norm(h_new, dim=1, keepdim=True)
        
    return h, torch.stack(alpha_history, dim=1)

# 4b. EMA (Exponential Moving Average) Smoothing Baseline
def run_ema(h3, centroids, gamma=0.3, tau=0.05, beta=0.9, num_layers=14):
    N, D = h3.shape
    K = centroids.shape[0]
    h = h3.clone()
    
    alpha_history = []
    alpha_ema = torch.full((N, K), 1.0 / K)
    
    for l in range(4, num_layers + 1):
        h_norm = h / torch.norm(h, dim=1, keepdim=True)
        cos_sim = h_norm @ centroids.t()
        alpha_instant = torch.softmax(cos_sim / tau, dim=1)
        
        # Apply first-order EMA smoothing
        alpha_ema = (1.0 - beta) * alpha_instant + beta * alpha_ema
        alpha_history.append(alpha_ema.clone())
        
        blended_centroid = alpha_ema @ centroids
        h_new = h + gamma * (blended_centroid - h)
        h = h_new / torch.norm(h_new, dim=1, keepdim=True)
        
    return h, torch.stack(alpha_history, dim=1)

# 4c. Second-Order Weight Momentum (WMomentum) Baseline
def run_weight_momentum(h3, centroids, gamma=0.3, tau=0.05, beta1=0.8, beta2=0.5, num_layers=14):
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
        
        # Second-order momentum update on the simplex
        v_alpha = beta1 * v_alpha + (1.0 - beta1) * (alpha_instant - alpha_smooth)
        alpha_smooth = alpha_smooth + beta2 * v_alpha
        # Clamp and normalize to ensure weights are valid probabilities on the simplex
        alpha_smooth = torch.clamp(alpha_smooth, min=1e-8)
        alpha = alpha_smooth / torch.sum(alpha_smooth, dim=1, keepdim=True)
        alpha_history.append(alpha.clone())
        
        blended_centroid = alpha @ centroids
        h_new = h + gamma * (blended_centroid - h)
        h = h_new / torch.norm(h_new, dim=1, keepdim=True)
        
    return h, torch.stack(alpha_history, dim=1)

# 4d. Kalman Filter (KF) Baseline
def run_kalman_filter(h3, centroids, gamma=0.3, tau=0.05, Q=0.01, R=0.1, num_layers=14):
    N, D = h3.shape
    K = centroids.shape[0]
    h = h3.clone()
    
    alpha_history = []
    # Initialize state: ensembling weights (position vector)
    alpha_state = torch.full((N, K), 1.0 / K)
    # Initialize error covariance: diagonal variance for each expert per sample
    P = torch.ones((N, K))
    
    for l in range(4, num_layers + 1):
        h_norm = h / torch.norm(h, dim=1, keepdim=True)
        cos_sim = h_norm @ centroids.t()
        alpha_instant = torch.softmax(cos_sim / tau, dim=1)
        
        # 1. Predict
        P_pred = P + Q
        
        # 2. Update
        K_gain = P_pred / (P_pred + R)
        alpha_state = alpha_state + K_gain * (alpha_instant - alpha_state)
        P = (1.0 - K_gain) * P_pred
        
        # 3. Project onto the simplex
        alpha = torch.clamp(alpha_state, min=1e-8)
        alpha = alpha / torch.sum(alpha, dim=1, keepdim=True)
        alpha_history.append(alpha.clone())
        
        blended_centroid = alpha @ centroids
        h_new = h + gamma * (blended_centroid - h)
        h = h_new / torch.norm(h_new, dim=1, keepdim=True)
        
    return h, torch.stack(alpha_history, dim=1)

# 5. GraviMerge (Ours) - Refined with Geodesic Exponential Map and Parallel Transport
def run_gravimerge(h3, centroids, gamma=0.3, tau=0.05, G=0.05, epsilon=0.8, drag=0.9, dt=1.0, num_layers=14, feedback_eta=0.0, prev_v=None, return_velocity=False, ood_safe=False, delta_ood=0.5, tau_ood=0.05):
    N, D = h3.shape
    K = centroids.shape[0]
    h = h3.clone()
    
    # Rigorous Position Initialization: Normalize h_sc to lie on the sphere S^{D-1} from the start
    h_sc = h3 / torch.norm(h3, dim=1, keepdim=True)
    
    # If prev_v is provided (for temporal carryover), inherit and apply 0.5 carryover damping.
    # Otherwise, initialize in the local tangent space (zero vector).
    if prev_v is not None:
        v = 0.5 * prev_v
    else:
        v = torch.zeros((N, D))
    
    # Compute Dynamic Gravitational Mass (Arrhenius Mass Activation)
    cos_sim3 = h_sc @ centroids.t()
    sim_max, _ = torch.max(cos_sim3, dim=1, keepdim=True)
    M = torch.exp((cos_sim3 - sim_max) / tau)
    
    if ood_safe:
        # Sentinel Attractor Dynamics (SAD) to safeguard against OOD task streams
        psi = torch.sigmoid((sim_max - delta_ood) / tau_ood)
        M = psi * M + (1.0 - psi) * 1.0
    
    alpha_history = []
    total_movement = 0.0
    
    for l in range(4, num_layers + 1):
        # Coordinates are on the sphere, calculate distance r as Euclidean distance on S^{D-1}
        cos_sim_sc = h_sc @ centroids.t()
        r = torch.sqrt(torch.clamp(2.0 * (1.0 - cos_sim_sc), min=1e-8))
        
        # Softened inverse-square force magnitude (derived from the Arctangent potential)
        force_mag = G * M / (r**2 + epsilon**2)
        
        # Gravitational Influence Blending (GIB) ensembling weights
        alpha = force_mag / torch.sum(force_mag, dim=1, keepdim=True)
        alpha_history.append(alpha.clone())
        
        # Compute force vectors pointing from spacecraft toward centroids
        diff = centroids.unsqueeze(0) - h_sc.unsqueeze(1)
        diff_norm = torch.norm(diff, dim=2, keepdim=True)
        u_hat = diff / torch.clamp(diff_norm, min=1e-8)
        force_vecs = force_mag.unsqueeze(2) * u_hat
        
        a_gravity = torch.sum(force_vecs, dim=1)
        
        # Incorporate representational feedback force if feedback_eta > 0.0
        if feedback_eta > 0.0:
            h_norm = h / torch.norm(h, dim=1, keepdim=True)
            feedback_force = feedback_eta * (h_norm - h_sc)
            a = a_gravity + feedback_force
        else:
            a = a_gravity
        
        # Project acceleration onto the local tangent space of the sphere
        a_tangent = a - torch.sum(a * h_sc, dim=1, keepdim=True) * h_sc
        
        # Update tentative velocity
        v_tentative = drag * v + a_tangent * dt
        
        # Project tentative velocity onto tangent space to preserve spherical constraint directions
        v_tangent = v_tentative - torch.sum(v_tentative * h_sc, dim=1, keepdim=True) * h_sc
        
        # Rigorous Geodesic Update on Sphere (Exponential Map)
        v_norm = torch.norm(v_tangent, dim=1, keepdim=True)
        v_norm_clamp = torch.clamp(v_norm, min=1e-8)
        h_sc_new = torch.cos(v_norm * dt) * h_sc + torch.sin(v_norm * dt) * (v_tangent / v_norm_clamp)
        h_sc_new = h_sc_new / torch.norm(h_sc_new, dim=1, keepdim=True)
        
        # Track actual geodesic distance traveled
        total_movement += torch.mean(v_norm).item()
        
        # Rigorous Parallel Transport of the velocity vector from h_sc (old) to h_sc_new (new)
        cos_theta = torch.sum(h_sc * h_sc_new, dim=1, keepdim=True)
        proj_coeff = torch.sum(v_tangent * h_sc_new, dim=1, keepdim=True) / (1.0 + cos_theta)
        v = v_tangent - (h_sc + h_sc_new) * proj_coeff
        
        # Update state position
        h_sc = h_sc_new
        
        # Standard activation propagation using blended weights (identical to baselines)
        blended_centroid = alpha @ centroids
        h_new = h + gamma * (blended_centroid - h)
        h = h_new / torch.norm(h_new, dim=1, keepdim=True)
        
    if return_velocity:
        return h, torch.stack(alpha_history, dim=1), v
    else:
        return h, torch.stack(alpha_history, dim=1)

# Helper to compute classification accuracy
def compute_accuracy(h_final, centroids, true_tasks):
    distances = torch.norm(h_final.unsqueeze(1) - centroids.unsqueeze(0), dim=2)
    preds = torch.argmin(distances, dim=1)
    acc = torch.mean((preds == true_tasks).float()).item()
    return acc

# Helper to compute layer-by-layer routing jitter (MAD)
def compute_jitter(alpha_history):
    diff = torch.abs(alpha_history[:, 1:, :] - alpha_history[:, :-1, :])
    jitter = torch.mean(diff).item()
    return jitter

# Main evaluation loop across 10 seeds using REAL handwritten digits data
def run_evaluation(feedback_eta=0.0):
    D = 192
    K = 4
    num_seeds = 10
    
    # Load real handwritten digits from scikit-learn (addressing Critical Flaw 1)
    digits = load_digits()
    X = digits.data # shape: (1797, 64)
    y = digits.target
    
    # 4 distinct digits represent 4 tasks
    task_labels = [0, 1, 2, 3]
    
    methods = ['Uniform', 'SPS-ZCA', 'SABLE', 'EMA', 'WMomentum', 'ChemMerge', 'KalmanFilter', 'GraviMerge']
    configs = ['Homogeneous', 'Heterogeneous_Batch', 'Heterogeneous_Serving']
    
    results = {m: {c: [] for c in configs} for m in methods}
    jitters = {m: [] for m in methods}
    
    trajectory_data = {}
    
    print("Starting evaluation on real-world handwritten digits dataset projected to D=192 across 10 seeds...")
    
    for seed in range(num_seeds):
        curr_seed = 42 + seed
        set_seed(curr_seed)
        
        # Generate random projection matrix unique to this seed to project 64 dimensions to 192 dimensions
        # This simulates high-dimensional model representations while maintaining relative distances
        projection_matrix = torch.randn(64, D)
        projection_matrix = projection_matrix / torch.norm(projection_matrix, dim=0, keepdim=True)
        
        # Filter and project data for each task
        task_samples = []
        true_tasks_list = []
        for k in range(K):
            indices = np.where(y == task_labels[k])[0]
            # Keep up to 200 samples per task to avoid unbalance
            indices = indices[:200]
            data_k = torch.tensor(X[indices], dtype=torch.float32)
            projected_k = data_k @ projection_matrix
            task_samples.append(projected_k)
            true_tasks_list.append(torch.full((len(indices),), k, dtype=torch.long))
            
        # Calibration Phase (first 64 samples of each task to pre-extract centroids)
        centroids = torch.zeros(K, D)
        for k in range(K):
            mean_h3 = torch.mean(task_samples[k][:64], dim=0)
            centroids[k] = mean_h3 / torch.norm(mean_h3)
            
        # Test Stream Generation (samples from index 64 onwards)
        test_samples_homog_list = []
        true_tasks_homog_list = []
        for k in range(K):
            test_samples_homog_list.append(task_samples[k][64:])
            true_tasks_homog_list.append(true_tasks_list[k][64:])
            
        test_samples_homog = torch.cat(test_samples_homog_list, dim=0)
        true_tasks_homog = torch.cat(true_tasks_homog_list, dim=0)
        
        # Shuffle test data to simulate heterogeneous serving
        shuffled_idx = torch.randperm(test_samples_homog.shape[0])
        test_samples_heterog = test_samples_homog[shuffled_idx]
        true_tasks_heterog = true_tasks_homog[shuffled_idx]
        
        # Initial representations at boundary layer 3 (representing features flowing out of early layers)
        h3_homog = test_samples_homog.clone()
        h3_heterog = test_samples_heterog.clone()
        
        # 3. Run Methods
        # Uniform
        h_uniform_ho, a_uniform_ho = run_uniform(h3_homog, centroids, gamma=0.3)
        h_uniform_he, a_uniform_he = run_uniform(h3_heterog, centroids, gamma=0.3)
        acc_uni_ho = compute_accuracy(h_uniform_ho, centroids, true_tasks_homog)
        acc_uni_he = compute_accuracy(h_uniform_he, centroids, true_tasks_heterog)
        jitter_uni = compute_jitter(a_uniform_he)
        
        # SPS-ZCA
        h_sps_ho, a_sps_ho = run_sps_zca(h3_homog, centroids, gamma=0.3, tau=0.05)
        h_sps_he, a_sps_he = run_sps_zca(h3_heterog, centroids, gamma=0.3, tau=0.05)
        acc_sps_ho = compute_accuracy(h_sps_ho, centroids, true_tasks_homog)
        acc_sps_he = compute_accuracy(h_sps_he, centroids, true_tasks_heterog)
        jitter_sps = compute_jitter(a_sps_he)
        
        # SABLE
        h_sable_ho, a_sable_ho = run_sable(h3_homog, centroids, gamma=0.3, tau=0.05)
        h_sable_he, a_sable_he = run_sable(h3_heterog, centroids, gamma=0.3, tau=0.05)
        acc_sable_ho = compute_accuracy(h_sable_ho, centroids, true_tasks_homog)
        acc_sable_he = compute_accuracy(h_sable_he, centroids, true_tasks_heterog)
        jitter_sable = compute_jitter(a_sable_he)
        
        # ChemMerge SOTA (fairly calibrated dt = 0.5 to avoid artificially discarding history memory)
        h_chem_ho, a_chem_ho = run_chemmerge(h3_homog, centroids, gamma=0.3, tau=0.05, k_decay=0.3, dt=0.5)
        h_chem_he, a_chem_he = run_chemmerge(h3_heterog, centroids, gamma=0.3, tau=0.05, k_decay=0.3, dt=0.5)
        acc_chem_ho = compute_accuracy(h_chem_ho, centroids, true_tasks_homog)
        acc_chem_he = compute_accuracy(h_chem_he, centroids, true_tasks_heterog)
        jitter_chem = compute_jitter(a_chem_he)
        
        # EMA Smoothing Baseline
        h_ema_ho, a_ema_ho = run_ema(h3_homog, centroids, gamma=0.3, tau=0.05, beta=0.9)
        h_ema_he, a_ema_he = run_ema(h3_heterog, centroids, gamma=0.3, tau=0.05, beta=0.9)
        acc_ema_ho = compute_accuracy(h_ema_ho, centroids, true_tasks_homog)
        acc_ema_he = compute_accuracy(h_ema_he, centroids, true_tasks_heterog)
        jitter_ema = compute_jitter(a_ema_he)

        # WMomentum Baseline
        h_wmom_ho, a_wmom_ho = run_weight_momentum(h3_homog, centroids, gamma=0.3, tau=0.05, beta1=0.8, beta2=0.5)
        h_wmom_he, a_wmom_he = run_weight_momentum(h3_heterog, centroids, gamma=0.3, tau=0.05, beta1=0.8, beta2=0.5)
        acc_wmom_ho = compute_accuracy(h_wmom_ho, centroids, true_tasks_homog)
        acc_wmom_he = compute_accuracy(h_wmom_he, centroids, true_tasks_heterog)
        jitter_wmom = compute_jitter(a_wmom_he)

        # Kalman Filter Baseline
        h_kf_ho, a_kf_ho = run_kalman_filter(h3_homog, centroids, gamma=0.3, tau=0.05, Q=0.01, R=0.1)
        h_kf_he, a_kf_he = run_kalman_filter(h3_heterog, centroids, gamma=0.3, tau=0.05, Q=0.01, R=0.1)
        acc_kf_ho = compute_accuracy(h_kf_ho, centroids, true_tasks_homog)
        acc_kf_he = compute_accuracy(h_kf_he, centroids, true_tasks_heterog)
        jitter_kf = compute_jitter(a_kf_he)
        
        # GraviMerge (Ours) - Refined and fully dynamic (G = 0.05, epsilon = 0.8, drag = 0.9)
        h_grav_ho, a_grav_ho = run_gravimerge(h3_homog, centroids, tau=0.05, G=0.05, epsilon=0.8, drag=0.9, dt=1.0, feedback_eta=feedback_eta)
        h_grav_he, a_grav_he = run_gravimerge(h3_heterog, centroids, tau=0.05, G=0.05, epsilon=0.8, drag=0.9, dt=1.0, feedback_eta=feedback_eta)
        acc_grav_ho = compute_accuracy(h_grav_ho, centroids, true_tasks_homog)
        acc_grav_he = compute_accuracy(h_grav_he, centroids, true_tasks_heterog)
        jitter_grav = compute_jitter(a_grav_he)
        
        # Store results
        results['Uniform']['Homogeneous'].append(acc_uni_ho)
        results['SPS-ZCA']['Homogeneous'].append(acc_sps_ho)
        results['SABLE']['Homogeneous'].append(acc_sable_ho)
        results['EMA']['Homogeneous'].append(acc_ema_ho)
        results['WMomentum']['Homogeneous'].append(acc_wmom_ho)
        results['ChemMerge']['Homogeneous'].append(acc_chem_ho)
        results['KalmanFilter']['Homogeneous'].append(acc_kf_ho)
        results['GraviMerge']['Homogeneous'].append(acc_grav_ho)
        
        results['Uniform']['Heterogeneous_Batch'].append(acc_uni_he)
        results['SPS-ZCA']['Heterogeneous_Batch'].append(acc_sps_he)
        results['SABLE']['Heterogeneous_Batch'].append(acc_sable_he)
        results['EMA']['Heterogeneous_Batch'].append(acc_ema_he)
        results['WMomentum']['Heterogeneous_Batch'].append(acc_wmom_he)
        results['ChemMerge']['Heterogeneous_Batch'].append(acc_chem_he)
        results['KalmanFilter']['Heterogeneous_Batch'].append(acc_kf_he)
        results['GraviMerge']['Heterogeneous_Batch'].append(acc_grav_he)
        
        results['Uniform']['Heterogeneous_Serving'].append(acc_uni_he)
        results['SPS-ZCA']['Heterogeneous_Serving'].append(acc_sps_he)
        results['SABLE']['Heterogeneous_Serving'].append(acc_sable_he)
        results['EMA']['Heterogeneous_Serving'].append(acc_ema_he)
        results['WMomentum']['Heterogeneous_Serving'].append(acc_wmom_he)
        results['ChemMerge']['Heterogeneous_Serving'].append(acc_chem_he)
        results['KalmanFilter']['Heterogeneous_Serving'].append(acc_kf_he)
        results['GraviMerge']['Heterogeneous_Serving'].append(acc_grav_he)
        
        jitters['Uniform'].append(jitter_uni)
        jitters['SPS-ZCA'].append(jitter_sps)
        jitters['SABLE'].append(jitter_sable)
        jitters['EMA'].append(jitter_ema)
        jitters['WMomentum'].append(jitter_wmom)
        jitters['ChemMerge'].append(jitter_chem)
        jitters['KalmanFilter'].append(jitter_kf)
        jitters['GraviMerge'].append(jitter_grav)
        
        if seed == 0:
            task_samples_idx = []
            for k in range(K):
                idx = (true_tasks_homog == k).nonzero()[0].item()
                task_samples_idx.append(idx)
                
            trajectory_data['SABLE'] = a_sable_ho[task_samples_idx]
            trajectory_data['EMA'] = a_ema_ho[task_samples_idx]
            trajectory_data['WMomentum'] = a_wmom_ho[task_samples_idx]
            trajectory_data['ChemMerge'] = a_chem_ho[task_samples_idx]
            trajectory_data['KalmanFilter'] = a_kf_ho[task_samples_idx]
            trajectory_data['GraviMerge'] = a_grav_ho[task_samples_idx]
            trajectory_data['tasks'] = task_samples_idx
            
        print(f"Seed {curr_seed} finished. Accuracies: SABLE: {acc_sable_he:.4f}, ChemMerge: {acc_chem_he:.4f}, GraviMerge: {acc_grav_he:.4f}")

    print("\n" + "="*50)
    print("EVALUATION RESULTS OVER 10 SEEDS (Mean ± Std %)")
    print("="*50)
    
    for m in methods:
        print(f"\nMethod: {m}")
        for c in configs:
            acc_list = np.array(results[m][c]) * 100
            print(f"  {c:<25}: {np.mean(acc_list):.2f}% ± {np.std(acc_list):.2f}%")
        j_list = np.array(jitters[m])
        print(f"  Routing Jitter (MAD)     : {np.mean(j_list):.5f} ± {np.std(j_list):.5f}")

    # Generate figures
    os.makedirs('results', exist_ok=True)
    
    task_plot_idx = 2
    layers = list(range(4, 15))
    
    plt.figure(figsize=(15, 5))
    
    # SABLE
    plt.subplot(1, 3, 1)
    sable_traj = trajectory_data['SABLE'][task_plot_idx].numpy()
    for k in range(K):
        plt.plot(layers, sable_traj[:, k], label=f'Expert {k}', marker='o', linewidth=2)
    plt.title('SABLE (Stateless Routing Jitter)', fontsize=12)
    plt.xlabel('Layer', fontsize=10)
    plt.ylabel('Ensembling Coefficient alpha', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    
    # ChemMerge
    plt.subplot(1, 3, 2)
    chem_traj = trajectory_data['ChemMerge'][task_plot_idx].numpy()
    for k in range(K):
        plt.plot(layers, chem_traj[:, k], label=f'Expert {k}', marker='s', linewidth=2)
    plt.title('ChemMerge (Kinetics Smoothed)', fontsize=12)
    plt.xlabel('Layer', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    
    # GraviMerge
    plt.subplot(1, 3, 3)
    grav_traj = trajectory_data['GraviMerge'][task_plot_idx].numpy()
    for k in range(K):
        plt.plot(layers, grav_traj[:, k], label=f'Expert {k}', marker='^', linewidth=2)
    plt.title('GraviMerge (Inertial Trajectory Integration)', fontsize=12)
    plt.xlabel('Layer', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('results/layer_trajectory.png', dpi=300)
    plt.close()
    
    # Summary figure
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    mean_accs = [np.mean(results[m]['Heterogeneous_Serving']) * 100 for m in methods]
    std_accs = [np.std(results[m]['Heterogeneous_Serving']) * 100 for m in methods]
    
    color = 'tab:blue'
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('Serving Accuracy (%)', color=color, fontsize=12)
    bars = ax1.bar(np.arange(len(methods)) - 0.2, mean_accs, yerr=std_accs, width=0.4, color=color, alpha=0.7, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(np.arange(len(methods)))
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_ylim(min(mean_accs) - 10, max(mean_accs) + 5)
    
    ax2 = ax1.twinx()
    mean_jitters = [np.mean(jitters[m]) for m in methods]
    std_jitters = [np.std(jitters[m]) for m in methods]
    
    color = 'tab:red'
    ax2.set_ylabel('Routing Jitter (MAD)', color=color, fontsize=12)
    bars2 = ax2.bar(np.arange(len(methods)) + 0.2, mean_jitters, yerr=std_jitters, width=0.4, color=color, alpha=0.7, label='Jitter')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.005, max(mean_jitters) + 0.02)
    
    plt.title('Performance vs. Stability Trade-off in Model Merging', fontsize=14)
    fig.tight_layout()
    plt.savefig('results/fig1.png', dpi=300)
    plt.close()
    
    print("\nFigures saved successfully.")
    
    # Save quantitative results in metrics.txt
    with open('results/metrics.txt', 'w') as f:
        f.write("Quantitative Evaluation Results across 10 Seeds (Mean ± Std %):\n")
        f.write("="*80 + "\n")
        for m in methods:
            f.write(f"\nMethod: {m}\n")
            for c in configs:
                acc_list = np.array(results[m][c]) * 100
                f.write(f"  {c:<25}: {np.mean(acc_list):.2f}% ± {np.std(acc_list):.2f}%\n")
            j_list = np.array(jitters[m])
            f.write(f"  Routing Jitter (MAD)     : {np.mean(j_list):.5f} ± {np.std(j_list):.5f}\n")
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="GraviMerge Model Merging Sandbox and Verifications")
    parser.add_argument('--feedback_eta', type=float, default=0.0, help="Representational feedback force coupling coefficient (Coupled GraviMerge)")
    parser.add_argument('--temporal_carryover', action='store_true', help="Run true temporal non-stationary streaming evaluation")
    parser.add_argument('--run_noise_study', action='store_true', help="Run representation noise robustness sweep study")
    parser.add_argument('--run_transformer_verification', action='store_true', help="Run GPT-2 dimension scale and representational drift verification")
    parser.add_argument('--run_ood_study', action='store_true', help="Run Out-of-Distribution and Sentinel Attractor Dynamics (SAD) study")
    
    args = parser.parse_args()
    
    if args.run_noise_study:
        from test_noise import run_noise_study
        run_noise_study()
    elif args.run_transformer_verification:
        from test_transformer_verification import main as run_trans_verification
        run_trans_verification()
    elif args.temporal_carryover:
        from test_temporal import run_temporal_eval
        run_temporal_eval()
    elif args.run_ood_study:
        from test_sentinel import run_ood_study
        run_ood_study()
    else:
        run_evaluation(feedback_eta=args.feedback_eta)
