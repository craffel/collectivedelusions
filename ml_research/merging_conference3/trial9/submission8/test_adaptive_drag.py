import torch
import numpy as np
from simulate_sandbox import load_digits, set_seed, compute_accuracy, compute_jitter

# GraviMerge with Adaptive Viscous Drag Scheduling
def run_gravimerge_adaptive_drag(h3, centroids, gamma=0.3, tau=0.05, G=0.05, epsilon=0.8, drag_base=0.95, adaptive_eta=0.15, dt=1.0, num_layers=14):
    N, D = h3.shape
    K = centroids.shape[0]
    h = h3.clone()
    
    # Rigorous Position Initialization: Normalize h_sc to lie on the sphere S^{D-1} from the start
    h_sc = h3 / torch.norm(h3, dim=1, keepdim=True)
    v = torch.zeros((N, D))
    
    # Compute Dynamic Gravitational Mass (Arrhenius Mass Activation)
    cos_sim3 = h_sc @ centroids.t()
    sim_max, _ = torch.max(cos_sim3, dim=1, keepdim=True)
    M = torch.exp((cos_sim3 - sim_max) / tau)
    
    alpha_history = []
    
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
        
        # Project acceleration onto the local tangent space of the sphere
        a_tangent = a_gravity - torch.sum(a_gravity * h_sc, dim=1, keepdim=True) * h_sc
        
        # Dynamic Drag Scheduling: drag decreases (damping increases) as we get closer to centroids
        # max_cos_sim is the cosine similarity of the probe to its closest attractor centroid
        max_cos_sim, _ = torch.max(cos_sim_sc, dim=1, keepdim=True)
        # We clamp max_cos_sim between 0.0 and 1.0 to keep drag in a valid physical regime
        clamped_cos = torch.clamp(max_cos_sim, min=0.0, max=1.0)
        drag_l = drag_base - adaptive_eta * clamped_cos
        
        # Update tentative velocity with dynamic drag
        v_tentative = drag_l * v + a_tangent * dt
        
        # Project tentative velocity onto tangent space to preserve spherical constraint directions
        v_tangent = v_tentative - torch.sum(v_tentative * h_sc, dim=1, keepdim=True) * h_sc
        
        # Rigorous Geodesic Update on Sphere (Exponential Map)
        v_norm = torch.norm(v_tangent, dim=1, keepdim=True)
        v_norm_clamp = torch.clamp(v_norm, min=1e-8)
        h_sc_new = torch.cos(v_norm * dt) * h_sc + torch.sin(v_norm * dt) * (v_tangent / v_norm_clamp)
        h_sc_new = h_sc_new / torch.norm(h_sc_new, dim=1, keepdim=True)
        
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
        
    return h, torch.stack(alpha_history, dim=1)

def test_adaptive_drag():
    D = 192
    K = 4
    num_seeds = 10
    
    digits = load_digits()
    X = digits.data
    y = digits.target
    task_labels = [0, 1, 2, 3]
    
    # Sweep adaptive_eta with drag_base = 0.95
    # When adaptive_eta = 0, this is static drag of 0.95 (which is quite responsive)
    # When adaptive_eta = 0.15, drag scales from 0.95 down to 0.80 near centroids (dynamic damping)
    # When adaptive_eta = 0.30, drag scales from 0.95 down to 0.65 near centroids (aggressive damping)
    drag_base = 0.95
    adaptive_etas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    print("Testing Adaptive Viscous Drag Scheduling of GraviMerge...")
    print(f"Base Drag: {drag_base}")
    print(f"{'Adaptive Eta':<15} | {'Accuracy (%)':<15} {'Jitter (MAD)':<15}")
    print("-" * 50)
    
    for eta in adaptive_etas:
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
            
            h_grav, alphas = run_gravimerge_adaptive_drag(
                h3_heterog, centroids, tau=0.05, G=0.05, epsilon=0.8, drag_base=drag_base, adaptive_eta=eta, dt=1.0
            )
            
            accs.append(compute_accuracy(h_grav, centroids, true_tasks_heterog))
            jitters.append(compute_jitter(alphas))
            
        print(f"{eta:<15.3f} | {np.mean(accs)*100:<15.2f} {np.mean(jitters):<15.6f}")

if __name__ == '__main__':
    test_adaptive_drag()
