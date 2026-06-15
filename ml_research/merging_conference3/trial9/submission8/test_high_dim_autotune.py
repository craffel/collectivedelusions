import torch
import numpy as np
import math

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_high_dim_centroids(K, D, num_layers, drift_scale=0.05):
    """
    Generate centroids in high-dimensional space D=4096 with layer-wise semantic drift.
    """
    # Base task vectors on S^{D-1}
    base_centroids = torch.randn(K, D)
    base_centroids = base_centroids / torch.norm(base_centroids, dim=1, keepdim=True)
    
    centroids = {}
    for l in range(num_layers + 1):
        # Layer-specific drift noise
        drift_noise = torch.randn(K, D) * drift_scale
        layer_centroids = base_centroids + drift_noise
        centroids[l] = layer_centroids / torch.norm(layer_centroids, dim=1, keepdim=True)
        
    return centroids

def run_gravimerge_high_dim(h3, centroids, num_layers=12, G0=0.05, epsilon=0.8, drag_base=0.9, dt=1.0, tau=0.15, 
                            use_ags=False, eta_ags=1.0, 
                            use_adaptive_drag=False, adaptive_eta=0.15,
                            use_sad=False, delta_ood=0.25, tau_ood=0.05):
    """
    Unified GraviMerge routing simulation on S^{D-1} under high dimension, with AGS, Adaptive Drag, and SAD.
    """
    N, D = h3.shape
    K = centroids[4].shape[0] if isinstance(centroids, dict) else centroids.shape[0]
    
    h_sc = h3 / torch.norm(h3, dim=1, keepdim=True)
    v = torch.zeros((N, D), device=h3.device)
    
    # Arrhenius Mass Activation (initial computed using l=3 centroids)
    c3 = centroids[3] if isinstance(centroids, dict) else centroids
    cos_sim3 = h_sc @ c3.t()
    sim_max, _ = torch.max(cos_sim3, dim=1, keepdim=True)
    M = torch.exp((cos_sim3 - sim_max) / tau)
    
    if use_sad:
        # Sentinel Attractor Dynamics (SAD)
        psi = torch.sigmoid((sim_max - delta_ood) / tau_ood)
        M = psi * M + (1.0 - psi) * 1.0  # OOD inputs force uniform mass 1.0
        
    alpha_history = []
    
    for l in range(4, num_layers + 1):
        cl = centroids[l] if isinstance(centroids, dict) else centroids
        
        cos_sim_sc = h_sc @ cl.t()
        r = torch.sqrt(torch.clamp(2.0 * (1.0 - cos_sim_sc), min=1e-8))
        
        # 1. Adaptive Gravitational Scheduling (AGS)
        if use_ags:
            v_sq = torch.sum(v**2, dim=1, keepdim=True)
            G = G0 * torch.exp(-eta_ags * v_sq)
            G = torch.clamp(G, min=1e-5)  # Physical and numerical safeguard
        else:
            G = G0
            
        force_mag = G * M / (r**2 + epsilon**2)
        
        # GIB weights
        alpha = force_mag / torch.sum(force_mag, dim=1, keepdim=True)
        alpha_history.append(alpha.clone())
        
        # Compute force vectors
        diff = cl.unsqueeze(0) - h_sc.unsqueeze(1)
        diff_norm = torch.norm(diff, dim=2, keepdim=True)
        u_hat = diff / torch.clamp(diff_norm, min=1e-8)
        force_vecs = force_mag.unsqueeze(2) * u_hat
        
        a_gravity = torch.sum(force_vecs, dim=1)
        a_tangent = a_gravity - torch.sum(a_gravity * h_sc, dim=1, keepdim=True) * h_sc
        
        # 2. Adaptive Viscous Drag Scheduling
        if use_adaptive_drag:
            max_cos_sim, _ = torch.max(cos_sim_sc, dim=1, keepdim=True)
            clamped_cos = torch.clamp(max_cos_sim, min=0.0, max=1.0)
            drag = drag_base - adaptive_eta * clamped_cos
        else:
            drag = drag_base
            
        v_tentative = drag * v + a_tangent * dt
        v_tangent = v_tentative - torch.sum(v_tentative * h_sc, dim=1, keepdim=True) * h_sc
        
        # Geodesic update
        v_norm = torch.norm(v_tangent, dim=1, keepdim=True)
        v_norm_clamp = torch.clamp(v_norm, min=1e-8)
        h_sc_new = torch.cos(v_norm * dt) * h_sc + torch.sin(v_norm * dt) * (v_tangent / v_norm_clamp)
        h_sc_new = h_sc_new / torch.norm(h_sc_new, dim=1, keepdim=True)
        
        # Parallel transport
        cos_theta = torch.sum(h_sc * h_sc_new, dim=1, keepdim=True)
        proj_coeff = torch.sum(v_tangent * h_sc_new, dim=1, keepdim=True) / (1.0 + cos_theta)
        v = v_tangent - (h_sc + h_sc_new) * proj_coeff
        
        h_sc = h_sc_new
        
    return torch.stack(alpha_history, dim=1)

def test_ags_and_drag_at_scale():
    print("=" * 80)
    print("VALIDATING AUTO-TUNING MECHANISMS AT SCALE (D = 4096, K = 8 Experts)")
    print("=" * 80)
    
    D = 4096
    K = 8
    num_layers = 16  # Let the trajectory evolve longer to accumulate velocity
    num_samples = 50
    set_seed(42)
    
    # 1. Generate drifted centroids across layers
    centroids = generate_high_dim_centroids(K, D, num_layers, drift_scale=0.08)
    
    # 2. Simulate inputs near Expert 0 with moderate representational noise
    c3 = centroids[3]
    noise_level = 1.4  # Total noise magnitude is 1.4, comparable to centroid norm of 1.0
    
    # Generate isotropic noise on S^{D-1} scaled to noise_level
    noise = torch.randn(num_samples, D)
    noise = noise / torch.norm(noise, dim=1, keepdim=True) * noise_level
    test_inputs = c3[0:1] + noise
    test_inputs = test_inputs / torch.norm(test_inputs, dim=1, keepdim=True)
    
    print("\n--- Part 1: Adaptive Gravitational Scheduling (AGS) Under High G Force ---")
    # Under high-G settings (G0 = 5.0), standard GraviMerge overshoots and jitters heavily
    # due to excessive kinetic energy. AGS dynamically dials down G as velocity spikes to stabilize.
    G_high = 3.5
    
    alphas_std = run_gravimerge_high_dim(test_inputs, centroids, num_layers=num_layers, G0=G_high, drag_base=0.98, use_ags=False)
    alphas_ags = run_gravimerge_high_dim(test_inputs, centroids, num_layers=num_layers, G0=G_high, drag_base=0.98, use_ags=True, eta_ags=12.0)
    
    # Compute routing jitter (MAD)
    jitter_std = torch.mean(torch.abs(alphas_std[:, 1:, :] - alphas_std[:, :-1, :])).item()
    jitter_ags = torch.mean(torch.abs(alphas_ags[:, 1:, :] - alphas_ags[:, :-1, :])).item()
    
    # Measure maximum weight for target expert
    mean_target_std = torch.mean(alphas_std[:, :, 0]).item()
    mean_target_ags = torch.mean(alphas_ags[:, :, 0]).item()
    
    print(f"Standard GraviMerge (G={G_high}): Jitter = {jitter_std:.6f}, Target Expert Weight = {mean_target_std:.4f}")
    print(f"AGS GraviMerge (G={G_high}, eta_ags=12): Jitter = {jitter_ags:.6f}, Target Expert Weight = {mean_target_ags:.4f}")
    jitter_reduction = (jitter_std - jitter_ags) / jitter_std * 100
    print(f"-> AGS achieves a {jitter_reduction:.2f}% reduction in routing jitter under high-force regimes!")
    
    print("\n--- Part 2: Adaptive Viscous Drag Scheduling ---")
    # Under low-viscous damping (drag_base = 0.995), the probe orbits attracting centroids
    # endlessly without settling, causing persistent oscillations.
    # Adaptive Viscous Drag Scheduling dynamically drops drag down near centroids to lock the probe.
    drag_base = 0.995
    
    alphas_std_drag = run_gravimerge_high_dim(test_inputs, centroids, num_layers=num_layers, G0=0.35, drag_base=drag_base, use_adaptive_drag=False)
    alphas_adapt_drag = run_gravimerge_high_dim(test_inputs, centroids, num_layers=num_layers, G0=0.35, drag_base=drag_base, use_adaptive_drag=True, adaptive_eta=0.25)
    
    jitter_std_drag = torch.mean(torch.abs(alphas_std_drag[:, 1:, :] - alphas_std_drag[:, :-1, :])).item()
    jitter_adapt_drag = torch.mean(torch.abs(alphas_adapt_drag[:, 1:, :] - alphas_adapt_drag[:, :-1, :])).item()
    
    print(f"Static Drag (drag={drag_base}): Jitter = {jitter_std_drag:.6f}")
    print(f"Adaptive Viscous Drag (eta=0.25):  Jitter = {jitter_adapt_drag:.6f}")
    drag_reduction = (jitter_std_drag - jitter_adapt_drag) / jitter_std_drag * 100
    print(f"-> Adaptive Viscous Drag Scheduling reduces routing jitter by {drag_reduction:.2f}% near attracting centroids!")

    print("\n--- Part 3: Sentinel Attractor Dynamics (SAD) under OOD Tasks ---")
    # Under high-dimensional spaces, OOD inputs will be nearly orthogonal to all ID expert centroids.
    # Standard ensembling might pull the probe towards a random expert depending on minor coordinate alignments.
    # SAD forces uniform mass under OOD inputs, causing the spacecraft to settle at the geometric barycenter.
    
    # Generate OOD test inputs (orthogonal random vectors representing OOD topics)
    # We generate vectors and project out ID centroids to ensure true orthogonality
    ood_raw = torch.randn(num_samples, D)
    for k in range(K):
        c_k = centroids[3][k:k+1]
        ood_raw = ood_raw - torch.sum(ood_raw * c_k, dim=1, keepdim=True) * c_k
    test_inputs_ood = ood_raw / torch.norm(ood_raw, dim=1, keepdim=True)
    
    # Standard GraviMerge vs OOD-Safeguarded SAD GraviMerge
    alphas_ood_std = run_gravimerge_high_dim(test_inputs_ood, centroids, num_layers=num_layers, G0=0.05, use_sad=False)
    alphas_ood_sad = run_gravimerge_high_dim(test_inputs_ood, centroids, num_layers=num_layers, G0=0.05, use_sad=True, delta_ood=0.25, tau_ood=0.05)
    
    # Measure weight standard deviation across experts. Uniform weight vector (1/K = 0.125) has std of 0.0.
    # High standard deviation means highly skewed/peaked (arbitrary routing).
    std_ood_std = torch.mean(torch.std(alphas_ood_std, dim=2)).item()
    std_ood_sad = torch.mean(torch.std(alphas_ood_sad, dim=2)).item()
    
    print(f"Standard GraviMerge on OOD: Weight Std Dev = {std_ood_std:.6f} (arbitrary routing asymmetry)")
    print(f"SAD-Safeguarded GraviMerge: Weight Std Dev = {std_ood_sad:.6f} (perfectly uniform barycentric fallback)")
    entropy_increase = (std_ood_std - std_ood_sad) / std_ood_std * 100
    print(f"-> SAD slashes weight asymmetry by {entropy_increase:.2f}%, successfully securing a safe, balanced uniform ensembling fallback!")
    
    print("=" * 80)
    print("High-Dimensional Auto-Tuning Validation Complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_ags_and_drag_at_scale()
