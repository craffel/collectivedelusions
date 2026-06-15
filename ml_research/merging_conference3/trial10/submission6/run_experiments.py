import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =====================================================================
# Analytical Coordinate Sandbox (ICS) Environment Setup
# =====================================================================
class CoordinateSandbox:
    def __init__(self, D=192, L=14, K=4, rho=0.0, overlap_v=0, device="cpu"):
        self.D = D
        self.L = L
        self.K = K
        self.rho = rho
        self.overlap_v = overlap_v
        self.device = device
        
        # Calibrated Hyperparameters for Unnormalized Signatures
        self.gamma_V = 0.05
        self.kappa_scale = 0.0636
        self.tau_0 = 0.05
        
        # Noise parameters (calibrated to scale)
        self.sigmas = torch.tensor([0.05, 0.15, 0.40, 1.20], device=device) * 0.1803
        self.biases = torch.tensor([0.0, 0.0, -0.90, -2.30], device=device)
        
        # Construct unnormalized task signatures (entries are 1.0 at active indices)
        self.v_k = self._generate_task_signatures()
        
        # Inject covariance
        self.v_prime = self._inject_covariance()
        
    def _generate_task_signatures(self):
        v_k = torch.zeros((self.K, self.D), device=self.device)
        S = self.D // self.K  # partition size (48)
        
        for k in range(self.K):
            if self.overlap_v == 0:
                # Orthogonal Manifolds
                start_idx = k * S
                end_idx = (k + 1) * S
            else:
                # Overlapping Manifolds
                start_idx = k * S - k * self.overlap_v
                end_idx = start_idx + S
            
            v_k[k, start_idx:end_idx] = 1.0  # Unnormalized: entries are 1.0
        return v_k
        
    def _inject_covariance(self):
        if self.rho == 0.0:
            return self.v_k.clone()
            
        # Toeplitz matrix construction: Sigma_{i,j} = rho^{|i-j|}
        idx = torch.arange(self.D, device=self.device)
        Sigma = self.rho ** torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        
        # Compute Sigma^{1/2} via eigendecomposition: Sigma = Q * Lambda * Q^T
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)  # positive semidefinite projection
        Sigma_half = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
        
        # Inject covariance into signatures: v'_k = Sigma^{1/2} v_k
        v_prime = (Sigma_half @ self.v_k.T).T
        # We do NOT normalize v_prime for the unnormalized regime, keeping physical coordinates large
        return v_prime

    def generate_stream(self, stream_type="homogeneous", length=200):
        """Generates task sequence and initial activations h_3"""
        task_seq = []
        if stream_type == "homogeneous":
            block_size = length // self.K
            for k in range(self.K):
                task_seq.extend([k] * block_size)
            while len(task_seq) < length:
                task_seq.append(self.K - 1)
            task_seq = task_seq[:length]
        else:
            task_seq = np.random.randint(0, self.K, size=length).tolist()
            
        h_3_stream = []
        for y_t in task_seq:
            sigma = self.sigmas[y_t]
            noise = torch.randn(self.D, device=self.device) * sigma
            h_3 = self.v_prime[y_t] + noise
            h_3_stream.append(h_3)
            
        return task_seq, torch.stack(h_3_stream)

# =====================================================================
# Stateful Router Definitions
# =====================================================================
class PIDMergeRouter(nn.Module):
    def __init__(self, num_tasks=4, init_Kp=0.5, init_Ki=0.15, init_Kd=0.2, init_temp=0.05, tau_min=0.01):
        super().__init__()
        self.num_tasks = num_tasks
        self.tau_min = tau_min
        
        up_val = torch.log(torch.exp(torch.tensor(init_Kp)) - 1.0)
        ui_val = torch.log(torch.exp(torch.tensor(init_Ki)) - 1.0)
        ud_val = torch.log(torch.exp(torch.tensor(init_Kd)) - 1.0)
        
        self.u_p = nn.Parameter(up_val)
        self.u_i = nn.Parameter(ui_val)
        self.u_d = nn.Parameter(ud_val)
        
        w_init = torch.log(torch.tensor(init_temp - tau_min))
        self.w = nn.Parameter(torch.full((num_tasks,), w_init.item()))
        
    def get_gains(self):
        Kp = F.softplus(self.u_p)
        Ki = F.softplus(self.u_i)
        Kd = F.softplus(self.u_d)
        return Kp, Ki, Kd
        
    def get_temperatures(self):
        return torch.exp(self.w) + self.tau_min

class PACKineticsRouter(nn.Module):
    def __init__(self, num_tasks=4, init_temp=0.05, tau_min=0.01):
        super().__init__()
        self.num_tasks = num_tasks
        self.tau_min = tau_min
        
        self.u = nn.Parameter(torch.zeros(num_tasks))
        self.W = nn.Parameter(torch.eye(num_tasks) * 0.5)
        
        w_init = torch.log(torch.tensor(init_temp - tau_min))
        self.w = nn.Parameter(torch.full((num_tasks,), w_init.item()))
        
    def get_decay(self):
        return torch.sigmoid(self.u)
        
    def get_temperatures(self):
        return torch.exp(self.w) + self.tau_min

# =====================================================================
# Vectorized Simulation / Evaluation Loop
# =====================================================================
def run_simulation(sandbox, task_seq, h_3_stream, router_type, router_model=None, decay_param=0.8, return_depth_jitter=False):
    """
    Runs the forward propagation of representation states through layers [4, 14]
    and computes the final alignment accuracies and temporal routing jitters.
    """
    T = len(task_seq)
    K = sandbox.K
    L = sandbox.L
    v_prime = sandbox.v_prime
    gamma_V = sandbox.gamma_V
    kappa_scale = sandbox.kappa_scale
    tau_0 = sandbox.tau_0
    
    # Pre-normalize v_prime for vectorized cosine similarity calculation
    v_norms = torch.clamp(torch.norm(v_prime, dim=1), min=1e-8)
    v_norm_all = v_prime / v_norms.unsqueeze(1)
    
    # Trackers for ensembling weights
    all_alphas = []
    align_accuracies = []
    all_depth_jitters = []
    
    # For stateful sequential tracking across stream steps t
    prev_s = torch.zeros(K, device=sandbox.device)
    
    for t in range(T):
        h_l = h_3_stream[t].clone()
        y_t = task_seq[t]
        
        # Stateful variables for layer-by-layer update within a single sample
        alpha_l = torch.full((K,), 1.0 / K, device=sandbox.device)
        e_l_curr = torch.zeros(K, device=sandbox.device)
        e_l_prev = torch.zeros(K, device=sandbox.device)
        e_l_prev2 = torch.zeros(K, device=sandbox.device)
        
        # State carried over temporally for sequence-wide tracking models
        s_l = prev_s.clone() if (router_type in ["pac-kinetics", "chemmerge"]) else torch.zeros(K, device=sandbox.device)
        
        alpha_prev = alpha_l.clone()
        depth_jit_t = torch.tensor(0.0, device=sandbox.device)
        
        for l in range(4, L + 1):
            # Vectorized Cosine Similarity Projection
            norms_h = torch.clamp(torch.norm(h_l), min=1e-8)
            h_norm = h_l / norms_h
            e_k = torch.mv(v_norm_all, h_norm)
                
            # Softmax Setpoint Computation
            w_l = F.softmax(e_k / tau_0, dim=0)
            
            # Router logic
            if router_type == "uniform":
                alpha_l = torch.full((K,), 1.0 / K, device=sandbox.device)
                
            elif router_type == "sable-raw":
                alpha_l = w_l.clone()
                
            elif router_type == "momentum-merge":
                alpha_l = decay_param * alpha_l + (1.0 - decay_param) * w_l
                
            elif router_type == "chemmerge":
                s_l = decay_param * s_l + (1.0 - decay_param) * e_k
                alpha_l = F.softmax(s_l / tau_0, dim=0)
                
            elif router_type == "pac-kinetics":
                if router_model is not None:
                    decay = router_model.get_decay()
                    W = router_model.W
                    temps = router_model.get_temperatures()
                else:
                    decay = torch.full((K,), 0.5, device=sandbox.device)
                    W = torch.eye(K, device=sandbox.device) * 0.5
                    temps = torch.full((K,), 0.05, device=sandbox.device)
                
                s_l = decay * s_l + torch.mv(W, e_k)
                alpha_l = F.softmax(s_l / temps, dim=0)
                
            elif router_type == "pid-merge":
                if router_model is not None:
                    Kp, Ki, Kd = router_model.get_gains()
                    temps = router_model.get_temperatures()
                else:
                    Kp, Ki, Kd = 0.5, 0.15, 0.2
                    temps = torch.full((K,), 0.05, device=sandbox.device)
                
                # Tracking error
                e_l_curr = w_l - alpha_l
                
                # Conditional Integration Clamping (Anti-Windup with dynamic K-scaled thresholds)
                epsilon = 0.08
                high_th = 1.0 - epsilon
                low_th = epsilon / K
                clamp_mask = ((alpha_l >= high_th) & (e_l_curr > 0)) | ((alpha_l <= low_th) & (e_l_curr < 0))
                Ki_tensor = torch.full_like(e_l_curr, Ki) if not isinstance(Ki, torch.Tensor) else Ki.expand_as(e_l_curr)
                Ki_eff = torch.where(clamp_mask, torch.zeros_like(e_l_curr), Ki_tensor)
                
                # Incremental PID update
                delta_s = Kp * (e_l_curr - e_l_prev) + Ki_eff * e_l_curr + Kd * (e_l_curr - 2 * e_l_prev + e_l_prev2)
                s_l = s_l + delta_s
                
                # Logit Mean-Centering Anti-Windup (Centered after temperature-scaling to preserve translation-invariance)
                s_scaled = s_l / temps
                s_centered = s_scaled - torch.mean(s_scaled)
                alpha_l = F.softmax(s_centered, dim=0)
                
                # Error history updates
                e_l_prev2 = e_l_prev.clone()
                e_l_prev = e_l_curr.clone()
                
            depth_jit_t = depth_jit_t + torch.norm(alpha_l - alpha_prev, p=1)
            alpha_prev = alpha_l.clone()
            
            # Vectorized Activation Blending
            blended_update = torch.sum(alpha_l.unsqueeze(1) * gamma_V * (v_prime - h_l.unsqueeze(0)), dim=0)
            h_l = h_l + blended_update
            
        all_alphas.append(alpha_l)
        all_depth_jitters.append(depth_jit_t / (L - 3))
        
        dist_sq = torch.sum((h_l - v_prime[y_t]) ** 2)
        align_acc = torch.exp(-kappa_scale * dist_sq)
        align_accuracies.append(align_acc)
        
        if router_type in ["pac-kinetics", "chemmerge"]:
            prev_s = s_l.detach()

    all_alphas = torch.stack(all_alphas)
    align_accuracies = torch.stack(align_accuracies)
    
    if return_depth_jitter:
        mean_depth_jit = torch.mean(torch.stack(all_depth_jitters)) if len(all_depth_jitters) > 0 else torch.tensor(0.0, device=sandbox.device)
        return align_accuracies, mean_depth_jit, all_alphas
    else:
        jitter_val = torch.tensor(0.0, device=sandbox.device)
        if T > 1:
            jitter_diffs = torch.norm(all_alphas[1:] - all_alphas[:-1], p=1, dim=1)
            jitter_val = torch.mean(jitter_diffs)
        return align_accuracies, jitter_val, all_alphas

# =====================================================================
# Calibration / Optimization Loop
# =====================================================================
def calibrate_router(sandbox, stream_y, stream_h, router_type, epochs=50, lr=0.01, beta_jitter=0.05):
    if router_type == "pid-merge":
        model = PIDMergeRouter(num_tasks=sandbox.K).to(sandbox.device)
        epochs_eff = 100
        beta_eff = 5.0
    elif router_type == "pac-kinetics":
        model = PACKineticsRouter(num_tasks=sandbox.K).to(sandbox.device)
        epochs_eff = epochs
        beta_eff = beta_jitter
    else:
        return None
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    use_depth_jit = (router_type == "pid-merge")
    
    for epoch in range(epochs_eff):
        optimizer.zero_grad()
        accs, jitter, _ = run_simulation(
            sandbox, stream_y, stream_h, router_type, router_model=model, return_depth_jitter=use_depth_jit
        )
        mean_acc = torch.mean(accs)
        loss = (1.0 - mean_acc) + beta_eff * jitter
        
        # Jury stability penalty for PID controller
        if router_type == "pid-merge":
            Kp, Ki, Kd = model.get_gains()
            temps = model.get_temperatures()
            # K_s <= 1 / (4 * tau_k)
            # Bound 1: K_d < 1 / K_s => K_d < 4 * tau_k
            violation_1 = torch.clamp(Kd - 4.0 * temps, min=0.0)
            # Bound 2: 2 * K_p + K_i + 4 * K_d < 2 / K_s => 2 * K_p + K_i + 4 * K_d < 8 * tau_k
            violation_2 = torch.clamp(2.0 * Kp + Ki + 4.0 * Kd - 8.0 * temps, min=0.0)
            stability_penalty = torch.mean(violation_1) + torch.mean(violation_2)
            loss = loss + 10.0 * stability_penalty
            
        loss.backward()
        optimizer.step()
        
    return model

# =====================================================================
# Run Comprehensive Evaluations across Seeds
# =====================================================================
def run_evaluation_sweep(rho=0.0, overlap_v=0, seeds=[42, 43, 44, 45, 46]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    methods = [
        "uniform",
        "sable-raw",
        "chemmerge",
        "momentum-merge",
        "pac-kinetics-tf", 
        "pid-merge-tf",    
        "pac-kinetics-opt", 
        "pid-merge-opt"     
    ]
    
    results = {m: {"homo_acc": [], "homo_jit": [], "hetero_acc": [], "hetero_jit": []} for m in methods}
    
    for seed in seeds:
        set_seed(seed)
        sandbox = CoordinateSandbox(rho=rho, overlap_v=overlap_v, device=device)
        
        # Calibration data (T=32)
        cal_y = []
        for k in range(sandbox.K):
            cal_y.extend([k] * 8)
        
        cal_h = []
        for y in cal_y:
            sigma = sandbox.sigmas[y]
            noise = torch.randn(sandbox.D, device=device) * sigma
            cal_h.append(sandbox.v_prime[y] + noise)
        cal_h = torch.stack(cal_h)
        
        # Calibrate optimized routers
        pid_opt_model = calibrate_router(sandbox, cal_y, cal_h, "pid-merge")
        pac_opt_model = calibrate_router(sandbox, cal_y, cal_h, "pac-kinetics")
        
        # Generate evaluation streams
        homo_y, homo_h = sandbox.generate_stream(stream_type="homogeneous", length=200)
        hetero_y, hetero_h = sandbox.generate_stream(stream_type="heterogeneous", length=200)
        
        # Evaluate each method
        for m in methods:
            if m == "uniform":
                acc_homo, jit_homo, _ = run_simulation(sandbox, homo_y, homo_h, "uniform")
                acc_hete, jit_hete, _ = run_simulation(sandbox, hetero_y, hetero_h, "uniform")
            elif m == "sable-raw":
                acc_homo, jit_homo, _ = run_simulation(sandbox, homo_y, homo_h, "sable-raw")
                acc_hete, jit_hete, _ = run_simulation(sandbox, hetero_y, hetero_h, "sable-raw")
            elif m == "chemmerge":
                acc_homo, jit_homo, _ = run_simulation(sandbox, homo_y, homo_h, "chemmerge")
                acc_hete, jit_hete, _ = run_simulation(sandbox, hetero_y, hetero_h, "chemmerge")
            elif m == "momentum-merge":
                acc_homo, jit_homo, _ = run_simulation(sandbox, homo_y, homo_h, "momentum-merge")
                acc_hete, jit_hete, _ = run_simulation(sandbox, hetero_y, hetero_h, "momentum-merge")
            elif m == "pac-kinetics-tf":
                acc_homo, jit_homo, _ = run_simulation(sandbox, homo_y, homo_h, "pac-kinetics")
                acc_hete, jit_hete, _ = run_simulation(sandbox, hetero_y, hetero_h, "pac-kinetics")
            elif m == "pid-merge-tf":
                acc_homo, jit_homo, _ = run_simulation(sandbox, homo_y, homo_h, "pid-merge")
                acc_hete, jit_hete, _ = run_simulation(sandbox, hetero_y, hetero_h, "pid-merge")
            elif m == "pac-kinetics-opt":
                acc_homo, jit_homo, _ = run_simulation(sandbox, homo_y, homo_h, "pac-kinetics", router_model=pac_opt_model)
                acc_hete, jit_hete, _ = run_simulation(sandbox, hetero_y, hetero_h, "pac-kinetics", router_model=pac_opt_model)
            elif m == "pid-merge-opt":
                acc_homo, jit_homo, _ = run_simulation(sandbox, homo_y, homo_h, "pid-merge", router_model=pid_opt_model)
                acc_hete, jit_hete, _ = run_simulation(sandbox, hetero_y, hetero_h, "pid-merge", router_model=pid_opt_model)
                
            results[m]["homo_acc"].append(torch.mean(acc_homo).item())
            results[m]["homo_jit"].append(jit_homo.item())
            results[m]["hetero_acc"].append(torch.mean(acc_hete).item())
            results[m]["hetero_jit"].append(jit_hete.item())
            
    # Aggregate statistics
    aggregated = {}
    for m in methods:
        aggregated[m] = {
            "homo_acc_mean": np.mean(results[m]["homo_acc"]) * 100.0,
            "homo_acc_std": np.std(results[m]["homo_acc"]) * 100.0,
            "homo_jit_mean": np.mean(results[m]["homo_jit"]),
            "homo_jit_std": np.std(results[m]["homo_jit"]),
            "hetero_acc_mean": np.mean(results[m]["hetero_acc"]) * 100.0,
            "hetero_acc_std": np.std(results[m]["hetero_acc"]) * 100.0,
            "hetero_jit_mean": np.mean(results[m]["hetero_jit"]),
            "hetero_jit_std": np.std(results[m]["hetero_jit"]),
        }
    return aggregated

# =====================================================================
# Generate Trajectory Tracking Plots
# =====================================================================
def generate_trajectory_plots(sandbox_orth, sandbox_over):
    device = "cpu"
    homo_y, homo_h = sandbox_orth.generate_stream(stream_type="homogeneous", length=200)
    
    _, _, alphas_sable = run_simulation(sandbox_orth, homo_y, homo_h, "sable-raw")
    _, _, alphas_mm = run_simulation(sandbox_orth, homo_y, homo_h, "momentum-merge")
    _, _, alphas_pid = run_simulation(sandbox_orth, homo_y, homo_h, "pid-merge")
    
    plt.figure(figsize=(12, 5))
    t_idx = np.arange(200)
    
    oracle_weights = np.zeros(200)
    for t in range(200):
        if homo_y[t] == 1:
            oracle_weights[t] = 1.0
            
    plt.plot(t_idx, oracle_weights, 'k--', label="Oracle Target Expert 1", alpha=0.5)
    plt.plot(t_idx, alphas_sable[:, 1].numpy(), label="SABLE (Stateless Raw)", color="red", alpha=0.6)
    plt.plot(t_idx, alphas_mm[:, 1].numpy(), label="Momentum-Merge (Open-Loop Lag)", color="orange", alpha=0.8)
    plt.plot(t_idx, alphas_pid[:, 1].numpy(), label="PID-Merge (Closed-Loop Proposed)", color="green", linewidth=2)
    
    plt.title("Active Ensembling Weight of Expert 1 under Task Transitions (Homogeneous Stream)")
    plt.xlabel("Query Sequence step (t)")
    plt.ylabel("Ensembling Weight (alpha)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig("results/fig1_trajectory_tracking.png", dpi=150)
    plt.close()
    
    # Layer-wise weight tracking trajectory for a single sample at transition
    t_switch = 51
    h_single = homo_h[t_switch].unsqueeze(0)
    
    def get_layerwise_alphas(sandbox, h, router_type):
        K = sandbox.K
        L = sandbox.L
        h_l = h[0].clone()
        alpha_l = torch.full((K,), 1.0 / K)
        s_l = torch.zeros(K)
        e_l_curr = torch.zeros(K)
        e_l_prev = torch.zeros(K)
        e_l_prev2 = torch.zeros(K)
        
        # Pre-normalize v_prime for vectorized cosine similarity
        v_norms = torch.clamp(torch.norm(sandbox.v_prime, dim=1), min=1e-8)
        v_norm_all = sandbox.v_prime / v_norms.unsqueeze(1)
        
        alphas = [alpha_l.clone()]
        for l in range(4, L + 1):
            norms_h = torch.clamp(torch.norm(h_l), min=1e-8)
            h_norm = h_l / norms_h
            e_k = torch.mv(v_norm_all, h_norm)
            w_l = F.softmax(e_k / sandbox.tau_0, dim=0)
            
            if router_type == "sable-raw":
                alpha_l = w_l.clone()
            elif router_type == "momentum-merge":
                alpha_l = 0.8 * alpha_l + 0.2 * w_l
            elif router_type == "pid-merge":
                e_l_curr = w_l - alpha_l
                # Conditional Integration Clamping (Anti-Windup with dynamic K-scaled thresholds)
                epsilon = 0.08
                high_th = 1.0 - epsilon
                low_th = epsilon / K
                clamp_mask = ((alpha_l >= high_th) & (e_l_curr > 0)) | ((alpha_l <= low_th) & (e_l_curr < 0))
                Ki_eff = torch.where(clamp_mask, torch.zeros_like(e_l_curr), torch.full_like(e_l_curr, 0.15))
                delta_s = 0.5 * (e_l_curr - e_l_prev) + Ki_eff * e_l_curr + 0.2 * (e_l_curr - 2 * e_l_prev + e_l_prev2)
                s_l = s_l + delta_s
                # Logit Mean-Centering Anti-Windup (Centered after temperature-scaling)
                s_scaled = s_l / 0.05
                s_centered = s_scaled - torch.mean(s_scaled)
                alpha_l = F.softmax(s_centered, dim=0)
                e_l_prev2 = e_l_prev.clone()
                e_l_prev = e_l_curr.clone()
                
            alphas.append(alpha_l.clone())
            
            blended_update = torch.sum(alpha_l.unsqueeze(1) * sandbox.gamma_V * (sandbox.v_prime - h_l.unsqueeze(0)), dim=0)
            h_l = h_l + blended_update
            
        return torch.stack(alphas)

    alphas_lw_sable = get_layerwise_alphas(sandbox_orth, h_single, "sable-raw")
    alphas_lw_mm = get_layerwise_alphas(sandbox_orth, h_single, "momentum-merge")
    alphas_lw_pid = get_layerwise_alphas(sandbox_orth, h_single, "pid-merge")
    
    plt.figure(figsize=(10, 5))
    layers = np.arange(3, 15)
    
    plt.plot(layers, alphas_lw_sable[:, 1].numpy(), 'r-o', label="SABLE (High Oscillations)")
    plt.plot(layers, alphas_lw_mm[:, 1].numpy(), 'y-s', label="Momentum-Merge (Severe Lag)")
    plt.plot(layers, alphas_lw_pid[:, 1].numpy(), 'g-^', label="PID-Merge (Clean, High-Speed Tracking)", linewidth=2)
    
    plt.title("Layer-by-Layer Expert 1 Weight Convergence (Immediately After Task Switch)")
    plt.xlabel("Layer index (l)")
    plt.ylabel("Ensembling Weight (alpha)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig("results/fig2_layerwise_convergence.png", dpi=150)
    plt.close()

# =====================================================================
# Main Execution Entrypoint
# =====================================================================
if __name__ == "__main__":
    print("================================================================")
    print("STARTING PID-MERGE COMPARATIVE EMPIRICAL STUDY...")
    print("================================================================")
    
    print("\n--- Evaluating Orthogonal Manifolds (rho = 0.0, overlap = 0) ---")
    results_orth = run_evaluation_sweep(rho=0.0, overlap_v=0)
    
    print("\n--- Evaluating Overlapping Manifolds (rho = 0.5, overlap = 12) ---")
    results_over = run_evaluation_sweep(rho=0.5, overlap_v=12)
    
    def build_results_table(agg_results):
        rows = []
        for m, stats in agg_results.items():
            rows.append({
                "Method": m,
                "Homo Acc (%)": f"{stats['homo_acc_mean']:.2f}% ± {stats['homo_acc_std']:.2f}%",
                "Homo Jitter": f"{stats['homo_jit_mean']:.4f} ± {stats['homo_jit_std']:.4f}",
                "Hetero Acc (%)": f"{stats['hetero_acc_mean']:.2f}% ± {stats['hetero_acc_std']:.2f}%",
                "Hetero Jitter": f"{stats['hetero_jit_mean']:.4f} ± {stats['hetero_jit_std']:.4f}"
            })
        return pd.DataFrame(rows)
        
    df_orth = build_results_table(results_orth)
    df_over = build_results_table(results_over)
    
    print("\n[Orthogonal Manifolds Results Table]")
    print(df_orth.to_string(index=False))
    
    print("\n[Overlapping Manifolds Results Table]")
    print(df_over.to_string(index=False))
    
    print("\n--- Generating Trajectory Plots ---")
    sandbox_orth_cpu = CoordinateSandbox(rho=0.0, overlap_v=0, device="cpu")
    sandbox_over_cpu = CoordinateSandbox(rho=0.5, overlap_v=12, device="cpu")
    generate_trajectory_plots(sandbox_orth_cpu, sandbox_over_cpu)
    print("Trajectory tracking figures saved successfully to results/ directory!")
    
    df_orth.to_csv("results/orthogonal_results.csv", index=False)
    df_over.to_csv("results/overlapping_results.csv", index=False)
    print("Tabular results written successfully to results/ directory!")
    print("================================================================")
    print("STUDY COMPLETE!")
    print("================================================================")
