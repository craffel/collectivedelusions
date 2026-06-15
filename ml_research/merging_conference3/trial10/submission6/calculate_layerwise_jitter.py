import torch
import torch.nn.functional as F
import numpy as np
from run_experiments import CoordinateSandbox, calibrate_router, run_simulation

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

def run_simulation_with_depth_jitter(sandbox, task_seq, h_3_stream, router_type, router_model=None, decay_param=0.8):
    T = len(task_seq)
    K = sandbox.K
    L = sandbox.L
    v_prime = sandbox.v_prime
    gamma_V = sandbox.gamma_V
    kappa_scale = sandbox.kappa_scale
    tau_0 = sandbox.tau_0
    
    v_norms = torch.clamp(torch.norm(v_prime, dim=1), min=1e-8)
    v_norm_all = v_prime / v_norms.unsqueeze(1)
    
    align_accuracies = []
    depth_jitters = []
    
    prev_s = torch.zeros(K)
    
    for t in range(T):
        h_l = h_3_stream[t].clone()
        y_t = task_seq[t]
        
        alpha_l = torch.full((K,), 1.0 / K)
        e_l_curr = torch.zeros(K)
        e_l_prev = torch.zeros(K)
        e_l_prev2 = torch.zeros(K)
        
        s_l = prev_s.clone() if (router_type in ["pac-kinetics", "chemmerge"]) else torch.zeros(K)
        
        alpha_history = [alpha_l.clone()]
        
        for l in range(4, L + 1):
            norms_h = torch.clamp(torch.norm(h_l), min=1e-8)
            h_norm = h_l / norms_h
            e_k = torch.mv(v_norm_all, h_norm)
            w_l = F.softmax(e_k / tau_0, dim=0)
            
            if router_type == "uniform":
                alpha_l = torch.full((K,), 1.0 / K)
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
                    decay = torch.full((K,), 0.5)
                    W = torch.eye(K) * 0.5
                    temps = torch.full((K,), 0.05)
                s_l = decay * s_l + torch.mv(W, e_k)
                alpha_l = F.softmax(s_l / temps, dim=0)
            elif router_type == "pid-merge":
                if router_model is not None:
                    Kp, Ki, Kd = router_model.get_gains()
                    temps = router_model.get_temperatures()
                else:
                    Kp, Ki, Kd = 0.5, 0.15, 0.2
                    temps = torch.full((K,), 0.05)
                e_l_curr = w_l - alpha_l
                # Conditional Integration Clamping (Anti-Windup with dynamic K-scaled thresholds)
                epsilon = 0.08
                high_th = 1.0 - epsilon
                low_th = epsilon / K
                clamp_mask = ((alpha_l >= high_th) & (e_l_curr > 0)) | ((alpha_l <= low_th) & (e_l_curr < 0))
                Ki_tensor = torch.full_like(e_l_curr, Ki) if not isinstance(Ki, torch.Tensor) else Ki.expand_as(e_l_curr)
                Ki_eff = torch.where(clamp_mask, torch.zeros_like(e_l_curr), Ki_tensor)
                delta_s = Kp * (e_l_curr - e_l_prev) + Ki_eff * e_l_curr + Kd * (e_l_curr - 2 * e_l_prev + e_l_prev2)
                s_l = s_l + delta_s
                # Logit Mean-Centering Anti-Windup (Centered after temperature-scaling to preserve translation-invariance)
                s_scaled = s_l / temps
                s_centered = s_scaled - torch.mean(s_scaled)
                alpha_l = F.softmax(s_centered, dim=0)
                e_l_prev2 = e_l_prev.clone()
                e_l_prev = e_l_curr.clone()
                
            alpha_history.append(alpha_l.clone())
            blended_update = torch.sum(alpha_l.unsqueeze(1) * gamma_V * (v_prime - h_l.unsqueeze(0)), dim=0)
            h_l = h_l + blended_update
            
        # Calculate depth-wise jitter for this sample
        depth_jit = 0.0
        for i in range(1, len(alpha_history)):
            depth_jit += torch.norm(alpha_history[i] - alpha_history[i-1], p=1).item()
        depth_jitters.append(depth_jit / (L - 3)) # average across layer transitions
        
        dist_sq = torch.sum((h_l - v_prime[y_t]) ** 2)
        align_acc = torch.exp(-kappa_scale * dist_sq)
        align_accuracies.append(align_acc)
        
        if router_type in ["pac-kinetics", "chemmerge"]:
            prev_s = s_l.detach()
            
    return np.mean(depth_jitters)

# Run evaluation on seeds
def evaluate_depth_jitter(rho=0.0, overlap_v=0, seeds=[42, 43, 44, 45, 46]):
    methods = ["uniform", "sable-raw", "chemmerge", "momentum-merge", "pac-kinetics-tf", "pid-merge-tf", "pid-merge-opt"]
    jitters = {m: [] for m in methods}
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        sandbox = CoordinateSandbox(rho=rho, overlap_v=overlap_v, device="cpu")
        
        cal_y = []
        for k in range(sandbox.K):
            cal_y.extend([k] * 8)
        cal_h = []
        for y in cal_y:
            sigma = sandbox.sigmas[y]
            noise = torch.randn(sandbox.D) * sigma
            cal_h.append(sandbox.v_prime[y] + noise)
        cal_h = torch.stack(cal_h)
        
        pid_opt_model = calibrate_router(sandbox, cal_y, cal_h, "pid-merge")
        
        homo_y, homo_h = sandbox.generate_stream(stream_type="homogeneous", length=200)
        hetero_y, hetero_h = sandbox.generate_stream(stream_type="heterogeneous", length=200)
        
        for m in methods:
            if m == "uniform":
                jit_homo = run_simulation_with_depth_jitter(sandbox, homo_y, homo_h, "uniform")
            elif m == "sable-raw":
                jit_homo = run_simulation_with_depth_jitter(sandbox, homo_y, homo_h, "sable-raw")
            elif m == "chemmerge":
                jit_homo = run_simulation_with_depth_jitter(sandbox, homo_y, homo_h, "chemmerge")
            elif m == "momentum-merge":
                jit_homo = run_simulation_with_depth_jitter(sandbox, homo_y, homo_h, "momentum-merge")
            elif m == "pac-kinetics-tf":
                jit_homo = run_simulation_with_depth_jitter(sandbox, homo_y, homo_h, "pac-kinetics")
            elif m == "pid-merge-tf":
                jit_homo = run_simulation_with_depth_jitter(sandbox, homo_y, homo_h, "pid-merge")
            elif m == "pid-merge-opt":
                jit_homo = run_simulation_with_depth_jitter(sandbox, homo_y, homo_h, "pid-merge", router_model=pid_opt_model)
            jitters[m].append(jit_homo)
            
    print(f"--- Depth-wise Jitter results (rho={rho}, overlap_v={overlap_v}) ---")
    for m in methods:
        print(f"{m}: {np.mean(jitters[m]):.5f} ± {np.std(jitters[m]):.5f}")

print("Orthogonal Manifolds:")
evaluate_depth_jitter(rho=0.0, overlap_v=0)
print("\nOverlapping Manifolds:")
evaluate_depth_jitter(rho=0.5, overlap_v=12)
