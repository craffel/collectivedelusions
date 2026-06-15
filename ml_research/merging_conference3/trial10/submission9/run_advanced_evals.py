import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from run_experiments import (
    AIR, get_task_signatures, extract_pca_bases, generate_stream, set_seed
)

# Define custom diagonal W model for large K
class AIRDiagonal(nn.Module):
    def __init__(self, K, tau_min=0.01):
        super().__init__()
        self.K = K
        self.tau_min = tau_min
        self.u = nn.Parameter(torch.zeros(K)) # log-retention
        self.W_diag = nn.Parameter(torch.ones(K)) # diagonal generative coordinate mapping
        self.p_e = nn.Parameter(torch.zeros(K)) # log-sensory precisions
        self.p_s = nn.Parameter(torch.zeros(K)) # log-prior precisions
        self.w = nn.Parameter(torch.zeros(K)) # log-temperatures
        self.mu_prev = None
        
    def reset(self, e1):
        self.mu_prev = e1.clone()
        
    def forward(self, e_t, return_logits=False):
        a = torch.sigmoid(self.u)
        Pi_e = torch.exp(self.p_e)
        Pi_s = torch.exp(self.p_s)
        tau = torch.exp(self.w) + self.tau_min
        
        # Analytical Hessian is diagonal:
        H_diag = (self.W_diag ** 2) * Pi_e + Pi_s
        
        mu_t_0 = a.unsqueeze(0) * self.mu_prev
        b_t = e_t * Pi_e.unsqueeze(0) * self.W_diag.unsqueeze(0) + Pi_s.unsqueeze(0) * mu_t_0
        
        # Closed-form single-step update:
        mu_t = b_t / H_diag.unsqueeze(0)
        
        self.mu_prev = mu_t.clone()
        logits = mu_t / tau.unsqueeze(0)
        if return_logits:
            return logits
        return F.softmax(logits, dim=1)

def get_task_signatures_k(K, D=192, config="orthogonal"):
    S = D // K
    v = torch.zeros(K, D)
    if config == "orthogonal" or config == "nonlinear":
        for k in range(K):
            v[k, k*S : (k+1)*S] = 1.0
    return v

def evaluate_output_custom(h14, v_signatures, target_y):
    K = v_signatures.shape[0]
    # For general K, use uniform/zero biases to prevent size mismatch
    biases = torch.zeros(K, device=h14.device)
    
    h_expanded = h14.unsqueeze(1)
    v_expanded = v_signatures.unsqueeze(0)
    dists = torch.sum((h_expanded - v_expanded) ** 2, dim=2)
    
    logits = -dists + biases.unsqueeze(0)
    preds = torch.argmax(logits, dim=1)
    cat_acc = (preds == target_y).float().mean()
    
    kappa_scale = 0.0385
    target_v = v_signatures[target_y]
    target_dists = torch.sum((h14 - target_v) ** 2, dim=1)
    align_accs = torch.exp(-kappa_scale * target_dists)
    mean_align_acc = align_accs.mean()
    
    return cat_acc, mean_align_acc, logits

def evaluate_router_custom(model, test_h3, test_target_y, v_signatures, V_bases):
    T_test, B, D = test_h3.shape
    K = len(V_bases)
    
    e = torch.zeros(T_test, B, K)
    for t in range(T_test):
        z_t = test_h3[t]
        z_norm = z_t / (torch.norm(z_t, p=2, dim=1, keepdim=True) + 1e-8)
        for k in range(K):
            proj = z_norm @ V_bases[k]
            e[t, :, k] = torch.norm(proj, p=2, dim=1)
            
    if isinstance(model, AIR) or isinstance(model, AIRDiagonal):
        model.reset(e[0])
    elif hasattr(model, 'reset'):
        model.reset()
        
    all_alphas = []
    all_align_accs = []
    all_cat_accs = []
    
    from run_experiments import (
        ExpertOracle, UniformMerging, SABLE, MomentumMerge, ChemMerge, PACKinetics
    )
    
    for t in range(T_test):
        if isinstance(model, AIR) or isinstance(model, AIRDiagonal):
            if t == 0:
                tau = torch.exp(model.w) + model.tau_min
                logits = model.mu_prev / tau.unsqueeze(0)
                alpha_t = F.softmax(logits, dim=1)
            else:
                alpha_t = model(e[t])
        elif isinstance(model, PACKinetics):
            alpha_t = model(e[t])
        elif isinstance(model, ExpertOracle):
            alpha_t = model.forward(e[t], test_target_y[t])
        elif isinstance(model, UniformMerging):
            alpha_t = model.forward(e[t])
        elif isinstance(model, SABLE):
            alpha_t = model.forward(e[t])
        elif isinstance(model, MomentumMerge):
            alpha_t = model.forward(e[t])
        elif isinstance(model, ChemMerge):
            alpha_t = model.forward(e[t])
            
        all_alphas.append(alpha_t.unsqueeze(0))
        
        from run_experiments import propagate_sandbox
        h14 = propagate_sandbox(test_h3[t], alpha_t, v_signatures)
        cat_acc, align_acc, logits = evaluate_output_custom(h14, v_signatures, test_target_y[t])
        all_align_accs.append(align_acc.item())
        all_cat_accs.append(cat_acc.item())
        
    all_alphas = torch.cat(all_alphas, dim=0)
    
    jitters = []
    for b in range(B):
        stream_alphas = all_alphas[:, b, :]
        diff = torch.abs(stream_alphas[1:] - stream_alphas[:-1])
        l1_diff = torch.sum(diff, dim=1)
        jitter = l1_diff.mean().item()
        jitters.append(jitter)
        
    mean_jitter = np.mean(jitters)
    mean_align_acc = np.mean(all_align_accs)
    mean_cat_acc = np.mean(all_cat_accs)
    
    return mean_cat_acc, mean_align_acc, mean_jitter, all_alphas

def train_router_custom(model, cal_h3, cal_target_y, V_bases, epochs=200, lr=0.01):
    T_cal, B, D = cal_h3.shape
    K = len(V_bases)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    e = torch.zeros(T_cal, B, K)
    for t in range(T_cal):
        z_t = cal_h3[t]
        z_norm = z_t / (torch.norm(z_t, p=2, dim=1, keepdim=True) + 1e-8)
        for k in range(K):
            proj = z_norm @ V_bases[k]
            e[t, :, k] = torch.norm(proj, p=2, dim=1)
            
    for epoch in range(epochs):
        optimizer.zero_grad()
        model.reset(e[0])
        ce_loss = 0.0
        smoothness_loss = 0.0
        
        prev_alpha = None
        for t in range(1, T_cal):
            logits_t = model(e[t], return_logits=True)
            ce_loss += F.cross_entropy(logits_t, cal_target_y[t])
            
            alpha_t = F.softmax(logits_t, dim=1)
            if prev_alpha is not None:
                smoothness_loss += torch.sum((alpha_t - prev_alpha) ** 2, dim=1).mean()
            prev_alpha = alpha_t
            
        ce_loss = ce_loss / (T_cal - 1)
        smoothness_loss = smoothness_loss / (T_cal - 2)
        
        loss = ce_loss + 0.05 * smoothness_loss
        loss.backward()
        optimizer.step()

def run_k16_experiment():
    print("\n=======================================================")
    print("RUNNING REGISTRY SCALING EXPERIMENT (K=16)")
    print("=======================================================")
    
    seeds = [42, 43, 44, 45, 46]
    K = 16
    D = 192
    T_test = 200
    B = 16
    cfg = "orthogonal"
    
    # 16 sigmas matching our noise scale range [0.05, 1.20]
    sigmas_16 = np.linspace(0.05, 1.20, K).tolist()
    
    methods_list = [
        "Oracle", "Uniform", "SABLE", "Momentum-Merge", "ChemMerge", 
        "AIR (Ours, T_cal=32)", "AIR (Ours, T_cal=128)", "AIR (Diagonal, T_cal=32)"
    ]
    
    results = {m: {'hom_acc': [], 'hom_jit': [], 'het_acc': [], 'het_jit': []} for m in methods_list}
    
    for seed in seeds:
        set_seed(seed)
        v_signatures = get_task_signatures_k(K, D, cfg)
        V_bases = extract_pca_bases(v_signatures, sigmas_16, d=2, config=cfg)
        
        # Calibration streams
        cal_h3_32, cal_target_y_32 = generate_stream(v_signatures, sigmas_16, stream_type="homogeneous", T=32, B=B, config=cfg)
        cal_h3_128, cal_target_y_128 = generate_stream(v_signatures, sigmas_16, stream_type="homogeneous", T=128, B=B, config=cfg)
        
        # Test streams
        hom_test_h3, hom_test_target_y = generate_stream(v_signatures, sigmas_16, stream_type="homogeneous", T=T_test, B=B, config=cfg)
        het_test_h3, het_test_target_y = generate_stream(v_signatures, sigmas_16, stream_type="heterogeneous", T=T_test, B=B, config=cfg)
        
        # Instantiate and train models
        # 1. AIR (Ours, T_cal=32)
        air_32 = AIR(K, N_steps=5, eta_test=0.1)
        train_router_custom(air_32, cal_h3_32, cal_target_y_32, V_bases, epochs=200, lr=0.01)
        
        # 2. AIR (Ours, T_cal=128)
        air_128 = AIR(K, N_steps=5, eta_test=0.1)
        train_router_custom(air_128, cal_h3_128, cal_target_y_128, V_bases, epochs=200, lr=0.01)
        
        # 3. AIR (Diagonal, T_cal=32)
        air_diag = AIRDiagonal(K)
        train_router_custom(air_diag, cal_h3_32, cal_target_y_32, V_bases, epochs=200, lr=0.01)
        
        from run_experiments import (
            ExpertOracle, UniformMerging, SABLE, MomentumMerge, ChemMerge
        )
        
        for mtd in methods_list:
            if mtd == "Oracle":
                model = ExpertOracle(K)
            elif mtd == "Uniform":
                model = UniformMerging(K)
            elif mtd == "SABLE":
                model = SABLE(K)
            elif mtd == "Momentum-Merge":
                model = MomentumMerge(K, beta=0.15)
            elif mtd == "ChemMerge":
                model = ChemMerge(K, beta=0.10)
            elif mtd == "AIR (Ours, T_cal=32)":
                model = air_32
            elif mtd == "AIR (Ours, T_cal=128)":
                model = air_128
            elif mtd == "AIR (Diagonal, T_cal=32)":
                model = air_diag
                
            # Evaluate on Homogeneous
            _, hom_align_acc, hom_jitter, _ = evaluate_router_custom(model, hom_test_h3, hom_test_target_y, v_signatures, V_bases)
            results[mtd]['hom_acc'].append(hom_align_acc * 100.0)
            results[mtd]['hom_jit'].append(hom_jitter)
            
            # Evaluate on Heterogeneous
            _, het_align_acc, het_jitter, _ = evaluate_router_custom(model, het_test_h3, het_test_target_y, v_signatures, V_bases)
            results[mtd]['het_acc'].append(het_align_acc * 100.0)
            results[mtd]['het_jit'].append(het_jitter)
            
    print("\n--- K=16 Scaling Experiment Results (Averaged over 5 seeds) ---")
    print("| Method | Homogeneous Acc (%) | Homogeneous Jitter | Heterogeneous Acc (%) | Heterogeneous Jitter |")
    print("|---|---|---|---|---|")
    for mtd in methods_list:
        hom_acc_mean, hom_acc_std = np.mean(results[mtd]['hom_acc']), np.std(results[mtd]['hom_acc'])
        hom_jit_mean, hom_jit_std = np.mean(results[mtd]['hom_jit']), np.std(results[mtd]['hom_jit'])
        het_acc_mean, het_acc_std = np.mean(results[mtd]['het_acc']), np.std(results[mtd]['het_acc'])
        het_jit_mean, het_jit_std = np.mean(results[mtd]['het_jit']), np.std(results[mtd]['het_jit'])
        print(f"| {mtd:<24} | {hom_acc_mean:.2f}% ± {hom_acc_std:.2f}% | {hom_jit_mean:.4f} ± {hom_jit_std:.4f} | {het_acc_mean:.2f}% ± {het_acc_std:.2f}% | {het_jit_mean:.4f} ± {het_jit_std:.4f} |")
        
    return results

def run_cross_calibration_experiment():
    print("\n=======================================================")
    print("RUNNING CROSS-SEQUENCE CALIBRATION STRESS TEST (K=4)")
    print("=======================================================")
    
    seeds = [42, 43, 44, 45, 46]
    sigmas = [0.05, 0.15, 0.40, 1.20]
    K = 4
    T_cal = 32
    T_test = 200
    B = 16
    cfg = "orthogonal"
    
    # We evaluate two calibration regimes:
    # 1. Calibrated on Homogeneous (stable sequences, sparse switches)
    # 2. Calibrated on Heterogeneous (dynamic sequences, rapid switches)
    regimes = ["Stable (Homogeneous)", "Dynamic (Heterogeneous)"]
    
    results = {r: {'hom_acc': [], 'hom_jit': [], 'het_acc': [], 'het_jit': []} for r in regimes}
    
    for seed in seeds:
        set_seed(seed)
        v_signatures = get_task_signatures(cfg)
        V_bases = extract_pca_bases(v_signatures, sigmas, config=cfg)
        
        # Calibration streams
        cal_hom_h3, cal_hom_target_y = generate_stream(v_signatures, sigmas, stream_type="homogeneous", T=T_cal, B=B, config=cfg)
        cal_het_h3, cal_het_target_y = generate_stream(v_signatures, sigmas, stream_type="heterogeneous", T=T_cal, B=B, config=cfg)
        
        # Test streams
        hom_test_h3, hom_test_target_y = generate_stream(v_signatures, sigmas, stream_type="homogeneous", T=T_test, B=B, config=cfg)
        het_test_h3, het_test_target_y = generate_stream(v_signatures, sigmas, stream_type="heterogeneous", T=T_test, B=B, config=cfg)
        
        # Model calibrated on Homogeneous (Stable)
        model_stable = AIR(K, N_steps=5, eta_test=0.1)
        train_router_custom(model_stable, cal_hom_h3, cal_hom_target_y, V_bases, epochs=200, lr=0.01)
        
        # Model calibrated on Heterogeneous (Dynamic)
        model_dynamic = AIR(K, N_steps=5, eta_test=0.1)
        train_router_custom(model_dynamic, cal_het_h3, cal_het_target_y, V_bases, epochs=200, lr=0.01)
        
        # Evaluate Stable Model
        _, hom_acc_s, hom_jit_s, _ = evaluate_router_custom(model_stable, hom_test_h3, hom_test_target_y, v_signatures, V_bases)
        _, het_acc_s, het_jit_s, _ = evaluate_router_custom(model_stable, het_test_h3, het_test_target_y, v_signatures, V_bases)
        results["Stable (Homogeneous)"]['hom_acc'].append(hom_acc_s * 100.0)
        results["Stable (Homogeneous)"]['hom_jit'].append(hom_jit_s)
        results["Stable (Homogeneous)"]['het_acc'].append(het_acc_s * 100.0)
        results["Stable (Homogeneous)"]['het_jit'].append(het_jit_s)
        
        # Evaluate Dynamic Model
        _, hom_acc_d, hom_jit_d, _ = evaluate_router_custom(model_dynamic, hom_test_h3, hom_test_target_y, v_signatures, V_bases)
        _, het_acc_d, het_jit_d, _ = evaluate_router_custom(model_dynamic, het_test_h3, het_test_target_y, v_signatures, V_bases)
        results["Dynamic (Heterogeneous)"]['hom_acc'].append(hom_acc_d * 100.0)
        results["Dynamic (Heterogeneous)"]['hom_jit'].append(hom_jit_d)
        results["Dynamic (Heterogeneous)"]['het_acc'].append(het_acc_d * 100.0)
        results["Dynamic (Heterogeneous)"]['het_jit'].append(het_jit_d)
        
    print("\n--- Cross-Sequence Calibration Robustness Results (Averaged over 5 seeds) ---")
    print("| Calibration Regime | Homogeneous Test Acc (%) | Homogeneous Test Jitter | Heterogeneous Test Acc (%) | Heterogeneous Test Jitter |")
    print("|---|---|---|---|---|")
    for r in regimes:
        hom_acc_mean, hom_acc_std = np.mean(results[r]['hom_acc']), np.std(results[r]['hom_acc'])
        hom_jit_mean, hom_jit_std = np.mean(results[r]['hom_jit']), np.std(results[r]['hom_jit'])
        het_acc_mean, het_acc_std = np.mean(results[r]['het_acc']), np.std(results[r]['het_acc'])
        het_jit_mean, het_jit_std = np.mean(results[r]['het_jit']), np.std(results[r]['het_jit'])
        print(f"| {r:<22} | {hom_acc_mean:.2f}% ± {hom_acc_std:.2f}% | {hom_jit_mean:.4f} ± {hom_jit_std:.4f} | {het_acc_mean:.2f}% ± {het_acc_std:.2f}% | {het_jit_mean:.4f} ± {het_jit_std:.4f} |")
        
    return results

if __name__ == "__main__":
    k16_res = run_k16_experiment()
    cross_res = run_cross_calibration_experiment()
