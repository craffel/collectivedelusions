import numpy as np
import torch
import os

# Set random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Simulation Parameters
D = 192               # Representation dimension
K = 4                 # Number of task experts
L = 14                # Total layers
L_frozen = 3          # Frozen layers
T = 1000              # Sequence length
kappa_scale = 1.50    # Calibrated scale parameter
sigma_noise = 0.20    # Input observation noise
sigma_layer_noise = 0.01 # Layer propagation noise
g_scale = 0.35         # LoRA projection scale
tau = 0.10            # Softmax temperature

# Generate task signatures (centroids)
def get_signatures(overlapping=True):
    signatures = np.zeros((K, D))
    S = D // K # 48
    if not overlapping:
        for k in range(K):
            signatures[k, k*S : (k+1)*S] = 1.0 / np.sqrt(S)
    else:
        V = 12
        for k in range(K):
            start = k*S - k*V
            end = start + S
            signatures[k, start:end] = 1.0 / np.sqrt(S)
    return torch.tensor(signatures, dtype=torch.float32)

# Generate query stream task labels
def get_query_stream(homogeneous=True):
    if homogeneous:
        y = []
        for k in range(K):
            y.extend([k] * (T // K))
        return np.array(y)
    else:
        np.random.seed(42)
        return np.random.randint(0, K, size=T)

# Run 2D-STEM simulation with specific beta_depth and beta_temp_0
def run_stem_ablation(seed, beta_depth, beta_temp_0, overlapping=True, homogeneous=True):
    set_seed(seed)
    v = get_signatures(overlapping=overlapping)
    y = get_query_stream(homogeneous=homogeneous)
    
    stem_alpha = {l: torch.ones(K) / K for l in range(L_frozen, L + 1)}
    accuracies = []
    coefficients = []
    
    for t in range(T):
        target_k = y[t]
        target_v = v[target_k]
        
        # Input generation with noise
        h_0 = target_v + torch.randn(D) * sigma_noise
        
        # Propagation through frozen layers
        h = h_0.clone()
        for l in range(1, L_frozen + 1):
            h = h + torch.randn(D) * sigma_layer_noise
            
        h_3 = h.clone()
        
        # Coordinate signals
        e_t = torch.zeros(K)
        for k in range(K):
            e_t[k] = torch.max(torch.tensor(0.0), torch.dot(h_3, v[k]) / (torch.norm(h_3) * torch.norm(v[k]) + 1e-6))
            
        if t == 0:
            Sim_t = torch.tensor(1.0)
            e_prev = e_t.clone()
        else:
            Sim_t = torch.dot(e_t, e_prev) / (torch.norm(e_t) * torch.norm(e_prev) + 1e-6)
            Sim_t = torch.clamp(Sim_t, 0.0, 1.0)
            e_prev = e_t.clone()
            
        h_l = h_3.clone()
        m_coeffs = []
        
        # Propagate through adapted layers
        for l in range(L_frozen + 1, L + 1):
            S = torch.zeros(K)
            for k in range(K):
                S[k] = torch.dot(h_l, v[k]) / (torch.norm(h_l) * torch.norm(v[k]) + 1e-6)
            S_noise = torch.randn(K) * 0.04
            w_l_t = torch.softmax((S + S_noise) / tau, dim=0)
            
            # Adaptive Temporal Gating with Power-Law exponent gamma = 3
            beta_temp_t = beta_temp_0 * (Sim_t.item() ** 3) if t > 0 else 0.0
            
            if l == L_frozen + 1:
                alpha_prev_depth = w_l_t.clone()
                
            alpha_prev_temp = stem_alpha[l]
            
            alpha = beta_depth * alpha_prev_depth + beta_temp_t * alpha_prev_temp + (1.0 - beta_depth - beta_temp_t) * w_l_t
            alpha_prev_depth = alpha.clone()
            stem_alpha[l] = alpha.clone()
            
            m_coeffs.append(alpha.clone())
            h_l = h_l + g_scale * torch.matmul(alpha, v - h_l) + torch.randn(D) * sigma_layer_noise
            
        dist_sq = torch.sum((h_l - target_v)**2)
        acc = torch.exp(-kappa_scale * dist_sq).item()
        accuracies.append(acc)
        coefficients.append(m_coeffs[-1].numpy())
        
    accs = np.array(accuracies)
    coeffs = np.array(coefficients)
    jitters = np.sum(np.abs(coeffs[1:] - coeffs[:-1]), axis=1)
    return np.mean(accs) * 100.0, np.mean(jitters)

def main():
    seeds = [42, 43, 44, 45, 46]
    beta_depth_vals = [0.2, 0.4, 0.5]
    beta_temp_vals = [0.2, 0.4, 0.5]
    
    print("Running hyperparameter ablation grid...")
    grid_results = {}
    
    for bd in beta_depth_vals:
        grid_results[bd] = {}
        for bt in beta_temp_vals:
            print(f"Ablating beta_depth={bd}, beta_temp={bt}...")
            hom_accs, hom_jits = [], []
            het_accs, het_jits = [], []
            for seed in seeds:
                ha, hj = run_stem_ablation(seed, bd, bt, overlapping=True, homogeneous=True)
                he_a, he_j = run_stem_ablation(seed, bd, bt, overlapping=True, homogeneous=False)
                hom_accs.append(ha)
                hom_jits.append(hj)
                het_accs.append(he_a)
                het_jits.append(he_j)
                
            grid_results[bd][bt] = {
                "hom_acc": np.mean(hom_accs),
                "hom_acc_std": np.std(hom_accs),
                "hom_jit": np.mean(hom_jits),
                "hom_jit_std": np.std(hom_jits),
                "het_acc": np.mean(het_accs),
                "het_acc_std": np.std(het_accs),
                "het_jit": np.mean(het_jits),
                "het_jit_std": np.std(het_jits)
            }
            
    print("\nLaTeX Table output:")
    print("\\begin{table}[h]")
    print("\\caption{Hyperparameter sensitivity sweep of the spatial momentum $\\beta_{\\text{depth}}$ and temporal momentum $\\beta_{\\text{temp}, 0}$ coefficients of \\textbf{2D-STEM} on the Overlapping manifolds configuration (mean $\\pm$ standard deviation across 5 independent seeds).}")
    print("\\label{tab:beta_ablation}")
    print("\\vskip 0.15in")
    print("\\begin{center}")
    print("\\begin{small}")
    print("\\begin{sc}")
    print("\\begin{tabular}{cccccc}")
    print("\\toprule")
    print(" & & \\multicolumn{2}{c}{Homogeneous Stream} & \\multicolumn{2}{c}{Heterogeneous Stream} \\\\")
    print("\\cmidrule(r){3-4} \\cmidrule(r){5-6}")
    print("$\\beta_{\\text{depth}}$ & $\\beta_{\\text{temp}, 0}$ & Accuracy (\\%) & Routing Jitter & Accuracy (\\%) & Routing Jitter \\\\")
    print("\\midrule")
    for bd in beta_depth_vals:
        for bt in beta_temp_vals:
            res = grid_results[bd][bt]
            bold_str = "\\mathbf{" if (bd == 0.4 and bt == 0.4) else ""
            bold_end = "}" if (bd == 0.4 and bt == 0.4) else ""
            print(f"{bd:.1f} & {bt:.1f} & {bold_str}{res['hom_acc']:.2f}\\% \\pm {res['hom_acc_std']:.2f}%{bold_end} & {bold_str}{res['hom_jit']:.4f} \\pm {res['hom_jit_std']:.4f}{bold_end} & {bold_str}{res['het_acc']:.2f}\\% \\pm {res['het_acc_std']:.2f}%{bold_end} & {bold_str}{res['het_jit']:.4f} \\pm {res['het_jit_std']:.4f}{bold_end} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{sc}")
    print("\\end{small}")
    print("\\end{center}")
    print("\\vskip -0.1in")
    print("\\end{table}")

if __name__ == "__main__":
    main()
