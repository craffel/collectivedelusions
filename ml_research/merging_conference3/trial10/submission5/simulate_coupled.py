import numpy as np
import pandas as pd

# Set random seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)

def generate_task_signatures(seed, D=192, K=4):
    set_seed(seed)
    signatures = []
    block_size = D // K
    for k in range(K):
        v = np.zeros(D)
        # Generate random values for active block
        start_idx = k * block_size
        end_idx = (k + 1) * block_size
        v[start_idx:end_idx] = np.random.normal(0, 1, block_size)
        # Normalize to unit L2 norm
        v = v / (np.linalg.norm(v) + 1e-8)
        signatures.append(v)
    return np.array(signatures)

def cos_sim(h, mu):
    norm_h = np.linalg.norm(h, axis=1, keepdims=True) + 1e-8
    norm_mu = np.linalg.norm(mu, axis=1, keepdims=True).T + 1e-8
    dot_prod = np.dot(h, mu.T)
    return dot_prod / (norm_h * norm_mu)

def run_simulation(seed, num_samples=1000, D=192, K=4, L=14, L_frozen=3, gamma_V=0.05, sigma_layer=0.015):
    set_seed(seed)
    
    # Task specific noise scales
    sigmas = [0.05, 0.15, 0.40, 1.20]
    
    # Generate orthogonal task signatures
    v = generate_task_signatures(seed, D, K)
    
    # ------------------ Calibration Phase ------------------
    cal_size = 64
    cal_features = []
    for k in range(K):
        noise = np.random.normal(0, sigmas[k], (cal_size, D))
        h3 = v[k] + noise
        cal_features.append(h3)
    
    layer_centroids = {}
    layer_centroids[3] = np.array([np.mean(cal_features[k], axis=0) for k in range(K)])
    
    current_cal = [np.copy(cal_features[k]) for k in range(K)]
    for l in range(4, L + 1):
        for k in range(K):
            h_prev = current_cal[k]
            h_next = h_prev + gamma_V * (v[k] - h_prev)
            layer_noise = np.random.normal(0, sigma_layer, h_next.shape)
            current_cal[k] = h_next + layer_noise
        
        if l < L:
            layer_centroids[l] = np.array([np.mean(current_cal[k], axis=0) for k in range(K)])
            
    # ------------------ Streaming Phase ------------------
    samples_per_task = num_samples // K
    stream_tasks = []
    for k in range(K):
        stream_tasks.extend([k] * samples_per_task)
    stream_tasks = np.array(stream_tasks)
    np.random.shuffle(stream_tasks)
    
    stream_h3 = np.zeros((num_samples, D))
    for i, k in enumerate(stream_tasks):
        noise = np.random.normal(0, sigmas[k], D)
        stream_h3[i] = v[k] + noise
        
    methods = [
        "Oracle",
        "Uniform",
        "SABLE_0.005",
        "SABLE_LC",
        "ChemMerge_0.05",
        "ChemMerge_0.05_Coupled",
        "ChemMerge_LC",
        "ChemMerge_LC_Coupled",
        "Momentum_Merge_Base",
        "Momentum_Merge_Base_Coupled",
        "Momentum_Merge_Advanced",
        "Momentum_Merge_Advanced_Coupled",
        "UGR",
        "UGR_Hybrid_Reset",
        "UGR_Softmax_Free",
        "UGR_Born_Target"
    ]
    
    results = {m: {"acc": [], "jitter_l5": [], "jitter_l4": []} for m in methods}
    
    for method in methods:
        h = np.copy(stream_h3)
        all_alphas = {l: np.zeros((num_samples, K)) for l in range(3, L + 1)}
        
        # State histories for the entire stream
        prev_ugr_s = np.ones(K) / np.sqrt(K)
        prev_ughr_s = np.ones(K) / np.sqrt(K)
        prev_ugrsf_s = np.ones(K) / np.sqrt(K)
        prev_ugrborn_s = np.ones(K) / np.sqrt(K)
        prev_chem_005_C = np.ones(K) * 0.25
        prev_chem_lc_C = np.ones(K) * 0.25
        prev_mom_base_alpha = np.ones(K) * 0.25
        prev_mom_adv_alpha = np.ones(K) * 0.25
        
        for i in range(num_samples):
            target_task = stream_tasks[i]
            
            # Reset defaults
            sample_C = np.ones(K) * 0.25
            sample_alpha = {}
            sample_alpha[3] = np.ones(K) * 0.25
            
            if method == "Oracle":
                sample_alpha[3] = np.zeros(K)
                sample_alpha[3][target_task] = 1.0
            elif method == "Uniform":
                sample_alpha[3] = np.ones(K) * 0.25
            elif method == "SABLE_0.005" or method == "SABLE_LC":
                sample_alpha[3] = np.ones(K) * 0.25
            elif method == "ChemMerge_0.05":
                sample_C = np.ones(K) * 0.25
                sample_alpha[3] = np.ones(K) * 0.25
            elif method == "ChemMerge_0.05_Coupled":
                sample_C = np.copy(prev_chem_005_C)
                sample_alpha[3] = sample_C / (np.sum(sample_C) + 1e-8)
            elif method == "ChemMerge_LC":
                sample_C = np.ones(K) * 0.25
                sample_alpha[3] = np.ones(K) * 0.25
            elif method == "ChemMerge_LC_Coupled":
                sample_C = np.copy(prev_chem_lc_C)
                sample_alpha[3] = sample_C / (np.sum(sample_C) + 1e-8)
            elif method == "Momentum_Merge_Base":
                sample_alpha[3] = np.ones(K) * 0.25
            elif method == "Momentum_Merge_Base_Coupled":
                sample_alpha[3] = np.copy(prev_mom_base_alpha)
            elif method == "Momentum_Merge_Advanced":
                # set at layer 4
                pass
            elif method == "Momentum_Merge_Advanced_Coupled":
                sample_alpha[3] = np.copy(prev_mom_adv_alpha)
            elif method == "UGR":
                if i == 0:
                    sample_s = np.ones(K) / np.sqrt(K)
                else:
                    sample_s = np.copy(prev_ugr_s)
                sample_alpha[3] = sample_s ** 2
            elif method == "UGR_Hybrid_Reset":
                if i == 0:
                    sample_s = np.ones(K) / np.sqrt(K)
                else:
                    sample_s = np.copy(prev_ughr_s)
                sample_alpha[3] = sample_s ** 2
            elif method == "UGR_Softmax_Free":
                if i == 0:
                    sample_s = np.ones(K) / np.sqrt(K)
                else:
                    sample_s = np.copy(prev_ugrsf_s)
                sample_alpha[3] = sample_s ** 2
            elif method == "UGR_Born_Target":
                if i == 0:
                    sample_s = np.ones(K) / np.sqrt(K)
                else:
                    sample_s = np.copy(prev_ugrborn_s)
                sample_alpha[3] = sample_s ** 2
            
            h_sample = np.copy(h[i])
            
            for l in range(4, L + 1):
                if "LC" in method or "Advanced" in method or method == "UGR" or method == "UGR_Softmax_Free":
                    centroids = layer_centroids[l-1]
                else:
                    centroids = layer_centroids[3]
                
                norm_h = np.linalg.norm(h_sample) + 1e-8
                norm_c = np.linalg.norm(centroids, axis=1) + 1e-8
                sim = np.dot(centroids, h_sample) / (norm_h * norm_c)
                
                if method == "UGR_Softmax_Free":
                    # Softmax-free target: ReLU on cosine similarity, L1 normalized
                    relu_sim = np.maximum(sim, 0.0)
                    sum_relu = np.sum(relu_sim)
                    if sum_relu > 1e-8:
                        w = relu_sim / sum_relu
                    else:
                        w = np.ones(K) / K
                else:
                    if "SABLE_0.005" in method:
                        temp = 0.005
                    elif "SABLE_LC" in method:
                        temp = 0.200
                    elif "ChemMerge_0.05" in method:
                        temp = 0.050
                    elif "ChemMerge_LC" in method:
                        temp = 0.050
                    elif "Momentum_Merge_Base" in method:
                        temp = 0.100
                    elif "Momentum_Merge_Advanced" in method:
                        temp = 0.005
                    elif "UGR" in method:
                        temp = 0.010
                    else:
                        temp = 1.0
                    
                    shifted_sim = (sim - np.max(sim)) / temp
                    w = np.exp(shifted_sim)
                    w = w / np.sum(w)
                
                if method == "Oracle":
                    sample_alpha[l] = np.zeros(K)
                    sample_alpha[l][target_task] = 1.0
                elif method == "Uniform":
                    sample_alpha[l] = np.ones(K) * 0.25
                elif "SABLE" in method:
                    sample_alpha[l] = w
                elif "ChemMerge" in method:
                    dt = 1.5
                    k_decay = 0.3
                    sample_C = np.clip(sample_C + dt * (w * (1.0 - sample_C) - k_decay * sample_C), 0.0, 1.0)
                    sample_alpha[l] = sample_C / (np.sum(sample_C) + 1e-8)
                elif "Momentum_Merge_Base" in method:
                    beta = 0.60
                    sample_alpha[l] = (1.0 - beta) * w + beta * sample_alpha[l-1]
                elif "Momentum_Merge_Advanced" in method:
                    beta = 0.60
                    sample_alpha[l] = (1.0 - beta) * w + beta * sample_alpha[l-1]
                elif "UGR" in method or method == "UGR_Softmax_Free":
                    eta_ugr = 0.80
                    if method == "UGR_Born_Target":
                        w_sphere = np.sqrt(np.clip(w, 0.0, 1.0))
                    else:
                        w_sphere = w / (np.linalg.norm(w) + 1e-6)
                    
                    if method == "UGR_Hybrid_Reset" and l == 4:
                        c_val = np.dot(sample_s, w_sphere)
                        if c_val < 0.70:
                            sample_s = np.ones(K) / np.sqrt(K)
                            sample_alpha[3] = sample_s ** 2
                            
                    c_val = np.dot(sample_s, w_sphere)
                    if np.abs(c_val) >= 1.0 - 1e-6:
                        pass
                    else:
                        v_orth = w_sphere - c_val * sample_s
                        u_vec = v_orth / (np.linalg.norm(v_orth) + 1e-8)
                        phi_angle = np.arccos(np.clip(c_val, -1.0, 1.0))
                        theta_angle = eta_ugr * phi_angle
                        sample_s = np.cos(theta_angle) * sample_s + np.sin(theta_angle) * u_vec
                    sample_alpha[l] = sample_s ** 2
                
                all_alphas[l][i] = sample_alpha[l]
                if l == 4:
                    if "Advanced" in method and "Coupled" not in method:
                        all_alphas[3][i] = sample_alpha[3]
                    elif "Coupled" in method or "UGR" in method:
                        all_alphas[3][i] = sample_alpha[3]
                
                blend_v = np.zeros(D)
                for k in range(K):
                    blend_v += sample_alpha[l][k] * v[k]
                
                h_sample = h_sample + gamma_V * (blend_v - h_sample)
                h_sample += np.random.normal(0, sigma_layer, D)
            
            # State propagation
            if method == "UGR":
                prev_ugr_s = np.copy(sample_s)
            elif method == "UGR_Hybrid_Reset":
                prev_ughr_s = np.copy(sample_s)
            elif method == "UGR_Softmax_Free":
                prev_ugrsf_s = np.copy(sample_s)
            elif method == "UGR_Born_Target":
                prev_ugrborn_s = np.copy(sample_s)
            elif method == "ChemMerge_0.05_Coupled":
                prev_chem_005_C = np.copy(sample_C)
            elif method == "ChemMerge_LC_Coupled":
                prev_chem_lc_C = np.copy(sample_C)
            elif method == "Momentum_Merge_Base_Coupled":
                prev_mom_base_alpha = np.copy(sample_alpha[L])
            elif method == "Momentum_Merge_Advanced_Coupled":
                prev_mom_adv_alpha = np.copy(sample_alpha[L])
                
            # Prediction
            logits = np.zeros(K)
            biases = [0.0, 0.0, -0.90, -2.30]
            for k_idx in range(K):
                logits[k_idx] = -np.linalg.norm(h_sample - v[k_idx])**2 + biases[k_idx]
            pred = np.argmax(logits)
            acc_val = 1.0 if pred == target_task else 0.0
            results[method]["acc"].append(acc_val)
            
        # Jitter calculation: L5 (starting at layer 5) and L4 (starting at layer 4)
        jitters_l5 = []
        jitters_l4 = []
        for i in range(num_samples):
            # Jitter L5
            sum_l5 = 0.0
            for l in range(5, L + 1):
                sum_l5 += np.sum((all_alphas[l][i] - all_alphas[l-1][i])**2)
            jitters_l5.append(sum_l5 / (L - L_frozen - 1))
            
            # Jitter L4 (including layer 3 to 4 transition)
            sum_l4 = 0.0
            for l in range(4, L + 1):
                sum_l4 += np.sum((all_alphas[l][i] - all_alphas[l-1][i])**2)
            jitters_l4.append(sum_l4 / (L - L_frozen))
            
        results[method]["jitter_l5"] = jitters_l5
        results[method]["jitter_l4"] = jitters_l4
        
    return results

if __name__ == "__main__":
    seeds = list(range(42, 52))
    all_seed_results = []
    
    for seed in seeds:
        res = run_simulation(seed)
        seed_data = {}
        for m in res:
            seed_data[f"{m}_acc"] = np.mean(res[m]["acc"])
            seed_data[f"{m}_jitter_l5"] = np.mean(res[m]["jitter_l5"])
            seed_data[f"{m}_jitter_l4"] = np.mean(res[m]["jitter_l4"])
        all_seed_results.append(seed_data)
        
    df = pd.DataFrame(all_seed_results)
    
    summary = {}
    for m in run_simulation(42).keys():
        summary[m] = {
            "Acc Mean": df[f"{m}_acc"].mean() * 100,
            "Acc Std": df[f"{m}_acc"].std() * 100,
            "Jitter L5 Mean (x10^-4)": df[f"{m}_jitter_l5"].mean() * 10000,
            "Jitter L5 Std (x10^-4)": df[f"{m}_jitter_l5"].std() * 10000,
            "Jitter L4 Mean (x10^-4)": df[f"{m}_jitter_l4"].mean() * 10000,
            "Jitter L4 Std (x10^-4)": df[f"{m}_jitter_l4"].std() * 10000,
        }
        
    summary_df = pd.DataFrame(summary).T
    print("\nSimulation Summary across 10 Seeds (including Coupled Baselines and Jitter deconstruction):")
    print(summary_df.to_string())
