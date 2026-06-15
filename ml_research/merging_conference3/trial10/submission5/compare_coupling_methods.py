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
        start_idx = k * block_size
        end_idx = (k + 1) * block_size
        v[start_idx:end_idx] = np.random.normal(0, 1, block_size)
        v = v / (np.linalg.norm(v) + 1e-8)
        signatures.append(v)
    return np.array(signatures)

def run_simulation(seed, num_samples=1000, D=192, K=4, L=14, L_frozen=3, gamma_V=0.05, sigma_layer=0.015):
    set_seed(seed)
    
    # Task specific noise scales
    sigmas = [0.05, 0.15, 0.40, 1.20]
    
    # Generate orthogonal task signatures
    v = generate_task_signatures(seed, D, K)
    
    # Calibration Phase
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
            
    # Streaming Phase
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
        
    methods = ["UGR_Cross_Layer", "UGR_Layer_Wise"]
    results = {m: {"acc": [], "jitter_l5": [], "jitter_l4": []} for m in methods}
    
    for method in methods:
        h = np.copy(stream_h3)
        all_alphas = {l: np.zeros((num_samples, K)) for l in range(3, L + 1)}
        
        # State histories
        prev_ugr_s = np.ones(K) / np.sqrt(K)  # for Cross-Layer
        prev_ugr_s_layers = {l: np.ones(K) / np.sqrt(K) for l in range(3, L + 1)}  # for Layer-Wise
        
        for i in range(num_samples):
            target_task = stream_tasks[i]
            
            sample_alpha = {}
            
            if method == "UGR_Cross_Layer":
                sample_s = np.copy(prev_ugr_s)
                sample_alpha[3] = sample_s ** 2
            elif method == "UGR_Layer_Wise":
                sample_alpha[3] = np.copy(prev_ugr_s_layers[3]) ** 2
            
            h_sample = np.copy(h[i])
            
            for l in range(4, L + 1):
                centroids = layer_centroids[l-1]
                
                norm_h = np.linalg.norm(h_sample) + 1e-8
                norm_c = np.linalg.norm(centroids, axis=1) + 1e-8
                sim = np.dot(centroids, h_sample) / (norm_h * norm_c)
                
                # localized Softmax target
                temp = 0.010
                shifted_sim = (sim - np.max(sim)) / temp
                w = np.exp(shifted_sim)
                w = w / np.sum(w)
                
                eta_ugr = 0.80
                w_sphere = w / (np.linalg.norm(w) + 1e-6)
                
                if method == "UGR_Cross_Layer":
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
                
                elif method == "UGR_Layer_Wise":
                    # Initialize layer l state from previous query's layer l state
                    sample_s_l = np.copy(prev_ugr_s_layers[l])
                    c_val = np.dot(sample_s_l, w_sphere)
                    if np.abs(c_val) >= 1.0 - 1e-6:
                        pass
                    else:
                        v_orth = w_sphere - c_val * sample_s_l
                        u_vec = v_orth / (np.linalg.norm(v_orth) + 1e-8)
                        phi_angle = np.arccos(np.clip(c_val, -1.0, 1.0))
                        theta_angle = eta_ugr * phi_angle
                        sample_s_l = np.cos(theta_angle) * sample_s_l + np.sin(theta_angle) * u_vec
                    sample_alpha[l] = sample_s_l ** 2
                    # Save updated layer l state for next query
                    prev_ugr_s_layers[l] = np.copy(sample_s_l)
                
                all_alphas[l][i] = sample_alpha[l]
                
                blend_v = np.zeros(D)
                for k in range(K):
                    blend_v += sample_alpha[l][k] * v[k]
                
                h_sample = h_sample + gamma_V * (blend_v - h_sample)
                h_sample += np.random.normal(0, sigma_layer, D)
                
            if method == "UGR_Cross_Layer":
                all_alphas[3][i] = prev_ugr_s ** 2
                prev_ugr_s = np.copy(sample_s)
            elif method == "UGR_Layer_Wise":
                all_alphas[3][i] = prev_ugr_s_layers[3] ** 2
                
            # Compute accuracy
            logits = np.zeros(K)
            biases = [0.0, 0.0, -0.90, -2.30]
            for k_idx in range(K):
                logits[k_idx] = -np.linalg.norm(h_sample - v[k_idx])**2 + biases[k_idx]
            pred = np.argmax(logits)
            acc_val = 1.0 if pred == target_task else 0.0
            results[method]["acc"].append(acc_val)
            
        # Compute jitters
        jitters_l5 = []
        jitters_l4 = []
        for i in range(num_samples):
            sum_l5 = 0.0
            for l in range(5, L + 1):
                sum_l5 += np.sum((all_alphas[l][i] - all_alphas[l-1][i])**2)
            jitters_l5.append(sum_l5 / (L - L_frozen - 1))
            
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
    for m in ["UGR_Cross_Layer", "UGR_Layer_Wise"]:
        summary[m] = {
            "Acc Mean (%)": df[f"{m}_acc"].mean() * 100,
            "Acc Std (%)": df[f"{m}_acc"].std() * 100,
            "Jitter L5 Mean (x10^-4)": df[f"{m}_jitter_l5"].mean() * 10000,
            "Jitter L5 Std (x10^-4)": df[f"{m}_jitter_l5"].std() * 10000,
            "Jitter L4 Mean (x10^-4)": df[f"{m}_jitter_l4"].mean() * 10000,
            "Jitter L4 Std (x10^-4)": df[f"{m}_jitter_l4"].std() * 10000,
        }
        
    summary_df = pd.DataFrame(summary).T
    print("\nSimulation Summary: Cross-Layer vs. Layer-Wise Geodesic Coupling (10 Seeds):")
    print(summary_df.to_string())
