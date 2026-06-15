import numpy as np
import pandas as pd
from simulate_coupled import generate_task_signatures, set_seed

def run_simulation_hybrid(seed, threshold, num_samples=1000, D=192, K=4, L=14, L_frozen=3, gamma_V=0.05, sigma_layer=0.015):
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
        "UGR_Standard",
        "UGR_Hybrid_Reset"
    ]
    
    results = {m: {"acc": [], "jitter_l5": [], "jitter_l4": []} for m in methods}
    
    for method in methods:
        h = np.copy(stream_h3)
        all_alphas = {l: np.zeros((num_samples, K)) for l in range(3, L + 1)}
        
        # State histories for the entire stream
        prev_s = np.ones(K) / np.sqrt(K)
        
        for i in range(num_samples):
            target_task = stream_tasks[i]
            
            sample_alpha = {}
            if i == 0:
                sample_s = np.ones(K) / np.sqrt(K)
            else:
                sample_s = np.copy(prev_s)
            sample_alpha[3] = sample_s ** 2
            
            h_sample = np.copy(h[i])
            
            for l in range(4, L + 1):
                centroids = layer_centroids[l-1]
                
                norm_h = np.linalg.norm(h_sample) + 1e-8
                norm_c = np.linalg.norm(centroids, axis=1) + 1e-8
                sim = np.dot(centroids, h_sample) / (norm_h * norm_c)
                
                temp = 0.010
                shifted_sim = (sim - np.max(sim)) / temp
                w = np.exp(shifted_sim)
                w = w / np.sum(w)
                
                eta_ugr = 0.80
                w_sphere = w / (np.linalg.norm(w) + 1e-6)
                
                # Apply hybrid reset at layer 4 if method is UGR_Hybrid_Reset
                if method == "UGR_Hybrid_Reset" and l == 4:
                    c_val = np.dot(sample_s, w_sphere)
                    # If cosine similarity is below threshold, it indicates a strong task transition
                    if c_val < threshold:
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
                    all_alphas[3][i] = sample_alpha[3]
                
                blend_v = np.zeros(D)
                for k in range(K):
                    blend_v += sample_alpha[l][k] * v[k]
                
                h_sample = h_sample + gamma_V * (blend_v - h_sample)
                h_sample += np.random.normal(0, sigma_layer, D)
            
            # State propagation
            prev_s = np.copy(sample_s)
                
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
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    seeds = list(range(42, 52))
    
    print("Sweeping Hybrid Reset Thresholds over 10 seeds...")
    for th in thresholds:
        all_seed_results = []
        for seed in seeds:
            res = run_simulation_hybrid(seed, th)
            seed_data = {}
            for m in res:
                seed_data[f"{m}_acc"] = np.mean(res[m]["acc"])
                seed_data[f"{m}_jitter_l5"] = np.mean(res[m]["jitter_l5"])
                seed_data[f"{m}_jitter_l4"] = np.mean(res[m]["jitter_l4"])
            all_seed_results.append(seed_data)
        
        df = pd.DataFrame(all_seed_results)
        acc_mean = df["UGR_Hybrid_Reset_acc"].mean() * 100
        jit_l5_mean = df["UGR_Hybrid_Reset_jitter_l5"].mean() * 10000
        jit_l4_mean = df["UGR_Hybrid_Reset_jitter_l4"].mean() * 10000
        
        print(f"Threshold: {th:.1f} | Acc: {acc_mean:.2f}% | Jitter L5: {jit_l5_mean:.2f} | Jitter L4: {jit_l4_mean:.2f}")
