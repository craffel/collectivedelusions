import numpy as np
import pandas as pd

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

def run_simulation_decomposition(seed, num_samples=1000, D=192, K=4, L=14, L_frozen=3, gamma_V=0.05, sigma_layer=0.015):
    set_seed(seed)
    sigmas = [0.05, 0.15, 0.40, 1.20]
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
        
    methods = [
        "SABLE_0.005",
        "SABLE_LC",
        "ChemMerge_0.05",
        "Momentum_Merge_Advanced",
        "UGR"
    ]
    
    results = {m: {"acc": [], "jitter_intra": [], "jitter_inter": [], "jitter_all": []} for m in methods}
    
    for method in methods:
        h = np.copy(stream_h3)
        all_alphas = {l: np.zeros((num_samples, K)) for l in range(3, L + 1)}
        
        # Init layer 3
        if method == "SABLE_0.005" or method == "SABLE_LC":
            all_alphas[3].fill(0.25)
        elif method == "ChemMerge_0.05":
            all_alphas[3].fill(0.25)
            sample_C = np.ones(K) * 0.25
        elif method == "Momentum_Merge_Advanced":
            pass
        elif method == "UGR":
            prev_ugr_s = np.ones(K) / np.sqrt(K)
            
        for i in range(num_samples):
            target_task = stream_tasks[i]
            
            if method == "UGR":
                if i == 0:
                    sample_s = np.ones(K) / np.sqrt(K)
                else:
                    sample_s = np.copy(prev_ugr_s)
                sample_alpha = {3: sample_s ** 2}
            else:
                sample_C = np.ones(K) * 0.25
                sample_alpha = {3: np.ones(K) * 0.25}
                
            h_sample = np.copy(h[i])
            
            for l in range(4, L + 1):
                if "LC" in method or "Advanced" in method or method == "UGR":
                    centroids = layer_centroids[l-1]
                else:
                    centroids = layer_centroids[3]
                
                # Cosine similarity
                norm_h = np.linalg.norm(h_sample) + 1e-8
                norm_c = np.linalg.norm(centroids, axis=1) + 1e-8
                sim = np.dot(centroids, h_sample) / (norm_h * norm_c)
                
                if method == "SABLE_0.005":
                    temp = 0.005
                elif method == "SABLE_LC":
                    temp = 0.200
                elif "ChemMerge" in method:
                    temp = 0.050
                elif method == "Momentum_Merge_Advanced":
                    temp = 0.005
                elif method == "UGR":
                    temp = 0.010
                
                shifted_sim = (sim - np.max(sim)) / temp
                w = np.exp(shifted_sim)
                w = w / np.sum(w)
                
                if method == "SABLE_0.005" or method == "SABLE_LC":
                    sample_alpha[l] = w
                elif "ChemMerge" in method:
                    dt = 1.5
                    k_decay = 0.3
                    sample_C = np.clip(sample_C + dt * (w * (1.0 - sample_C) - k_decay * sample_C), 0.0, 1.0)
                    sample_alpha[l] = sample_C / (np.sum(sample_C) + 1e-8)
                elif method == "Momentum_Merge_Advanced":
                    beta = 0.60
                    sample_alpha[l] = (1.0 - beta) * w + beta * sample_alpha[l-1]
                elif method == "UGR":
                    eta_ugr = 0.80
                    w_sphere = w / (np.linalg.norm(w) + 1e-6)
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
                if l == 4 and ("Advanced" in method or method == "UGR"):
                    all_alphas[3][i] = sample_alpha[3]
                    
                blend_v = np.zeros(D)
                for k in range(K):
                    blend_v += sample_alpha[l][k] * v[k]
                h_sample = h_sample + gamma_V * (blend_v - h_sample)
                h_sample += np.random.normal(0, sigma_layer, D)
                
            if method == "UGR":
                prev_ugr_s = np.copy(sample_s)
                
            logits = np.zeros(K)
            biases = [0.0, 0.0, -0.90, -2.30]
            for k_idx in range(K):
                logits[k_idx] = -np.linalg.norm(h_sample - v[k_idx])**2 + biases[k_idx]
            pred = np.argmax(logits)
            acc_val = 1.0 if pred == target_task else 0.0
            results[method]["acc"].append(acc_val)
            
        # Calculate routing jitter decomposed
        for i in range(num_samples):
            sample_jitter = 0.0
            for l in range(5, L + 1):
                sample_jitter += np.sum((all_alphas[l][i] - all_alphas[l-1][i])**2)
            avg_jitter = sample_jitter / (L - L_frozen - 1)
            
            results[method]["jitter_all"].append(avg_jitter)
            if i > 0:
                if stream_tasks[i] == stream_tasks[i-1]:
                    # Intra-task sequence (stability)
                    results[method]["jitter_intra"].append(avg_jitter)
                else:
                    # Inter-task sequence (boundary agility)
                    results[method]["jitter_inter"].append(avg_jitter)
                    
    return results

if __name__ == "__main__":
    seeds = list(range(42, 52))
    all_results = []
    
    for seed in seeds:
        print(f"Running seed {seed}...")
        res = run_simulation_decomposition(seed)
        seed_data = {}
        for m in res:
            seed_data[f"{m}_acc"] = np.mean(res[m]["acc"])
            seed_data[f"{m}_jitter_all"] = np.mean(res[m]["jitter_all"])
            seed_data[f"{m}_jitter_intra"] = np.mean(res[m]["jitter_intra"])
            seed_data[f"{m}_jitter_inter"] = np.mean(res[m]["jitter_inter"])
        all_results.append(seed_data)
        
    df = pd.DataFrame(all_results)
    
    print("\nDECOMPOSED JITTER RESULTS ACROSS 10 SEEDS:")
    for m in ["SABLE_0.005", "SABLE_LC", "ChemMerge_0.05", "Momentum_Merge_Advanced", "UGR"]:
        print(f"\n--- {m} ---")
        print(f"Accuracy:        {df[f'{m}_acc'].mean()*100:.2f}% ± {df[f'{m}_acc'].std()*100:.2f}%")
        print(f"Overall Jitter:  {df[f'{m}_jitter_all'].mean()*10000:.4f} ± {df[f'{m}_jitter_all'].std()*10000:.4f}")
        print(f"Intra-Task Jitter (Stability): {df[f'{m}_jitter_intra'].mean()*10000:.4f} ± {df[f'{m}_jitter_intra'].std()*10000:.4f}")
        print(f"Inter-Task Jitter (Agility):   {df[f'{m}_jitter_inter'].mean()*10000:.4f} ± {df[f'{m}_jitter_inter'].std()*10000:.4f}")
