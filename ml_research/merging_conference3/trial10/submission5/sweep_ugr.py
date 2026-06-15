import numpy as np

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

def run_simulation_ugr(seed, eta_ugr, temp, num_samples=1000, D=192, K=4, L=14, L_frozen=3, gamma_V=0.05, sigma_layer=0.015):
    set_seed(seed)
    sigmas = [0.05, 0.15, 0.40, 1.20]
    v = generate_task_signatures(seed, D, K)
    
    # Calibration
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
            
    # Stream
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
        
    h = np.copy(stream_h3)
    all_alphas = {l: np.zeros((num_samples, K)) for l in range(3, L + 1)}
    
    prev_ugr_s = np.ones(K) / np.sqrt(K)
    accs = []
    
    for i in range(num_samples):
        target_task = stream_tasks[i]
        
        if i == 0:
            sample_s = np.ones(K) / np.sqrt(K)
        else:
            sample_s = np.copy(prev_ugr_s)
        sample_alpha = {}
        sample_alpha[3] = sample_s ** 2
        
        h_sample = np.copy(h[i])
        
        for l in range(4, L + 1):
            centroids = layer_centroids[l-1]
            
            # Cosine similarity
            norm_h = np.linalg.norm(h_sample) + 1e-8
            norm_c = np.linalg.norm(centroids, axis=1) + 1e-8
            sim = np.dot(centroids, h_sample) / (norm_h * norm_c)
            
            # Softmax
            shifted_sim = (sim - np.max(sim)) / temp
            w = np.exp(shifted_sim)
            w = w / np.sum(w)
            
            # Map bottom-up similarity to target vector on sphere
            w_sphere = w / (np.linalg.norm(w) + 1e-6)
            # Compute alignment
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
            
        prev_ugr_s = np.copy(sample_s)
        
        logits = np.zeros(K)
        biases = [0.0, 0.0, -0.90, -2.30]
        for k_idx in range(K):
            logits[k_idx] = -np.linalg.norm(h_sample - v[k_idx])**2 + biases[k_idx]
        pred = np.argmax(logits)
        acc_val = 1.0 if pred == target_task else 0.0
        accs.append(acc_val)
        
    jitters = []
    for i in range(num_samples):
        sample_jitter = 0.0
        for l in range(5, L + 1):
            sample_jitter += np.sum((all_alphas[l][i] - all_alphas[l-1][i])**2)
        jitters.append(sample_jitter / (L - L_frozen - 1))
        
    return np.mean(accs), np.mean(jitters)

if __name__ == "__main__":
    etas = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    temps = [0.001, 0.005, 0.010, 0.050, 0.100, 0.150, 0.200, 0.300]
    
    seeds = [42, 43, 44] # use 3 seeds for speed in sweep
    
    best_acc = 0.0
    best_jitter = 1.0
    best_eta = None
    best_temp = None
    
    results = []
    
    for eta in etas:
        for temp in temps:
            acc_list = []
            jitter_list = []
            for seed in seeds:
                acc, jitter = run_simulation_ugr(seed, eta, temp)
                acc_list.append(acc)
                jitter_list.append(jitter)
            mean_acc = np.mean(acc_list) * 100
            mean_jitter = np.mean(jitter_list)
            print(f"eta={eta:.2f}, temp={temp:.3f} | Acc={mean_acc:.2f}%, Jitter={mean_jitter:.6f}")
            results.append({"eta": eta, "temp": temp, "acc": mean_acc, "jitter": mean_jitter})
            
    results_sorted = sorted(results, key=lambda x: -x["acc"])
    print("\nTop 5 Hyperparameter Combinations by Accuracy:")
    for r in results_sorted[:5]:
        print(f"eta={r['eta']:.2f}, temp={r['temp']:.3f} | Acc={r['acc']:.2f}%, Jitter={r['jitter']:.6f}")
        
    results_sorted_jitter = sorted(results, key=lambda x: x["jitter"])
    print("\nTop 5 Hyperparameter Combinations by Jitter (Stability):")
    for r in results_sorted_jitter[:5]:
        print(f"eta={r['eta']:.2f}, temp={r['temp']:.3f} | Acc={r['acc']:.2f}%, Jitter={r['jitter']:.6f}")
