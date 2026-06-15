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

def cos_sim(h, mu):
    # Compute cosine similarity between h (N x D) and mu (K x D)
    # Output is N x K similarity matrix
    norm_h = np.linalg.norm(h, axis=1, keepdims=True) + 1e-8
    norm_mu = np.linalg.norm(mu, axis=1, keepdims=True).T + 1e-8
    dot_prod = np.dot(h, mu.T)
    return dot_prod / (norm_h * norm_mu)

def softmax(x, temp):
    # Softmax of rows of x scaled by temp
    # x shape: N x K
    # Output shape: N x K
    shifted_x = (x - np.max(x, axis=1, keepdims=True)) / temp
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def run_simulation(seed, num_samples=1000, D=192, K=4, L=14, L_frozen=3, gamma_V=0.05, sigma_layer=0.015):
    set_seed(seed)
    
    # Task specific noise scales
    sigmas = [0.05, 0.15, 0.40, 1.20]
    
    # Generate orthogonal task signatures
    v = generate_task_signatures(seed, D, K)
    
    # ------------------ Calibration Phase ------------------
    # Generate 64 calibration samples per task
    cal_size = 64
    cal_features = [] # List of activations at layer 3 for each task
    for k in range(K):
        noise = np.random.normal(0, sigmas[k], (cal_size, D))
        h3 = v[k] + noise
        cal_features.append(h3)
    
    # Propagate calibration samples stand-alone (oracle ensembling) to compute layer-wise centroids
    # layer_centroids[l] is of shape K x D for l in range(3, L)
    layer_centroids = {}
    
    # For l=3 (before any adapted layer)
    layer_centroids[3] = np.array([np.mean(cal_features[k], axis=0) for k in range(K)])
    
    # Propagate layer-by-layer
    current_cal = [np.copy(cal_features[k]) for k in range(K)]
    for l in range(4, L + 1):
        for k in range(K):
            # standalone execution: alpha_k = 1.0, others 0.0
            h_prev = current_cal[k]
            # CAB update step
            h_next = h_prev + gamma_V * (v[k] - h_prev)
            # Add layer noise
            layer_noise = np.random.normal(0, sigma_layer, h_next.shape)
            current_cal[k] = h_next + layer_noise
        
        # Save centroids entering layer l (which is the output of layer l-1)
        if l < L:
            layer_centroids[l] = np.array([np.mean(current_cal[k], axis=0) for k in range(K)])
            
    # ------------------ Streaming Phase ------------------
    # Create stream of shuffled samples: exactly 250 of each task
    samples_per_task = num_samples // K
    stream_tasks = []
    for k in range(K):
        stream_tasks.extend([k] * samples_per_task)
    stream_tasks = np.array(stream_tasks)
    # Shuffle stream tasks
    np.random.shuffle(stream_tasks)
    
    # Pre-generate stream inputs at layer 3
    stream_h3 = np.zeros((num_samples, D))
    for i, k in enumerate(stream_tasks):
        noise = np.random.normal(0, sigmas[k], D)
        stream_h3[i] = v[k] + noise
        
    # We will evaluate different ensembling methods:
    methods = [
        "Oracle",
        "Uniform",
        "SABLE_0.005",
        "SABLE_LC",
        "ChemMerge_0.05",
        "ChemMerge_LC",
        "Momentum_Merge_Base",
        "Momentum_Merge_Advanced",
        "UGR" # Our proposal
    ]
    
    results = {m: {"acc": [], "jitter": []} for m in methods}
    
    # Run each ensembling method on the exact same stream
    for method in methods:
        h = np.copy(stream_h3)
        
        # Initialize ensembling weights history
        # alpha_history[l] will store the weights at layer l for all samples
        alpha_history = {}
        
        # SABLE or other models use different state initialization/persistence
        # For stateful methods, we keep track of states across samples
        if method == "ChemMerge_0.05" or method == "ChemMerge_LC":
            # State is concentration: num_samples x K
            # Concentrations are initialized to 0.25 at layer 3
            C = np.ones((num_samples, K)) * 0.25
        elif method == "UGR":
            # State vector is on sphere: num_samples x K
            # Initialize state vector at layer 3 for first sample
            s_state = np.ones((num_samples, K)) / np.sqrt(K)
        
        # We process layer by layer, but wait, for sequential temporal coupling in UGR:
        # We must process sample-by-sample for temporal state tracking, or we can vectorize if stateless.
        # ChemMerge does not couple across samples (it resets to 0.25 for each sample at layer 3).
        # Momentum-Merge does not couple across samples (it resets to 0.25 or raw similarity at layer 3).
        # Only UGR couples across samples! "s_t^{(3)} = s_{t-1}^{(14)}"
        # So we can process sample-by-sample, which is extremely robust and allows exact implementation of all methods.
        
        all_alphas = {l: np.zeros((num_samples, K)) for l in range(3, L + 1)}
        
        # Initialize layer 3 weights
        if method == "Oracle":
            for i, k in enumerate(stream_tasks):
                all_alphas[3][i, k] = 1.0
        elif method == "Uniform":
            all_alphas[3].fill(0.25)
        elif method == "SABLE_0.005" or method == "SABLE_LC":
            all_alphas[3].fill(0.25)
        elif method == "ChemMerge_0.05" or method == "ChemMerge_LC":
            all_alphas[3].fill(0.25)
        elif method == "Momentum_Merge_Base":
            all_alphas[3].fill(0.25)
        elif method == "Momentum_Merge_Advanced":
            # Will be set dynamically at layer 4
            pass
        elif method == "UGR":
            # Will be set dynamically based on temporal coupling
            pass
            
        # Running the simulation sample-by-sample
        prev_ugr_s = np.ones(K) / np.sqrt(K) # final state of previous sample for UGR
        
        for i in range(num_samples):
            target_task = stream_tasks[i]
            
            # SABLE, ChemMerge, Momentum-Merge initialization for this sample
            sample_C = np.ones(K) * 0.25
            sample_alpha = {}
            sample_alpha[3] = np.ones(K) * 0.25
            
            if method == "Oracle":
                sample_alpha[3] = np.zeros(K)
                sample_alpha[3][target_task] = 1.0
            elif method == "Uniform":
                sample_alpha[3] = np.ones(K) * 0.25
            elif method == "Momentum_Merge_Advanced":
                # Will be initialized to w_4
                pass
            elif method == "UGR":
                if i == 0:
                    sample_s = np.ones(K) / np.sqrt(K)
                else:
                    sample_s = np.copy(prev_ugr_s)
                sample_alpha[3] = sample_s ** 2
            
            h_sample = np.copy(h[i]) # 1 x D
            
            for l in range(4, L + 1):
                # 1. Compute bottom-up similarity
                if "LC" in method or "Advanced" in method or method == "UGR":
                    centroids = layer_centroids[l-1]
                else:
                    centroids = layer_centroids[3]
                
                # Cosine similarity
                norm_h = np.linalg.norm(h_sample) + 1e-8
                norm_c = np.linalg.norm(centroids, axis=1) + 1e-8
                sim = np.dot(centroids, h_sample) / (norm_h * norm_c)
                
                # Softmax temp
                if method == "SABLE_0.005":
                    temp = 0.005
                elif method == "SABLE_LC":
                    temp = 0.200
                elif "ChemMerge" in method:
                    temp = 0.050
                elif method == "Momentum_Merge_Base":
                    temp = 0.100
                elif method == "Momentum_Merge_Advanced":
                    temp = 0.005
                elif method == "UGR":
                    temp = 0.010 # Optimal swept gating temperature
                else:
                    temp = 1.0
                
                # Softmax normalized similarity
                shifted_sim = (sim - np.max(sim)) / temp
                w = np.exp(shifted_sim)
                w = w / np.sum(w)
                
                # 2. Update ensembling weights
                if method == "Oracle":
                    sample_alpha[l] = np.zeros(K)
                    sample_alpha[l][target_task] = 1.0
                elif method == "Uniform":
                    sample_alpha[l] = np.ones(K) * 0.25
                elif method == "SABLE_0.005" or method == "SABLE_LC":
                    sample_alpha[l] = w
                elif "ChemMerge" in method:
                    dt = 1.5
                    k_decay = 0.3
                    sample_C = np.clip(sample_C + dt * (w * (1.0 - sample_C) - k_decay * sample_C), 0.0, 1.0)
                    sample_alpha[l] = sample_C / (np.sum(sample_C) + 1e-8)
                elif method == "Momentum_Merge_Base":
                    beta = 0.60
                    sample_alpha[l] = (1.0 - beta) * w + beta * sample_alpha[l-1]
                elif method == "Momentum_Merge_Advanced":
                    beta = 0.60
                    sample_alpha[l] = (1.0 - beta) * w + beta * sample_alpha[l-1]
                elif method == "UGR":
                    # Unitary Geodesic Routing
                    eta_ugr = 0.80 # Optimal swept inertia coeff
                    # Map bottom-up similarity to target vector on sphere
                    w_sphere = w / (np.linalg.norm(w) + 1e-6)
                    # Compute alignment
                    c_val = np.dot(sample_s, w_sphere)
                    if np.abs(c_val) >= 1.0 - 1e-6:
                        # Collinear, state remains same
                        pass
                    else:
                        # Orthonormalize target component
                        v_orth = w_sphere - c_val * sample_s
                        u_vec = v_orth / (np.linalg.norm(v_orth) + 1e-8)
                        # Angle interpolation
                        phi_angle = np.arccos(np.clip(c_val, -1.0, 1.0))
                        theta_angle = eta_ugr * phi_angle
                        # Rotate state
                        sample_s = np.cos(theta_angle) * sample_s + np.sin(theta_angle) * u_vec
                    
                    sample_alpha[l] = sample_s ** 2
                
                # Save computed alphas for jitter calculation
                all_alphas[l][i] = sample_alpha[l]
                if l == 4 and "Advanced" in method:
                    all_alphas[3][i] = sample_alpha[3]
                elif l == 4 and method == "UGR":
                    all_alphas[3][i] = sample_alpha[3]
                
                # 3. Activation Blending forward pass
                # Blend task signatures v_k
                blend_v = np.zeros(D)
                for k in range(K):
                    blend_v += sample_alpha[l][k] * v[k]
                
                h_sample = h_sample + gamma_V * (blend_v - h_sample)
                # Add layer noise
                h_sample += np.random.normal(0, sigma_layer, D)
            
            # Save final state for temporal coupling if UGR
            if method == "UGR":
                prev_ugr_s = np.copy(sample_s)
                
            # Final prediction and soft alignment accuracy
            logits = np.zeros(K)
            biases = [0.0, 0.0, -0.90, -2.30]
            for k_idx in range(K):
                logits[k_idx] = -np.linalg.norm(h_sample - v[k_idx])**2 + biases[k_idx]
            pred = np.argmax(logits)
            acc_val = 1.0 if pred == target_task else 0.0
            results[method]["acc"].append(acc_val)
            
        # Calculate routing jitter for this method
        # Eq. 13: Jitter = N_adapt^-1 \sum_{l=5}^{14} \sum_{k=1}^K (alpha_k^(l) - alpha_k^(l-1))^2
        # Shape of all_alphas[l]: num_samples x K
        jitters = []
        for i in range(num_samples):
            sample_jitter = 0.0
            for l in range(5, L + 1):
                sample_jitter += np.sum((all_alphas[l][i] - all_alphas[l-1][i])**2)
            jitters.append(sample_jitter / (L - L_frozen - 1))
        
        results[method]["jitter"] = jitters
        
    return results

if __name__ == "__main__":
    import pandas as pd
    
    seeds = list(range(42, 52)) # 10 seeds
    all_seed_results = []
    
    for seed in seeds:
        print(f"Running seed {seed}...")
        res = run_simulation(seed)
        seed_data = {}
        for m in res:
            seed_data[f"{m}_acc"] = np.mean(res[m]["acc"])
            seed_data[f"{m}_jitter"] = np.mean(res[m]["jitter"])
        all_seed_results.append(seed_data)
        
    df = pd.DataFrame(all_seed_results)
    
    # Calculate mean and std across seeds
    summary = {}
    for m in run_simulation(42).keys():
        summary[m] = {
            "Acc Mean": df[f"{m}_acc"].mean() * 100,
            "Acc Std": df[f"{m}_acc"].std() * 100,
            "Jitter Mean": df[f"{m}_jitter"].mean(),
            "Jitter Std": df[f"{m}_jitter"].std()
        }
        
    summary_df = pd.DataFrame(summary).T
    print("\nSimulation Summary across 10 Seeds:")
    print(summary_df.to_string())
