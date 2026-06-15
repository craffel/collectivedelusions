import numpy as np

def run_simulation(seed=42, rho=0.0):
    np.random.seed(seed)
    
    D = 192
    K = 4
    L = 14
    d = 48
    
    # 1. Intrinsic orthogonal task signatures
    v_orth = np.zeros((K, D))
    for k in range(K):
        v_orth[k, k*d:(k+1)*d] = 1.0 / np.sqrt(d)
        
    # 2. Covariance injection / entanglement
    if rho > 0.0:
        # Toeplitz covariance matrix Sigma
        Sigma = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                Sigma[i, j] = rho ** abs(i - j)
        # Symmetric square root of Sigma via SVD
        U, S, Vt = np.linalg.svd(Sigma)
        Sigma_half = U @ np.diag(np.sqrt(S)) @ Vt
        
        v = np.zeros((K, D))
        for k in range(K):
            v[k] = Sigma_half @ v_orth[k]
            # Normalize to preserve unit norm
            v[k] /= np.linalg.norm(v[k])
    else:
        v = v_orth.copy()
        
    sigmas = [0.05, 0.15, 0.40, 1.20]
    biases = [0.0, 0.0, 0.0, -2.1]
    
    # Test split: 250 samples per task (1000 total)
    N_test = 250
    test_samples = []
    test_labels = []
    for k in range(K):
        for _ in range(N_test):
            eps = np.random.normal(0, sigmas[k], D)
            test_samples.append(v[k] + eps)
            test_labels.append(k)
            
    test_samples = np.array(test_samples)
    test_labels = np.array(test_labels)
    
    # Linear ramp for progressive specialization gamma^(l)
    gammas = [0.1 + 0.9 * (l - 4) / 10 for l in range(1, L + 1)] # indexed 1 to L
    
    # Cosine similarity helper
    def cosine_similarity(x, y):
        # x: [B, D], y: [K, D]
        # returns [B, K]
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True)
        # Avoid division by zero
        x_norm[x_norm == 0] = 1e-8
        y_norm[y_norm == 0] = 1e-8
        return (x @ y.T) / (x_norm @ y_norm.T)

    # ------------------ 1. Expert Ceiling (Oracle) ------------------
    correct_oracle = 0
    for b in range(len(test_samples)):
        y = test_labels[b]
        h = test_samples[b].copy()
        # Propagate through layers 4 to 14
        for l in range(4, L + 1):
            gamma = gammas[l-1]
            # Perfect expert routing: alpha is one-hot
            h = h + gamma * (v[y] - h)
        # Final logits
        logits = -np.sum((h - v)**2, axis=1) + biases
        pred = np.argmax(logits)
        if pred == y:
            correct_oracle += 1
            
    acc_oracle = correct_oracle / len(test_samples)
    
    # ------------------ 2. Uniform Merging ------------------
    correct_uniform = 0
    for b in range(len(test_samples)):
        y = test_labels[b]
        h = test_samples[b].copy()
        for l in range(4, L + 1):
            gamma = gammas[l-1]
            # Uniform merging: alpha is [0.25, 0.25, 0.25, 0.25]
            update = np.zeros(D)
            for k in range(K):
                update += 0.25 * gamma * (v[k] - h)
            h = h + update
        logits = -np.sum((h - v)**2, axis=1) + biases
        pred = np.argmax(logits)
        if pred == y:
            correct_uniform += 1
            
    acc_uniform = correct_uniform / len(test_samples)
    
    # ------------------ 3. SABLE ------------------
    correct_sable = 0
    tau_sable = 0.05
    for b in range(len(test_samples)):
        y = test_labels[b]
        h = test_samples[b].copy()
        
        # SABLE routing weights computed at layer 3
        # h at layer 3 is identical to h^(0)
        sims = cosine_similarity(h[None, :], v)[0]
        # Softmax with tau
        alpha = np.exp(sims / tau_sable) / np.sum(np.exp(sims / tau_sable))
        
        for l in range(4, L + 1):
            gamma = gammas[l-1]
            update = np.zeros(D)
            for k in range(K):
                update += alpha[k] * gamma * (v[k] - h)
            h = h + update
            
        logits = -np.sum((h - v)**2, axis=1) + biases
        pred = np.argmax(logits)
        if pred == y:
            correct_sable += 1
            
    acc_sable = correct_sable / len(test_samples)
    
    # ------------------ 4. ChemMerge ------------------
    correct_chem = 0
    tau_chem = 0.01
    dt = 1.5
    k_decay = 0.3
    
    for b in range(len(test_samples)):
        y = test_labels[b]
        h = test_samples[b].copy()
        
        # Concentrations initialized uniformly at Layer 3
        C = np.ones(K) * 0.25
        
        for l in range(4, L + 1):
            # Compute similarity to early centroids (v)
            sims = cosine_similarity(h[None, :], v)[0]
            # Arrhenius rate equation
            rates = np.exp(sims / tau_chem) / np.sum(np.exp(sims / tau_chem))
            # Update concentration using Explicit Euler and clipping
            C_next = C + dt * (rates * (1.0 - C) - k_decay * C)
            C = np.clip(C_next, 0.0, 1.0)
            
            # Law of Mass Action for blending coefficients
            alpha = C / np.sum(C)
            
            gamma = gammas[l-1]
            update = np.zeros(D)
            for k in range(K):
                update += alpha[k] * gamma * (v[k] - h)
            h = h + update
            
        logits = -np.sum((h - v)**2, axis=1) + biases
        pred = np.argmax(logits)
        if pred == y:
            correct_chem += 1
            
    acc_chem = correct_chem / len(test_samples)
    
    print(f"Expert Ceiling: {acc_oracle:.4f}")
    print(f"Uniform Merging: {acc_uniform:.4f}")
    print(f"SABLE: {acc_sable:.4f}")
    print(f"ChemMerge: {acc_chem:.4f}")

run_simulation()
