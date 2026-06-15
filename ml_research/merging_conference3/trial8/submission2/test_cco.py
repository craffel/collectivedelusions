import numpy as np

def main():
    np.random.seed(42)
    K = 4
    D = 192
    
    # Base orthogonal centroids
    centroids = np.zeros((K, D))
    for k in range(K):
        centroids[k, k*48 : (k+1)*48] = 1.0

    dispersions = [0.15, 0.45, 0.25, 0.30]
    expected_scales = [0.98, 0.72, 0.88, 0.82]

    epsilons = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    # Shared task-agnostic feature vector for entanglement
    shared_vector = np.random.normal(0, 0.5, D)
    shared_vector /= np.linalg.norm(shared_vector)

    print(f"{'Epsilon':<8} | {'Method':<20} | {'Routing Acc (%)':<15} | {'Flicker Rate (%)':<15}")
    print("-" * 65)

    for epsilon in epsilons:
        # Generate entangled samples for each task
        N = 1000
        samples_per_task = N // K
        X_features = []
        y_labels = []
        
        # Centroids used for projection
        entangled_centroids = []
        for k in range(K):
            c_orth = np.zeros(D)
            c_orth[k*48 : (k+1)*48] = 1.0
            c_entangled = (1.0 - epsilon) * c_orth + epsilon * shared_vector
            c_entangled /= np.linalg.norm(c_entangled)
            entangled_centroids.append(c_entangled)
            
            # Generate samples from the entangled centroids with noise
            noise = np.random.normal(0, dispersions[k], (samples_per_task, D))
            samples = c_entangled + noise
            samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
            X_features.append(samples)
            y_labels.extend([k] * samples_per_task)
            
        X_features = np.vstack(X_features)
        y_labels = np.array(y_labels)
        entangled_centroids = np.array(entangled_centroids)

        # 1. Gram-Schmidt CCO Centroids
        gs_centroids = np.zeros_like(entangled_centroids)
        for k in range(K):
            temp = entangled_centroids[k].copy()
            for j in range(k):
                proj = np.dot(entangled_centroids[k], gs_centroids[j])
                temp -= proj * gs_centroids[j]
            gs_centroids[k] = temp / np.linalg.norm(temp)

        # 2. Löwdin Symmetric Orthogonalization (Symmetric Manifold De-Entangling, SMD)
        # Overlap matrix S = C @ C.T (K x K)
        S_matrix = np.dot(entangled_centroids, entangled_centroids.T)
        eigvals, eigvecs = np.linalg.eigh(S_matrix)
        # S^{-1/2} = V @ diag(1 / sqrt(val)) @ V.T
        # Add small epsilon to avoid division by zero
        S_inv_sqrt = np.dot(eigvecs * (1.0 / np.sqrt(eigvals + 1e-12)), eigvecs.T)
        smd_centroids = np.dot(S_inv_sqrt, entangled_centroids)
        # Re-normalize to ensure unit norm
        smd_centroids = smd_centroids / np.linalg.norm(smd_centroids, axis=1, keepdims=True)

        methods = {
            "Nearest-Centroid": {"centroids": entangled_centroids, "use_idc": False},
            "ZCA-IDC": {"centroids": entangled_centroids, "use_idc": True},
            "ZCA-IDC-GS-CCO": {"centroids": gs_centroids, "use_idc": True},
            "ZCA-IDC-SMD (Ours)": {"centroids": smd_centroids, "use_idc": True}
        }

        for name, cfg in methods.items():
            preds = []
            
            for b, h in enumerate(X_features):
                cos_sims = []
                for k in range(K):
                    sim = np.dot(h, cfg["centroids"][k]) / (np.linalg.norm(h) * np.linalg.norm(cfg["centroids"][k]))
                    if cfg["use_idc"]:
                        sim = sim / expected_scales[k]
                    cos_sims.append(sim)
                
                cos_sims = np.array(cos_sims)
                exp_sims = np.exp((cos_sims - np.max(cos_sims)) / 0.01)
                alpha = exp_sims / np.sum(exp_sims)
                preds.append(np.argmax(alpha))

            preds = np.array(preds)
            routing_acc = np.mean(preds == y_labels) * 100

            flicker_events = 0
            total_pairs = 0
            for k in range(K):
                task_start = k * samples_per_task
                task_end = (k + 1) * samples_per_task
                task_preds = preds[task_start:task_end]
                for i in range(len(task_preds) - 1):
                    if task_preds[i] != task_preds[i+1]:
                        flicker_events += 1
                    total_pairs += 1
            flicker_rate = (flicker_events / total_pairs) * 100

            print(f"{epsilon:<8.1f} | {name:<20} | {routing_acc:<15.2f} | {flicker_rate:<15.2f}")

if __name__ == "__main__":
    main()
