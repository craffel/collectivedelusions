import os
import time
import numpy as np
import matplotlib.pyplot as plt

def generate_generalized_subspace_data(K, num_samples_per_task, seed=42):
    """
    Generates 192-dimensional representation vectors for K tasks,
    residing in orthogonal coordinate blocks of dimension 192 // K.
    """
    np.random.seed(seed)
    D = 192
    block_size = D // K
    
    v = []
    for k in range(K):
        vk = np.zeros(D)
        vk[k*block_size : (k+1)*block_size] = 1.0 / np.sqrt(block_size)
        v.append(vk)
    v = np.array(v)
    
    # Base noise scales cycled through the 4 core task difficulties
    sigmas_base = [0.05, 0.15, 0.40, 1.20]
    sigmas = [sigmas_base[k % 4] for k in range(K)]
    
    X = []
    y_true_task = []
    y_true_class = []
    
    for k in range(K):
        for _ in range(num_samples_per_task):
            noise = np.random.normal(0, sigmas[k], D)
            xk = v[k] + noise
            xk = xk / np.linalg.norm(xk)
            
            X.append(xk)
            y_true_task.append(k)
            y_true_class.append(np.random.randint(0, 10))
            
    return np.array(X), np.array(y_true_task), np.array(y_true_class), v, sigmas

def get_generalized_pfsr_weights(X_test, v, temp=0.05):
    """SABLE baseline."""
    probs = []
    for x in X_test:
        sims = np.dot(v, x)
        sims_stable = sims - np.max(sims)
        exp_sims = np.exp(sims_stable / temp)
        probs.append(exp_sims / np.sum(exp_sims))
    return np.array(probs)

def get_generalized_sps_zca_weights(X_test, v, expected_sims, temp=0.001):
    """SPS-ZCA baseline with IDC calibration."""
    probs = []
    for x in X_test:
        sims = np.dot(v, x)
        cal_sims = sims / expected_sims
        cal_sims_stable = cal_sims - np.max(cal_sims)
        exp_sims = np.exp(cal_sims_stable / temp)
        probs.append(exp_sims / np.sum(exp_sims))
    return np.array(probs)

def run_generalized_chemmerge_kinetics(X_test, v, temp=0.01, delta_t=1.5, k_decay=0.3, L=14):
    """ChemMerge ensembling kinetics."""
    num_samples = len(X_test)
    K = len(v)
    
    # Init uniformly
    C_layer = np.ones((num_samples, K)) / K
    
    for l in range(4, L + 1):
        # Cosine similarity
        sims = np.dot(X_test, v.T)
        sims_stable = sims - np.max(sims, axis=-1, keepdims=True)
        
        # Arrhenius rate
        exp_u = np.exp(sims_stable / temp)
        k_rate = exp_u / np.sum(exp_u, axis=-1, keepdims=True)
        
        # Euler update with clipping
        C_layer = C_layer + delta_t * (k_rate * (1.0 - C_layer) - k_decay * C_layer)
        C_layer = np.clip(C_layer, 0.0, 1.0)
        
    alpha = C_layer / np.sum(C_layer, axis=-1, keepdims=True)
    return alpha

def evaluate_generalized_accuracies(X, y_true_task, y_true_class, v, alpha_weights, sigmas, seed=42):
    """
    Evaluates joint and task-specific classification accuracies
    using the generalized calibrated expert logit model.
    """
    num_samples = len(X)
    K = len(v)
    
    # Define logit scaling parameters based on base difficulties
    expert_params_base = {
        0: (15.0, 0.1),
        1: (15.0, 0.1),
        2: (23.5, 0.5),
        3: (11.6, 0.5)
    }
    expert_params = {k: expert_params_base[k % 4] for k in range(K)}
    
    task_correct = [0]*K
    task_total = [0]*K
    
    np.random.seed(seed)
    
    for i in range(num_samples):
        t = y_true_task[i]
        c = y_true_class[i]
        x = X[i]
        
        expert_logits = []
        for k in range(K):
            logits = np.random.normal(0, 1.0, 10)
            if k == t:
                proj = np.dot(x, v[t])
                scale, noise_std = expert_params[t]
                logits[c] = scale * proj + np.random.normal(0, noise_std)
            else:
                logits = np.random.normal(0, 1.5, 10)
            expert_logits.append(logits)
            
        alpha = alpha_weights[i]
        blended = np.zeros(10)
        for k in range(K):
            blended += alpha[k] * expert_logits[k]
            
        pred_class = np.argmax(blended)
        if pred_class == c:
            task_correct[t] += 1
        task_total[t] += 1
        
    accs = [task_correct[k] / task_total[k] if task_total[k] > 0 else 0 for k in range(K)]
    return np.mean(accs)

def run_expert_scaling_suite():
    print("====================================================")
    print("RUNNING SYSTEMATIC EXPERT SCALING STUDY (K = 4, 8, 12, 16)")
    print("====================================================")
    
    K_values = [4, 8, 12, 16]
    seeds = list(range(42, 47)) # 5 seeds for statistical stability and speed
    num_samples_per_task = 150
    
    methods = ["Uniform Merging", "SABLE", "SPS-ZCA", "ChemMerge (Ours)"]
    
    accuracies = {m: {K: [] for K in K_values} for m in methods}
    routing_times = {m: {K: [] for K in K_values} for m in methods if m != "Uniform Merging"}
    
    for K in K_values:
        print(f"\n--- Evaluating K = {K} Experts ---")
        
        for seed in seeds:
            X_test, y_test_task, y_test_class, v, sigmas = generate_generalized_subspace_data(
                K, num_samples_per_task, seed=seed
            )
            num_samples = len(X_test)
            
            # Calibration for SPS-ZCA
            expected_sims = []
            for k in range(K):
                cal_samples = []
                for _ in range(64):
                    noise = np.random.normal(0, sigmas[k], 192)
                    xk = v[k] + noise
                    xk = xk / np.linalg.norm(xk)
                    cal_samples.append(xk)
                cal_sims = [np.dot(xk, v[k]) for xk in cal_samples]
                expected_sims.append(np.mean(cal_sims))
            expected_sims = np.array(expected_sims)
            
            # 1. Uniform Merging
            weights_uniform = np.ones((num_samples, K)) / K
            acc_uniform = evaluate_generalized_accuracies(X_test, y_test_task, y_test_class, v, weights_uniform, sigmas, seed=seed)
            accuracies["Uniform Merging"][K].append(acc_uniform)
            
            # 2. SABLE
            t0 = time.perf_counter()
            weights_sable = get_generalized_pfsr_weights(X_test, v, temp=0.05)
            t_sable = (time.perf_counter() - t0) * 1000.0 # to ms
            acc_sable = evaluate_generalized_accuracies(X_test, y_test_task, y_test_class, v, weights_sable, sigmas, seed=seed)
            accuracies["SABLE"][K].append(acc_sable)
            routing_times["SABLE"][K].append(t_sable)
            
            # 3. SPS-ZCA
            t0 = time.perf_counter()
            weights_zca = get_generalized_sps_zca_weights(X_test, v, expected_sims, temp=0.001)
            t_zca = (time.perf_counter() - t0) * 1000.0
            acc_zca = evaluate_generalized_accuracies(X_test, y_test_task, y_test_class, v, weights_zca, sigmas, seed=seed)
            accuracies["SPS-ZCA"][K].append(acc_zca)
            routing_times["SPS-ZCA"][K].append(t_zca)
            
            # 4. ChemMerge (Ours)
            t0 = time.perf_counter()
            weights_chem = run_generalized_chemmerge_kinetics(X_test, v, temp=0.01)
            t_chem = (time.perf_counter() - t0) * 1000.0
            acc_chem = evaluate_generalized_accuracies(X_test, y_test_task, y_test_class, v, weights_chem, sigmas, seed=seed)
            accuracies["ChemMerge (Ours)"][K].append(acc_chem)
            routing_times["ChemMerge (Ours)"][K].append(t_chem)
            
        # Summary for this K
        print(f"K = {K} Accuracies:")
        for m in methods:
            mean_acc = np.mean(accuracies[m][K]) * 100.0
            std_acc = np.std(accuracies[m][K]) * 100.0
            latency_str = f"Routing Latency: {np.mean(routing_times[m][K]):.2f} ms" if m != "Uniform Merging" else "N/A"
            print(f"  {m:<20}: {mean_acc:.2f}% +- {std_acc:.2f}% | {latency_str}")
            
    # Print Markdown Table
    print("\n\n=== EXPERT SCALING PERFORMANCE AND ROUTING LATENCY TABLE ===")
    print("| Expert Count K | Uniform Merging | SABLE (SOTA) | SPS-ZCA (SOTA) | ChemMerge (Ours) |")
    print("| :---: | :---: | :---: | :---: | :---: |")
    for K in K_values:
        row = f"| K = {K} | "
        for m in methods:
            mean_acc = np.mean(accuracies[m][K]) * 100.0
            std_acc = np.std(accuracies[m][K]) * 100.0
            row += f"{mean_acc:.2f}% &plusmn; {std_acc:.2f}%"
            if m != "Uniform Merging":
                row += f" ({np.mean(routing_times[m][K]):.1f}ms)"
            row += " | "
        print(row)
        
    # Generate Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Accuracy Plot
    colors = {"Uniform Merging": "#7f7f7f", "SABLE": "#1f77b4", "SPS-ZCA": "#ff7f0e", "ChemMerge (Ours)": "#d62728"}
    markers = {"Uniform Merging": "o", "SABLE": "s", "SPS-ZCA": "^", "ChemMerge (Ours)": "D"}
    
    for m in methods:
        means = [np.mean(accuracies[m][K]) * 100.0 for K in K_values]
        stds = [np.std(accuracies[m][K]) * 100.0 for K in K_values]
        ax1.errorbar(
            K_values, means, yerr=stds, label=m, color=colors[m],
            marker=markers[m], linewidth=2, elinewidth=1.5, capsize=4
        )
        
    ax1.set_title("Joint Mean Accuracy vs. Expert Count $K$", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Expert Adapters ($K$)", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_xticks(K_values)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)
    
    # 2. Latency Plot
    for m in ["SABLE", "SPS-ZCA", "ChemMerge (Ours)"]:
        means = [np.mean(routing_times[m][K]) for K in K_values]
        stds = [np.std(routing_times[m][K]) for K in K_values]
        ax2.errorbar(
            K_values, means, yerr=stds, label=m, color=colors[m],
            marker=markers[m], linewidth=2, elinewidth=1.5, capsize=4
        )
        
    ax2.set_title("Ensembling Weight Routing Latency vs. $K$", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Expert Adapters ($K$)", fontsize=11)
    ax2.set_ylabel("Total Batch Routing Latency (ms)", fontsize=11)
    ax2.set_xticks(K_values)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("submission/results", exist_ok=True)
    plt.savefig("results/expert_scaling.png", dpi=150)
    plt.savefig("submission/results/expert_scaling.png", dpi=150)
    print("\nExpert scaling plots saved successfully!")

if __name__ == "__main__":
    run_expert_scaling_suite()
