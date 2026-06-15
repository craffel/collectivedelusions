import numpy as np
import matplotlib.pyplot as plt
from run_experiments import (
    generate_subspace_data,
    get_pfsr_weights,
    evaluate_accuracies,
    run_chemmerge_kinetics
)

def run_static_ema_baseline(X_test, v, beta=0.3, temp=0.05, L=14):
    """
    Static EMA routing baseline.
    Applies a standard, static-coefficient EMA directly to stateless PFSR weights:
    alpha^(l) = beta * w_stateless + (1 - beta) * alpha^(l-1)
    """
    num_samples = len(X_test)
    K = len(v)
    
    # 1. Compute stateless routing weights (SABLE-like PFSR weights)
    w_stateless = get_pfsr_weights(X_test, v, temp=temp)
    
    # 2. Apply EMA layer-by-layer
    alpha_layer = np.ones((num_samples, K)) / K
    trajectory = [alpha_layer.copy()]
    
    for l in range(4, L + 1):
        alpha_next = beta * w_stateless + (1.0 - beta) * alpha_layer
        alpha_layer = alpha_next
        trajectory.append(alpha_layer.copy())
        
    return alpha_layer, trajectory

def compute_routing_jitter(trajectory):
    """
    Computes mean routing weight jitter (layer-to-layer ensembling weight variance)
    """
    traj = np.array(trajectory)  # Shape: (num_layers, num_samples, K)
    L = traj.shape[0]
    num_samples = traj.shape[1]
    
    jitters = []
    for i in range(num_samples):
        jitter_sample = 0
        for l in range(1, L):
            jitter_sample += np.sum((traj[l, i] - traj[l-1, i])**2)
        jitters.append(jitter_sample / (L - 1))
    return np.mean(jitters)

def run_ema_evaluation():
    print("Evaluating Static EMA Routing Baseline vs. ChemMerge...")
    seeds = [42, 43, 44, 45, 46]
    betas = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Store results
    ema_accs = {b: [] for b in betas}
    ema_jitters = {b: [] for b in betas}
    
    chem_accs = []
    chem_jitters = []
    
    sable_accs = []
    sable_jitters = []
    
    for seed in seeds:
        X_train, y_train_task, y_train_class, v = generate_subspace_data(1000, seed=seed)
        X_test, y_test_task, y_test_class, _ = generate_subspace_data(250, seed=seed)
        
        # 1. Stateless SABLE
        weights_sable = get_pfsr_weights(X_test, v, temp=0.05)
        _, acc_sable = evaluate_accuracies(X_test, y_test_task, y_test_class, v, weights_sable, seed=seed)
        sable_accs.append(acc_sable * 100)
        # SABLE is stateless, so its weights don't change layer-to-layer if evaluated statically,
        # but to measure its intrinsic jitter under equivalent sensitivity, we evaluate SABLE's trajectory.
        # Here we just treat SABLE as having a constant trajectory of its weights to measure its static variance.
        sable_trajectory = [weights_sable for _ in range(12)] # L=14, so 12 blocks (Layer 3 to 14)
        sable_jitters.append(compute_routing_jitter(sable_trajectory))
        
        # 2. ChemMerge
        weights_chem, chem_traj = run_chemmerge_kinetics(X_test, v, temp=0.01)
        _, acc_chem = evaluate_accuracies(X_test, y_test_task, y_test_class, v, weights_chem, seed=seed)
        chem_accs.append(acc_chem * 100)
        chem_jitters.append(compute_routing_jitter(chem_traj))
        
        # 3. Static EMA Baseline for various betas
        for b in betas:
            weights_ema, ema_traj = run_static_ema_baseline(X_test, v, beta=b, temp=0.05)
            _, acc_ema = evaluate_accuracies(X_test, y_test_task, y_test_class, v, weights_ema, seed=seed)
            ema_accs[b].append(acc_ema * 100)
            ema_jitters[b].append(compute_routing_jitter(ema_traj))
            
    print("\n--- RESULTS OVER 5 SEEDS ---")
    print(f"{'Method':<25} | {'Joint Accuracy (%)':<20} | {'Routing Jitter':<15}")
    print("-" * 66)
    print(f"{'SABLE (Stateless)':<25} | {np.mean(sable_accs):5.2f}% +/- {np.std(sable_accs):4.2f}% | {np.mean(sable_jitters):6.4f}")
    print(f"{'ChemMerge (Ours)':<25} | {np.mean(chem_accs):5.2f}% +/- {np.std(chem_accs):4.2f}% | {np.mean(chem_jitters):6.4f}")
    for b in betas:
        print(f"Static EMA (beta={b:4.2f})    | {np.mean(ema_accs[b]):5.2f}% +/- {np.std(ema_accs[b]):4.2f}% | {np.mean(ema_jitters[b]):6.4f}")

if __name__ == "__main__":
    run_ema_evaluation()
