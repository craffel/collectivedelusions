import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Running Temporal Transition Lag Simulation for Sequential B=1 Serving...")
    np.random.seed(42)
    os.makedirs("results", exist_ok=True)

    # 1. Configuration matching simulate.py
    K = 4
    D = 192
    
    # Orthogonal centroids
    centroids = np.zeros((K, D))
    for k in range(K):
        centroids[k, k*48 : (k+1)*48] = 1.0

    expert_ceilings = {0: 1.00, 1: 1.00, 2: 0.88, 3: 0.312}
    uniform_accuracies = {0: 0.695, 1: 0.45, 2: 0.405, 3: 0.168}
    dispersions = [0.05, 0.35, 0.15, 0.20]
    expected_scales = [0.98, 0.72, 0.88, 0.82]

    # Create a sequential stream with 4 blocks of 100 samples.
    # Each block corresponds to a single task, representing high temporal locality
    # with sharp task transitions at t = 100, 200, 300.
    block_length = 100
    N = K * block_length
    
    y_true_seq = []
    X_seq = []
    for k in range(K):
        noise = np.random.normal(0, dispersions[k], (block_length, D))
        samples = centroids[k] + noise
        X_seq.append(samples)
        y_true_seq.extend([k] * block_length)
        
    X_seq = np.vstack(X_seq)
    y_true_seq = np.array(y_true_seq)

    # Helper function to compute accuracy based on active expert
    def compute_sample_accuracy(y, active_expert, quant_penalty=0.995):
        A_yy = expert_ceilings[y] * quant_penalty
        if active_expert == y:
            return A_yy
        else:
            # Interference/degraded accuracy
            A_y_interf = (uniform_accuracies[y] - 0.0625 * A_yy) / 0.9375
            return A_y_interf

    # Evaluate different gamma values
    gammas = [0.0, 0.2, 0.5, 0.8, 0.9, 0.95]
    results = {}

    for gamma in gammas:
        smoothed_coords = None
        routing_history = []
        accuracies = []
        flicker_events = 0
        transition_lags = []
        
        # We track transition step triggers
        # Transitions occur at t = 100 (switch to 1), 200 (switch to 2), 300 (switch to 3)
        transition_points = [100, 200, 300]
        lag_trackers = {p: -1 for p in transition_points}

        for t in range(N):
            h = X_seq[t]
            y_true = y_true_seq[t]
            
            # Compute instantaneous coordinates u'
            cos_sims = []
            for k in range(K):
                sim = np.dot(h, centroids[k]) / (np.linalg.norm(h) * np.linalg.norm(centroids[k]))
                sim_calib = sim / expected_scales[k]
                cos_sims.append(sim_calib)
            cos_sims = np.array(cos_sims)
            
            # Apply EWMA coordinate smoothing
            if t == 0 or smoothed_coords is None:
                smoothed_coords = cos_sims.copy()
            else:
                smoothed_coords = (1.0 - gamma) * cos_sims + gamma * smoothed_coords
                
            # Determine active expert (argmax of coordinates)
            active_expert = np.argmax(smoothed_coords)
            routing_history.append(active_expert)
            
            # Measure flicker (adjacent routing change)
            if t > 0:
                if active_expert != routing_history[-2]:
                    # Only count as flicker if it's NOT a true task boundary transition
                    if t not in transition_points:
                        flicker_events += 1
                        
            # Measure accuracies
            sample_acc = compute_sample_accuracy(y_true, active_expert)
            accuracies.append(sample_acc)
            
            # Track transition lags
            for tp in transition_points:
                if t >= tp and lag_trackers[tp] == -1:
                    target_task = y_true_seq[tp]
                    if active_expert == target_task:
                        lag_trackers[tp] = t - tp

        # Calculate flicker rate over non-transition steps
        # Non-transition steps = N - 1 - len(transition_points) = 396
        flicker_rate = (flicker_events / (N - 1 - len(transition_points))) * 100
        mean_accuracy = np.mean(accuracies) * 100
        mean_lag = np.mean(list(lag_trackers.values()))
        
        results[gamma] = {
            "flicker_rate": flicker_rate,
            "mean_accuracy": mean_accuracy,
            "mean_lag": mean_lag,
            "lag_details": list(lag_trackers.values())
        }
        
        print(f"Gamma: {gamma:.2f} | Accuracy: {mean_accuracy:.2f}% | Flicker: {flicker_rate:.2f}% | Mean Lag: {mean_lag:.2f} steps")

    # 2. Plot results
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = '#2ca25f'
    ax1.set_xlabel('EWMA Smoothing Coefficient (γ)')
    ax1.set_ylabel('Sequential Routing Flicker Rate (%)', color=color)
    ax1.plot(gammas, [results[g]["flicker_rate"] for g in gammas], marker='o', color=color, linewidth=2.5, label='Flicker Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-1, 20)
    ax1.grid(linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  
    color = '#de2d26'
    ax2.set_ylabel('Temporal Transition Delay (B=1 steps)', color=color)
    ax2.plot(gammas, [results[g]["mean_lag"] for g in gammas], marker='s', color=color, linewidth=2.5, label='Transition Lag')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-1, 15)

    plt.title('Sequential Serving (B=1): Routing Flicker vs Transition Lag Trade-off')
    fig.tight_layout()
    plt.savefig("results/fig9_temporal_transition_lag.png", dpi=150)
    plt.close()
    print("Saved results/fig9_temporal_transition_lag.png")

if __name__ == '__main__':
    main()
