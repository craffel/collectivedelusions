import numpy as np

def simulate_drift(steps=10000, alpha=0.15, pf=0.0004, beta=0.0002, seed=42):
    np.random.seed(seed)
    
    # Let prototype A start at 0.0, and target contaminated centroid B be at 1.0
    mu_A_clean = 0.0
    mu_B_target = 1.0
    
    # 1. No Anchoring
    mu_A_no_anchor = mu_A_clean
    drift_no_anchor = []
    
    # 2. With Anchoring
    mu_A_anchored = mu_A_clean
    drift_anchored = []
    
    for t in range(steps):
        # Determine if gating failure occurs
        failure = np.random.rand() < pf
        
        # No anchor update
        if failure:
            # Batch contaminated feature
            feat = mu_B_target + np.random.normal(0, 0.1)
            mu_A_no_anchor = (1 - alpha) * mu_A_no_anchor + alpha * feat
        drift_no_anchor.append(abs(mu_A_no_anchor - mu_A_clean))
        
        # Anchored update
        if failure:
            feat = mu_B_target + np.random.normal(0, 0.1)
            mu_A_anchored = (1 - alpha) * mu_A_anchored + alpha * feat
        # Anchor pull
        mu_A_anchored = mu_A_anchored - beta * (mu_A_anchored - mu_A_clean)
        drift_anchored.append(abs(mu_A_anchored - mu_A_clean))
        
    return drift_no_anchor, drift_anchored

if __name__ == "__main__":
    no_anchor, anchored = simulate_drift()
    
    print("Long-term drift simulation results (normalized distance to clean prototype):")
    print(f"Step 1000  | Without Anchoring: {no_anchor[999]:.4f} | With Anchoring: {anchored[999]:.4f}")
    print(f"Step 5000  | Without Anchoring: {no_anchor[4999]:.4f} | With Anchoring: {anchored[4999]:.4f}")
    print(f"Step 10000 | Without Anchoring: {no_anchor[9999]:.4f} | With Anchoring: {anchored[9999]:.4f}")
