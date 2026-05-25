# Tuning script to search for better hyperparameters for Method G
import run_experiments as re
import torch

# We can import the pre-trained models, datasets, and variables from run_experiments directly!
# Since run_experiments runs on import, we can capture the loaded datasets/models and run custom searches.

print("\n--- Tuning Hyperparameters for SAK-AHR (Method G) ---")

grid = {
    'lr': [0.01, 0.02, 0.03, 0.05],
    'eps_stab': [0.01, 0.02, 0.05, 0.10],
    'rho_sam': [0.005, 0.01, 0.02, 0.05],
    'alpha_ema': [0.5, 0.8, 0.9, 0.95]
}

best_overall = 0.0
best_config = None

# We can run a random search or a subset of interest
import random
random.seed(2026)

# Test 20 configurations
for i in range(25):
    lr = random.choice(grid['lr'])
    eps_stab = random.choice(grid['eps_stab'])
    rho_sam = random.choice(grid['rho_sam'])
    alpha_ema = random.choice(grid['alpha_ema'])
    
    config = {
        "use_ahr": True,
        "use_cosface_experts": False,
        "use_ttbn": True,
        "on_the_fly_preconditioning": True,
        "use_ema_kronecker": True,
        "alpha_ema": alpha_ema,
        "steps": 5,
        "lr": lr,
        "beta": 1.5,
        "gamma_c": 0.02,
        "eps_stab": eps_stab,
        "use_sam": True,
        "rho_sam": rho_sam
    }
    
    # Run evaluation
    res, _ = re.run_evaluation(f"Tune_{i}", config)
    overall = res[-1]
    
    print(f"Config {i}: lr={lr}, eps_stab={eps_stab}, rho_sam={rho_sam}, alpha_ema={alpha_ema} => Overall={overall:.4f}%")
    
    if overall > best_overall:
        best_overall = overall
        best_config = (lr, eps_stab, rho_sam, alpha_ema)

print("\n--- Best Config Found ---")
print(f"Overall Accuracy: {best_overall:.4f}%")
print(f"Best parameters: lr={best_config[0]}, eps_stab={best_config[1]}, rho_sam={best_config[2]}, alpha_ema={best_config[3]}")
