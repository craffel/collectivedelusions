import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
sigma2_unified = 1.0  # Normalized task-specific variance
sigma2_c = 2.0        # Variance of task centroids (cross-task dispersion)
K_range = np.arange(1, 31)
num_simulations = 1000

# Arrays to store results
sim_alpha_means = []
sim_alpha_stds = []
sim_sigma_between_means = []

for K in K_range:
    alphas = []
    sigmas_between = []
    for _ in range(num_simulations):
        if K == 1:
            sigma_between = 0.0
        else:
            # Draw K task centroids from N(0, sigma2_c)
            centroids = np.random.normal(0, np.sqrt(sigma2_c), K)
            mixture_mean = np.mean(centroids)
            sigma_between = np.var(centroids)  # Biased sample variance
            
        alpha = np.sqrt(sigma2_unified / (sigma2_unified + sigma_between))
        alphas.append(alpha)
        sigmas_between.append(sigma_between)
        
    sim_alpha_means.append(np.mean(alphas))
    sim_alpha_stds.append(np.std(alphas))
    sim_sigma_between_means.append(np.mean(sigmas_between))

sim_alpha_means = np.array(sim_alpha_means)
sim_alpha_stds = np.array(sim_alpha_stds)
sim_sigma_between_means = np.array(sim_sigma_between_means)

# Theoretical curves
theo_sigma_between = (K_range - 1.0) / K_range * sigma2_c
theo_alpha = np.sqrt(sigma2_unified / (sigma2_unified + theo_sigma_between))
alpha_limit = np.sqrt(sigma2_unified / (sigma2_unified + sigma2_c))

# Create publication-grade plot
plt.figure(figsize=(7, 4.5))
plt.grid(True, linestyle='--', alpha=0.6)

# Plot simulated mean with standard deviation shading
plt.fill_between(K_range, sim_alpha_means - sim_alpha_stds, sim_alpha_means + sim_alpha_stds, 
                 color='#377eb8', alpha=0.15, label='Simulation $\pm$ 1 SD')
plt.plot(K_range, sim_alpha_means, 'o', color='#377eb8', markersize=4, label='Simulation Mean')

# Plot theoretical expected curve
plt.plot(K_range, theo_alpha, '-', color='#e41a1c', linewidth=2, label='Theoretical Expectation')

# Plot asymptotic limit line
plt.axhline(y=alpha_limit, color='#4daf4a', linestyle='--', linewidth=1.5, 
            label=f'Asymptotic Limit $\\alpha_\\infty = {alpha_limit:.3f}$')

plt.title('Scalability of Attenuation Factor $\\alpha$ with Number of Tasks $K$', fontsize=11, fontweight='bold', pad=10)
plt.xlabel('Number of Merged Tasks ($K$)', fontsize=10)
plt.ylabel('Activation Attenuation Factor ($\\alpha$)', fontsize=10)
plt.xlim(0.8, 30.2)
plt.ylim(0.45, 1.05)
plt.xticks(np.append([1], np.arange(5, 31, 5)))
plt.legend(loc='upper right', fontsize=9, framealpha=0.9)
plt.tight_layout()

# Save figure
plt.savefig('sweep_scalability.png', dpi=300)
print("Saved sweep_scalability.png successfully!")
