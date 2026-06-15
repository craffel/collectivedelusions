# 4. Experiment Check

## Rigor and Convincingness of the Experimental Setup

As an empiricist reviewer, the experimental setup is overall highly commendable, featuring a balanced combination of:
1. **A highly-controlled, high-fidelity simulation sandbox:** A diagnostic setup that simulates non-convex loss-space geometry, layer sensitivities, and transductive noise, allowing the authors to isolate the effects of noise under exact ground-truth optimal parameters.
2. **A real-world physical weight-merging deployment:** Direct evaluation on actual physical weights of pre-trained OpenAI CLIP ViT-B/32 (86M parameters) and CLIP ViT-L/14 (307M parameters) backbones across 8 diverse visual task datasets (SUN397, Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, and DTD).

However, some aspects of the experimental setup and statistical evaluation warrant closer scrutiny:

1. **Inherent Design Bias in the Simulation Sandbox:**
   - *Critique:* In the Architectural Calibration of the simulation, the true optimal parameters $\lambda^*_k$ are generated using a decaying spatial covariance matrix $\Sigma_{\text{true}}$ with parameter $0.5^{|l-l'|}$. As the authors explicitly admit, this synthetic structure introduces an inherent design bias that naturally favors spatially-smooth regularizers like GP-BayesMerge. While the physical weight experiments are unbiased and successful, the simulated results are somewhat circular because the ground-truth optimal configuration is *defined* using the exact spatial covariance assumptions that GP-BayesMerge is built to exploit.
2. **Inadequate Number of Random Seeds given Low Compute Costs:**
   - *Critique:* The authors report mean and standard deviation values across **only 3 random seeds** (42, 100, 2026). While 3 seeds is a common default in computationally expensive deep learning training, test-time adaptation for model merging is extremely fast and lightweight. The authors state that GP-BayesMerge achieves near-peak performance in fewer than 50 steps, reducing test-time adaptation latency on standard devices to less than 0.15 seconds. Given that each run takes a fraction of a second, there is no computational barrier to running **at least 10 or 20 random seeds** to obtain statistically robust confidence intervals and run rigorous statistical significance tests (e.g., t-tests). Relying on only 3 seeds for such a computationally cheap adaptation represents a missed opportunity for empirical rigor.

## Baselines: Strength and Tuning
The baselines compared are highly relevant and represent the state-of-the-art in parameter-space model merging and test-time adaptation:
- **Task Arithmetic (Uniform):** The standard training-free baseline.
- **Task-Wise & Layer-Wise AdaMerging / AdaMerging++:** The standard unconstrained test-time optimization baselines.
- **RegCalMerge (ESR):** A state-of-the-art heuristic spatial smoothing baseline.
- **PolyMerge:** A state-of-the-art rigid subspace projection baseline.
- **Flat Spatial Averaging:** A baseline forcing uniform coefficients across layers.

However, the paper is completely silent on how the hyperparameters of the baseline methods (especially the smoothing parameters of RegCalMerge and the degree of PolyMerge) were tuned. For the comparison to be fair, the baselines should be optimized to their peak capacity on the same calibration batches.

## Comprehensiveness of Ablations and Sweeps
The ablation studies and hyperparameter sweeps are exceptionally thorough:
- **Noise Sensitivity Sweep (Figure 2a):** Applying relative perturbations to optimized coefficients shows that unconstrained AdaMerging lies in a sharp, fragile minimum, whereas GP-BayesMerge exhibits a robust, flat basin.
- **Regularization Strength Sweep (Figure 2b & Figure 7b):** Sweeping $\alpha$ across a logarithmic scale in both simulation and physical weights demonstrates a stable generalizing basin centered at $\alpha = 1.0$.
- **Spatial Lengthscale Sweep (Figure 2c & Figure 7a):** Sweeping $\ell$ confirms the smooth continuous transition from independent weight decay ($\ell \le 0.05$) to flat spatial averaging ($\ell \ge 0.8$), with the optimal balance at $\ell \approx 0.3$.
- **Sensitivity to Spatial Depth Correlation Base (Table 3):** Simulating uncorrelated networks (correlation base = 0.0) shows that GP-BayesMerge still outperforms unconstrained optimization ($80.24\%$ vs. $77.85\%$) due to the diagonal proximity penalty, proving its robustness under structural mismatches.
- **Sensitivity to Posterior Variance (Table 4):** Sweeping $\sigma_q^2$ shows that $\sigma_q^2 \in [10^{-4}, 10^{-3}]$ is optimal for representation-space dropout, cutting calibration error in half.
- **RBF vs. OU Kernel Comparison:** Showing that the tridiagonal OU kernel achieves statistically equivalent classification performance ($82.21 \pm 0.25\%$) to the dense RBF kernel ($82.35 \pm 0.24\%$) while enabling a massive $800\times$ speedup at scale is a highly convincing empirical contribution.

## Support for the Central Claims
The empirical results provide exceptionally strong, clear, and consistent support for the paper's core claims:
1. **The Overfitting-Optimizer Paradox is convincingly demonstrated:** In both Table 1 and Table 2, unconstrained Layer-Wise AdaMerging is highly volatile across seeds (std dev of $\pm 1.15\%$ on average and $\pm 1.84\%$ on physical SVHN) and experiences severe generalization collapse on SVHN (accuracies drop below Task Arithmetic).
2. **GP-BayesMerge effectively stabilizes optimization:** GP-BayesMerge achieves the highest classification accuracies on physical weights ($82.35 \pm 0.24\%$) and reduces seed-to-seed standard deviation dramatically (e.g., SVHN standard deviation drops from $\pm 1.84\%$ to $\pm 0.35\%$).
3. **Kronecker Multi-Task joint prior (MT-GP-BayesMerge) is superior:** Utilizing online activation CKA task-correlation matrices, MT-GP-BayesMerge achieves peak physical performance ($82.68 \pm 0.18\%$) with the lowest standard deviation across seeds.
4. **Randomized Posterior Evaluation cuts calibration error:** The stochastic ensemble classifier evaluation (Table 4) cuts Expected Calibration Error (ECE) on physical SVHN in half ($8.45\% \to 4.12\%$) and outperforms standard AdaMerging ($16.42\%$) by $4\times$, proving the empirical effectiveness of the randomized PAC-Bayes formulation.
