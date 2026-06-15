# Paper Outline: Deconstructing Out-of-Distribution Task Rejection in Dynamic Model Merging: Covariance Shrinkage and Sample Complexity Audits

## Abstract
- **Context:** Serving multi-task adapters (e.g., LoRA) on edge hardware via dynamic model merging.
- **Problem:** OOD task rejection in prior works (such as SPS-ZCA) relies on unregularized coordinate-space diagonal GMMs. These models overfit small calibration sets ($N=64$), causing catastrophic false-rejection rate spikes under mild covariate shift (representation noise).
- **Proposed Solution:** SRC-DE (Shrinkage-Regularized Coordinate Density Estimation), which dynamically stabilizes GMM covariance matrices using analytical Ledoit-Wolf-style shrinkage.
- **Results:** SRC-DE recovers OOD rejection AUC under representation noise ($\sigma^2=0.10$) from 0.6370 to 0.6823 (+4.53% absolute improvement) over the unregularized diagonal GMM, and improves moderate-sample (N=32) multi-component AUC by +4.14% absolute while reducing estimation variance under EM component splitting, establishing a new standard of methodological robustness on the edge.

## 1. Introduction
- **PEFT and Adapter Serving:** The shift toward on-device deployment of multiple task-specific LoRA adapters over a single frozen backbone.
- **The Routing Paradox and Coordinate Space:** How prior work uses early-layer centroids and coordinate space to route queries in a single pass.
- **The Methodological Critique:**
  - Expose the assumption that coordinate spaces remain stable under covariate shift.
  - Expose the overfitting vulnerability of fitting GMMs on low-resource calibration splits ($N=64$).
- **Our Contribution (SRC-DE):** Integrating analytical Ledoit-Wolf shrinkage to resolve variance underestimation, evaluating performance under systematic perturbation.

## 2. Related Work
- **PEFT Serving on the Edge:** Review LoRA, S-LoRA, Punica, AdapterHub.
- **Model Merging & Activation Blending:** Weight-averaging vs. activation blending (Task Arithmetic, ZipIt, SABLE, PFSR).
- **Task Routing and OOD Detection:** Nearest-centroid routing, coordinate-space density estimation.
- **High-Dimensional Statistics and Covariance Shrinkage:** Standard covariance regularizers, Ledoit-Wolf shrinkage and its application to low-data GMM parameter estimation.

## 3. Methodology
- **Coordinate Space Projection:** Formula for $u_{k, b} = \text{cos\_sim}(h^{(3)}_b, \mu^{(3)}_k)$.
- **Mathematical Analysis of GMM Overfitting:** How maximum likelihood estimates of diagonal variance $\sigma^2_{k, m, j}$ underestimate the true manifold variance when $N$ is small, leading to boundary collapse.
- **Shrinkage-Regularized Coordinate Density Estimation (SRC-DE):**
  - Define the shrunk covariance: $\Sigma^{*}_{k, m} = (1 - \alpha) \Sigma_{k, m} + \alpha \nu I$.
  - Provide the analytical formula for optimal shrinkage intensity $\alpha_{\text{opt}}$.
- **Online Inference and Routing-Fallback:** Dual-path routing based on log-likelihood thresholding $\log P(u_b \mid \text{GMM}_k) \ge \eta$.

## 4. Experiments
- **Experimental Setup:** Vision Transformer backbone, high-conflict task suite (MNIST, FashionMNIST, CIFAR-10, SVHN).
- **Baselines:** Raw Cosine, Unregularized GMM (SPS-ZCA), Ridge GMM (Ridge regularizer $\gamma I$).
- **Experiment 1 (Robustness to Covariate Shift):** Quantitative analysis showing GMM collapse under noise and how SRC-DE restores AUC. Reference Figure 1 (ROC curves) and Figure 2 (AUC vs. Noise).
- **Experiment 2 (Sample Complexity Analysis):** Quantitative audit showing SRC-DE's superior sample efficiency at $N=8, 16$. Reference Figure 3 (AUC vs. Sample Size).
- **Methodological Discovery:** The sklearn GMM precision Cholesky bug and how we resolved it.

## 5. Conclusion
- **Summary:** Retrospective on the importance of testing under covariate shift.
- **Future Directions:** Full covariance shrinkage, dynamic adaptation of shrinkage target, streaming calibration.
