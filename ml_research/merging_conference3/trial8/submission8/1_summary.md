# 1. Summary of the Paper

## 1.1 Core Objective
The paper addresses the challenge of **robust and sample-efficient Out-of-Distribution (OOD) task rejection inside dynamic model merging serving frameworks on resource-constrained edge hardware**. When serving multi-task parameter-efficient experts (e.g., LoRA adapters) on top of a shared, frozen backbone model (e.g., a Vision Transformer), routing must occur dynamically. Current state-of-the-art frameworks project early-layer representations onto task centroids to obtain task-similarity coordinates, using a coordinate-space diagonal Gaussian Mixture Model (GMM) to reject out-of-distribution queries and prevent downstream activation-blending/routing corruption.

## 1.2 Identified Methodology Bottleneck (The Critique)
Adopting the rigorous perspective of *The Methodologist*, the authors deconstruct current OOD rejection modules and reveal two fundamental flaws:
1. **The Clean Sandbox Confounder:** Prior works evaluate OOD performance under unrealistically clean conditions, ignoring realistic representation-level covariate shifts (sensor noise, lighting changes, etc.) that occur during on-device serving.
2. **Low-Resource Calibration Overfitting:** To keep calibration training-free and quick, diagonal GMMs are fit on tiny calibration sets ($N \le 64$). This causes maximum likelihood estimated (MLE) coordinate variances (especially on inactive dimensions) to collapse toward zero, rendering density evaluation highly unstable. Under minor covariate shifts, unregularized log-likelihoods drop catastrophically, triggering severe False Positive Rate (FPR) spikes and falsely rejecting valid in-distribution samples.

## 1.3 Key Technical Contributions (The Remedy)
To resolve this variance collapse, the authors propose **SRC-DE (Shrinkage-Regularized Coordinate Density Estimation)**:
- **Ledoit-Wolf Covariance Shrinkage:** Replaces heuristic L2/Ridge GMM regularizers with an analytical, parameter-free covariance shrinkage target. By calculating the optimal shrinkage intensity $\alpha_{\text{opt}}$ using sample fourth moments (soft EM kurtosis), it dynamically regularizes diagonal covariances.
- **Duality of Shrinkage Targets:**
  - *Global Coordinate-Wise Diagonal Target:* Shrinks component variances toward the global, registry-wide variance of each coordinate dimension, preserving individual scales and mitigating sphericity over-regularization bias in small registries ($K=4$).
  - *Spherical Diagonal Target:* Shrinks component variances toward an isotropic average variance, serving as a robust prior under high-dimensional registries ($K \ge 16$).
- **Symmetric Noise Protocol:** Exposes the "unequal noise confounder" in prior OOD pipelines (where noise was only added to ID test features, breaking classifiers and driving AUC below 0.5) and replaces it with a mathematically rigorous symmetric noise protocol.
- **Scikit-Learn GMM Cholesky Bug Fix:** Identifies a silent software defect in `sklearn.mixture.GaussianMixture` where manual covariance updates are ignored because the internal cached Cholesky precision matrix (`precisions_cholesky_`) is not rebuilt post-fit.
- **Rigorous Evaluation:** Evaluates 20 random seeds over 4 high-conflict vision datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) and high-dimensional scaling simulations, showing substantial AUC and system-level utility improvements, and validating Layer 3 as the optimal routing depth.
