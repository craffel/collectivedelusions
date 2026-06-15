# 1. Summary of the Paper

## Main Topic and Context
The paper addresses the problem of **test-time model merging**, which aims to combine multiple task-specific expert neural networks (fine-tuned from a common base model) into a single multi-task network without requiring additional training on raw datasets. Specifically, it focuses on the layer-wise optimization of merging coefficients using small, unlabeled calibration batches at test time.

## Exposing the Overfitting-Optimizer Paradox
The authors identify a fundamental vulnerability in existing optimization-based test-time merging methods (e.g., AdaMerging). Because the number of learnable parameters (layer-wise and task-wise merging coefficients) is large relative to the small size of the test-time calibration batch (typically $N \le 64$), unconstrained first-order optimizers (like Adam) aggressively overfit the transductive noise of the calibration stream. This leads to:
1. Highly volatile, high-frequency spatial oscillations of merging coefficients across adjacent layers.
2. Catastrophic generalization collapse on unseen target test data.
This phenomenon is termed the **Overfitting-Optimizer Paradox**.

## Proposed Approach: GP-BayesMerge
To resolve this, the authors propose **GP-BayesMerge**, a Gaussian Process (GP) PAC-Bayes framework for robust test-time model merging.
1. **PAC-Bayes Formulation:** The authors reformulate test-time adaptation as a Bayesian inference problem over the coefficient space. Utilizing Alquier's linear PAC-Bayes bound, they derive a linear Kullback-Leibler (KL) complexity penalty on the coefficient posterior.
2. **Continuous GP Spatial Prior:** They model the prior distribution of the layer-wise merging coefficients as a continuous Gaussian Process over normalized network depth.
3. **Quadratic Precision-Matrix Regularization:** Under an isotropic Gaussian posterior centered at the optimized coefficients, the KL divergence simplifies to a quadratic form governed by the GP precision matrix $\Sigma_{\ell}^{-1}$. This precision matrix mathematically unifies:
   - **Proximity Penalty:** Positive diagonal elements act as weight decay, penalizing large deviations from the task-arithmetic prior mean $\mu_0$.
   - **Spatial Smoothness:** Negative adjacent off-diagonal elements function as a finite-difference Laplacian, penalizing high-frequency spatial noise across adjacent layers.
4. **Extensions:**
   - **MT-GP-BayesMerge:** A multi-task joint prior modeled via a Kronecker product of a task correlation matrix $B$ and the spatial covariance matrix $\Sigma_{\ell}$. The task correlation matrix $B$ is estimated online and data-free using Centered Kernel Alignment (CKA) similarities on target calibration activations.
   - **Non-Stationary Block-Wise GP Prior:** Decouples layer correlations across functional block boundaries to prevent over-smoothing.
   - **Ornstein-Uhlenbeck (OU) Kernel:** Provides a strictly tridiagonal precision matrix with an exact closed-form analytical inverse, enabling linear-time $O(L)$ assembly and bypassing $O(L^3)$ dense matrix inversion.
   - **Randomized Posteriors Evaluation:** Sampling coefficients from the randomized PAC-Bayes posterior at test time to act as a representation-space dropout and improve calibration.

## Key Findings and Claims
1. **Overfitting-Optimizer Paradox Resolution:** Unconstrained AdaMerging on physical SVHN weights drops performance to $46.64\%$, while GP-BayesMerge recovers and preserves it at $73.38 \pm 1.55\%$.
2. **Performance Improvements:** GP-BayesMerge and MT-GP-BayesMerge achieve state-of-the-art average accuracies of $84.76\%$ and $84.55\%$ (in simulation) and $82.35\%$ and $82.68\%$ (on physical weights), outperforming heuristic penalties (RegCalMerge) and rigid subspace constraints (PolyMerge).
3. **Stability Boost:** The proposed spatial priors significantly reduce standard deviations across random seeds (reducing average variance to $0.37\%$ in simulation and $0.24\%$ on physical weights).
4. **Computational Efficiency:** The offline covariance inversion cost is negligible ($<0.2$~ms for 80 layers), and the OU kernel allows linear-time scaling. The optimization converges rapidly in fewer than 50 steps, reducing adaptation latency to $<0.15$ seconds.

## Explicitly Claimed Contributions and Evidence
- **Identification of the Overfitting-Optimizer Paradox:** Supported by simulated and physical results demonstrating generalization collapse of unconstrained layer-wise optimization.
- **Derivation of GP-BayesMerge from PAC-Bayes Theory:** Formally proven in Section 3 and Appendix A/B.
- **Multi-task Extension (MT-GP-BayesMerge):** Demonstrated to achieve superior stability by utilizing online CKA similarities.
- **Physical Validation:** Empirical validation on actual physical weights of a CLIP ViT-B/32 image encoder across 8 diverse datasets.
