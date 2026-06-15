# Paper Summary: GP-BayesMerge

## 1. Overview and Core Motivation
This paper addresses the problem of test-time parameter-space model merging, which integrates multiple task-specific expert models into a single multi-task network without expensive retraining. 

Recent test-time model merging methods (e.g., AdaMerging) treat layer-wise merging coefficients as learnable parameters optimized on small, unlabeled calibration batches via unsupervised surrogate losses (typically prediction entropy). However, this work exposes the **Overfitting-Optimizer Paradox**: unconstrained first-order optimization aggressively fits the transductive noise of small test-time calibration batches. This leads to highly volatile, jagged layer-wise coefficient profiles that destroy the model's structural and representational integrity, resulting in catastrophic generalization collapse on unseen target test domains (particularly street view house numbers, SVHN).

To resolve this paradox, the authors propose **GP-BayesMerge**, a mathematically rigorous Gaussian Process (GP) PAC-Bayes framework for robust test-time model merging.

## 2. Proposed Methodology
The core idea is to reformulate test-time adaptation as a Bayesian inference problem over the low-dimensional control space of merging coefficients:
* **PAC-Bayes Generalization Bound:** Utilizing Alquier’s linear PAC-Bayes generalization bounds, the authors derive an optimization objective where the complexity penalty is proportional to the Kullback-Leibler (KL) divergence between the coefficient posterior and a continuous GP spatial prior over network depth.
* **Continuous GP Spatial Prior:** The prior over layer-wise merging coefficients is modeled as a continuous GP over normalized network depth (using a Squared Exponential or Ornstein-Uhlenbeck kernel). Evaluating the KL divergence under an isotropic Gaussian posterior centered at the optimized coefficients simplifies the complexity penalty to a quadratic form governed by the GP precision matrix $\Sigma_{\ell}^{-1}$.
* **Precision Matrix Dual Action:** The precision matrix $\Sigma_{\ell}^{-1}$ naturally acts as a unified regularizer:
  1. *Proximity Penalty (Diagonal elements):* Penalizes deviations from the uniform task-arithmetic prior mean.
  2. *Spatial Smoothness (Off-diagonal elements):* Acts as a finite-difference Laplacian smoother, penalizing high-frequency spatial noise across adjacent layers.
* **Multi-Task Joint GP Prior:** To model cross-task representational conflicts, the authors generalize the framework using a joint prior governed by the Kronecker product of a task correlation matrix $B$ (estimated dynamically online using activation CKA) and the spatial GP covariance $\Sigma_{\ell}$.
* **Surrogate-to-Target Risk Bound:** To bridge the theoretical gap between minimizing unsupervised entropy and true classification error, the paper proves a theorem bounding true target risk under Margin-Preserving Support and Classifier Calibration assumptions.

## 3. Key Findings & Empirical Results
The paper evaluates GP-BayesMerge under both a highly controlled non-convex simulation (representing a 12-layer Vision Transformer on MNIST, FashionMNIST, CIFAR-10, and SVHN) and actual physical weight merging of pre-trained CLIP ViT-B/32 (86M params) and CLIP ViT-L/14 (307M params) models across 8 real-world datasets:
* **Resolving Overfitting:** GP-BayesMerge completely resolves the Overfitting-Optimizer Paradox, eliminating wild spatial coefficient oscillations and stabilizing SVHN performance.
* **State-of-the-Art Accuracy:** On physical weights, it achieves superior average accuracy ($82.35\%$) and dramatically reduced seed-to-seed variance ($\le 0.24\%$) compared to standard AdaMerging and heuristic baselines (RegCalMerge, PolyMerge).
* **Multi-Task Superiority:** MT-GP-BayesMerge (utilizing online task correlation $B_{\text{online}}$ estimated dynamically from CKA activations of only 16 unlabeled calibration samples) achieves the highest physical accuracy of $82.68\%$ and outstanding stability ($0.18\%$ variance).
* **Scalability:** The authors show that the lengthscale scales stably as $\ell \propto 1/L$ and verify that the Ornstein-Uhlenbeck (OU) kernel enables exact $O(L)$ tridiagonal inversion, delivering perfect scalability with $<0.2$ ms latency for ultra-deep models.
