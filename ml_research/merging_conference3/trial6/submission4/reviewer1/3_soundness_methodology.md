# 3. Soundness and Methodology

## Clarity of the Description
The methodology of the paper is exceptionally well-structured, precise, and clearly written:
- Mathematical formulations are explicitly defined. The singular value decomposition (SVD) for PCA projection, the normalization onto the unit sphere, and the TSAR regularizer are mathematically sound and easy to follow.
- The paper is highly transparent about its architectural simplifications, providing formal proofs for both the **layer-averaging collapse** (showing that multi-layer dynamic routing is equivalent to a single global router at deployment) and the **mathematical equivalence to output-level logit ensembling** in the linear sandbox.
- The explanations of key phenomena, such as **heterogeneity collapse** (coefficient cancellation in mixed streams) and **multi-task gradient imbalance** (hard tasks hijacking the gradients of easy tasks), are logically and mathematically rigorous.

## Appropriateness of Methods
The choice of methods is highly appropriate and well-aligned with the goal of low-data router calibration:
- **Low-Dimensional Projection (PCA & Random Gaussian):** Drastically reduces the parameter footprint of the router (to just 20 parameters for a single-layer global router), which is a vital design choice to prevent overfitting in data-scarce splits.
- **Task-Space Centroids as Anchors:** Using the centroids of pre-trained expert representations as spatial anchors is geometrically intuitive. Pre-trained features possess high inter-task separability, making centroids stable coordinate references even under extreme data scarcity.
- **Projecting Conflicting Gradients (PCGrad):** Highly appropriate for multi-task optimization. Gradient conflict and cross-talk are ubiquitous when joint losses include tasks of highly varying difficulty (e.g., MNIST vs. SVHN), and PCGrad explicitly projects out conflicting gradient components.
- **Scaled Sigmoid Activation:** Using a non-negative bounded activation function is an elegant, zero-runtime-overhead solution to prevent coefficient cancellation under heterogeneous streaming, bypassing the latency-heavy batch partitioning approach.

## Potential Technical Flaws and Limitations
While the methodology is highly robust, several key limitations and minor technical approximations are worth highlighting:
1. **Physical Model Merging is Restricted to Head-Level Merging (Logit Ensembling):**
   - In Section 3.3 and Appendix F, the authors perform "physical weight-space model merging" on a Vision Transformer (ViT-Tiny). However, they restrict this fusion to the linear classification heads on top of frozen backbone features.
   - As the authors transparently derive in Equation 10, head-level weight merging is mathematically identical to output-level logit ensembling. There is no actual parameter-level fusion or weight interpolation applied to the internal, non-linear layers of the deep transformer (such as self-attention projection matrices or feed-forward MLPs).
   - Thus, while the paper is titled and framed around "model merging," its physical validation does not address the unique structural and representation flow challenges of merging weights *inside* deep non-linear networks. This is a critical limitation, as deep weight merging is much more complex than simple output-level ensembling.
2. **Uncentered PCA Projection Approximation:**
   - In Equation 1 and 2 (Section 3.1), the forward PCA projection is applied to uncentered features $z(x)_b$ prior to unit-sphere normalization, rather than subtracting the global mean $\mu_z$.
   - The authors show that because the translation vector $\mu_P = \mu_z P$ resides inside the norm divisor, this uncentered projection introduces a sample-dependent non-linear coordinate distortion. While they argue that this distortion is harmless due to task-manifold concentration, it is still a mathematical approximation to standard PCA.
3. **SVHN Expert Performance Ceiling:**
   - In the main experiments, the SVHN expert ceiling is deliberately set to a very low accuracy (19.28%) by simulating high noise. While the authors perform a stress test and show in Section 17 that the results hold under a realistic SVHN expert ceiling (90.40%), the main paper's reliance on a highly degraded simulation task could be seen as an artificial setup, even though it serves as an interesting optimization stress test.
4. **PCGrad Computational Complexity:**
   - Standard PCGrad requires $K$ separate backward passes per step, scaling the training cost to $O(K)$. While the authors propose Task Grouping and Stochastic Sampling in Appendix B to mitigate this, the standard joint optimization remains computationally expensive for massive-scale systems during calibration.

## Reproducibility
The work exhibits an exceptionally high standard of reproducibility:
- **Hyperparameter Specification:** Complete and detailed hyperparameters (optimizer, learning rates, weight decays, training epochs, batch sizes, etc.) are specified for both the classifier training (Section 9) and router calibration (Section 3.4).
- **Control Variables:** All experiments are evaluated across 5 independent random seeds with reported means and standard deviations, which ensures statistical significance and rules out seed-cherrypicking.
- **Clear Implementation Guidelines:** The step-by-step description in Section 4.6 (Real-World Deployment Guidelines) provides a clear blueprint for practitioners to implement and run TSAR on real datasets.
