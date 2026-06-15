# 1. Summary of the Paper

## Main Topic
The paper addresses the challenge of **dynamic model ensembling/merging** for specialized, pre-trained neural networks (e.g., LoRA specialists). It challenges the current trend of using increasingly complex, over-parameterized, and trainable routing networks which require calibration datasets, multi-epoch optimization (e.g., AdamW), and hyperparameter tuning. The authors apply Occam's razor to advocate for simple, closed-form, and parameter-free linear algebra projections to route online representations dynamically.

## Proposed Approaches
1. **Parameter-Free Task-Space Projection (PFSR):** A 100% training-free and data-free dynamic routing method.
   - **Centroid Extraction via SVD:** Performs SVD on each expert's classifier weights ($W_k = U_k \Sigma_k V_k^T$) and defines the task centroid as the first column of $V_k$ (the top right-singular vector), which captures the principal axis of prototype variation and avoids the sum-to-zero cancellation of simple averaging.
   - **Coordinate Projection & Absolute Value:** Normalizes online features ($\tilde{z}_b$) and projects them onto the normalized centroids: $u_{k, b} = |\bar{v}_k \cdot \tilde{z}_b|$. Taking the absolute value resolves prototype sign cancellation.
   - **Gating:** Applies temperature-scaled Softmax to compute ensembling weights: $\alpha_{k, b} = \text{Softmax}(u_{k, b} / \tau)$.
2. **Löwdin-Orthogonalized Task-Space Projection (OTSP):** An advanced orthogonalization extension.
   - Applies Löwdin Symmetric Orthogonalization to the extracted task centroids offline to produce an orthonormal basis $\{q_k\}_{k=1}^K$ that is mathematically closest to the original centroids in a least-squares sense, preserving perfect order-invariance and symmetry across specialists.
   - Projects feature representations onto this orthonormal basis: $u'_{k, b} = |q_k \cdot \tilde{z}_b|$.
3. **Self-Calibrated Temperature Scheduling:** To avoid manual tuning of the scaling temperature $\tau$, the authors propose sample-wise dynamic temperature scheduling: $\tau_b = \gamma \cdot \text{std}_k(u_{k,b})$.
4. **Top-$k$ Sparse Gating:** Restricts ensembling coefficients to the top-$k$ coordinates (e.g., $k=2$) and renormalizes, ensuring that at most $k$ specialists are active to preserve systems-level memory and compute savings.
5. **Anisotropic Feature Noise Spherization via Offline Covariance Whitening:** For non-spherical feature manifolds, the authors propose estimating the empirical covariance matrix $\hat{\Sigma}$ and applying a Mahalanobis whitening transformation ($\hat{\Sigma}^{-1/2}$) offline to both task centroids and representations.

## Key Findings and Evidence
- **SVD Centroid Extraction is Mandatory:** Simple averaging of classifier weights (Naive Mean Centroid) collapses to near $\mathbf{0}$ due to the sum-to-zero symmetry of prototypes, yielding near-random routing accuracy ($25.18\% \pm 1.10\%$). In contrast, SVD centroid extraction maintains perfect **100.00%** routing accuracy under uncorrupted disjoint subspaces (Table 4.1).
- **Redundancy of Orthogonalization under Symmetry:** Under symmetric task correlation, OTSP and PFSR make identical routing decisions (Table 4.1 matches perfectly at 100.00% routing, and Section 4.3 matches perfectly at 94.62% routing under symmetric overlap).
- **The Noise Amplification & Noise Spillover Penalties:** In asymmetric task overlap settings under isotropic representation noise, OTSP systematically underperforms PFSR by 0.2% to 1.6% (Table 4.2). This is caused by:
  - *Noise Amplification:* Near-singular overlap makes $S$ ill-conditioned, and $S^{-1/2}$ scales up the variance of online projection coordinate noise.
  - *Noise Spillover:* Orthogonalization spreads noise across coordinate axes, contaminating clean specialists' coordinates.
- **Deconstruction of Vectorization Collapse:** An unregularized, unnormalized LinearRouter baseline drops to $55.57\% \pm 1.68\%$ accuracy under sample-wise vectorized streaming ($B=1$) due to overfitting on the small calibration split (64 samples). The authors show that simplex-constraint normalization (e.g., Softmax) naturally immunizes PFSR, OTSP, and modern trained routers against this instability (Table 4.1).
- **Characterization of the Orthogonal Masking Effect:** Under perfectly disjoint orthogonal subspaces, any positive ensembling weight on the correct expert yields the ceiling joint classification accuracy ($74.46\% \pm 0.81\%$). The authors explain that classification accuracy is flat and uninformative in this regime, establishing routing accuracy as the primary evaluation metric.
- **Implicit Regularization of Zero-Initialization:** Training a parametric Softmax router with zero-initialized weights acts as an implicit uniform maximum-entropy prior, shielding it from small-sample overfitting (Table 4.1).
- **Real-World Generalization:** On a 1,250-sample ResNet-18 ImageNet feature manifold, PFSR and OTSP generalize seamlessly, achieving $92.00\%$ and $92.08\%$ routing accuracy respectively (Table 4.4).
- **Anisotropic Noise Spherization:** Toy simulations show that anisotropic feature noise collapses OTSP routing accuracy to $77.10\%$, but offline covariance whitening successfully recovers performance to $89.45\%$ (Section 4.6).

## Explicitly Claimed Contributions
1. **Introduction of PFSR:** A training-free, data-free, closed-form linear projection router that extracts task centroids using SVD (Table 4.1).
2. **Analysis of OTSP and Orthogonalization Limits:** Proved mathematically (Section 3.7, 3.8) and empirically (Table 4.2) that OTSP is redundant under symmetric layouts and underperforms under asymmetric layouts due to Noise Amplification and Spillover.
3. **Deconstruction of Vectorization Collapse:** Showed that simplex-simplex constraints are mathematically required to prevent unconstrained coordinate collapse under vectorized streaming (Section 4.2).
4. **Deconstruction of the Orthogonal Masking Effect:** Explained why classification accuracy is flat in disjoint orthogonal sandboxes, making routing accuracy the key metric (Section 4.2).
5. **Implicit Regularization Analysis:** Showed that zero-initialization of Softmax gating acts as a maximum-entropy prior (Table 4.1).
6. **Self-Calibrated Temperature, Top-$k$ Sparse Gating, and Covariance Whitening:** Designed robust mitigation strategies to address practical deployment gaps (Sections 4.3, 4.4, 4.6).
7. **Real-World Verification:** Evaluated and proved performance on real ImageNet-1K features from a pre-trained ResNet-18 final layer (Section 4.5).
