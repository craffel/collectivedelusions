# 2. Novelty Check and Delta from Prior Work

## Key Novel Aspects
1. **SVD-Based Parameter-Free Task Centroids:** 
   While previous parameter-free routing methods might project onto simple mean activations or class averages, the paper proposes utilizing Singular Value Decomposition (SVD) on final classifier weight matrices to extract task centroids ($v_k = V_{k, 1}$). This elegantly captures the principal direction of maximum class-prototype variance while completely bypassing prototype sum-to-zero cancellations.
2. **Löwdin Symmetric Orthogonalization in Representation Space:**
   The paper is the first to apply Löwdin Symmetric Orthogonalization (an order-invariant, least-squares optimal orthonormalization technique from quantum chemistry) to representation-space task projection. This is a highly creative and theoretically elegant addition to the model-merging literature, distinct from standard Gram-Schmidt orthogonalization (which is sequentially asymmetric and order-dependent).
3. **Rigorous Theoretical Deconstruction of Orthogonalization Limits:**
   Perhaps the most novel and refreshing contribution of the paper is its rigorous theoretical critique of its own proposed orthogonalization. Instead of claiming OTSP is superior, the authors mathematically prove:
   - **Symmetric Equivalence:** Under constant symmetric task correlation ($S = (1-s)I_K + s\mathbf{1}\mathbf{1}^T$), OTSP and PFSR are mathematically guaranteed to yield the exact same routing argmax for all samples, making Löwdin orthogonalization redundant.
   - **SNR Equivalence:** Under isotropic representation noise, the coordinate-difference Signal-to-Noise Ratio (SNR) of OTSP and PFSR is identical ($\text{SNR}_{\text{OTSP}} = \text{SNR}_{\text{PFSR}} = \frac{\sqrt{1-s}}{\sigma \sqrt{2}}$), demonstrating a perfect cancellation where margin expansion is offset by coordinate noise amplification.
   - **Noise Amplification & Spillover Penalties:** Under asymmetric layouts, OTSP's orthogonalization actively degrades performance because near-singular overlap makes $S^{-1/2}$ ill-conditioned, scaling up online coordinate noise variance.
4. **Systems-Level Operational Insights:**
   The paper deconstructs the **Orthogonal Masking Effect** (why joint classification accuracy is flat under disjoint orthogonal subspaces, necessitating routing accuracy as the primary metric) and the **Vectorization Collapse** of unconstrained linear routers (proving the simplex constraint is mathematically necessary under $B=1$ vectorized streaming).

## Delta from Prior Work
- **Delta from Static Model Merging (Task Arithmetic, TIES-Merging, DARE, RegMean):** 
  Static merging combines model weights permanently, producing a single, compromise model. In contrast, PFSR/OTSP perform sample-wise, input-dependent dynamic ensembling online in a single forward pass without permanent weight blending.
- **Delta from Trainable Routing (MoE, QWS-Merge, Parametric Routers):** 
  Trainable dynamic ensembling networks require specialized calibration splits, offline backpropagation loops, AdamW optimization, hyperparameter schedules, and suffer from small-sample inductive overfitting. PFSR and OTSP are **100% parameter-free, training-free, and data-free**, requiring zero calibration splits or optimization epochs.
- **Delta from Gram-Schmidt Orthogonalization:**
  Standard Gram-Schmidt is sequentially asymmetric (the resulting orthonormal basis depends on the arbitrary index order of experts). OTSP's Löwdin orthogonalization is perfectly symmetric and order-invariant.

## Characterization of Novelty
We characterize the novelty of this work as **significant and highly refreshing**.
In a literature saturated with increasingly complex, empirical-heavy, and over-parameterized neural routing layers, this paper stands out by successfully applying Occam's razor. The novelty does not merely lie in introducing a new routing algorithm, but in the **extraordinary mathematical rigor** and self-critical analysis used to map the limits of coordinate projection, proving closed-form SNR bounds, and exposing why the simpler method (PFSR) is more robust than the theoretically elegant, more complex orthogonalized variant (OTSP). This level of mathematical depth and theoretical honesty is a major departure from typical incremental empirical machine learning papers.
