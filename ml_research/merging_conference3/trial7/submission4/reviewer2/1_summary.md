# Comprehensive Summary of the Submission

## 1. Main Topic and Research Question
The paper addresses a core challenge in the field of **dynamic model merging and ensembling**: how to combine the specialized capabilities of pre-trained models on a sample-by-sample basis at runtime without the excessive overhead of training parameterized routing networks. The research is motivated by a critical reaction to the growing complexity of dynamic routing methods, which often rely on complex, over-parameterized neural networks (e.g., wave-superposition layers, multi-layer classifiers) requiring specialized calibration splits, multi-epoch optimization loops, and fine-tuning.

The authors ask: *Can we perform high-fidelity, sample-wise dynamic model ensembling without training a single parameter or using a single calibration sample?*

To answer this, they present and analyze **Parameter-Free Task-Space Projection (PFSR)**, a remarkably simple and training-free method, and its advanced orthogonalization extension, **L{\"o}wdin-Orthogonalized Task-Space Projection (OTSP)**.

---

## 2. Methodology and Approach
The proposed approach consists of two primary formulations and is detailed in five clean mathematical steps:

### A. Parameter-Free Task-Space Projection (PFSR)
PFSR uses raw, unorthogonalized task centroids extracted from the static weights of the specialist models:
1. **Centroid Extraction via SVD**: Rather than taking a simple average of the expert classifier weights $W_k$ (which suffers from sum-to-zero cancellation), PFSR computes the Singular Value Decomposition (SVD) of $W_k$ and extracts the top right-singular vector (the first principal component) as the raw task centroid $v_k$, which is then normalized to unit-norm ($\bar{v}_k$).
2. **Absolute Value Projection**: At runtime, penultimate feature representations $z_b$ are normalized ($\tilde{z}_b$) and projected onto the centroids. To handle class prototypes pointing in opposite directions within the task's subspace, PFSR takes the absolute value: $u_{k, b} = |\bar{v}_k \cdot \tilde{z}_b|$.
3. **Gating**: Sample-wise merging coefficients are computed via a temperature-scaled Softmax over the projection coordinates: $\alpha_{k, b} = \text{Softmax}(u_{k, b} / \tau)$.

### B. L{\"o}wdin-Orthogonalized Task-Space Projection (OTSP)
OTSP extends PFSR to decouple task coordinate axes under task overlap:
- It computes the pairwise cosine similarity of task centroids to construct a Gram overlap matrix $S$.
- It performs **L{\"o}wdin Symmetric Orthogonalization** to compute an optimal, order-invariant orthonormal task basis $Q = S^{-1/2} \bar{V}$ in closed form.
- Input features are projected onto this orthonormal basis: $u'_{k, b} = |q_k \cdot \tilde{z}_b|$.

---

## 3. Key Findings
The paper presents several major theoretical and empirical findings across its simulation sweeps and real-world proof-of-concept:
1. **Redundancy under Symmetric Overlap**: Under symmetric task correlations, OTSP and PFSR have identical routing coordinates up to a constant positive scaling factor. Their argmax decisions are mathematically identical, making L{\"o}wdin orthogonalization redundant.
2. **Noise Amplification Penalty**: Under active isotropic representation noise in asymmetric layouts, OTSP systematically underperforms PFSR (by 0.2% to 1.6%). The near-singular overlap matrix $S$ becomes ill-conditioned, and its inverse square root $S^{-1/2}$ dramatically amplifies the variance of isotropic feature noise: $\text{Var}(q_k \cdot \eta_b) = \sigma^2 (S^{-1})_{kk}$.
3. **Noise Spillover Penalty**: In asymmetric layouts, the orthogonalization step spreads noise across axes, allowing high noise from one expert to contaminate clean coordinate axes, whereas PFSR is immune.
4. **Vectorization Collapse**: Under sample-wise vectorized ensembling ($B=1$), unconstrained linear routers collapse to 55.57% accuracy due to severe small-sample inductive overfitting and a lack of constraint normalization.
5. **Orthogonal Masking Effect**: In perfectly disjoint orthogonal sandboxes, any positive weight on the correct expert yields the ceiling prediction because logits of other experts on out-of-subspace features are exactly $0.0$. This makes joint classification accuracy flat (74.46%) and insensitive to routing quality, establishing routing accuracy as the primary informative metric.
6. **Uniform Merging vs. Dynamic Routing**: Under active overlap, Uniform Merging slightly dominates average classification accuracy due to ensembling prediction-averaging. However, PFSR/OTSP provide high routing specificity, enabling massive savings in compute/latency in large registries by loading only selected experts.
7. **Real-World Generalization**: Evaluated on a 1,250-sample ImageNet-1K ResNet-18 manifold (Dogs/Cats/Vehicles), PFSR and OTSP achieve high routing accuracy (~92%).
8. **Anisotropic Feature Noise**: Under anisotropic noise, covariance whitening (spherizing features) is highly effective, restoring routing accuracy from 77.10% to 89.45%.

---

## 4. Explicitly Claimed Contributions and Evidence
The authors claim five core contributions, supported by the following evidence in the text:
* **SVD-Based Parameter-Free Task-Space Projection (PFSR)**: Shown to achieve 100% routing accuracy in the disjoint setup (Table 1), whereas Naive Mean Centroid collapses to 25.18% due to sum-to-zero cancellation.
* **Analysis of OTSP and Orthogonalization Limits**: Derived mathematically in Sections 3.6, 3.7, and 3.8. Backed by empirical sweeps under asymmetric layouts in Table 2, proving OTSP's systematic underperformance due to the Noise Amplification Penalty.
* **Deconstruction of Vectorized Streaming Stability (B=1)**: Demonstrated via the collapse of the LinearRouter (Unreg) to 55.57% (Table 1 & Figure 1).
* **Analysis of the Orthogonal Masking Effect**: Explained in Section 4.2 to justify the identical 74.46% classification accuracy across multiple methods in Table 1.
* **Implicit Regularization of Zero-Initialization**: Demonstrated by L3-Softmax Well-Reg (Zero-Init) achieving 67.22% routing accuracy (Table 1), showing that simple zero-initialization acts as a powerful maximum-entropy prior.
* **Top-k Sparse Gating & Self-Calibrated Temperature**: Proposed and analyzed in Sections 3.5 & 4.3 as a means to scale dynamic ensembling with high systems-level efficiency.
* **Real-world Proof of Concept**: Validated on ResNet-18 in Section 4.4, with OTSP gaining +0.08% routing accuracy due to positive Dogs/Cats overlap.
* **Mitigation of Anisotropic Noise**: Validated on a toy simulation in Section 4.5, where covariance whitening gains +12.35% absolute routing accuracy.
