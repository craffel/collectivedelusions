# 2. Novelty Check

## Originality and Contextualization
The paper addresses a highly active area of research: post-hoc model merging and adaptive ensembling. Overfitting on small calibration sets is a critical issue in layer-wise adaptive merging. 

The paper positions its novelty primarily as a resolution of the limitations of **Rademacher-Bounded Polynomial Merging (RBPM)** (Chatterjee et al., 2024), which restricts layer-wise coefficients using a low-degree polynomial subspace. The authors argue that:
1. Polynomial trajectories (even at low degrees like $d=2$) suffer from severe **boundary runaway** (Runge's-like phenomena) at the early and final layers because of global parabolic shape constraints.
2. The Fourier (RB-FTM) and Discrete Cosine (RB-DCTM) parameterizations use bounded trigonometric bases, naturally stabilizing the boundaries.

---

## Evaluation of Specific Novelty Claims

### 1. Trigonometric/Fourier Trajectory Parameterization (RB-FTM)
*   **Claim:** Projecting layer-wise coefficients onto a low-frequency continuous Fourier subspace across network depth is a novel and stable trajectory representation.
*   **Assessment:** **High Novelty.** While Fourier analysis is a mature field and low-pass filtering is standard in signal processing, applying a continuous Fourier series parameterization to ensembling trajectories across the discrete coordinate of neural network depth is highly creative and original. This is a significant improvement over the rigid polynomial representations of RBPM and the overparameterized independent layers of AdaMerging.

### 2. Discrete Cosine Trajectory Merging (RB-DCTM)
*   **Claim:** Utilizing a half-period cosine basis (DCT) resolves the artificial periodic boundary identity ($\alpha_k(0) = \alpha_k(1)$) of Fourier series, while keeping the boundaries stable and derivatives flat (Neumann boundary conditions).
*   **Assessment:** **Exceptional Novelty.** The transition from Fourier (which is periodic and forces identical ensembling weights at the first and last layers) to DCT is mathematically elegant and architecturally justified. The identification of implicit homogeneous Neumann boundary conditions ($h'(0) = h'(1) = 0$) as a regularizer that shields the early representation extraction and final classification projections from high-frequency noise is a highly original and insightful observation.

### 3. Trajectory-Space Rademacher Complexity Bounds (Theorems 3.1 & 3.4)
*   **Claim:** Deriving analytical empirical Rademacher complexity bounds for Fourier and DCT trajectory classes over depth coordinates, showing they scale with $\sqrt{\ln(F)/L}$ independent of the underlying network parameter count.
*   **Assessment:** **Solid Novelty.** Chatterjee et al. (2024) derived bounds for polynomial trajectories, but extending this statistical learning framework to trigonometric Fourier and Cosine bases requires non-trivial mathematical derivations (using Massart's Finite Lemma over a $4F+2$ or $2F+2$ bounded coordinate basis). Proving that the DCT variant (Theorem 3.4) achieves a strictly tighter bound because of its cosine-only representation is a strong, valuable theoretical contribution.

### 4. Spectral Lasso ($L_1$) Regularization on Harmonics
*   **Claim:** Directly coupling the Rademacher complexity bounds to a practical $L_1$ penalty on harmonic coefficients, while leaving the base uniform coefficient $a_{k,0}$ unpenalized, represents a novel physical regularization mechanism.
*   **Assessment:** **Good Novelty.** Applying $L_1$ regularization to promote sparsity in coefficients is standard. However, the selective application of $L_1$ strictly to the *harmonic* components ($a_{k,f}, b_{k,f}$ for $f \ge 1$) while exempting $a_{k,0}$ is a highly practical and chemically clean design. Initializing $a_{k,0} = 1/K$ and forcing harmonics to $0$ ensures that the optimization starts exactly at the robust Static Uniform baseline.

---

## Comparison with Prior Art

| Feature | Task Arithmetic / TIES (Wortsman et al., 2022) | AdaMerging / PolyMerge (Yang et al., 2024) | RBPM (Chatterjee et al., 2024) | **RB-FTM / RB-DCTM (Ours)** |
| :--- | :--- | :--- | :--- | :--- |
| **Trajectory Type** | None (Static / Flat) | Independent or Polynomial | Low-degree Polynomial | **Harmonic/Trigonometric (Fourier & Cosine)** |
| **Boundary Stability** | Perfect (Flat) | Poor / Overparameterized | Severe Runaway / Oscillations | **Excellent (Sinusoidal bounds & Neumann flat derivatives)** |
| **Complexity Guarantees** | Not Applicable | None | Rademacher Polynomial Bounds | **Rademacher Fourier & Cosine Bounds (Tighter for DCT)** |
| **Calibration Regularizer** | None | Prediction Entropy | Analytical Polynomial Penalty | **Analytical Spectral Lasso ($L_1$) strictly on harmonics** |
| **Coordinate Alignment Needed?** | Yes | Yes | Yes | **Yes (But highly robust to misalignment under DCT)** |

---

## Novelty Conclusion
The paper's novelty is highly convincing. It does not merely combine existing techniques; it uses rigorous statistical learning theory to diagnose and solve a concrete practical failure mode of a state-of-the-art baseline (the boundary runaway of RBPM). The progression from Fourier to DCT trajectories to resolve periodic boundary constraints while maintaining Neumann flat derivatives represents a solid step forward in our understanding of layer-wise weight merging.
