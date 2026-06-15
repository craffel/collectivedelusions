# 2. Novelty and Literature Delta Check

## Key Novel Aspects
1. **First Spectral Analysis on Riemannian Manifolds:** While previous works investigated representational isotropy and spectral alignment in Euclidean spaces (e.g., SAIM), this work is the first to explicitly attempt and analyze spectral balancing within the Lie algebra tangent space $\mathfrak{so}(d)$ of the orthogonal group $\mathrm{O}(d)$.
2. **Identification of the Tangent-Space Spectral Pitfall:** The paper uncovers a profound, counter-intuitive negative result: operations that stabilize representations in flat Euclidean spaces (such as smoothing/balancing singular values to restore isotropy) become highly destructive when mapped through non-linear retraction operators like the Cayley transform.
3. **Rigorous Mathematical Proofs of SVD Incompatibility:** The authors derive the *Kernel Distortion Theorem* (Theorem 3.2) and *Spectrum Distortion Theorem* (Theorem 3.3) to explain how standard numerical SVD solvers introduce coordinate gauge mismatches that are incompatible with skew-symmetric algebraic closure.
4. **Symmetry-Preserving Alternatives:** The paper introduces real Schur decomposition and a complex Hermitian eigensolver to perform projection-free spectral operations within Lie algebras.

## Delta from Prior Work
- **Delta from OrthoMerge (Yang et al., 2026):** OrthoMerge performs manifold interpolation of rotations without modifying their spectra. The delta is the addition of spectral operations (balancing and pruning) in the Lie algebra, along with a deep theoretical diagnostic of why naive balancing fails.
- **Delta from SAIM (2026):** SAIM balances the singular value spectrum to restore representational isotropy, but is strictly restricted to linear, Euclidean weight additions. This work attempts to translate SAIM to curved manifolds, showing that doing so is fundamentally destructive due to curvature distortion.
- **Delta from standard SVD/Schur math:** While real Schur and complex Hermitian decompositions are standard linear algebra techniques, their specific application as a symmetry-preserving, projection-free alternative to SVD for deep-learning manifold model merging is novel.

## Characterization of Novelty and Critical Critique

### 1. The SVD-Distortion Theorems are a Mathematical Side-Show (Critical Gap)
The authors spend considerable effort proving the **Kernel Distortion Theorem** and **Spectrum Distortion Theorem** as the primary mathematical reasons why standard SVD is structurally incompatible with spectral operations in Lie algebras. However, their own empirical results in Section 4.4 and Section 4.5 completely undermine the importance of these theorems:
- **RIMO-Schur-Balanced** (which uses Schur decomposition to guarantee perfect skew-symmetry, zero projection error, and zero kernel distortion) **still collapses catastrophically to $12.36\%$ (soft regularized) and $16.50\%$ (hard constraints)**.
- **RIMO-Complex-Balanced** (which utilizes a complex Hermitian solver with a symmetric gauge to avoid kernel distortion) **also collapses to $20.04\%$**.
- This demonstrates that SVD projection errors and coordinate gauge mismatches are *not* the primary cause of the performance collapse. The catastrophic collapse is entirely driven by the **non-linear coordinate inflation under the Cayley map** (Cayley mapping noise propagation), which affects Schur and SVD-based balancing equally. 
- Consequently, while Theorems 3.2 and 3.3 are mathematically elegant, they represent a theoretical distraction from the actual physical bottleneck of tangent-space operations. The "novelty" of these theorems is diminished because the problem they address (projection and gauge distortion) is a minor numerical issue compared to the fundamental non-linear noise propagation under the Cayley map.

### 2. RIMO-Pruned is Conceptually Incremental
While the theoretical discussion of the pitfall is deep, the proposed practical solution—**RIMO-Pruned**—is highly incremental. It consists of performing standard low-rank truncation (keeping the top singular value pairs and zeroing the rest). This is a well-known technique in deep learning (e.g., SVD-based low-rank approximation) and does not represent a significant conceptual leap. Moreover, it completely abandons the goal of "restoring representational isotropy" (the core idea from SAIM they wanted to translate), opting instead to preserve the low-rank structure. Thus, the actual successful method presented is a simple low-rank truncation scheme rather than a novel manifold-level isotropic operator.

### 3. Lack of Utility: Why Manifold Merging if Task Arithmetic is Better?
The ultimate goal of any merging method is to improve multi-task performance. However, under both training regimes, the proposed geometric framework is consistently out-performed by standard flat-space **Task Arithmetic (TA)**:
- Under standard Euclidean training: Task Arithmetic ($\lambda = 0.3$) achieves **$91.11\%$**, while RIMO-Pruned achieves **$90.47\%$** (a $0.64\%$ drop).
- Under Orthogonal Regularization: Task Arithmetic ($\lambda = 1.0$) achieves **$94.00\%$**, while OrthoMerge / RIMO ($t=1.0$) achieves **$84.55\%$** (a massive $9.45\%$ gap), and RIMO-Pruned achieves **$91.49\%$** (a $2.51\%$ gap).
- This raises a severe practical question: Why should practitioners adopt a complex, mathematically intensive pipeline requiring expensive $O(d^3)$ SVD/Schur operations and restrictive orthogonal training constraints when simple, training-free, $O(1)$ Euclidean Task Arithmetic outperforms it across the board? The paper struggles to establish a strong practical motivation for the proposed framework.
