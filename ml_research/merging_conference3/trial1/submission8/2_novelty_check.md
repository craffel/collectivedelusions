# Novelty and Originality Evaluation

## 1. Positioning and Baseline Comparison
The paper sits at the intersection of three major areas in machine learning:
1. **Model Merging:** Traditional flat Euclidean methods such as Task Arithmetic (TA) \cite{ilharco2022editing}, TIES-Merging \cite{ties2023}, and DARE \cite{dare2024}.
2. **Geometric Parameterizations:** Methods that constrain parameters to Lie groups, specifically Orthogonal Fine-Tuning (OFT) \cite{oft2023} and OrthoMerge \cite{yang2026orthomerge}.
3. **Representational Isotropy / Spectral Analysis:** Methods that balance singular values of weight matrices to reduce interference, specifically Sharpness-Aware Isotropic Merging (SAIM) \cite{saim2026}.

## 2. Novelty of the Contributions
The paper's novelty is highly significant and can be broken down into three main pillars:

### Pillar A: The First Study of Representational Isotropy on Non-Linear Manifolds
While Euclidean spectral balancing (like SAIM) is well-studied, this work is the **first** to evaluate whether these concepts translate to curved manifold structures (the orthogonal group $\mathrm{O}(d)$ and Lie algebra $\mathfrak{so}(d)$). The paper establishes a clear geometric boundary: operations that are linear-safe in Euclidean spaces can become highly destructive in tangent spaces of non-linear manifolds due to curvature mapping. This represents a substantial conceptual advance.

### Pillar B: Exposing and Formalizing a New Fundamental Pitfall
Instead of presenting a purely positive, heuristic contribution, this work provides a rigorous **negative result**—the tangent-space spectral balancing pitfall—and formalizes it mathematically:
1. **The Kernel Distortion Theorem:** Explains why standard SVD solvers introduce non-symmetric coordinate gauges in multi-dimensional null spaces, injecting destructive skew-symmetric noise under projection.
2. **The Spectrum Distortion Theorem:** Explains why non-uniform modifications of Lie algebra singular values violate the structural skew-symmetry relation, causing projection to distort the active spectrum.
By proving these theorems, the paper elevates what could have been seen as a simple empirical failure into a rigorous, general mathematical truth about standard SVD and Lie algebras.

### Pillar C: Symmetry-Preserving Decompositions and Rank-Preserving Mitigations
The paper does not stop at the negative result but introduces three elegant technical innovations to address it:
1. **Real Schur Decomposition:** A projection-free, symmetry-preserving alternative to standard SVD.
2. **Complex Hermitian Solver:** A highly novel, GPU-parallelizable solver utilizing complex eigen-decomposition scaled by $j$, which runs up to $12.2\times$ faster than Schur and $8.1\times$ faster than SVD, resolving the scalability bottleneck of manifold merging.
3. **Rank-Preserving Spectral Pruning (RIMO-Pruned):** A highly effective, low-rank-preserving mitigation that achieves robust performance without test-time calibration.

### Pillar D: Key Insights on Orthogonality, AdaMerging, and Hard Constraints
* **Post-Hoc SVD Projection Collapse:** Shows that attempting to bypass orthogonal pre-training via post-hoc projection collapses performance, establishing that native manifold-respecting models are necessary.
* **AdaMerging Overfitting:** Identifies a critical domain-specific overfitting vulnerability in state-of-the-art AdaMerging during disjoint setups, contrasting it with RIMO-Pruned's native robustness.
* **Hard Orthogonal Constraints:** Introduces a pilot of projected SGD on the Stiefel manifold to completely eliminate residual coordinate warp.

## 3. Verdict on Originality
The originality of this paper is **excellent**. It does not merely combine existing techniques; it systematically analyzes their combination, discovers a fundamental mathematical conflict (the spectral balancing pitfall), derives two new theorems to prove it, and introduces novel geometric mitigations (real Schur, complex Hermitian, and rank-preserving pruning). The combination of a deep diagnostic study with rigorous proofs and elegant engineering mitigations represents a top-tier contribution.
