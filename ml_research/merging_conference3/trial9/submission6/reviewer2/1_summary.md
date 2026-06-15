# Intermediate Evaluation 1: Summary of the Paper

## Main Topic and Domain
The submission operates in the domain of **model merging and parameter ensembling** (specifically within Multi-Task Learning and Parameter-Efficient Fine-Tuning/PEFT). It focuses on the ensembling of representation-space low-rank projection operators (which filter noise, align representations, or compress states) in deep neural networks. 

The paper identifies a fundamental mathematical and empirical flaw in traditional "flat" ensembling methods (e.g., simple weight averaging or activation blending like SABLE): when ensembling low-rank projection operators, flat linear combinations violate the underlying geometry of the projection manifold, resulting in eigenvalue shrinkage. In deep sequential architectures, this shrinkage accumulates exponentially, leading to a phenomenon termed **projected coordinate collapse**, which decays representation norms to zero and collapses accuracy.

## Proposed Approach (C-Lie-MM)
To resolve this, the authors propose **Continuous Riemannian-Geometric Homotopical Model Merging via Grassmannian Geodesic Blending (C-Lie-MM)**. The key technical elements of the proposed framework include:
1. **Grassmannian Representation:** Treats each task-specific projection basis (an orthonormal matrix $V_k \in \mathbb{R}^{D \times d}$ representing a $d$-dimensional subspace) as a point on the Grassmannian manifold $\mathcal{G}(d, D)$.
2. **Fixed Reference Centroid:** Instead of dynamically choosing a reference point (which introduces step-function discontinuities, rendering gradients zero almost everywhere), C-Lie-MM computes a static offline reference point $Y_0$ under the projection (chordal) metric. This surrogate Karcher mean is computed once per epoch (or offline) as the span of the top $d$ eigenvectors of the average projection matrix $P_{\text{avg}} = \frac{1}{K}\sum_k V_k V_k^T$.
3. **Logarithmic Tangent Mapping:** Maps all expert bases $\{V_k\}_{k=1}^K$ onto the tangent space of $Y_0$ using a robust, cut-locus-aware Grassmannian logarithm map ($\log_{Y_0}(V_k)$). This is done entirely offline.
4. **Online Tangent Blending:** Linearly blends these tangent matrices in the flat tangent space $T_{Y_0}\mathcal{G}(d, D)$ using sample-wise routing weights $\alpha_k(z)$ governed by a temperature-calibrated Gibbs routing policy:
   $$H_{\text{merged}}(\alpha) = \sum_{k=1}^K \alpha_k(z) H_k$$
5. **Exponential Manifold Projection:** Maps the blended tangent matrix back onto the manifold via the Grassmannian exponential map:
   $$Y_{\text{merged}} = \exp_{Y_0}(H_{\text{merged}})$$
   This guarantees that the resulting merged projection operator $P_{\text{merged}} = Y_{\text{merged}} Y_{\text{merged}}^T$ is strictly symmetric, idempotent ($P_{\text{merged}}^2 = P_{\text{merged}}$), and of rank $d$, completely eliminating coordinate collapse.

## Key Findings and Claims
1. **Empirical Verification of Collapse:** In a 14-layer Analytical Coordinate Sandbox simulation ($D=192$, $d=8$, $K=4$ tasks), traditional Uniform Merging collapses to exactly $25.00\% \pm 0.00\%$ accuracy (random guessing on 4 classes) under severe manifold overlap (overlap=12) due to exponential representation decay.
2. **Elimination of Coordinate Collapse:** C-Lie-MM completely resolves coordinate collapse, achieving $71.00\% \pm 2.37\%$ accuracy on orthogonal manifolds and $70.30\% \pm 4.01\%$ accuracy on highly overlapping manifolds.
3. **Immunity to Heterogeneity Collapse:** Because ensembling coefficients and the barycenter are evaluated sample-wise during the forward pass, C-Lie-MM maintains identical high performance under both Homogeneous and Heterogeneous mixed-task deployment streams, unlike static parameter-level merging.
4. **Out-of-Distribution (OOD) Robustness:** Under high routing uncertainty, routing weights distribute uniformly ($\alpha_k \to 1/K$). The ensembled tangent matrix converges to the Karcher mean centroid ($H_{\text{merged}} \to 0$), causing the exponential map to project representations directly onto $Y_0$, preserving shared semantic features.
5. **Practical Serving and Polynomial Approximation:** The paper presents a coordinate-free formulation of the exponential map that bypasses online SVD via Chebyshev polynomial expansions (order $M=6$). This reduces the operation to hardware-accelerated GEMMs, delivering a $12.6\times$ speedup and making C-Lie-MM highly viable for real-time edge serving.
6. **Varying Rank and Sign Ambiguity Solutions:** Proposes padding (subspace expansion) or truncation (subspace compression) to handle varying expert ranks, and a continuous tracking-based sign alignment wrapper to stabilize SVD gradients under joint backpropagation.
7. **Simulated GLUE LoRA Benchmark:** On a high-fidelity sequential simulation scaled to RoBERTa-Large dimensions ($D=1024$), C-Lie-MM achieves near-oracle multi-task performance ($97.0\%$), while traditional parameter-merging (Task Arithmetic, TIES-Merging, SABLE) collapses to $49.8\%$--$55.0\%$ due to cumulative representation distortion.

## Explicit Claimed Contributions (with Evidence in Paper)
- **Contribution 1: Mathematical formulation of coordinate collapse.** (Evidence: Section 3.2, Propositions 3.1 and 3.2. Proves eigenvalue shrinkage and exponential norm decay under flat ensembling).
- **Contribution 2: The C-Lie-MM framework with a fixed reference point.** (Evidence: Section 3.3 and 3.4. Formulates the offline-online split, cut-locus logarithm, and proves manifold preservation in Theorem 3.4).
- **Contribution 3: Empirical validation in an Analytical Sandbox.** (Evidence: Section 4.2, Table 1. Demonstrates accuracy curves, prevents coordinate collapse, and tracks the idempotency error $\Delta_{\text{idem}} \approx 10^{-7}$).
- **Contribution 4: Complete immunity to heterogeneity collapse.** (Evidence: Section 4.2, Table 1. Shows identical results in Homo and Hetero columns).
- **Contribution 5: Practical deployment solutions.** (Evidence: Section 3.8, Section 4.3. Validates SVD-free Chebyshev polynomials, SVD sign tracking, selective layer application, and varying ranks under heterogeneous settings).
- **Contribution 6: Validation on high-fidelity Simulated GLUE.** (Evidence: Section 4.4, Table 2. Shows C-Lie-MM outperforming flat ensembling by $+42.0\%$).
