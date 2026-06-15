# 2. Novelty Check

## Key Novel Aspects & Delta from Prior Work
1. **Curved Subspace Ensembling**: Unlike standard weight merging (Task Arithmetic, TIES-Merging) or activation blending (SABLE) which assume a flat Euclidean geometry, C-Lie-MM explicitly models task-specific projections on the non-Euclidean Grassmannian Manifold $\mathcal{G}(d, D)$. It blends bases along exact Riemannian geodesics instead of linear interpolation.
2. **Continuous Homotopic Flow (Fixed Reference Point)**: To bypass the discontinuous, non-differentiable "argmax jump" of dynamic reference points in standard manifold barycenter calculations, the paper introduces a static reference point $Y_0$ computed offline as the projection-metric Karcher mean. This maintains continuous differentiability ($C^1$) and allows offline pre-computation of the logarithm map.
3. **Chebyshev Polynomial Approximations**: To bypass the high GPU latency and branch-divergence overhead of online sample-wise SVD operations, the paper derives a square-root-free, coordinate-free formulation of the exponential map approximated via Chebyshev polynomials, reducing online execution to standard GEMMs.
4. **Sign Tracking and Regularization**: Introduces continuous tracking-based sign alignment to prevent SVD sign ambiguity and Tikhonov regularization to handle near-singular matrix inversions smoothly.

## Characterization of Novelty
While the mathematical formulation is undeniably rigorous, the overall novelty is **incremental and heavily over-engineered**. 
- The core concept of dynamic routing and activation blending is already established (e.g., SABLE, MoE).
- Applying Riemannian manifold mechanics to neural network representations is a known technique (e.g., federated subspace tracking, Stiefel manifold optimization).
- The "SVD-free" polynomial approximations are only necessary because the authors chose a complex manifold mapping in the first place. This creates a circular design loop: introducing mathematical complexity (manifold ensembling with online SVD) and then adding further engineering complexity (Chebyshev polynomials and a custom Triton GPU kernel) to mitigate the performance bottleneck of that very same choice.
- In realistic deep learning models, standard architectural features like residual connections, LayerNorm, and non-linearities already act as natural buffers against the coordinate collapse that the paper claims to resolve. When these realistic features are present, the empirical advantage of this complex machinery over simple, flat ensembling is modest ($+7.7\%$ in the sandbox and $+0.3\%$ under severe overlap in the baseline comparison). 

Thus, the novelty is conceptually complex but practically marginal, representing an over-engineered solution to a problem that can be largely mitigated by simpler architectural choices (such as residuals/LayerNorm) or temperature tuning in flat models.
