# 1. Summary of the Paper

## Main Topic
The paper addresses the challenge of **model merging**, specifically focusing on parameter-space operations that attempt to combine multiple task-specific expert models without joint retraining. While most existing model merging techniques (such as Task Arithmetic, TIES-Merging, and DARE) operate in flat Euclidean space, they are prone to "representation drift." To preserve representational structure, geometric model merging methods constrain parameter updates to curved manifolds like the orthogonal group $\mathrm{O}(d)$. 

This work presents a diagnostic and theoretical investigation into the **limits of translating representational isotropy techniques (specifically spectral balancing, such as SAIM) from Euclidean spaces to curved manifolds**, using the orthogonal group $\mathrm{O}(d)$ and its Lie algebra tangent space $\mathfrak{so}(d)$ as a diagnostic environment.

## Proposed Approach
The authors formalize a framework called **RIMO** (Riemannian Isometry-respecting Manifold Operations) to merge models on the manifold of $\mathrm{O}(d)$ and its Lie algebra tangent space $\mathfrak{so}(d)$. The pipeline consists of:
1. **Orthogonal Procrustes Decoupling:** Decoupling unconstrained weights into an orthogonal rotation $R_k \in \mathrm{O}(d)$ and a linear residual component $\rho_k$.
2. **Tangent Projection:** Mapping the rotations to the Lie algebra tangent space $\mathfrak{so}(d)$ via the inverse Cayley transform to obtain skew-symmetric generators $Q_k$.
3. **Aggregation:** Linearly combining generators in $\mathfrak{so}(d)$ with a Frobenius norm magnitude correction.
4. **Spectral Balancing (RIMO):** Performing Singular Value Decomposition (SVD) on the aggregated generator, shifting the singular value spectrum towards its mean to restore isotropy (similar to SAIM), reconstructing, and projecting back to skew-symmetry.
5. **Spectral Pruning (RIMO-Pruned):** An alternative mitigation that keeps the top singular value pairs and zeroes the remaining inactive ones.
6. **Retraction and Re-assembly:** Mapping back to $\mathrm{O}(d)$ via the forward Cayley transform, averaging linear residuals, and reconstructing the final merged weights.

The authors also introduce symmetry-preserving decompositions—**Real Schur Decomposition** and a **Complex Hermitian Solver**—as alternatives to standard SVD to avoid numerical projection errors.

## Key Findings
1. **Sensitivity to the Orthogonality Condition:** Riemannian model merging requires native, manifold-respecting models. If individual experts are not regularized during training, Procrustes decoupling yields extremely large residuals, acting as unconstrained Euclidean noise that collapses accuracy. Soft orthogonal regularization ($\lambda_{ortho} = 2.0$) stabilizes merging, whereas post-hoc SVD projection of an unconstrained base model collapses performance ($15.00\%$), proving to be functionally destructive.
2. **The Tangent Space Spectral Balancing Pitfall:** Standard spectral balancing (inflating smaller singular values to force isotropy) in Lie algebra tangent spaces catastrophically collapses accuracy ($13.66\%$ on MLP, $18.44\%$ on ViT).
3. **Mathematical Explanations of the Pitfall:**
   - **Kernel Distortion Theorem (Theorem 3.2):** Standard numerical SVD solvers introduce non-symmetric coordinate gauges in the multi-dimensional null space, and subsequent skew-symmetric projection injects destructive rotational noise.
   - **Spectrum Distortion Theorem (Theorem 3.3):** Non-uniform singular value modifications violate the compatibility relation $R \Sigma = -\Sigma R^T$, meaning post-decomposition skew-symmetric projection inevitably distorts the spectrum.
   - **Cayley Mapping Noise Propagation:** Under the non-linear forward Cayley map, inflating small or zero singular values in the tangent space translates to spurious, non-zero rotations across thousands of inactive high-dimensional planes, scrambling representation alignment.
4. **Quadratic Scaling of the Pitfall:** The noise power from inflating inactive planes scales quadratically with the network's representation dimension $d$, making the pitfall worse for large-scale models.
5. **Mitigation via Pruning:** Setting inactive singular values to exactly zero (**RIMO-Pruned**) bypasses the pitfall, recovering robust accuracies ($91.49\%$ on MLP and $88.16\%$ on ViT) while smoothing active components.

## Explicitly Claimed Contributions (with Evidence)
1. **Rigorous study of the limits of representational isotropy on curved manifolds**, revealing and detailing the tangent-space spectral balancing pitfall in Lie algebras (supported by mathematical derivation and Split-MNIST experiments).
2. **Mathematical formalization of the SVD-projection incompatibility** via the *Kernel Distortion Theorem* and *Spectrum Distortion Theorem*, along with the implementation of **real Schur decomposition** and a parallel **complex Hermitian solver** (supported by proofs in Appendix A, CPU/GPU latency benchmarks in Section 4.5, and accuracy results in Tables 1 & 2).
3. **Demonstration of the necessity of native manifold training** and the failure of post-hoc SVD projection (supported by empirical analysis of residual norms and accuracies in Section 4.3).
4. **Evaluation of SOTA test-time adaptive merging (AdaMerging)** under disjoint settings, revealing its overfitting tendencies, and demonstrating that **RIMO-Pruned** is a robust alternative (supported by comparative results in Tables 1 & 2, and Section 4.6).
