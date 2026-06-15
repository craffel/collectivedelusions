# 1. Summary of the Paper

This paper presents an investigation into performing model merging on the manifold of the orthogonal group $\mathrm{O}(d)$ and its Lie algebra tangent space $\mathfrak{so}(d)$. While Euclidean parameter averaging methods (like Task Arithmetic) are standard in the literature, the authors argue they can cause representation drift. They evaluate the feasibility of translating Euclidean spectral-balancing techniques (like SAIM) to curved manifolds.

To do this, they analyze a pipeline called **RIMO** (Riemannian Isometry-respecting Manifold Operations), which performs Singular Value Decomposition (SVD) on skew-symmetric Lie algebra generators to balance rotation magnitudes.

### Key Findings and Contributions
1. **The Orthogonality Condition**: Manifold-level merging (OrthoMerge/RIMO) is highly sensitive to parameter non-orthogonality. Merging models trained with soft orthogonal regularization ($\lambda_{ortho} = 2.0$) achieves $84.55\%$ accuracy, whereas merging standard unconstrained models collapses performance to $42.07\%$. Naive post-hoc SVD projection of standard models onto $\mathrm{O}(d)$ is highly destructive ($15.00\%$).
2. **The Tangent Space Spectral Pitfall**: Modifying the singular spectrum in the Lie algebra tangent space $\mathfrak{so}(d)$ via SVD-based isotropic balancing ($t > 1.0$) catastrophically scrambles representations, dropping accuracy to $13.66\%$ (MLP) and $18.44\%$ (Vision Transformer).
3. **Theoretical Formalization**: The authors prove via the *Kernel Distortion Theorem* and *Spectrum Distortion Theorem* that standard SVD solvers introduce non-symmetric coordinate gauges in multi-dimensional null spaces. The subsequent projection required to restore skew-symmetry inevitably distorts the spectrum. Under the non-linear Cayley map, inflating small singular values propagates as massive, high-dimensional rotational noise across inactive dimensions.
4. **Proposed Mitigations**:
   - *Real Schur Decomposition* (RIMO-Schur-Balanced/Pruned) and a *Complex Hermitian Solver* (RIMO-Complex-Balanced/Pruned) are proposed to perform symmetry-preserving, projection-free spectral modifications.
   - *Rank-Preserving Spectral Pruning* (RIMO-Pruned) is introduced to keep inactive dimensions at exactly zero singular value, bypassing the noise injection and recovering robust merged accuracies ($91.49\%$ on MLP, $88.16\%$ on ViT).
5. **Evaluation**: Empirical tests on Split-MNIST (using MLP and a custom micro-ViT) and Split-CIFAR-10.
