# 3. Soundness and Methodology

### Technical Rigor
The mathematical formulations, theorems, and proofs presented in the paper and its appendix are highly rigorous. The authors correctly identify the group properties, Lie algebraic structures, and the behavior of numerical SVD solvers. The proofs for Proposition 3.1 (closure), Theorem 3.2 (Kernel Distortion), and Theorem 3.3 (Spectrum Distortion) are mathematically sound and correct.

### Methodological Concerns & Over-Engineering
From an architectural and engineering perspective, the methodology is excessively complex and heavily over-engineered:
1. **Unnecessary Complexity**: The merging pipeline requires a staggering number of complex steps:
   - Extract orthogonal rotations via Orthogonal Procrustes decoupling (requires an SVD).
   - Project rotations to the Lie algebra tangent space via the inverse Cayley transform.
   - Compute magnitude-corrected Lie algebra averages.
   - Perform a second SVD, real Schur decomposition, or complex Hermitian eigen-decomposition on the aggregated generator.
   - Selectively prune singular values (rank pruning).
   - Reconstruct and project back to ensure skew-symmetry.
   - Retract back to the Lie group via the forward Cayley transform.
   - Average linear residuals in Euclidean space.
   - Reconstruct final weights.
2. **Special Training Requirements**: For this manifold-level merging to work at all, the model parameters must be kept near the orthogonal manifold during fine-tuning. This requires training the models with either:
   - A soft orthogonal regularization constraint ($\lambda_{ortho} = 2.0$) in the loss function, which restricts representational capacity and complicates hyperparameter tuning.
   - Or a hard orthogonal constraint (Riemannian optimization on the Stiefel manifold or Cayley parameterizations), which significantly increases training-time optimization difficulty and sensitivity.
   - Standard unconstrained models cannot be merged this way; the post-hoc SVD projection collapses performance.

This creates a highly fragile pipeline where both the pre-training/fine-tuning phases and the post-hoc merging phase are heavily burdened by geometric constraints and complex matrix decompositions.
