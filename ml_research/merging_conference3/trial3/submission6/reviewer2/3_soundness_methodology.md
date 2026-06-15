# 3. Soundness and Methodology

## Clarity of the Description
The methodology is exceptionally well-written, structured, and presented with a high level of mathematical rigor. The progression from problem formulation to Taylor expansion, subspace projection, closed-form analytical solution, and finally finite-difference estimation is logical and easy to follow. Important equations are derived from first principles, and all symbols are clearly defined.

## Appropriateness of Methods
The proposed methods are highly appropriate and address the key bottlenecks of model merging from first principles:
- **Subspace projection** is an elegant, mathematically sound method to bypass the curse of dimensionality. It leverages the fact that task vectors define the active directions of interest, allowing the full cross-parameter Hessian to be computed within a tractable $K \times K$ space.
- **Gradient Subtraction** is highly appropriate for real-world scenarios. By explicitly subtracting the unperturbed expert gradient, it prevents the amplification of optimization noise during finite-difference estimation, which is a major issue in partially converged models.
- **Global Normalization (ACM-GlobalNorm)** successfully balances different task losses while preserving the relative sensitivity profile across layers.
- **Lasso Regularization via Proximal ISTA Updates** is a mathematically rigorous solution to address numerical ill-conditioning in low-parameter layers (like LayerNorm), driving non-essential coefficients to exactly zero.

## Potential Technical Flaws & Crucial Assumptions
The authors are remarkably honest and thorough in identifying, analyzing, and discussing the limitations and potential technical flaws of their own method:

1. **The Layer Block-Diagonal Hessian Assumption (Assumption 3.1):**
   - *Assumption:* Cross-layer second-order interactions are negligible.
   - *Analysis:* In deep architectures, layers are highly coupled. The authors identify a **Block-Jacobi Coupling Mismatch**, where performing layer-wise independent updates under coupled simultaneous perturbations introduces projection errors.
   - *Investigation:* To resolve this, they implemented a sequential block Gauss-Seidel coordination descent (Equation 28). They discovered that sequential updates cascade representation drift and cause **Hessian Reference-Point Collapse** (where upstream modifications shift the activation manifold, invalidating downstream Hessians), leading to a performance drop (36.65%). This deep analysis proves the block-diagonal assumption is necessary to maintain stability in a single-step (Block-Jacobi) solution.

2. **Ill-Conditioning in Low-Parameter Layers:**
   - *Assumption:* The projected Hessian $A^l$ is well-conditioned.
   - *Analysis:* In layers with very few parameters, such as the final LayerNorm (Layer 13, with only 384 parameters), the projected Hessian matrix $A^{13}$ is extremely ill-conditioned (condition number $>10^4$). This causes massive coefficient blowups (e.g., SVHN solved to $-95.691$). 
   - *Mitigation:* The authors honestly characterize this blowup as a numerical artifact of ill-conditioning and successfully resolve it by proposing L1-regularized Lasso ACM solved via proximal gradient (ISTA) updates, proving that mathematical regularizers can stabilize the system.

3. **The Local-Global Gap on Non-Convex Manifolds:**
   - *Assumption:* The local quadratic Taylor expansion remains valid across large distances in the parameter space.
   - *Analysis:* On highly non-convex, fully converged physical manifolds, the Taylor remainder scales cubically ($O(V_{\max}^3)$) with task vector magnitude. Because merged models lie far from any individual expert, the quadratic surrogate breaks down. This explains why unguided Task Arithmetic (tuned to 0.4) performs comparably or slightly better than ACM on physical ViT-Tiny (60.72% vs. 57.76% for ACM-GlobalNorm and 60.89% for Vanilla ACM).

## Reproducibility
The work meets a very high standard of reproducibility:
- The authors provide full details of the hyperparameter sweeps (Ridge candidate range, Lasso candidate range, static scale factor candidate range).
- They explain the calibration/validation split heuristic used to select hyperparameters (24 calibration samples, 8 validation samples), preventing test-set leakage.
- Appendix A contains the full procedural algorithm of ACM, and Appendix C describes implementation details, making it straightforward for researchers to implement.
- No "hidden" or heuristic tricks are used; all derivations are fully disclosed.
