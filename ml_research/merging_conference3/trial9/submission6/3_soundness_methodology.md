# 3. Soundness and Methodology Check

## Mathematical Rigor and Soundness
The paper is exceptionally sound and mathematically rigorous. The theoretical claims are backed by solid proofs and formal derivations:

1. **Eigenvalue Shrinkage and Norm Decay (Propositions 3.1 & 3.2):**
   The proof that flat blending of projection matrices shrinks eigenvalues below 1 is mathematically correct. The quadratic form analysis using the Rayleigh quotient is clean. The subsequent derivation of exponential norm decay in deep sequential networks logically and rigorously demonstrates why flat blending leads to representation decay.

2. **Manifold Preservation (Theorem 3.3):**
   The proof that C-Lie-MM yields a mathematically correct, orthogonal projection matrix of rank $d$ ($P_{\text{merged}} = Y_{\text{merged}} Y_{\text{merged}}^T$) is correct. Since $Y_{\text{merged}}$ is guaranteed to have orthonormal columns via the exponential map, $P_{\text{merged}}$ is symmetric, positive semi-definite, idempotent ($P_{\text{merged}}^2 = P_{\text{merged}}$), and has exactly $d$ eigenvalues equal to 1, maintaining strict adherence to the Grassmannian manifold $\mathcal{G}(d, D)$.

3. **Differentiability and Continuity (Theorem 3.4):**
   The paper correctly proves that using a fixed reference point $Y_0$ restores differentiability ($C^1$ smoothness) and continuity. By detaching $Y_0$ from the gradient graph and updating it block-coordinate-wise at epoch boundaries (or via EMA), the framework bypasses the SVD gradient explosion issue. The discussion on SVD sign-tracking and the differentiability of the sign function under infinitesimal rotations is thorough. The soft-sign tracking alternative ($s_{i,\text{soft}} = \tanh(\beta \langle u_i^{(t)}, u_i^{(t-1)} \rangle)$) provides absolute mathematical completeness for gradient flow.

4. **Tangent Space Metric Distortion (Proposition 3.3):**
   The metric warp analysis is mathematically advanced, utilizing the Rauch Comparison Theorem and Jacobi fields to derive a strict theoretical bound on the distance distortion between the curved manifold and the flat tangent space under maximum sectional curvature of $\kappa = 2$:
   $$\left( \frac{\sin(\sqrt{2}\theta_{\max})}{\sqrt{2}\theta_{\max}} \right) \|H_a - H_b\|_F \le d_{\mathcal{G}}(V_a, V_b) \le \|H_a - H_b\|_F$$
   This provides a rigorous justification for choosing the Karcher mean (projection-metric centroid) as the central coordinate reference point $Y_0$, as it mathematically minimizes the maximum geodesic distance $\theta_{\max}$, thereby minimizing metric distortion.

5. **Chebyshev Polynomial Approximation (Proposition 3.5):**
   The derivation of uniform error bounds for Chebyshev approximations on the interval $[0, \pi/2]$ and their sub-linear ($O(\sqrt{d})$) scaling with respect to subspace rank $d$ is correct. The proof demonstrates that order $M=6$ achieves double-precision machine epsilon accuracy while bypassing online SVD.

## Key Assumptions and Potential Weaknesses
- **Assumption 3.1 (Latent Subspace Non-Orthogonality):**
  The assumption that task experts do not lie on the cut locus of $Y_0$ (i.e., they are not perfectly orthogonal, $\theta_i < \pi/2$) is realistic for PEFT adapters fine-tuned from a shared backbone. To ensure numerical safety under extreme edge cases, the authors incorporate symmetric positive-definite Tikhonov regularization:
  $$ (A^T A + \epsilon I_d)^{-1} A^T $$ (for $A = Y_0^T V_k$), which guarantees a non-singular, smooth numerical transition.
- **Heterogeneous Subspace Ranks ($d_k$):**
  The Grassmannian manifold requires a fixed rank $d$. In real-world multi-task setups, different adapters can have different ranks. The authors address this via two proposed strategies: zero-padding (subspace expansion to $d_{\max}$) and spectral truncation (subspace compression to $d_{\min}$). While these are sound, they add structural complexity when deployed in diverse, unconstrained environments. The authors provide a robust empirical validation of these strategies in their sandbox, demonstrating their stability and performance.
- **Geodesic Convexity Boundaries of the Karcher Mean:**
  The authors acknowledge and analyze the geodesic convexity boundary of the Karcher mean on the Grassmannian. Since the sectional curvature is bounded in $[0, 2]$, the convexity radius is $r < \frac{\pi}{2\sqrt{2}} \approx 1.11$ radians. Under highly dissimilar or nearly orthogonal expert subspaces, the principal angles can exceed this, risking local minima. The authors formally justify that their offline SVD-based centroid under the projection metric ($P_{\text{avg}} = \frac{1}{K}\sum_k V_k V_k^T$) serves as an exceptionally stable, robust, and computationally cheap surrogate that treats all experts with perfect symmetry and completely bypasses these local minima and uniqueness issues.
