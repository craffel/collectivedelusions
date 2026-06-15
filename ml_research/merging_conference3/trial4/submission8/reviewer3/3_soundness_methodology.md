# Soundness and Methodology Evaluation

## Clarity of the Description
The methodology is exceptionally clear, with every mathematical term, variable, and mapping precisely defined. The authors provide a complete step-by-step description of the CR-PolySACM adaptation loop, detailing the exact formulation of:
- The polynomial depth-dependent mapping (Eq. 2)
- The high-dimensional quantization noise decomposition (Eq. 5)
- The Taylor expansion-based multi-task loss decomposition (Eq. 12)
- The clipping-regularized scale normalization (Eq. 14, 15, 16)
- The two-step sharpness-aware parameter update process.

---

## Appropriateness of Methods
From a theoretical perspective, the methods are highly appropriate and elegant:
- **Subspace Constraint:** Restricting the search space using a depth-dependent polynomial manifold is a mathematically rigorous way to introduce a smooth spatial inductive bias, directly countering the high-dimensional overparameterization and local overfitting problems of standard test-time adaptation.
- **Flatness Regularization:** Minimizing loss landscape sharpness is the correct theoretical mechanism to combat post-training quantization rounding noise, as the quadratic loss increase under weight perturbations is bounded by the eigenvalues of the Hessian matrix.
- **Scale-Invariant Perturbation (CR-SACM):** Scaling perturbations inversely by $(V_k^l)^2$ (via gradient scaling and normalization) is the mathematically correct way to achieve uniform weight-space perturbations across layers, and the clipping threshold $\beta$ is a highly effective way to prevent numerical singularity and gradient explosion.

---

## Potential Technical Concerns & Theoretical Analysis

As a theory-minded reviewer, we analyze the core assumptions and potential technical risks:

1. **The Gauss-Newton Hessian Approximation:**
   The paper assumes $\mathcal{H}_{\mathbf{p}} \approx J_{\mathbf{p}}^T \mathcal{H}_W J_{\mathbf{p}}$, which neglects the term $\sum_i \nabla_W \mathcal{L}_i \nabla^2_{\mathbf{p}} W_{\text{merged}, i}(\mathbf{p})$. Because $W_{\text{merged}}^l(\mathbf{p})$ is a non-linear function of $\mathbf{p}$ (due to the logistic sigmoid mapping $\sigma$), the second-order derivative $\nabla^2_{\mathbf{p}} W_{\text{merged}}$ is non-zero and scales with $\sigma''(\cdot)$. However, because the optimization is conducted for a short horizon ($T=40$) and the coefficients remain within the stable, active interior region ($[0.18, 0.81]$) where $\sigma''$ is strictly bounded, this term is indeed negligible, validating the Gauss-Newton approximation.

2. **Subspace Orthogonality & Coupling Assumptions:**
   In Eq. 12, the cross term $\boldsymbol{\epsilon}^T J_{\mathbf{p}}^T \mathcal{H}_W \delta_{\perp}$ is omitted. This assumes that the low-dimensional subspace spanned by the task vectors is weakly coupled with the massive $D$-dimensional orthogonal complement space. In deep neural networks, because the task-vector subspace has extremely low rank ($3K = 12$) relative to the parameter space ($D \approx 5.7\times 10^6$), the random projection of the orthogonal noise $\delta_{\perp}$ has near-zero alignment with the Jacobian column space. Thus, the coupling is theoretically negligible, justifying this omission.

3. **Sigmoid Saturation & Parameter Freezing:**
   Because gradients scale with $\sigma'(\cdot) = \lambda_k^l (1 - \lambda_k^l)$, if any coefficient converges to the boundaries (0.0 or 1.0), its update would freeze. The authors address this in Section 3.1 and Appendix A, proving that adjacent layer polynomial constraints act as a global structural regularizer that naturally keeps individual layer coefficients within the active interior range ($[0.18, 0.81]$), preventing parameter freezing.

4. **Sum-to-One Normalization & Weight Scale Inflation:**
   Standard merging uses a sum-to-one constraint ($\sum \lambda_k = 1$), whereas CR-PolySACM allows independent $[0,1]$ scaling, meaning the scale of the merged weights is not strictly bounded. The authors demonstrate that the unsupervised entropy minimization loss $\mathcal{L}_{\text{entropy}}$ naturally acts as an implicit scale regularizer: if coefficients scale excessively, the resulting activation drift disrupts normalization statistics and triggers an extreme spike in prediction entropy, which is heavily penalized by the gradient. This is backed by empirical measurements showing that the average coefficient sum stabilizes at $1.42 \pm 0.04$, far below the maximum limit of $4.0$.

---

## Reproducibility
The work is highly reproducible. The authors explicitly state all relevant hyperparameters and optimization settings:
- Optimization steps: $T=40$
- Learning rate: $\eta = 10^{-2}$
- Perturbation radius: $\rho = 0.05$
- Clipping threshold: $\beta = 0.10$
- Calibration data size: $N=64$ (with $B=16$ samples per task)
- Backbone architecture: Vision Transformer (`vit_tiny_patch16_224` from `timm`)
- Individual expert training details, convergence ceilings, and 6 evaluation quantization schemas.
This comprehensive disclosure ensures that any researcher can easily replicate the findings.
