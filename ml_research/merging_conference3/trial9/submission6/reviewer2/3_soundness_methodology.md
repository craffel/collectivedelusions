# Intermediate Evaluation 3: Technical Soundness and Methodology

## Clarity of Description and Mathematical Rigor
The methodology section (Section 3) of the submission is written with an exceptionally high level of mathematical clarity and rigor. The exposition of the Grassmannian manifold $\mathcal{G}(d, D)$ is correct, and the definitions of projection matrices as symmetric, positive semi-definite, and idempotent operators are standard and precise.

The mathematical proofs provided are elegant and technically sound:
- **Proposition 3.1 (Eigenvalue Shrinkage in Flat Blending):** The proof is correct. Expressing the eigenvalue as a Rayleigh quotient and utilizing the properties of orthogonal projection quadratic forms rigorously demonstrates that flat linear blending of non-identical projection matrices shrinks eigenvalues strictly below 1 for any eigenvector outside the subspace intersection.
- **Proposition 3.2 (Exponential Norm Decay in Deep Networks):** The proof is sound, showing that sequential application of flat-blended projection operators act as lossy filters, compounding eigenvalue shrinkage exponentially ($(\lambda_{\max})^L$). This provides a robust, mathematically complete explanation for the "projected coordinate collapse" observed in deep ensembling models.
- **Theorem 3.4 (Manifold Preservation):** The proof is simple and correct, verifying that the output of the exponential map $Y_{\text{merged}}$ maintains orthonormal columns, which guarantees that $P_{\text{merged}} = Y_{\text{merged}} Y_{\text{merged}}^T$ is symmetric, idempotent, and of rank $d$.
- **Theorem 3.5 (Differentiability and Continuity):** The continuity and differentiability ($C^1$) of the C-Lie-MM mapping on the simplex are correctly justified. Because $Y_0$ is fixed, the logarithmic maps are constant, making $H_{\text{merged}}(\alpha)$ linear in $\alpha$. Since the exponential map is differentiable within the injectivity radius ($\|H\|_2 < \pi/2$), the composition is smooth, enabling gradient-based backpropagation.

---

## Evaluation of Specific Methodological Innovations

### 1. Robust Cut-Locus-Aware Logarithm Map (Eq. 14--19)
The standard formulation of the Grassmannian logarithm map is known to degenerate numerically when the target point $Y_1$ has orthogonal components relative to $Y_0$. The proposed closed-form formulation resolves this elegantly. By using the SVD of the inner product matrix $Y_0^T Y_1$ and extracting $M_{\perp} = Y_1 - Y_0(Y_0^T Y_1)$, the left singular vectors $U_{\perp}$ are recovered using the cosecant of the principal angles:
$$U_{\perp} = M_{\perp} U_0 \operatorname{diag}(\csc(\Theta))$$
Setting $\csc(0) = 0$ is numerically stable because the corresponding angle $\theta_i = 0$ also nullifies the component in the final tangent matrix $H$. Clamping the singular values to $[-1+\epsilon, 1-\epsilon]$ is a standard, robust engineering safeguard.

### 2. Projection-Metric (Chordal) Centroid $Y_0$
To avoid non-uniqueness and the expensive iterative solvers (e.g., Karcher flow) of the geodesic Karcher mean, the authors propose a surrogate centroid under the projection (chordal) metric:
$$Y_0 = \arg\max_{Y \in \mathcal{G}(d, D)} \operatorname{Tr}(Y Y^T P_{\text{avg}})$$
where $P_{\text{avg}} = \frac{1}{K}\sum_k V_k V_k^T$. By the Ky Fan theorem, $Y_0$ is retrieved in closed form via the SVD of $P_{\text{avg}}$ (taking its top $d$ eigenvectors). This is a highly appropriate, elegant, and standard solution in matrix manifold statistics.

### 3. SVD Sign Ambiguity Alignment
During end-to-end backpropagation, the SVD's $2^d$ sign ambiguity can cause gradient explosions or step discontinuities. The proposed **continuous tracking-based sign alignment protocol**:
$$s_i^{(t)} = \operatorname{sign}\left((u_i^{(t)})^T u_i^{(t-1)}\right)$$
is mathematically sound and ensures $C^1$ smoothness. The proposed **soft-sign alignment** using $\tanh(\beta x)$ offers an infinitely differentiable ($C^\infty$) transition, completely resolving the non-differentiable boundary where columns rotate near perfect orthogonality.

### 4. SVD-Free Chebyshev Polynomial Approximation (Eq. 23--26)
This is one of the strongest practical contributions. The authors show that the square root $\sqrt{H^T H}$ never needs to be evaluated because the power series expansions of $\cos(X)$ and $X^{-1}\sin(X)$ contain only even powers of $X$, which translate directly into integer powers of the symmetric matrix $M = H^T H \in \mathbb{R}^{d \times d}$.
Using Chebyshev polynomials to approximate these functions provides guaranteed uniform convergence and minimal approximation error over the injectivity radius, bypassing online SVD entirely.

---

## Potential Mathematical Nuances and Constructive Clarifications

### 1. Zero-Tangent Sum at the Centroid
The authors claim that under uniform routing weights ($\alpha_k = 1/K$), the merged tangent matrix approaches zero because "the sum of tangent vectors at the Karcher mean is approximately zero ($\sum_k H_k \approx 0$)."
*   **Scholarly Nuance:** While $\sum_k H_k = 0$ holds *exactly* as the first-order optimality condition for the true **geodesic Karcher mean**, it only holds *approximately* for the **projection-metric (chordal) centroid** $Y_0$. For highly overlapping manifolds, this approximation is extremely tight. However, the authors should add a brief mathematical footnote clarifying that this zero-sum property is an approximation under the chordal surrogate, although its practical effect in driving $H_{\text{merged}} \to 0$ remains highly robust.

### 2. Sectional Curvature Bounds on the Grassmannian
In Proposition 3.3, the authors state that the sectional curvatures of the Grassmannian are bounded in $[0, 2]$, leading to a maximum sectional curvature of $\kappa = 2$.
*   **Scholarly Nuance:** For the Grassmannian $\mathcal{G}(d, D)$ equipped with its standard Riemannian (canonically scaled) metric, the sectional curvature bounds are indeed $[0, 2]$ when $d \ge 2$ and $D-d \ge 2$. Under $d=1$ (which is the real projective space $\mathbb{R}\text{P}^{D-1}$), the sectional curvature is constant and equal to $1$. The authors should briefly clarify that these bounds assume $d \ge 2$ and $D-d \ge 2$, which is the standard multi-dimensional subspace setting under LoRA.

---

## Reproducibility Assessment
The paper demonstrates an exceptionally high standard of reproducibility:
1. **Analytical Sandbox Specifications:** The authors provide exact dimensions ($D=192$, $d=8$, $L=14$), tasks ($K=4$), overlaps (0 and 12), and training configurations, allowing direct replication.
2. **Pseudocode & Algorithms:** The mathematical descriptions of the Cut-Locus-Aware Logarithm, the projection centroid, and the Chebyshev polynomial approximation are written as step-by-step constructive procedures.
3. **PEFT Integration Blueprint:** Section 4.5 provides a clear, actionable guide on how to extract projection bases from standard LoRA weights ($W = W_{\text{base}} + B A$) by SVD-ing $A^T$, facilitating direct integration into libraries like Hugging Face PEFT.
