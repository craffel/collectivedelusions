# Soundness and Methodology Evaluation

## 1. Technical Soundness of Theoretical Claims

The paper is exceptionally strong and rigorous in its theoretical analysis. Let us evaluate the three core theoretical proofs provided in the appendix:

### Proposition 3.1: Algebraic Closure of the Inverse Cayley Transform
* **Statement:** For any orthogonal $R \in \mathrm{O}(d)$ with $\det(R + I_d) \ne 0$, the inverse Cayley transform $Q = (R - I_d)(R + I_d)^{-1}$ is skew-symmetric ($Q^T = -Q$).
* **Review:** The proof is fully correct. It utilizes the transpose properties and the fact that since $I+R$ and $I-R$ are rational functions of the same matrix $R$, they commute. Additionally, $R$ commutes with $(I+R)^{-1}$ and $(I-R)$. The derivation is clean, standard, and mathematically sound.

### Theorem 3.2: Kernel Distortion Theorem
* **Statement:** SVD-based inflation of the kernel components of $Q_{\text{com}}$ combined with skew-symmetric projection $\mathcal{P}$ injects skew-symmetric noise of the form $\frac{\hat{\sigma}}{2} V_{\text{kernel}} (P - P^T) V_{\text{kernel}}^T$, where $P \in \mathrm{O}(m)$ is an arbitrary orthogonal gauge transformation.
* **Review:** The proof is highly elegant and correct. Since $U_{\text{kernel}}$ and $V_{\text{kernel}}$ both form orthonormal bases for the same kernel space $\mathcal{K}$, the existence of an orthogonal matrix $P \in \mathrm{O}(m)$ such that $U_{\text{kernel}} = V_{\text{kernel}} P$ is guaranteed by fundamental linear algebra. The projection is zero if and only if $P$ is symmetric. In numerical LAPACK libraries (like those underpinning PyTorch's SVD solver), left and right singular vectors for the null space are computed independently or via arbitrary orthonormal bases, meaning $P$ is almost never symmetric in practice for $m \ge 2$. Therefore, SVD-based inflation of inactive dimensions injects spurious skew-symmetric noise. This is mathematically flawless.

### Theorem 3.3: Spectrum Distortion Theorem
* **Statement:** Non-uniform singular value modifications of $Q$ in $\mathfrak{so}(d)$ under standard SVD followed by projection $\mathcal{P}$ distort the singular value spectrum unless the compatibility condition $R \hat{\Sigma} = -\hat{\Sigma} R^T$ is met, which is violated for $t > 1.0$ unless $R$ is a pure block-rotation of angle $\theta = \pi/2$.
* **Review:** This is a spectacular proof. The author shows that a real skew-symmetric matrix satisfies $R \Sigma = -\Sigma R^T$ for its left/right singular bases relationship $R = U^T V$. Under a non-uniform modification (such as the isotropic smoothing $\hat{\Sigma}$), this relationship is violated unless $R^T = -R$, which restricts $R$ to have eigenvalues of exactly $\pm j$ (a rotation of exactly $\pi/2$ in every plane). For any general model updates, this condition is violated, proving that SVD-based modifications are structurally incompatible with Lie algebras and that post-hoc projection inevitably distorts the spectrum. This is mathematically solid and extremely insightful.

---

## 2. Methodology Gaps and Areas of Criticism

While the mathematical proofs are solid, we identify several minor theoretical and numerical gaps in the methodology:

### Gap A: Singularity of the Inverse Cayley Transform
The inverse Cayley transform is defined if and only if $\det(R + I_d) \ne 0$. If $R$ has an eigenvalue of $-1$ (which corresponds to a rotation of exactly $\pi$ radians in some plane), the matrix $R + I_d$ is singular, and the transform cannot be computed.
* **Criticism:** The paper does not address how it handles the case where $R_k + I_d$ is near-singular. While OFT/OrthoMerge typically operate with weight matrices close to the identity (where eigenvalues are close to $1$), unconstrained standard models or heavily fine-tuned experts could occasionally have eigenvalues close to $-1$, leading to numerical instability or division by zero during inversion.
* **Constructive Suggestion:** The authors should discuss standard numerical safeguards, such as adding a tiny identity pertubation $R_k + (1 + \epsilon) I_d$, or acknowledge this as a limitation of the Cayley parameterization compared to the matrix exponential/logarithm map (which is defined everywhere on $\mathrm{O}(d)$ and maps to $\mathfrak{so}(d)$ without singularities).

### Gap B: Block-Diagonal Boundaries on Rectangular Weights
The authors apply block-diagonal operations on square sub-matrices of size $b \times b$ to handle rectangular weight matrices.
* **Criticism:** While block-diagonal partitioning is computationally efficient, it restricts the degrees of freedom of the orthogonal transformation to localized sub-spaces. The authors state that this "introduces zero coordinate clipping or boundary distortions," but they should more clearly discuss whether this block-diagonal boundary restriction affects the representational capacity of the model, especially when merging models trained on highly divergent tasks.

### Gap C: Residual Scale Sensitivity
The residual components are averaged as $\rho_{\text{merged}} = \frac{s}{N} \sum \rho_k$. In the results, the authors tune $s$ (as $\rho_{\text{scale}}$) to $0.2$ for the standard unconstrained regime and $1.0$ for the regularized regime.
* **Criticism:** The need to scale down the residual ($s=0.2$) in the standard regime to make RIMO-Pruned work suggests that the orthogonal-residual decoupling is highly sensitive to the magnitude of the residual. If the residual scale is set to $1.0$ in the standard regime, does the performance collapse even for RIMO-Pruned? This sensitivity suggests that the geometric merging framework is not fully robust to unregularized weights, which limits its general applicability.

## 3. Overall Rating
**Soundness Rating: Excellent**
The paper is highly rigorous and correct. The theoretical results are mathematically sound and provide a deep, complete explanation of the empirical phenomena. The methodology is appropriate, and the proposed symmetry-preserving alternatives (real Schur, complex Hermitian) and rank-preserving pruning are elegant and well-justified.
