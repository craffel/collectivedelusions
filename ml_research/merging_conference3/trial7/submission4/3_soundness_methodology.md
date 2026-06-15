# Soundness and Methodology Check: Löwdin-Orthogonalized Task-Space Projection

## 1. Overall Rating
**Excellent.**

The mathematical foundations of this paper are exceptionally solid. The derivations are presented with rigorous step-by-step clarity, and the assumptions are realistic. The proofs of symmetric equivalence and SNR equivalence are mathematically sound and are verified by empirical results.

## 2. Mathematical Soundness and Proofs
- **SVD Centroid Extraction (Step 1):** The explanation of prototype sum-to-zero cancellation is highly accurate. In trained classifiers, final weight vectors (prototypes) are pushed apart symmetrically to maximize classification boundaries. Taking a simple average of these vectors will indeed collapse toward $\mathbf{0}$. SVD-based centroid extraction ($v_k = V_{k,1}$) is mathematically justified as it captures the principal direction of maximum prototype variance, which is perfectly stable.
- **Symmetric Orthogonalization (Step 3):** The formulation of the symmetric inverse square root ($S^{-1/2} = U \Lambda^{-1/2} U^T$) is mathematically standard and correct.
- **Absolute Projection Sign Equivalence Proof (Section 3.7):** The updated proof of sign equivalence under absolute value projections is extremely elegant and flawless:
  $$y_1^2 - y_2^2 = (a x_1 + b x_2)^2 - (b x_1 + a x_2)^2 = (a^2 - b^2) (x_1^2 - x_2^2)$$
  Since the diagonal element $a$ and off-diagonal element $b$ satisfy $a^2 - b^2 = \frac{1}{\sqrt{1-s^2}} > 0$ for all symmetric similarities $s \in [0,1)$, the sign of $|y_1| - |y_2|$ matches the sign of $|x_1| - |x_2|$, guaranteeing that OTSP and PFSR make identical routing decisions for every single sample under symmetric correlation.
- **Signal-to-Noise Ratio (SNR) Equivalence Proof (Section 3.8):** Under isotropic noise $\eta_b \sim \mathcal{N}(0, \sigma^2 I_D)$, the authors prove that:
  $$\text{SNR}_{\text{OTSP}} = \text{SNR}_{\text{PFSR}} = \frac{\sqrt{1-s}}{\sigma \sqrt{2}}$$
  This is a brilliant closed-form geometric result. Although Löwdin orthogonalization expands the clean coordinate routing margin by a factor of $\frac{1}{\sqrt{1-s}}$, it simultaneously amplifies the projection noise variance by exactly the same factor, resulting in an exact cancellation of routing benefits. This derivation is mathematically sound and extremely clean.
- **Noise Amplification Penalty derivation (Section 3.6):** The variance derivation $\text{Var}(q_k \cdot \eta_b) = \sigma^2 (S^{-1})_{kk}$ is correct and establishes a clear analogy to the well-known problem of multicollinearity in classical linear regression, demonstrating high theoretical maturity.

## 3. Alternative Centroid Formulations (Weights vs. Activations)
Section 3.1 contains a highly detailed, comprehensive comparison of SVD on classifier weights vs. sample activations (mean, SVD, and K-Means) and their respective practical and computational trade-offs:
1. *SVD on Classifier Weights (Primary):* 100% data-free, instantaneous ($O(KCD)$), but requires direct parameter access to classification heads.
2. *Mean/SVD on Activations:* Requires a small representative calibration dataset, but captures contextualized feature distributions, is robust to out-of-distribution drift, and can be applied at any arbitrary intermediate layer where explicit classification heads do not exist.
3. *K-Means on Activations:* Captures multi-modal task distributions, allowing for highly flexible routing in complex multi-task environments, though computationally more expensive.
This discussion shows that the authors did not just propose one heuristic, but rather deeply analyzed the broader methodological landscape and its systems-level trade-offs.
