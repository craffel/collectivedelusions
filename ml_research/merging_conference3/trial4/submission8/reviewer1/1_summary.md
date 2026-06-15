# Intermediate Review: 1. Summary

## Main Topic and Motivation
The paper addresses a critical and overlooked vulnerability in test-time adaptive model merging (TTA) frameworks: **Quantization-Operator Overfitting**. While existing test-time adaptation methods (such as AdaMerging) successfully learn blending coefficients for separate task-specific expert models in weight space, they tend to optimize these coefficients into extremely sharp local minima. When these merged models are quantized to low-precision formats (such as INT8 or INT4) for edge deployment, the rounding noise causes a catastrophic collapse in multi-task accuracy. 

To resolve this issue, the paper proposes a unified framework called **CR-PolySACM (Clipping-Regularized Sharpness-Aware Subspace Model Merging)**, also referred to as **PolySACM**. This framework combines:
1. **Differentiable Polynomial Subspace Parameterization (PolyMerge):** Constraining layer-wise blending coefficients to a low-degree polynomial of network depth to reduce optimization complexity from 56 parameters to 12.
2. **Clipping-Regularized Sharpness-Aware Minimization (CR-SACM):** Minimizing local loss sharpness while correcting a "task-vector norm scale pathology" where a 50-fold discrepancy in layer-wise task-vector norms makes standard flatness optimizers blind to highly sensitive, low-norm layers (like the final layer norm layer).

---

## Technical Approach
1. **Polynomial Subspace Constraint (PolyMerge):**
   The blending coefficients $\lambda_k^l$ for task $k$ at layer $l$ are parameterized by a quadratic polynomial of normalized depth $d_l = \frac{l-1}{L-1}$:
   $$
   \lambda_k^l(\mathbf{p}_k) = \sigma\left(a_k + b_k \left(\frac{l-1}{L-1}\right) + c_k \left(\frac{l-1}{L-1}\right)^2\right)
   $$
   This reduces the parameters from $L \times K$ independent variables (e.g., 56 for 14 layers and 4 tasks) to only $3 \times K$ polynomial coefficients (e.g., 12 parameters).
   
2. **Clipping-Regularized SACM (CR-SACM):**
   The paper identifies that the final layer norm (Layer 13) has an extremely small task-vector norm (0.014–0.020) compared to intermediate transformer blocks (0.40–0.68). This "scale pathology" causes standard unnormalized flatness optimizers to be blind to Layer 13's high sensitivity. To fix this, CR-SACM scales the adversarial coefficient perturbation inversely by the clipping-regularized task-vector norm:
   $$
   V_{\text{clipped}, k}^l = \max\left(\|\tau_k^l\|_2, \beta\right)
   $$
   where $\beta = 0.10$. The perturbation is computed as:
   $$
   \epsilon_k^l = \rho \frac{\hat{g}_k^l}{\|\hat{\mathbf{g}}\|_2 V_{\text{clipped}, k}^l} \quad \text{with} \quad \hat{g}_k^l = \frac{1}{V_{\text{clipped}, k}^l} \frac{\partial \mathcal{L}}{\partial \lambda_k^l}
   $$

---

## Key Findings and Claims
1. **Quantization-Operator Overfitting:** Unconstrained TTA model merging methods like AdaMerging optimize blending coefficients into sharp local minima that are fragile under PTQ rounding noise, particularly in low-precision (e.g., INT4/INT8) regimes.
2. **Task-Vector Norm Scale Pathology:** Standard unnormalized flatness-aware optimization fails because of a massive scale mismatch in task-vector norms between layer groups (especially intermediate transformer blocks vs. the final layer norm).
3. **Subspace Constraint Benefits (PolyMerge):** Restricting blending coefficients to a low-degree polynomial subspace serves as a powerful global structural regularizer, yielding dramatic improvements in generalization and robustness over unconstrained TTA.
4. **CR-PolySACM Performance:** In aggressive INT4 symmetric per-channel quantization, CR-PolySACM achieves a joint mean accuracy of **19.07%**, outperforming the previous state-of-the-art PolyMerge baseline (**18.10%**) by **+0.97%**.

---

## Explicitly Claimed Contributions and Evidence
- **Identification of Quantization-Operator Overfitting:** Supported by empirical results showing AdaMerging's collapse under INT4/INT8 quantization formats compared to static Uniform TA.
- **Formulation of Task-Vector Norm Scale Pathology & CR-SACM:** Supported by diagnostic norm measurements (showing the 50-fold scale discrepancy) and ablation studies showing that unconstrained HessMerge degrades performance as regularization strength increases, whereas CR-SACM stabilizes it.
- **Introduction of CR-PolySACM:** Supported by multi-task evaluations across six quantization schemas (FP32, INT8 Sym/Asym, INT4 Sym) on Vision Transformer models.
- **State-of-the-Art INT4 Merging:** Supported by Table 1, where CR-PolySACM achieves 19.07% in INT4 compared to other adaptive baselines.
