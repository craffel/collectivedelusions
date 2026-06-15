# 3. Soundness and Methodology Evaluation

## Clarity of the Description
The methodology section is exceptionally well-written, clear, and mathematically rigorous. Specifically:
- **Quantization Equations:** The mathematical formulations for symmetric and asymmetric quantization (equations 5–9) are precise and map correctly to standard integer ranges (e.g., $[-2^{b-1}, 2^{b-1}-1]$ for asymmetric).
- **Double Quantization (DQ):** The inclusion of a detailed formulation for Double Quantization (equations 10–14) is highly commendable. It describes exactly how scale factors are themselves quantized using an 8-bit symmetric precision schema.
- **Autograd Detachment:** The text clearly and explicitly states that the scale factors ($s$) and zero-points ($z$) are detached from the PyTorch autograd graph during backpropagation. This is a crucial detail for correctness, preventing highly unstable or exploding gradients through dynamic min/max tracking.
- **Continuous Noise vs. Discrete Inference:** The paper clarifies that scale and zero-point perturbations ($\epsilon_s$ and $\epsilon_z$) are strictly active during the test-time calibration phase. Upon convergence, the noise is disabled ($\epsilon_s = 0$, $\epsilon_z = 0$) and standard integer rounding is restored, ensuring complete compatibility with commodity integer-arithmetic edge pipelines.

## Appropriateness of Methods
- **Test-Time Adaptation (TTA) via Entropy Minimization:** Using prediction entropy minimization on a tiny, unlabeled stream ($N_{\text{cal}} = 64$) is highly appropriate and practical for edge deployment, where labeled training data is typically unavailable.
- **Task-Consensus Regularization (TCR):** The objective incorporates an absolute proximity penalty ($\beta = 0.1$, keeping weights close to the uniform $\lambda_{\text{init}} = 0.3$) and a group consensus penalty ($\gamma = 0.5$, pulling weights toward the layer average consensus $\bar{\lambda}^l$). This dual penalty is well-designed to prevent parameter drift and avoid any single task dominating the optimization path.
- **Straight-Through Estimator (STE) with SOS:** Propagating gradients through hard rounding using the STE is standard and appropriate. Combining it with stochastically sampled operators (SOS) effectively introduces a "gradient dropout" effect, regularizing the continuous coefficient search and driving updates toward a flat, robust minimum.
- **First-Order vs. Zeroth-Order Discussion:** The methodological discussion at the end of Section 4 comparing first-order STE co-optimization with zeroth-order methods (like CMA-ES) is a great inclusion. It properly justifies using STE for rapid convergence (15 steps) on strict on-device latency budgets, where zeroth-order methods would exhibit prohibitive sample complexity.

## Technical Nuances & Potential Weaknesses
- **Scale Noise Divison:** In equation 5, the asymmetric scale factor is computed as $s_{\text{asym}} = \frac{\max(W) - \min(W)}{2^b - 1} \cdot (1 + \epsilon_s)$. Since $\epsilon_s \sim \mathcal{N}(0, 0.01^2)$, $1 + \epsilon_s$ is practically always positive. However, to ensure absolute technical robustness, a clipping operation (e.g., clamping $s_{\text{asym}}$ to be $\ge \epsilon_{\text{eps}}$) should be mentioned or utilized in code to guarantee there is no division-by-zero during weight scaling.
- **Continuous Zero-Point Noise:** In equation 6, $\epsilon_z \sim \mathcal{N}(0, 0.02^2)$ is added directly to the rounded zero-point. This results in a continuous (non-integer) zero-point during the forward pass of test-time adaptation. While mathematically valid as a regularizer, it means the simulated forward pass during optimization slightly diverges from a strict integer-only forward pass. The authors' clarification that this noise is disabled post-optimization is essential to resolve any potential compiler mismatches.

## Reproducibility
The methodology is exceptionally reproducible. Every hyperparameter, noise standard deviation, learning rate, optimization step, and data budget is explicitly stated, leaving no ambiguity for future researchers attempting to reproduce the framework.
