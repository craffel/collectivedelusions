# 3_soundness_methodology.md: Soundness and Methodology Evaluation

## Clarity of the Description
The methodology of Q-PolyMerge is exceptionally well-written, clear, and mathematically precise. 
- The partitioning of parameters, the formulation of expert task vectors, and the polynomial mapping are introduced systematically.
- The use of the Vandermonde matrix to map continuous low-dimensional parameters to layer-wise coefficients is standard and clearly diagrammed in Section 3 and the appendix.
- The symmetric uniform post-training quantization operator is defined precisely (Equation 5), including the scale factors and rounding equations.
- The Straight-Through Estimator (STE) chain rule derivation is clear and mathematically sound.
- The 1+1 Evolution Strategy (1+1 ES) algorithm is formulated step-by-step, including step-size adaptation.
- The appendix provides a detailed and rigorous mathematical blueprint for fully-integerized activation and operator execution (CMSIS-NN/Integer LayerNorm) and generalizes standard monomial bases to Chebyshev orthogonal polynomials.

---

## Appropriateness of Methods
The methods used are highly appropriate for the targeted constraints:
- **Low-degree Polynomials (e.g., $d=2$):** Constraining coefficients to a continuous polynomial trajectory is a highly effective way of regularizing high-dimensional parameter search spaces under data-impoverished streams.
- **Symmetric Uniform PTQ (INT8 and INT4 per-channel):** These match standard edge-hardware deployment pipelines, where per-channel quantization is critical under low bit-widths (INT4) to preserve coordinate alignment.
- **Straight-Through Estimator (STE):** This is the industry-standard way to enable gradient-based optimization in non-differentiable quantized networks.
- **1+1 Evolution Strategy:** This is a highly appropriate choice for microcontrollers because it only requires forward inference passes, avoiding gradient storage, backward pass computations, and activation caching.

---

## Potential Technical Flaws and Limitations
While the methodology is highly robust, several subtle technical nuances and minor flaws/limitations are worth noting:

1. **Monomial Conditioning at Higher Degrees:**
   As the authors themselves identify and prove in Appendix C, the standard monomial basis $\{1, \bar{l}, \bar{l}^2, \dots\}$ becomes severely ill-conditioned at degrees $d \ge 5$ (condition number $> 3000$). While this is not an issue for their proposed quadratic model ($d=2$, condition number $\approx 20$), it represents a fundamental numerical barrier if they were to scale to more complex, high-degree continuous trajectories. The transition to Chebyshev polynomials (proposed in Appendix C) resolves this, but was not empirically validated in the main experiments.
   
2. **High Volatility of Zero-Order Search:**
   As shown in Appendix B.1, the zero-order 1+1 ES pathway exhibits high standard deviations across trials (e.g., MNIST standard deviation of $\pm 21.11\%$ under 8-bit). This volatility is a known risk of isotropic random search in non-smooth, step-cliff quantization landscapes. Although the authors propose advanced safeguards (CMA-ES, Multi-Candidate Population Search, Historical Momentum Filtering), these remain theoretical suggestions in the text rather than empirically implemented and tested additions.
   
3. **Software Emulation of Float Activations:**
   The proposed pipeline is "hybrid-precision," maintaining weights as integers but activations and normalization layer-outputs as floating-point formats. On edge microcontrollers lacking dedicated vector or floating-point hardware, executing activations in float requires software emulation, which can introduce massive latency and energy penalties. While the authors outline a highly thorough, step-by-step blueprint for a fully-integerized pipeline (W8A8/W4A8) in Appendix B.6, the paper's experiments still rely on floating-point activations.

---

## Reproducibility
The reproducibility of the work is **excellent**:
- The authors list detailed network structures (pre-trained `vit_tiny_patch16_224` containing 5.7M parameters) and clear hyperparameter selections.
- The dataset splits (512 training images, 16 unlabeled calibration images, 2000 test images per task) are deterministic and clearly structured.
- We have inspected the underlying codebase (`run_experiments.py`), which is extremely clean, well-commented, and implements every single baseline and optimization pathway (including Adam STE, 1+1 ES, block-wise constant, and degree ablations) precisely as described.
- The use of 3 independent random seeds (42, 100, 2026) ensures statistical transparency and prevents selective reporting of lucky runs.
- Standard deviation is explicitly reported across all tables.
