# 1. Summary of the Paper

## Main Topic and Approach
The paper introduces **Q-PolyMerge**, a parameter-efficient, quantization-aware test-time adaptation (TTA) framework for multi-task model merging on resource-constrained edge devices. Multi-task model merging consolidates multiple single-task experts (fine-tuned from a shared base model) directly in the parameter space without requiring joint multi-task training. 

However, merging experts in full precision and then quantizing the model (Merge-then-Quantize, M-then-Q) introduces severe quantization noise, while merging pre-quantized experts (Quantize-then-Merge, Q-then-M) breaks continuous linear mode connectivity (LMC). Current TTA approaches (e.g., AdaMerging) optimize layer-wise merging coefficients on a tiny streaming set of unlabeled on-device calibration images via entropy minimization, but they suffer from what the authors term the **Overfitting-Optimizer Paradox**: unconstrained optimization over high-dimensional layer-wise spaces (e.g., 56 parameters for ViT-Tiny across 14 blocks and 4 tasks) easily fits transductive noise, producing jagged, physically nonsensical coefficient trajectories that fail to generalize.

To mitigate this, **Q-PolyMerge** restricts the search space of merging coefficients to a low-dimensional continuous polynomial subspace of normalized layer depth. For a layer index $l$ and task $k$, the coefficient is represented as:
$$\lambda_{k, l}(\boldsymbol{\alpha}) = \sum_{j=0}^d \alpha_{k, j} \cdot \left( \frac{l}{L-1} \right)^j$$
where $\boldsymbol{\alpha} \in \mathbb{R}^{K \times (d+1)}$ are the learnable polynomial parameters of degree $d$ (typically $d=2$). This formulation reduces the parameter search space by 78.6% (from 56 to 12 parameters) for a standard ViT-Tiny. 

The authors propose two optimization pathways:
1. **First-Order Optimization via Straight-Through Estimator (Adam STE):** Gradient descent through the non-differentiable rounding operator of the symmetric uniform post-training quantization (PTQ) pipeline.
2. **Zero-Order Optimization via 1+1 Evolution Strategy (1+1 ES):** A derivative-free black-box optimization scheme requiring only forward passes, which eliminates the SRAM activation cache footprint required for backpropagation.

## Key Findings
1. **SRAM Footprint Reduction:** The zero-order 1+1 ES pathway bypasses activation caching entirely, reducing peak volatile memory (SRAM) requirements during TTA from **158.40 MB** (under first-order Adam STE) to just **4.05 MB** (under 4-bit per-channel PTQ) on a ViT-Tiny backbone at batch size $B=16$. This represents a $97.5\%$ reduction, making TTA physically viable on microcontrollers.
2. **Optimization Stability:** Constraining the coefficients to a smooth low-degree polynomial subspace stabilizes gradient-based optimization and derivative-free search. Under 4-bit per-channel PTQ, Q-PolyMerge (Adam STE) achieves an average accuracy of **48.87%**, outperforming unconstrained Q-Merge by **+2.85%** and naive post-merge quantization (M-then-Q) by **+5.95%**. 
3. **Low-Pass Filtering Effect:** Bounding the trajectories to a smooth polynomial acts as a qualitative low-pass filter, removing high-frequency optimization noise and preventing degenerate entropy-collapse states (where the model collapses to predicting a single dominant class with high confidence).
4. **Generalization of ES:** In the 8-bit regime, Q-PolyMerge (ES) matches the accuracy of unconstrained Q-Merge (ES) while reducing the standard deviation across seeds by 47% (to 4.35%), demonstrating significantly smoother and physically stable coefficient schedules.

## Explicitly Claimed Contributions (with Evidence)
1. **Introduction of Q-PolyMerge:** A hybrid-precision (weight-quantized, activation-float) framework that projects merging coefficients onto a low-degree continuous polynomial subspace of layer depth.
   - *Evidence:* Formulated in Sec. 3.2, with empirical evaluation in Sec. 4.
2. **First-Order & Zero-Order On-Device Pathways:** Combining STE-based first-order gradient descent and derivative-free 1+1 ES to adapt the parameters on unlabeled test-time streams.
   - *Evidence:* Outlined in Sec. 3.5 & 3.6, and evaluated under 8-bit per-tensor and 4-bit per-channel PTQ formats.
3. **Derivation & Validation of SRAM Footprint:** A detailed mathematical derivation of the 158.40 MB activation cache required for backpropagation on a ViT-Tiny, compared to the memory-efficient forward-only zero-order pathway.
   - *Evidence:* Presented in Appendix A.2 and Table 1.
4. **Algorithmic Solutions for 4-Bit ES Search:** Implementing Coordinate Descent with Greedy Backtracking to help zero-order search navigate the highly fragmented rounding landscape of 4-bit per-channel PTQ.
   - *Evidence:* Discussed in Appendix B.1 and evaluated in Table 3 (Block-wise vs. Continuous).
5. **Robustness to Stream Skew:** Proving that the polynomial constraint acts as a robust prior that prevents degenerate entropy collapse even under extreme class skew in the streaming calibration data.
   - *Evidence:* Discussed qualitatively in Appendix B.5.
6. **Theoretical Scaling Blueprint:** Outlining Chebyshev orthogonal scaling pathways and continuous localized B-splines to extend Q-PolyMerge to deeper foundation models (e.g., LLaMA, CLIP).
   - *Evidence:* Formulated in Appendix B.7 & B.8.
