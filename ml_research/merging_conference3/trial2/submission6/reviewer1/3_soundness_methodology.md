# 3. Soundness and Methodology

This document evaluates the technical soundness, clarity, and reproducibility of the methodology proposed in **Quantization-Aware Model Merging (Q-Merge)**.

## Clarity of the Description
The methodology is exceptionally well-documented and mathematically rigorous.
*   **Formulation:** The equations describing the layer-wise task arithmetic parameterization (Eq. 1-2), the symmetric uniform per-channel post-training quantization (Eq. 3-4), and the test-time joint entropy optimization (Eq. 5-6) are clean, precise, and standard.
*   **Gradient Derivation:** Section 3.4.2 provides an incredibly thorough mathematical derivation of the dual-path gradient flow through both the direct weight coordinate path and the dynamic per-channel scale factors ($S^l_c$) under the Straight-Through Estimator (STE) approximation. This makes the underlying mechanics of how PyTorch Autograd propagates gradients to the coefficients mathematically explicit and highly transparent.
*   **Discussion of Trade-offs:** The paper does not gloss over real-world implementation details. It explicitly discusses:
    *   *Trivial class collapse* during unsupervised entropy minimization and how the low-dimensional parameterization ($\Lambda$) serves as a powerful regularizer preventing it.
    *   *Backpropagation memory complexity* and three practical mitigation strategies (Gradient Checkpointing, Forward-Mode AD, and zero-order 1+1 ES).
    *   A pragmatic *hardware decision-tree guide* (when to use STE vs. 1+1 ES) based on whether the edge processor supports backpropagation and activation caching.

## Appropriateness of Methods
The choice of methods is highly appropriate and scientifically justified:
*   **Entropy Minimization:** Minimizing joint prediction entropy over an unlabeled calibration stream is a standard and highly successful proxy for accuracy in test-time adaptation literature (e.g., TENT, AdaMerging).
*   **Straight-Through Estimator (STE):** Bypassing the non-differentiable rounding operator with STE is the industry-standard method for training quantized neural networks. Applying it to optimize low-dimensional merging coefficients ($\Lambda$) is elegant and highly effective.
*   **Per-Channel Quantization:** Using per-channel instead of per-tensor quantization for low-bit (4-bit) representations is the standard way to protect against dynamic range crushing by outlier weights. It is highly appropriate for merging task-specific experts, which inherently contain localized outlier updates.

## Potential Technical Flaws & How They Are Addressed

### 1. Backpropagation Activation Caching on Edge Devices
*   *Issue:* Standard reverse-mode AD requires caching activation maps, which can exceed the memory capacity of low-power edge accelerators.
*   *Resolution:* The authors address this by:
    1.  Analyzing the precise runtime memory footprint (ViT-Tiny peak activation caching is only $2.8$ MB, and weight-only INT4 storage is only $2.85$ MB).
    2.  Proposing Forward-Mode AD (Jacobian-Vector Products) to compute exact gradients concurrently during the forward pass, which completely eliminates activation caching.
    3.  Evaluating the derivative-free 1+1 ES optimizer as a zero-activation-overhead alternative for ultra-low-power microcontrollers (MCUs) that lack backpropagation support.

### 2. Discretization of Scale Factors
*   *Issue:* On specialized fixed-point edge processors, scale factors ($S^l_c$) themselves cannot remain in high-precision floats (FP32/FP16) and must be quantized to integer multipliers and shift operators.
*   *Resolution:* In Appendix C, the authors formulate the discretization of scale factors into fixed-point representations. They show through a sensitivity analysis that at $16$-bit fraction scales, there is virtually zero performance degradation ($0.07\%$ absolute), and even at $8$-bit fraction scales, Q-Merge retains $61.98\%$ accuracy, proving the framework's compatibility with integer-only hardware.

### 3. Dynamic Activation Quantization
*   *Issue:* Weight-only quantization (W4A16) is highly viable, but some accelerators require fully integer-only operations (W8A8 or W4A4).
*   *Resolution:* In Appendix D, the authors extend Q-Merge to dynamic activation quantization using dynamic per-tensor scaling. Their empirical validation on seed 42 reveals that W8A8 is highly stable (retaining $73.12\%$ accuracy), while W4A4 is more challenging due to dynamic activation clipping noise, causing a drop to $58.45\%$ and slower convergence. This provides systems engineers with highly valuable design recommendations.

### 4. Non-Stationary Calibration Streams
*   *Issue:* Unsupervised entropy minimization on highly unbalanced calibration streams (e.g., dominated by one task) can cause catastrophic forgetting on the other tasks.
*   *Resolution:* In Appendix B, the authors propose a low-overhead on-device **Confidence-Based FIFO Stratification** heuristic to buffer and balance incoming samples before executing adaptation. In Section 4.6, they validate this heuristic empirically, demonstrating that it successfully restores perfect balance and achieves superior multi-task accuracy ($76.95\%$ 8-bit, $59.77\%$ 4-bit) even under extreme stream imbalance.

## Reproducibility
The reproducibility of this submission is **excellent**.
*   All hyperparameters (learning rates, batch sizes, optimizer choices, mutation step sizes, and calibration split sizes) are explicitly listed in Section 4.1.
*   The code architecture, layer-wise grouping ($L=14$ groups for ViT-Tiny), and backbone details are thoroughly documented.
*   The authors conduct all experiments across **three independent random seeds (42, 100, 2026)** and report standard deviations, ensuring high statistical reliability.
