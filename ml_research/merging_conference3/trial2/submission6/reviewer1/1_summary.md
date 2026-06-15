# 1. Summary of the Paper

## Main Topic
The paper introduces **Quantization-Aware Model Merging (Q-Merge)**, a calibration-free, test-time adaptation framework that optimizes layer-wise merging coefficients directly under a non-differentiable post-training quantization (PTQ) operator. The goal is to address the severe accuracy degradation that occurs when weight merging (e.g., Task Arithmetic or AdaMerging) is combined with low-bit post-training quantization (such as 8-bit or 4-bit formats) for edge deployment.

## Approach
Q-Merge fuses multiple task-specific expert models (fine-tuned from a shared pre-trained base) into a single multi-task network. Instead of static uniform or full-precision coefficient optimization followed by naive quantization, Q-Merge optimizes continuous layer-wise merging coefficients $\Lambda$ directly under the discrete quantization rounding operator. 
*   **Objective:** Minimizes the joint Shannon entropy of predictions over a compact, unlabeled calibration stream (64 images total, i.e., 16 images per task) as an unsupervised proxy for multi-task accuracy.
*   **Optimization Paradigms:** The authors formulate and compare two optimization strategies:
    1.  *Zero-Order:* A derivative-free black-box mutation strategy (1+1 Evolution Strategy) with step size adapted via Rechenberg’s 1/5th success rule.
    2.  *First-Order:* Gradient-based optimization utilizing the **Straight-Through Estimator (STE)** to approximate the gradient of the rounding operator as an identity mapping ($\frac{\partial \text{round}(x)}{\partial x} \approx 1$), updating continuous coefficients using Adam.
*   **Quantization Scheme:** Standard symmetric uniform Post-Training Quantization (PTQ) is applied to weights at 8-bit (INT8) and 4-bit (INT4) levels. Critically, to avoid catastrophic collapse at 4-bit, the authors utilize per-channel (channel-wise) weight quantization.

## Key Findings
1.  **Overcoming the 8-Bit Quantization Gap:** Under 8-bit PTQ, Q-Merge with Adam GD (STE) achieves **74.30%** average multi-task accuracy, significantly outperforming the unquantized uniform FP16 baseline (**71.88%**) and unquantized optimized AdaMerging (ES) ceiling (**73.21%**), while recovering $99.9\%$ of the unquantized Adam-optimized ceiling (**74.38%**).
2.  **Differentiable vs. Derivative-Free Performance:** Direct first-order gradient feedback using STE is substantially superior to zero-order 1+1 ES. Under 8-bit quantization, Adam GD achieves $74.30\%$ accuracy (with $0.38\%$ standard deviation), whereas 1+1 ES achieves $72.57\%$ (with $1.06\%$ standard deviation). Under 4-bit quantization, Adam GD achieves $63.36\%$, while 1+1 ES struggles at $57.83\%$.
3.  **Unlocking 4-Bit Model Merging:** Under aggressive 4-bit quantization, per-channel quantization preserves weight mode connectivity. Q-Merge with Adam GD (STE) achieves **63.36%** average accuracy, outperforming the naive post-merge quantization baseline (**56.66%**) by **6.70%** absolute and post-hoc quantized AdaMerging (**62.01%**) by **1.35%** absolute.
4.  **Fully Integer-Quantized Weight Pipeline:** Quantizing the task-specific classification heads (comprising only 0.03% of parameters) post-hoc to 8-bit results in negligible performance loss ($<0.01\%$ drop), enabling a fully compressed integer weight pipeline (W8A16 or W4A16).
5.  **Low Data and Imbalance Robustness:** The method requires as few as 8 images per task to converge and is highly robust to severe stream imbalance (e.g., 95% of the stream dominated by a single task) due to the strong structural regularization of the layer-wise blending parameterization.

## Explicitly Claimed Contributions
1.  **Formulation of Q-Merge:** The first framework to optimize layer-wise model-merging coefficients directly under a non-differentiable quantization operator.
2.  **First-Order STE Backpropagation for Merging:** Demonstrating that propagating gradients through a discrete rounding operator is highly stable, converges faster, and achieves over $2.7\times$ lower seed-to-seed variance compared to zero-order random-walk mutations.
3.  **Feasibility of 4-Bit Merging:** Showing that standard per-channel quantization is an absolute necessity to prevent "4-bit collapse" in model merging, and that combining it with Q-Merge (STE) makes aggressive INT4 merging highly viable and practical.
4.  **Comprehensive Evaluations:** Thorough ablation studies including standalone advanced PTQ baselines (AdaRound), sequential integration (Q-Merge + AdaRound), calibration size sensitivity, scale discretization analysis, joint weight-activation quantization (W8A8/W4A4), and non-stationary calibration stream validation.
