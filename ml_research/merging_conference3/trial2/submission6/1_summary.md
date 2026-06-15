# Paper Summary: Q-Merge

**Title:** Q-Merge: A Pragmatic Approach to Quantization-Aware Model Merging under Extreme Deployment Constraints

## Overview
This paper addresses the critical intersection of **model merging** and **Post-Training Quantization (PTQ)**. While model merging is a zero-shot, parameter-efficient approach to create multi-task networks, deploying these models on resource-constrained edge hardware requires low-bit quantization. Standard workflows suffer from a dilemma: merging first then quantizing (M-then-Q) degrades multi-task accuracy due to rounding noise, while quantizing first then merging (Q-then-M) breaks linear mode connectivity.

To solve this, the author proposes **Quantization-Aware Model Merging (Q-Merge)**. Q-Merge optimizes layer-wise merging coefficients directly under the non-differentiable quantization operator at test-time using a small, unlabeled calibration stream.

---

## Key Methodology
1. **Layer-Wise Task Arithmetic:** Fuses task-specific experts into a single backbone model, parameterized by a layer-wise coefficient tensor $\Lambda = \{\lambda^l_k\} \in [0, 1]^{L \times K}$.
2. **Per-Channel Symmetric Uniform PTQ:** Employs standard per-channel (channel-wise) weight quantization to 8-bit (INT8) or 4-bit (INT4). Dynamic channel-specific scale factors $S^l_c$ are computed as a function of the continuous weights.
3. **Test-Time Joint Entropy Optimization:** Minimizes prediction entropy on an unlabeled, compact calibration set (64 images total, 16 per task) to find optimal blending coefficients $\Lambda$.
4. **Optimization Paradigms:**
   - **Zero-Order (1+1 ES):** A black-box mutation strategy utilizing Rechenberg's 1/5th success rule.
   - **First-Order (Adam GD with STE):** Propagates gradients through the non-differentiable rounding operator using the Straight-Through Estimator (STE), utilizing dynamic scale factor derivatives.

---

## Key Findings & Quantitative Results
* **Overcoming the 8-Bit Quantization Gap:** Naive post-merge quantization degrades accuracy. Under 8-bit PTQ, Q-Merge with Adam GD + STE achieves **74.30%** average multi-task accuracy, remarkably exceeding the unquantized FP16 baseline (**71.88%**) and standard unquantized AdaMerging (**73.21%**), while recovering $99.9\%$ of the unquantized Adam-optimized ceiling (**74.38%**).
* **Optimizer Confounding Isolated:** When compared under the same optimizer, 8-bit Q-Merge matches the unquantized Adam ceiling within $0.08\%$ ($74.30\%$ vs $74.38\%$), proving that the "super-ceiling" boost is driven by unlocking superior first-order gradient descent via the STE.
* **First-Order Superiority:** Adam GD with STE converges faster (within 20 iterations vs 40+ for 1+1 ES), achieves higher final accuracy (74.30% vs 72.57% in 8-bit, 63.36% vs 57.83% in 4-bit), and exhibits $2.7\times$ lower seed-to-seed variance.
* **Unlocking 4-Bit Merging:** Standard per-channel quantization prevents the "4-bit collapse" seen in per-tensor quantization. Under 4-bit PTQ, Q-Merge with Adam GD achieves **63.36%** accuracy, outperforming naive post-merge quantization (**56.66%**) by **6.70%** absolute and post-hoc quantized AdaMerging (**62.01%**) by **1.35%** absolute.
* **Systems Feasibility & Complementarity:** 
  - Fully integer-quantized weight inference (with 8-bit quantized task heads) achieves **74.30% (8-bit)** and **63.35% (4-bit)** with virtually $0.00\%$ degradation.
  - Highly compatible with advanced PTQ: executing Q-Merge followed by AdaRound yields a state-of-the-art **64.46%** average accuracy in 4-bit.
  - Robust to non-stationary calibration streams (retaining ~76% under 95% single-task skew) and extremely data-efficient (stable down to 8 images per task).
