# 1_summary.md: Paper Summary

## Main Topic and Objective
The paper addresses the challenge of deploying multiple task-specific expert deep learning models on resource-constrained edge devices. In particular, it focuses on the intersection of two fields: **multi-task model merging** (specifically, combining independent single-task expert networks sharing a base initialization into a single unified network directly in parameter space) and **post-training quantization (PTQ)** to low-bit formats (such as 8-bit or 4-bit integers). 

When combining these paradigms, practitioners face a dilemma:
1. **Merge-then-Quantize (M-then-Q):** Merging in full precision and then quantizing introduces severe quantization noise, degrading test accuracy.
2. **Quantize-then-Merge (Q-then-M):** Merging already-quantized integer experts is mathematically invalid because independent integer quantization breaks the continuous linear mode connectivity (LMC) between the experts, resulting in representational collapse.

While on-device test-time adaptation (TTA) methods (e.g., AdaMerging, Q-Merge) optimize layer-wise merging coefficients on small unlabeled calibration streams, they suffer from the **Overfitting-Optimizer Paradox**: on tiny on-device calibration streams (e.g., 16 images), optimizing high-dimensional layer-wise spaces (e.g., 56 parameters in a 14-layer ViT-Tiny with 4 tasks) easily overfits transductive noise, producing highly jagged, physically nonsensical coefficient schedules that fail to generalize.

To resolve this, the authors propose **Q-PolyMerge**, a hybrid-precision (weight-quantized, activation-float) framework that restricts the search space of merging coefficients to a low-dimensional continuous polynomial subspace of normalized layer depth. Bounding these trajectories to a smooth polynomial (e.g., quadratic) reduces the search space by over 78% (from 56 to 12 parameters) and acts as a robust low-pass filter, mitigating overfitting and stabilizing gradient-based or zero-order search.

---

## Technical Approach
1. **Continuous Polynomial Subspace Parameterization:** Parameterizes the merging coefficients $\lambda_{k, l}$ as a polynomial of degree $d$ evaluated at the normalized depth $\bar{l} = \frac{l}{L-1} \in [0, 1]$:
   $$\lambda_{k, l}(\boldsymbol{\alpha}) = \sum_{j=0}^d \alpha_{k, j} \cdot \left( \frac{l}{L-1} \right)^j$$
   where $\boldsymbol{\alpha} \in \mathbb{R}^{K \times (d+1)}$ represents the learnable parameters. For a quadratic polynomial ($d=2$), this reduces the parameters from $L \times K$ to $(d+1) \times K$.
2. **Symmetric Uniform Post-Training Quantization:** Weights are compressed to 8-bit (symmetric uniform per-tensor) or 4-bit (symmetric uniform per-channel).
3. **First-Order Optimization Pathway (Adam STE):** Optimizes the low-dimensional parameters $\boldsymbol{\alpha}$ using standard backpropagation approximated via the Straight-Through Estimator (STE) to flow gradients through the rounding operator.
4. **Zero-Order Optimization Pathway (1+1 ES):** Uses a derivative-free 1+1 Evolution Strategy to search the 12-dimensional parameter space, requiring only forward passes and thus incurring zero activation caching or gradient storage overhead.
5. **Integer-Weight Edge Pipeline:** Compresses weights of both the backbone and task-specific heads to low-bit formats, yielding a 100% integer weight pipeline (while activations remain in floating-point during inference for stability).

---

## Key Findings and Claims
1. **Overfitting Mitigation:** By restricting the coefficients to a smooth polynomial subspace, Q-PolyMerge acts as a low-pass filter, avoiding the highly jagged layer coefficient profiles of unconstrained methods and stabilizing test-time adaptation.
2. **Superior Performance under Low-Bit PTQ:** 
   - Under 4-bit PTQ, Q-PolyMerge (Adam STE) achieves **48.87%** average accuracy, outperforming unconstrained Q-Merge (46.02%) and naive post-merge quantization (42.92%).
   - Under 8-bit PTQ, Q-PolyMerge achieves **59.76%** average accuracy, matching unconstrained Q-Merge (60.03%) while reducing standard deviation by over 47%.
3. **On-Device SRAM Viability:** Bypassing backpropagation and activation caching using the zero-order 1+1 ES pathway achieves a **95.8% to 97.5% reduction in peak SRAM footprint** (e.g., 4.05 MB vs. 162.72 MB under 4-bit), making it highly viable for resource-constrained microcontrollers.
4. **Fast Convergence and Compute Efficiency:** Due to the low-dimensional search space, zero-order search converges in 100 iterations (100 forward passes, 0 backward passes), making it **16.7% cheaper** than first-order gradient descent (equivalent to 120 forward-pass computations).

---

## Explicitly Claimed Contributions (with Evidence)
1. **Identification of the Overfitting-Optimizer Paradox:** The authors demonstrate that unconstrained TTA over layer-wise merging coefficients under quantization easily overfits to small, unlabeled on-device calibration streams. They back this up with qualitative visualization of highly jagged coefficient profiles in Appendix B.2 and Figure 2.
2. **The Q-PolyMerge Framework:** They propose the first framework that restricts merging coefficients to a low-dimensional continuous polynomial subspace. They formulate this mathematically and provide scaling pathways using orthogonal Chebyshev polynomials for deeper models.
3. **Dual Optimization Pathways:** They develop first-order (Adam STE) and zero-order (1+1 ES) optimization routines. They prove that under 1+1 ES, the 12-dimensional polynomial search space avoids the curse of dimensionality, unlike the unconstrained 56-dimensional space which collapses.
4. **Rigorous Empirical Benchmark:** They evaluate on a Vision Transformer (ViT-Tiny) across 4 benchmarks (MNIST, FashionMNIST, CIFAR-10, SVHN) under 8-bit and 4-bit PTQ over 3 random seeds, providing a robust comparison with 23 baseline configurations.
5. **Hardware-Level and Systems Profiling:** They provide modeled latency and energy consumption analyses on ARM Cortex-M7 and RISC-V GAP8 edge processors, showing 16.7% latency and energy reductions alongside a 97% active SRAM memory footprint savings.
