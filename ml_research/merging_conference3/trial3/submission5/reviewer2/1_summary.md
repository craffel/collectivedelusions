# Paper Summary: Q-PolyMerge

## 1. Main Topic and Scope
The paper addresses the challenge of deploying multi-task merged models on resource-constrained edge devices (such as microcontrollers) under strict memory, storage, and thermal budgets. Specifically, it focuses on **quantization-aware test-time model merging**, where multiple task-specific expert models (fine-tuned from a shared base model) are consolidated into a single low-bit integer model (e.g., INT8 or INT4) directly in parameter space, and then adaptively optimized at test-time using small streams of unlabeled calibration data.

## 2. Proposed Approach: Q-PolyMerge
Traditional test-time adaptation (TTA) methods like AdaMerging optimize merging coefficients layer-by-layer. However, the authors identify a key vulnerability they term the **Overfitting-Optimizer Paradox**: on tiny, transient, unlabeled on-device calibration streams (e.g., 16 images), unconstrained optimization over high-dimensional layer-wise spaces (e.g., 56 parameters for a ViT-Tiny) easily overfits to transductive noise, resulting in jagged, physically unstable coefficient schedules that generalize poorly.

To resolve this, **Q-PolyMerge** restricts the search space of merging coefficients to a low-dimensional continuous polynomial subspace of normalized layer depth.
- **Polynomial Parameterization:** The merging coefficient for task $k$ at layer $l$ is defined as a polynomial of degree $d$ evaluated at normalized depth $\bar{l} = \frac{l}{L-1}$:
  $$\lambda_{k, l}(\boldsymbol{\alpha}) = \sum_{j=0}^d \alpha_{k, j} \cdot \left( \frac{l}{L-1} \right)^j$$
  This projects the learnable dimensions from $L \times K$ (layer-wise) down to $(d+1) \times K$. For a quadratic polynomial ($d=2$), this represents a **78.6% search space reduction** (e.g., from 56 to 12 parameters).
- **Low-Pass Filtering:** Bounding the coefficients to a smooth polynomial acts as a mathematical low-pass filter, preventing high-frequency optimization noise and mitigating degenerate entropy collapse.
- **On-Device Optimization Pathways:**
  1. *First-Order Gradient Descent (Adam STE):* Optimizes parameters using the Straight-Through Estimator to backpropagate pseudo-gradients through the non-differentiable quantization operator.
  2. *Zero-Order Derivative-Free Search (1+1 Evolution Strategy):* Bypasses backpropagation and activation caching completely. By operating in the low-dimensional polynomial subspace, 1+1 ES runs efficiently with zero gradient overhead, making on-device TTA memory-viable.
- **Integer-Weight Pipeline:** Stores 100% of parameters as low-bit integers (W8 or W4), while keeping activations and normalization in floating-point (FP16/FP32) during inference for numerical stability.

## 3. Key Findings
- **4-Bit PTQ Benchmark:** On a Vision Transformer (ViT-Tiny) benchmark across four tasks (MNIST, FashionMNIST, CIFAR-10, SVHN), Q-PolyMerge (Adam STE) achieves an average accuracy of **48.87%**, outperforming unconstrained Q-Merge by **+2.85%** and naive post-merge quantization (M-then-Q) by **+5.95%**.
- **8-Bit PTQ Benchmark:** Q-PolyMerge (Adam STE) achieves **59.76%** average accuracy, matching unconstrained Q-Merge (60.03%) while reducing standard deviation by over 47% (from 2.36% to 1.22%) and recovering the unquantized continuous PolyMerge ceiling (61.00%).
- **Peak SRAM Reduction:** Bypassing backpropagation via the zero-order 1+1 ES pathway reduces the peak volatile memory (SRAM) footprint by **95.8% to 97.5%** (requiring only 4.05 MB under 4-bit PTQ compared to 162.72 MB for first-order gradient descent), making it compatible with typical edge hardware.
- **Compute Efficiency:** Due to the compact 12-parameter search space, 1+1 ES converges in just 100 iterations (100 forward passes, 0 backward passes), making it **16.7% computationally cheaper** than 40 steps of first-order backpropagation (equivalent to 120 forward passes).

## 4. Explicitly Claimed Contributions (with Evidence)
1. **Identification of the Overfitting-Optimizer Paradox:** The authors demonstrate that high-dimensional layer-wise test-time adaptation overfits to tiny calibration streams (16 images), creating jagged, unstable schedules. Evidence is provided via qualitative visualizations of learned coefficient profiles showing highly oscillating layer trajectories for unconstrained Q-Merge.
2. **The Q-PolyMerge Framework:** A parameter-efficient formulation projecting merging coefficients onto continuous polynomial subspaces. Supported by mathematical formulations of the monomial and Chebyshev scaling pathways.
3. **Dual Optimization Pathways:** Proposing both STE first-order gradients and 1+1 ES zero-order search, demonstrating that the low-dimensional projection enables gradient-free optimization to succeed where high-dimensional unconstrained search collapses. Supported by quantitative results in Tables 1 and 2.
4. **Systems-Level Edge Profiling:** Providing theoretical peak SRAM analysis, modeled hardware latency/energy profiles, and a concrete blueprint for fully-integerized activation execution. Supported by memory derivations and hardware metrics for ARM Cortex-M7 and RISC-V GAP8.
