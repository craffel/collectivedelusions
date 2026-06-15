# 2. Novelty Check and Assessment of Delta

## Characterization of Novelty
The overall novelty of **Q-PolyMerge** is characterized as **incremental but highly pragmatic**. It does not introduce an entirely new machine learning paradigm (like a new family of models or a completely novel optimization theory). Rather, it identifies a severe and previously neglected engineering flaw in test-time model merging—specifically, the **Overfitting-Optimizer Paradox** under data-scarce test-time adaptation—and proposes an elegant, mathematically grounded structural constraint (projecting parameters onto a continuous polynomial subspace of layer depth) to resolve it.

---

## Key Novel Aspects

### 1. Continuous Polynomial Subspace Constraint for Merging Coefficients
While prior works (e.g., AdaMerging, Q-Merge) optimize merging coefficients layer-by-layer, they treat each layer block's coefficient as an independent parameter. This leads to a high-dimensional optimization landscape. The core novel idea of Q-PolyMerge is to enforce **layer-depth continuity** on the merging coefficients. By representing the layer coefficients as a continuous polynomial of normalized depth:
$$\lambda_{k, l} = \sum_{j=0}^d \alpha_{k, j} \left(\frac{l}{L-1}\right)^j$$
the paper introduces a structural prior that incorporates the inductive bias that adjacent layers in a deep neural network share functional similarity and should have smoothly transitioning task mixtures. This smooth constraint acts as a low-pass filter to reject optimization noise.

### 2. Integration of Polynomial Subspace with the Straight-Through Estimator (STE)
The paper details the backpropagation of pseudo-gradients through the non-differentiable uniform quantization rounding operator to optimize the continuous polynomial parameters $\boldsymbol{\alpha}$:
$$\frac{\partial \mathcal{L}_{\text{TTA}}}{\partial \alpha_{k, j}} \approx \sum_{l=0}^{L-1} \left\langle \frac{\partial \mathcal{L}_{\text{TTA}}}{\partial \Theta^q_{\text{merged}, l}}, \mathbf{\Delta}_{k, l} \right\rangle \cdot \left( \frac{l}{L-1} \right)^j$$
This shows how the polynomial basis elements weight the accumulated layer-wise task-vector gradients, allowing the optimizer to globally adjust the entire backbone trajectory using a single gradient update.

### 3. Enabling Highly Efficient Zero-Order On-Device Optimization
Derivative-free black-box optimizers (like 1+1 ES or CMA-ES) are notoriously susceptible to the curse of dimensionality, making them fail when searching over unconstrained spaces (e.g., 56 parameters in a 14-layer ViT). By reducing the search space to a low-dimensional 12-parameter space, Q-PolyMerge successfully unlocks the potential of zero-order optimization for test-time adaptation on edge microcontrollers, demonstrating stable convergence in just 100 iterations.

---

## The "Delta" from Prior Work

### 1. Delta from AdaMerging (Yang et al., 2024)
- **AdaMerging:** Optimizes unconstrained layer-wise coefficients $\lambda_{k,l}$ in a full-precision space. When applied to low-bit PTQ, it is typically optimized offline (server-side) in FP16 and then quantized post-hoc, or optimized unconstrained on-device.
- **Q-PolyMerge:** Explicitly targets the *quantized* weight landscape and constrains the search space to a polynomial subspace. This reduces parameters by over 78% (e.g., from 56 to 12). Under data scarcity (e.g., 16 images), Q-PolyMerge prevents the transductive overfitting that plagues AdaMerging.

### 2. Delta from Q-Merge (Yang et al., 2024 / Concurrent literature)
- **Q-Merge:** Performs direct unconstrained quantization-aware merging, optimizing independent coefficients $\lambda_{k,l}$ under the quantization operator using STE or 1+1 ES.
- **Q-PolyMerge:** Introduces the continuous polynomial trajectory constraint over normalized depth. Under unconstrained Q-Merge, the learned schedules are highly jagged and noisy (overfitting). Under Q-PolyMerge, they are smooth quadratic curves, providing lower variance across seeds (by over 47% under 8-bit Adam) and better generalization. Under zero-order search, unconstrained Q-Merge ES collapses (due to high dimensionality), while Q-PolyMerge ES remains stable.

### 3. Delta from Concurrent Low-Bit Merging Methods (TVQ, E-PMQ, 1bit-Merging)
- **TVQ, E-PMQ, 1bit-Merging:** These are static, offline methods designed to align representations or compress task vectors. They do not address dynamic, stream-based test-time adaptation (TTA), nor do they address the severe SRAM memory constraints of physical edge nodes (which Q-PolyMerge resolves by providing a zero-gradient-memory zero-order pathway).
- **Q-PolyMerge:** Specifically designed for dynamic on-device TTA under extreme data scarcity, using the polynomial prior to regularize transductive search.
