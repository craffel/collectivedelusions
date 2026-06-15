# 3. Soundness and Methodology Evaluation

## Methodological Strengths
1. **Mathematical Formulation:** The continuous polynomial parameterization is clearly described, and the normalization of layer depth ($\bar{l} = \frac{l}{L-1} \in [0, 1]$) is mathematically elegant. It guarantees scale invariance across networks of different depths and prevents numerical divergence in deeper architectures.
2. **First-Order Gradient Derivation:** The paper provides a clear, step-by-step derivation of the gradient flow through the continuous polynomial constraint and the straight-through estimator (STE) on the uniform rounding operator:
   $$\frac{\partial \mathcal{L}_{\text{TTA}}}{\partial \alpha_{k, j}} \approx \sum_{l=0}^{L-1} \left\langle \frac{\partial \mathcal{L}_{\text{TTA}}}{\partial \Theta^q_{\text{merged}, l}}, \mathbf{\Delta}_{k, l} \right\rangle \cdot \left( \frac{l}{L-1} \right)^j$$
3. **Pragmatic Regularization:** The use of trajectory clamping (restricting $\lambda_{k, l}$ to $[-0.5, 1.5]$) and $L2$ weight decay on the polynomial parameters $\boldsymbol{\alpha}$ is a sensible and practical safeguard to protect the dynamic range of weight values prior to uniform PTQ rounding.
4. **Reproducibility:** The experimental hyperparameters (seeds, sample sizes, learning rates, epochs) and architectures (ViT-Tiny patch16_224) are fully documented, making the research highly reproducible.

---

## Major Soundness Concerns and Technical Flaws

### 1. The Zero-Order On-Device Adaptation "Utility Paradox"
The authors build a substantial portion of their narrative around on-device viability and how Q-PolyMerge makes test-time adaptation feasible on low-power microcontrollers. They argue that first-order gradient descent (Adam STE) is physically unviable on-device due to its huge SRAM footprint (158.40 MB activation cache), leaving zero-order search (1+1 ES) as the **only physically viable on-device pathway**.

However, a rigorous empirical comparison reveals a critical contradiction in the actual utility of this zero-order pathway:
- **Under 8-Bit PTQ:** 
  - The naive, unadapted **Merge-then-Quantize (M-then-Q)** baseline (which requires **zero** test-time computation and **zero** on-device SRAM) achieves an average accuracy of **55.11 $\pm$ 0.22%**.
  - The proposed **Q-PolyMerge (ES)**, after running 100 iterations of zero-order adaptation on-device, achieves only **51.03 $\pm$ 4.35%** average accuracy.
  - *This means that performing on-device zero-order adaptation actively degrades performance by **-4.08%** compared to the naive unadapted model, while introducing substantial computational and runtime overhead.*

- **Under 4-Bit PTQ:**
  - The naive, unadapted **M-then-Q** baseline achieves **42.92 $\pm$ 2.06%** average accuracy.
  - The proposed **Q-PolyMerge (ES)** achieves **43.05 $\pm$ 1.90%** average accuracy.
  - *The on-device adaptation provides a negligible **+0.13%** improvement, which is completely within the standard deviation (1.90%). Essentially, there is no statistically significant benefit to performing zero-order adaptation under 4-bit quantization.*

**Conclusion:** The authors' only physically viable on-device adaptation pathway (zero-order ES) is either **actively harmful (8-bit)** or **useless (4-bit)** compared to simply doing nothing (the naive unadapted M-then-Q baseline).
Since first-order Adam is physically impossible to run on-device, and the only edge-viable method (ES) does not outperform the unadapted baseline, the core claim that Q-PolyMerge provides a viable on-device adaptation pipeline is severely undermined.

### 2. Optimization Failure of Zero-Order Search vs. Capacity of Polynomial Subspace
The empirical results reveal that the polynomial subspace *does* have the capacity to represent high-performing models:
- **Q-PolyMerge (Adam STE)** achieves **59.76%** under 8-bit and **48.87%** under 4-bit.
This proves that there exist configurations of the 12 polynomial parameters that significantly outperform both the unadapted M-then-Q baseline (55.11% in 8-bit, 42.92% in 4-bit) and the unconstrained Q-Merge (Adam) baseline.

However, the zero-order search (1+1 ES) fails to locate these high-performing parameters, collapsing to **51.03%** and **43.05%**. This represents a severe **optimization failure** of the 1+1 ES algorithm in navigating the highly discontinuous, step-like quantization landscape. Even with "Coordinate Descent with Greedy Backtracking" implemented for 4-bit ES, it barely matches the naive unadapted baseline.
The authors need to address this gap: if the edge-viable optimizer cannot find the good parameter configurations, the theoretical capacity of the polynomial subspace remains academic.

### 3. Floating-Point Activation Assumption on Microcontrollers
The proposed "integer-weight edge pipeline" keeps activations and layer normalizations in floating-point (FP16/FP32). 
The authors state: *"A truly edge-optimal execution pipeline would require transitioning to a fully-integerized arithmetic pipeline (e.g., W8A8 or W4A8 execution)..."*
However, on ultra-low-power microcontrollers lacking hardware FPUs (e.g., ARM Cortex-M0 or many RISC-V nodes), executing activations and normalization in floating-point requires software emulation. This software emulation introduces severe latency and energy penalties that often exceed the energy saved by compressing weights to low bits. Thus, the practical utility of the "integer-weight but floating-point activation" pipeline on the lowest-power edge nodes remains questionable.
While the authors provide a "Blueprint for Fully-Integerized Activation and Operator Execution" in Appendix B.5, this is purely theoretical and is not implemented or validated in their experiments.
