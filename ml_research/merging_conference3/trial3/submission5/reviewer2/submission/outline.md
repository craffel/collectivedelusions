# Paper Outline: Q-PolyMerge

## Title
**Q-PolyMerge: Quantization-Aware Continuous Polynomial Subspace Model Merging for Extreme Edge Efficiency**

## 1. Introduction
- **The Edge Deployment Dilemma:** The explosion of task-specific, fine-tuned expert models (e.g., Vision Transformers) creates a high demand for multi-task unified models.
- **Model Merging:** Weight-space merging (e.g., Task Arithmetic) avoids the cost of multi-task training by combining experts, but edge devices strictly require low-bit quantization (INT8, INT4) to meet severe memory, power, and storage constraints.
- **The Conflict:** Merging full-precision models followed by post-training quantization (PTQ) degrades multi-task accuracy, whereas merging pre-quantized models fails due to disrupted linear mode connectivity in integer weight spaces.
- **The Overfitting-Optimizer Paradox:** Test-time adaptation (TTA) approaches like AdaMerging and Q-Merge attempt to optimize merging coefficients layer-by-layer. However, when optimizing on tiny, unlabeled calibration streams (e.g., 16 images), unconstrained optimization over $L \times K$ parameters easily fits transductive statistical noise, producing jagged, physically nonsensical layer coefficients that fail to generalize.
- **Our Contribution (The Pragmatist's Perspective):** We propose **Q-PolyMerge**, which constrains the coefficient search space to a low-degree polynomial subspace of normalized layer depth. This:
  1. Reduces the search space by over 78% (from $L \times K$ to $(d+1) \times K$ parameters).
  2. Functions as a robust low-pass filter, preventing overfitting to transductive noise and stabilizing the merged network.
  3. Enables zero-order black-box optimization (1+1 ES) to succeed on edge devices with zero activation caching and minimal compute.
  4. Provides a fully integerized pipeline, post-hoc quantizing classification heads to 8-bit for a 100% integer edge model.

## 2. Related Work
- **Linear Mode Connectivity (LMC) & Model Soups:** Foundations of weight averaging (Wortsman et al., Frankle & Carbin, Ainsworth et al.).
- **Multi-Task Model Merging:** Task Arithmetic (Ilharco et al.), RegMean (Jin et al.), Fisher averaging (Matena & Raffel), TIES-Merging (Yadav et al.).
- **Adaptive Test-Time Merging:** AdaMerging (Yang et al.), SyMerge (Jung et al.). Highlights the Overfitting-Optimizer Paradox under transductive TTA streams.
- **Post-Training Quantization (PTQ):** Standard PTQ techniques (Gholami et al., Jacob et al.), and specialized LLM quantization (AWQ, GPTQ, SmoothQuant).
- **Quantization in Merging:** Q-Merge, and emergent papers like 1bit-Merging, TVQ, HDRQ, E-PMQ.

## 3. Methodology
- **Problem Setup:** Base model $\Theta_{\text{base}}$ and $K$ experts fine-tuned on distinct datasets. $L$ sequential layers.
- **Polynomial Parametrization:** Instead of optimizing independent layer coefficients $\lambda_{k,l}$, we define:
  $$\lambda_{k, l}(\boldsymbol{\alpha}) = \sum_{j=0}^d \alpha_{k, j} \cdot \left( \frac{l}{L-1} \right)^j$$
  where $\boldsymbol{\alpha} \in \mathbb{R}^{K \times (d+1)}$ are the learnable polynomial parameters.
- **Quantization Operator:** Symmetric uniform PTQ. INT8 per-tensor for standard weights, and INT4 per-channel for low-bit weights to prevent outlier experts from ruining accuracy. Rounding/clamping functions.
- **Optimization Objective:** Minimize Shannon entropy $\mathcal{L}_{\text{TTA}}$ on unlabeled calibration streams.
- **First-Order Optimization via Straight-Through Estimator (STE):** Details on using backprop with standard Adam on $\boldsymbol{\alpha}$ by passing gradients through the round function using STE.
- **Zero-Order Optimization via 1+1 Evolution Strategy (1+1 ES):** Black-box search. Highlight that lowering the parameter space to $(d+1) \times K$ (e.g., 12 parameters) makes derivative-free 1+1 ES extremely efficient and feasible on hardware that cannot do backprop or cache activations.
- **100% Integer Weight Pipeline:** Detail post-hoc 8-bit quantization of task heads.

## 4. Experiments and Analysis
- **Experimental Setup:** Vision Transformer (`vit_tiny_patch16_224`, 5.7M parameters) grouped into $L=14$ layers. $K=4$ tasks: MNIST, FashionMNIST, CIFAR-10, SVHN. Disjoint splits (train, unlabeled calibration, test) across 3 seeds (42, 100, 2026).
- **Quantization Regimes:** 8-bit PTQ (Table 2) and 4-bit PTQ (Table 3).
- **Main Performance Findings:**
  - In 8-bit PTQ: Q-PolyMerge (Adam STE) achieves **60.84%** average accuracy, outperforming unconstrained Q-Merge (58.09%) and uniform merging (54.74%).
  - In 4-bit PTQ: Extreme degradation for all models due to the tiny size of `vit_tiny` (only 5.7M parameters), but Q-PolyMerge achieves the highest robustness and stability.
  - Zero-order efficiency: Q-PolyMerge (8-Bit ES) achieves **59.13%**, outperforming Q-Merge (8-Bit ES) at **56.14%**, proving the value of the low-dimensional search space for black-box search.
- **Visualization & Qualitative Analysis:**
  - Refers to `results/accuracy_comparison.png` showing the clear boost in task performance.
  - Refers to `results/coefficient_profile.png`, proving that unconstrained Q-Merge learns noisy, jagged trajectories (overfitting the transductive noise), while Q-PolyMerge maintains smooth, physically meaningful quadratic curves.

## 5. Conclusion & Discussion
- **Summary:** Summary of Q-PolyMerge as a highly practical, parameter-efficient, and robust solution for edge deployment.
- **Practical Impact:** Substantial savings in on-device storage (up to 75% for 4-bit, 50% for 8-bit) and inference memory bandwidth, with zero test-time training storage overhead.
- **Future Directions:** Extensions to larger models (e.g., LLMs, LLaMA-based experts) and hardware-in-the-loop validation on actual microcontrollers.
