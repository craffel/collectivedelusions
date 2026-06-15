# Review Step 3: Soundness and Methodology Evaluation

## 1. Assessment of Technical Soundness
The methodology proposed in this paper is mathematically rigorous, technically sound, and extremely well-justified. The authors build a complete bridge between digital signal processing and parameter optimization in deep learning, and each design choice is supported by thorough theoretical and empirical justifications.

## 2. Key Strengths of the Methodology

* **Mathematical Rigor of the DCT-II and IDCT-III:**
  The paper defines the orthonormal DCT-II (forward) and IDCT-III (inverse) transformations with precision. The normalization factor $\gamma_j$ is correctly incorporated to guarantee that the forward and inverse transforms are exact mathematical duals. This preserves orthonormality, meaning the transformation matrix $T_{DCT}$ satisfies $T_{DCT}^T T_{DCT} = I$, and its condition number is exactly $1.0$ (perfect conditioning) at any scale.

* **Rigorous Comparison of Numerical Conditioning:**
  The critique of PolyMerge's polynomial basis is mathematically solid. Polynomial bases like standard power series suffer from extreme collinearity. Reconstructing coefficients requires solving equations governed by a Vandermonde matrix, whose condition number grows exponentially with polynomial degree. This leads to unstable, non-convex optimization landscapes for deep architectures ($L \ge 48$). The paper includes a dedicated figure (Figure 1) displaying condition numbers for polynomial Vandermonde bases, Chebyshev polynomial bases, and the DCT-II basis. It shows that while Chebyshev polynomials improve conditioning slightly over Vandermonde, they still grow exponentially on uniform grids and suffer from Runge's boundary run-away phenomenon. The DCT-II basis, being perfectly orthonormal on uniform grids, maintains a condition number of $1.0$ at all scales.

* **Analysis of Boundary Conditions and Symmetry:**
  The choice of DCT-II over DFT and DST is deeply analyzed and validated. DFT introduces artificial periodic wrap-around requirements, leading to high-frequency Gibbs-like oscillations. DST assumes an odd-boundary symmetric extension which forces the boundary coefficients to zero, leading to severe artificial underfitting at the first and last layers. The DCT-II's even-symmetry extension smoothly handles boundary spatial derivatives, allowing the critical endpoints (input layers and output classification heads) to adapt freely without artificial constraints or gradient spikes. The authors substantiate this with an empirical comparison against DST, showing a $15\%$ slower convergence rate and $4.5\%$ absolute reduction in multi-task accuracy for DST.

* **Addressing Architectural Heterogeneity:**
  A common critique of continuous trajectory constraints is that they ignore the physical reality of heterogeneous network layers (e.g., MHA projections vs. MLP feedforward layers). The paper elegantly resolves this through **Block-wise and Layer-type Spectral Merging**. By partitioning the layers into homogeneous subsets and applying independent DCT-II transforms within each subset, SpectralMerge successfully preserves block-specific sensitivities while maintaining spectral regularization within each family.

* **PEFT-Induced Step-Function Discontinuity:**
  The paper provides a brilliant, highly rigorous diagnosis of why hard spectral filters (SpectralMerge-LP and LP-Adaptive) collapse to random guessing ($29.00\%$) in the pre-trained ResNet-18 CIFAR-10 evaluation, while soft spectral decay (SpectralMerge-Reg) achieves a blowout accuracy of $54.00\%$. Under localized fine-tuning (where only deep layers are updated), the parameter sensitivity across layers forms a step function. From DSP principles, a step function has infinite frequency support. Thus, hard-cutoff filters catastrophically underfit by removing the required high-frequency components. SpectralMerge-Reg, through its soft decay, allows the validation gradients to activate specific localized high-frequency coordinates while still penalizing unneeded high-frequency noise. This mathematical explanation is extremely elegant and reveals high technical depth.

* **Computational Complexity and Scalability:**
  The paper analyzes the computational complexity, noting that a 1D DCT-II and IDCT-III can be computed in $\mathcal{O}(L \log L)$ time using the Fast Cosine Transform or $\mathcal{O}(L^2)$ time. For standard deep networks, this transform requires less than $0.05$ milliseconds, which is less than $0.0001\%$ of a single model forward pass. This guarantees that SpectralMerge introduces zero computational or latency bottlenecks during training or test-time adaptation. The paper also discusses scaling to gradient-based optimization and parameter-wise merging in the future (e.g., via Wavelets).

## 3. Methodological Strength: Robustness to Objective Noise and Optimizer Selection

In previous iterations of parameterized merging, optimization under data-scarce (few-shot) regimes relied heavily on derivative-free local search algorithms such as the Nelder-Mead simplex search. However, Nelder-Mead is notoriously sensitive to stochastic validation noise due to its reliance on strict inequality rankings of simplex vertices, which are easily corrupted under small validation sample sizes ($M \in [5, 50]$). 

The authors have addressed this optimization vulnerability exceptionally well in this work:
* **Gradient-Based Optimization via Adam:** They deliberately employ momentum-based gradient optimization (Adam) across all evaluation settings (simulations, physical MLP, and ResNet-18 checkpoints). By computing exact analytical gradients backpropagated through the IDCT mapping chain rule, the optimizer averages gradients over multiple steps, smoothing out stochastic validation noise and enabling highly stable updates.
* **Comparative Optimizer Analysis:** In Appendix A, the authors include a detailed empirical and theoretical comparison between Adam and Nelder-Mead. Under extremely noisy few-shot regimes ($M=5$), Nelder-Mead's vertex rankings are corrupted, causing premature simplex collapse and stalling at $82.10\%$ accuracy, whereas Adam with gradient smoothing converges rapidly and stably to $86.20\%$.
* **Computational Scalability:** The authors demonstrate that while derivative-free simplex search scales poorly as $O(2^d)$ in runtime complexity (making large-scale model merging unviable), gradient-based automatic differentiation scales linearly, allowing OFS-Tune to scale smoothly to dozens or hundreds of tasks.

This represents a major methodological strength, providing a theoretically sound and empirically validated optimization framework that is robust to small-sample validation noise.

## 4. Rating of Soundness
**Excellent.** The mathematical formulation is flawless, the DSP concepts are mapped to deep learning with high rigor, and the theoretical and empirical analyses of numerical conditioning, boundary extensions, layer-type heterogeneity, and step-function discontinuities are exemplary.
