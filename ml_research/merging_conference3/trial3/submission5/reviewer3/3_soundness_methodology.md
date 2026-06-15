# 3. Soundness and Methodology Evaluation

## Technical Soundness and Proof of Claims
From a systems and engineering standpoint, the methodology is clearly described, and the derivations of SRAM footprints and activation memory are highly meticulous. However, from a **theoretical perspective**, the paper is mathematically loose and lacks rigorous theoretical grounding. It frequently uses high-level mathematical terms as qualitative metaphors rather than establishing formal proofs or analytical guarantees.

### 1. Mismatch of "Low-Pass Filter" Claims
The authors repeatedly state that constraining the merging coefficients to a low-degree polynomial subspace "acts as a mathematical low-pass filter, filtering out high-frequency optimization noise and preventing overfitting." 
* **The Critique:** This is an analogy, not a mathematical fact. A formal "low-pass filter" operates in the frequency domain (e.g., Fourier or spectral representation of the parameter updates). The paper does not analyze the frequency spectrum of the optimization updates, nor does it define what "high-frequency optimization noise" means in the context of layer-wise coefficient search. Without a formal spectral analysis of the parameter trajectories, the claim that Q-PolyMerge "mathematically removes optimization noise" is unsubstantiated. It is simply a smooth parameter constraint.

### 2. Discontinuity-Continuity Mismatch in 4-Bit ES Search
The weight rounding operator $q(W)$ is a step function. Consequently, the test-time loss landscape $\mathcal{L}_{\text{TTA}}(\boldsymbol{\alpha})$ under 4-bit quantization is highly non-smooth, consisting of flat plateaus (where weight rounding wipes out small parameter changes) and sharp, discontinuous step-cliffs.
* **The Critique:** The authors propose a *smooth, continuous* polynomial constraint. While this regularizes first-order gradient descent (using Straight-Through Estimators), under the zero-order pathway (1+1 ES), Table 3 reveals that the discontinuous **Block-wise Constant (ES) baseline slightly outperforms Q-PolyMerge (ES) by 0.28%** (43.33% vs 43.05%).
* **Theoretical Implication:** This empirical anomaly indicates a fundamental theoretical limitation: **a smooth polynomial trajectory is mathematically mismatched for zero-order search in a discontinuous rounding landscape**. The hard boundaries of the block-wise constant parameterization provide localized step-perturbations that help random search escape flat local plateaus, whereas the smooth polynomial constraint restricts mutations to a global continuous trajectory, which smooths out and restricts the fine-grained mutations needed to navigate the fragmented rounding landscape. This contradicts the authors' general claim that the continuous polynomial trajectory is "mathematically superior" across both pathways.

### 3. Lack of Theoretical Foundations for the "Overfitting-Optimizer Paradox"
The "Overfitting-Optimizer Paradox" is a central concept of the paper, explaining why unconstrained optimization over $L \times K$ parameters overfits on small calibration streams (e.g., 16 images) compared to the $(d+1) \times K$ polynomial subspace.
* **The Critique:** While this is intuitively clear, the paper provides no formal generalization bounds, Rademacher complexity analysis, or statistical learning theory guarantees to mathematically prove why the polynomial subspace generalizes better. In transductive learning settings, the relationship between parameter dimensionality, calibration stream size, and generalization error can be analyzed theoretically (e.g., using margin-based generalization bounds under quantization). Bypassing this formal analysis in favor of purely empirical observations weakens the theoretical rigor of the paper.

### 4. Speculative Chebyshev Conditioning Analysis
In Appendix B.7, the authors analyze the 2-norm condition number of the monomial Vandermonde matrix $V_{L, d}$ to justify the future transition to orthogonal Chebyshev polynomials for deep models.
* **The Critique:** 
  1. Under the quadratic setting ($d=2$) used in all experiments, the condition number is only $\approx 20$, which is highly well-conditioned. Thus, the ill-conditioning problem does not exist for the presented results.
  2. The paper claims that increasing the degree ($d \ge 5$) causes "Runge's phenomenon, where wild oscillations occur near the boundary layers of the architecture." However, Runge's phenomenon is a classic property of polynomial interpolation of specific continuous functions on equidistant grids. Model merging is a variational optimization problem (minimizing entropy), not an interpolation of a fixed target function. The authors provide no proof or evidence that Runge's phenomenon actually occurs for neural network layer coefficients, making this theoretical connection highly speculative.
  3. No experiments are conducted with Chebyshev polynomials to show that they actually improve optimization stability or performance, even for deeper architectures.

### 5. No Convergence Guarantees for STE under Vandermonde Constraint
The Straight-Through Estimator (STE) utilizes a mismatched forward pass (rounding) and backward pass (identity mapping), which is known to cause biased gradients and unstable convergence in non-convex landscapes. When these biased gradients are backpropagated through a polynomial Vandermonde mapping:
$$\frac{\partial \mathcal{L}_{\text{TTA}}}{\partial \alpha_{k, j}} \approx \sum_{l=0}^{L-1} \left\langle \frac{\partial \mathcal{L}_{\text{TTA}}}{\partial \Theta^q_{\text{merged}, l}}, \mathbf{\Delta}_{k, l} \right\rangle \cdot \left( \frac{l}{L-1} \right)^j$$
and then clamped to $[-0.5, 1.5]$, the cumulative bias can lead to divergence. The paper provides no convergence proofs or stability boundaries for Adam STE under this specific linear projection constraint, leaving the optimization pathway theoretically ungrounded.

## Reproducibility
The methodology is extremely clear, with explicit formulas and detailed parameters (such as weight decay $\beta = 10^{-4}$ and clamping interval $[-0.5, 1.5]$). However, the reproducibility is primarily empirical, and the lack of a formal mathematical skeleton makes it difficult to extend the theoretical claims to other settings without further trial-and-error.
