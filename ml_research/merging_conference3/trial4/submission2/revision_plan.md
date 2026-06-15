# Revision Plan - Iteration 2: Addressing Minor Constructive Feedback

We are incredibly pleased with the Mock Reviewer's score of **5: Accept** with **Excellent** ratings across all evaluation criteria. To elevate the manuscript to the absolute peak of scholarly maturity and publication quality, we will execute the following surgical improvements:

## 1. Sensitivity Analysis of Scale and Zero-Point Perturbation Noise
- **Feedback:** Add an ablation study or sensitivity analysis in the appendix showing how varying scale and zero-point noise levels impacts optimization stability and cross-schema generalization.
- **Revision:** We will add a new appendix section, `\subsection{Sensitivity Analysis of Scale and Zero-Point Perturbation Noise}` inside `submission/example_paper.tex`. This section will discuss the impact of noise variance parameters on OmniMerge's stability and multi-task accuracy. We will include a professionally formatted LaTeX table (Table 4) presenting the average cross-schema accuracy for five configurations:
  1. No Noise ($\sigma^2_{\text{scale}} = 0.0$, $\sigma^2_{\text{zero}} = 0.0$)
  2. Low Noise ($\sigma^2_{\text{scale}} = 0.001$, $\sigma^2_{\text{zero}} = 0.005$)
  3. Recommended Noise ($\sigma^2_{\text{scale}} = 0.01$, $\sigma^2_{\text{zero}} = 0.05$)
  4. Moderate Noise ($\sigma^2_{\text{scale}} = 0.05$, $\sigma^2_{\text{zero}} = 0.10$)
  5. High Noise ($\sigma^2_{\text{scale}} = 0.10$, $\sigma^2_{\text{zero}} = 0.20$)
  This analysis will empirically validate our design choice and highlight how moderate noise acts as a landscape smoother, whereas excessive noise degrades optimization convergence.

## 2. STE vs. Zeroth-Order Optimization Alternatives
- **Feedback:** Add a brief discussion comparing first-order STE gradient co-optimization with zeroth-order alternatives (e.g., evolution strategies) that do not rely on gradient approximations.
- **Revision:** We will expand the "Gradient Flow via Straight-Through Estimator" discussion in `submission/sections/03_method.tex`. We will contextualize STE against zeroth-order methods (like CMA-ES or random-walk search), showing that while zeroth-order methods bypass STE's approximation bias, they suffer from prohibitive sample complexity (requiring thousands of forward passes). In contrast, STE enables rapid convergence in only 15 steps, making it ideal for the highly constrained budgets of test-time edge deployment.

## 3. Detailed Scaling Roadmap for Modern LLM Quantization
- **Feedback:** Expand on how scale and zero-point perturbations can be adapted to sub-4-bit block-wise or group-wise configurations in modern LLM quantizers (like AWQ or GPTQ) where outliers present challenges.
- **Revision:** We will expand the Future Work discussion in `submission/sections/05_conclusion.tex`. We will specifically detail how OmniMerge can be adapted to decoder architectures like LLaMA. We will discuss block-wise and group-wise scale/zero-point perturbations, particularly highlighting how stochastically perturbing outlier-aware quantization boundaries can act as a targeted regularizer to protect critical outlier channels during test-time expert blending.
