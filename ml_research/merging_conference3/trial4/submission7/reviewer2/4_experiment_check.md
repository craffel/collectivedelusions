# 4. Experimental Evaluation and Verification

## Critical Evaluation of the Setup
The experimental design is highly rigorous, utilizing **30 independent random seeds** ($42 \le \text{seed} \le 71$) to report mean accuracies and standard deviations across five evaluation suites. This stands in sharp contrast to many modern deep learning papers that report results on a single seed.
The deconstruction of the optimization budgets (Table in Section 4.4) is a standout highlight, standardizing Adam and L-BFGS-B across online and offline settings to prove that the performance gains of OFS-Tune are driven by its structural trajectory constraints rather than optimization advantages.

## Baselines
The paper includes an exceptionally comprehensive set of baselines:
1. **Static Uniform baseline (Task Arithmetic):** The standard starting point for model merging.
2. **Online AdaMerging:** The primary unconstrained online TTA baseline.
3. **Online PolyMerge:** The primary polynomial-constrained online TTA baseline.
4. **Offline OFS-Unconstrained (Ablation):** A vital baseline that isolates the value of the polynomial trajectory constraint from the effect of having labeled few-shot data.
5. **Temporal Smoothing / Parameter EMA (Ablation):** An essential baseline in physical weight-space validation that tests whether online methods can be rescued from transductive noise using parameter smoothing ($\beta = 0.90$).

## Do the Results Support the Claims?
Yes, the empirical results strongly support the paper's core theses:
- **Task Suite Bias:** Table 1 clearly shows that the relative rankings of methods shift dramatically between suites. For instance, while online methods appear highly successful in the control Suite E (AdaMerging = 84.09%, PolyMerge = 84.96% vs. Uniform = 72.00%), AdaMerging's performance in Suite B (62.58% ± 5.71%) is highly unstable and lags behind PolyMerge by $5.93\%$.
- **Transductive Overfitting:** Figure 3 visually captures how unconstrained AdaMerging and OFS-Unconstrained oscillate wildly from layer to layer, fitting local noise, while OFS-Tune establishes a smooth, linear trajectory that tightly tracks the ground truth.
- **The Simulated-to-Physical Gap:** In the physical connected basin (Regime B), online methods actually *degrade* the robust pre-trained weights, dropping from the Uniform baseline of 82.20% down to 78.80% (AdaMerging) and 79.30% (PolyMerge) because of the misalignment of unsupervised entropy. OFS-Tune successfully avoids this, achieving 83.00% accuracy.

## Specific Gaps and Weaknesses
From a practitioner's perspective, there are several notable empirical gaps and weaknesses in the evaluation:

1. **Missed Opportunity on Calibrated ViT-B/32 Weights:**
   The authors already fine-tuned independent and pre-trained experts on a 12-layer Vision Transformer (ViT-B/32) backbone to empirically calibrate the curvature parameters of their Model II Landscape (described in Appendix A.2). Given that these fine-tuned ViT-B/32 weights were already available, it is a **major missed opportunity** not to run physical weight-space merging experiments directly on these ViT-B/32 models. Doing so would have closed the "toy-scale" gap and provided a highly convincing, medium-scale physical weight-space validation, rather than relying solely on a 5-layer CNN.
2. **Highly Saturated and Simple Datasets:**
   The physical weight-space validation is restricted to MNIST and FashionMNIST. MNIST is a highly saturated, simple grayscale digit dataset where models easily achieve 97-98% standalone classification accuracy with basic translation-invariant features. Merging independent experts on simple digit classification might not represent the complex, high-conflict representational clashing observed in modern multi-task learning (e.g., merging LLMs fine-tuned on code generation versus mathematical reasoning). Evaluated on more complex benchmarks (like CIFAR-100 or TinyImageNet) in physical space, the performance dynamics of these methods might shift.
3. **No Physical Validation for Localized Trajectory constraints:**
   In Section 4.3, the authors discuss the "non-smooth zig-zag" optimal trajectory and show that localized Piecewise Splines and Block-wise Parameter Sharing recover performance in simulation (Table 4). However, they **do not validate these localized formulations in physical weight-space experiments**. A practitioner would want to see if grouping attention layers and MLP layers separately in a physical Transformer backbone (like a ViT) actually yields the promised structural benefits in practice.
