# Peer Review

## Strengths and Weaknesses

### Strengths
1. **Pioneering Identification of the Overfitting-Optimizer Paradox:** The paper identifies and formalizes a critical, previously unaddressed failure mode in test-time adaptive model merging (e.g., AdaMerging): that unconstrained layer-wise optimization of coefficients on small, unlabeled target streams is highly prone to transductive overfitting, yielding jagged, physically unrealistic coefficient trajectories that collapse generalization performance on held-out test data.
2. **Exceptional Empirical Rigor and Statistical Soundness:** The primary simulation sweeps are executed across **30 independent random seeds** and report detailed standard deviations. The authors also perform formal paired t-tests over all 120 task evaluations to establish high statistical significance ($p < 10^{-12}$), setting an outstanding standard for empirical research in this domain.
3. **Multi-Axis and Multi-Optimizer Sweeps:** The paper maps a beautiful, physically grounded bias-variance curve by sweeping the polynomial degree ($d \in \{0, 1, 2, 3\}$) and evaluates both first-order gradient-based optimization (Adam GD) and zero-order derivative-free optimization (1+1 ES), providing rich scientific and optimization insights.
4. **End-to-End Physical Validation:** Rather than remaining purely in simulation, the authors validate their findings on a real PyTorch Residual MLP (over 10 seeds) and a physical pre-trained CLIP Vision Transformer (`openai/clip-vit-base-patch32`) using real test-set images and tokenized prompt embeddings, executing a fully differentiable test-time adaptation pipeline.
5. **Discrete vs. Continuous Tuning Advantage:** The authors rightly point out the "Hyperparameter Calibration Dilemma" in test-time adaptation: standard regularizers like TV and L2 require dense continuous tuning, which is impossible at test time because labeled validation sets are unavailable. PolyMerge replaces this with a discrete, robust architectural choice (degree $d$).
6. **Theoretical Rigor:** The Appendix provides complete formal proofs for Proposition 3.1 (low-pass filtering of white Gaussian and alternating noise) and Section 9 (projected Hessian curvature flatness), successfully linking the structural constraints to flatter local loss basins and robust generalization.

### Weaknesses
1. **Task-Specific Adaptation Collapse in MLP Validation (Table 5):** 
   In the PyTorch Residual MLP validation, the unoptimized static Task Arithmetic baseline achieves a multi-task test accuracy of **85.90% $\pm$ 3.28%**. However, unconstrained TTA (85.63% $\pm$ 2.70%), TV Regularization (85.67% $\pm$ 2.25%), and PolyMerge ($d=2$) (85.43% $\pm$ 2.18%) all *underperform* the static baseline in terms of final generalization accuracy. Although the authors claim PolyMerge "stably minimizes entropy," they do not critically discuss why the entire test-time adaptation process fails to beat or even match the unoptimized starting point on this architecture, which raises questions about the practical utility of TTA on this MLP task.
2. **Underfitting Bottleneck of Global Polynomials in CLIP Validation (Table 6):**
   In the physical CLIP validation, Global PolyMerge ($d=2$) drops multi-task accuracy from 94.00% (static baseline) to 89.00%, and PolyMerge ($d=4$) only recovers to 90.00%. This is a significant drop that reveals a clear limitation: global polynomials are too rigid and suffer from a severe underfitting bottleneck on functional weights where layer-wise sensitivities are highly heterogeneous and non-monotonic. While the authors propose SplineMerge (Piecewise Constant) to resolve this (achieving 96.00% accuracy), it indicates that the core "PolyMerge" global polynomial framework is insufficient for realistic foundation models.
3. **No Confidence Intervals / Multiple Seeds for CLIP Validation (Table 6):**
   Unlike the simulation sweeps (30 seeds) and MLP validation (10 seeds), Table 6 reports single-run accuracies without any statistical error bars or multiple seeds. Since it evaluates a very small stream of only 50 images per dataset, the results might have high statistical sampling variance. Running the physical foundation model validation across multiple seeds is necessary to guarantee statistical soundness.
4. **Lack of Broader Physical Baselines:**
   While the baseline selection in the simulation is comprehensive, the physical CLIP validation (Table 6) only compares PolyMerge against Task Arithmetic, Unconstrained TTA, and TV Regularized Adam. Since a physical setup was implemented, the authors should have compared PolyMerge against other competitive or concurrent physical model merging baselines (e.g., AdaMerging++, L2 regularization, or SyMerge) to provide a fairer and stronger evaluation.

---

## Soundness
**Rating:** Good

**Justification:**
The paper is methodologically highly sound. The mathematical formulations are rigorous, the emulation setups (convex and coupled non-convex) are carefully calibrated, and the physical validations on PyTorch Residual MLP and CLIP ViT-B/32 ensure that findings hold under physical backpropagation and real-world weight-space dynamics. The theoretical proofs (Proposition 3.1 and projected Hessian curvature analysis) are correct and elegant. 
However, the soundness rating is capped at "Good" due to:
- The unexplained discrepancy in Table 5 where all adapted models underperform the static baseline.
- The severe underfitting bottleneck of global PolyMerge on real CLIP weights (Table 6), proving global polynomials are too rigid for pre-trained foundation models.
- The lack of multiple seeds and error bars for the physical CLIP validation, which is evaluated on a small sample size of 50 images.

---

## Presentation
**Rating:** Excellent

**Justification:**
The paper is exceptionally well-written, clear, and logically structured. The introduction immediately and honestly clarifies the simulation vs. physical setup, avoiding any scientific ambiguity. The related work is comprehensive and positions the paper perfectly. The figures are high-signal, beautifully depicting coefficient profiles (Figure 1), bias-variance curves (Figure 2), and TTA loss trajectories (Figure 3). Complete PyTorch integration code and detailed experimental configurations are provided in the Appendix, ensuring maximum clarity and reproducibility.

---

## Significance
**Rating:** Excellent

**Justification:**
Model merging is rapidly becoming the dominant paradigm for combining task-specific expert models without retraining costs. As the field shifts toward test-time adaptive merging, identifying and resolving the Overfitting-Optimizer Paradox is of paramount importance. By demonstrating that unconstrained optimization leads to transductive overfitting and proposing continuous/piecewise continuous subspace constraints (PolyMerge and SplineMerge) as a robust, hyperparameter-free alternative, this paper provides a highly valuable and durable contribution. The release of a lightweight, CPU-only weight-merging simulator will also democratize TTA merging research, allowing researchers to prototype new optimizers in seconds.

---

## Originality
**Rating:** Good

**Justification:**
While subspace learning, polynomial interpolation, and piecewise splines are classic mathematical techniques, their specific application to regularize weight-space test-time adaptation in model merging is highly creative and original. Framing the spatial high-frequency oscillations as transductive noise and proving that a Vandermonde projection acts as a spatial low-pass filter is a novel and elegant insight. The transition from global polynomials to localized splines (SplineMerge) to capture block-wise layer heterogeneity is a mature and well-thought-out contribution.

---

## Overall Recommendation
**Rating:** 5: Accept

**Justification:**
This is a technically solid, highly rigorous, and exceptionally well-written paper that addresses a highly relevant problem (the Overfitting-Optimizer Paradox) in test-time adaptive model merging. 
- **Empirical Rigor:** The paper sets a gold standard for TTA research by evaluating simulated benchmarks across 30 random seeds, executing formal t-tests, and measuring wall-clock step latency.
- **Physical Validation:** The authors complement their simulations with functional PyTorch Residual MLP and CLIP ViT-B/32 validations on real images.
- **Subspace Advantage:** The authors demonstrate that PolyMerge provides a powerful, discrete-degree alternative to penalty-term regularizers, which are impossible to tune during online TTA. It also provides an exponential complexity advantage for zero-order black-box optimization (1+1 ES), significantly outperforming TV-regularized ES.
- **SplineMerge Breakthrough:** The introduction of SplineMerge successfully captures block-wise layer transitions in pre-trained CLIP weights, resolving the underfitting bottleneck of global polynomials and matching the peak performance of unconstrained TTA while suppressing spatial roughness.

While there are some minor empirical discrepancies (such as TTA underperforming static baselines in the MLP task, and a lack of multiple seeds for CLIP validation), these are minor limitations in an otherwise outstanding, comprehensive, and highly significant paper. I strongly recommend accepting this submission.
