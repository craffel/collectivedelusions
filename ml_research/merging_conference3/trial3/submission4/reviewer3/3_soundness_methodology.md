# 3_soundness_methodology.md: Soundness and Methodology Evaluation

## Clarity of the Description
The methodology of this paper is described with exceptional clarity and mathematical rigor:
- **Mathematical Formulations:** Every step of the framework is explicitly formulated, from task-vector generation, on-the-fly magnitude pruning, and percentile thresholding, to the four proposed unsupervised regularized objectives (Equations 1 to 19).
- **Algorithmic Tracing:** Algorithm 1 provides a cohesive, step-by-step overview of the dual-engine co-optimization loop (first-order STE and zero-order ES), clarifying exactly how continuous coefficients and binary masks interact during test-time adaptation.
- **PEFT Alignment Formulation:** The step-by-step mathematical tracing of the Orthogonal Procrustes SVD Alignment (Section 4.3.3) is extremely clean, making it immediately implementable for readers with negligible computational overhead.

## Appropriateness of Methods
The experimental design and chosen methods are highly appropriate for investigating edge-device deployment limits:
- **Realistic Edge Models:** Testing on a compact Vision Transformer (\texttt{vit\_tiny}) and ResNet-18 represents realistic, tightly constrained edge backbones, avoiding the academic trap of only testing massive models where capacity hides structural weaknesses.
- **Unsupervised Test-Time Calibration:** Minimizing Predicted Shannon Entropy on a tiny calibration set of 16 unlabeled images per task is highly pragmatic since ground-truth labels are rarely available in production environments.
- **First- and Zero-Order Optimization Engines:** Exploring both Straight-Through Estimator (STE) and 1+1 Evolution Strategy (1+1 ES) is highly appropriate to address the non-differentiable pruning threshold, providing a complete picture of gradient-based vs. gradient-free adaptation behavior.
- **Robust Baselines:** Evaluating against standard baselines (Uniform, AdaMerging, and the decoupled Prune-then-Merge baseline) allows for a fair assessment of whether joint test-time adaptation is actually superior to simpler decoupling.

## Potential Technical Flaws & Limitations
- **Noisy Input Experts:** The SVHN expert achieves only 19.59% accuracy. This indicates that it is barely converged, and when its weights are merged, it acts as a "poison pill," corrupting the shared backbone with parameter-space noise. However, the authors do not hide this; they actively diagnose this as "The Noisy Expert Noise Injection Constraint" (Section 4.2.4), turning a potential flaw into an insightful systems-level lesson.
- **Aggressive High-Sparsity Bounds:** Applying 80% unstructured pruning on a tiny 5.7M parameter ViT backbone is extremely aggressive and almost guaranteed to damage representations. However, as an "extreme stress-test" of weight-space limits, this choice is scientifically valuable.
- **Unconstrained Entropy Vulnerability:** Unconstrained entropy minimization is known to be vulnerable to degenerate solutions (e.g., predicting a single class with high confidence). The authors acknowledge this and propose four regularized objectives (MMI, soft pseudo-labeling, LRA, and CBC) and empirically validate them, which successfully addresses this potential vulnerability.

## Reproducibility
The paper exhibits an exceptionally high standard of reproducibility:
- **Detailed Hyperparameters:** All hyperparameters (optimizer configurations, learning rates, ES scaling factors $\alpha_{\text{up}}/\beta_{\text{down}}$, regularization strengths $\gamma$, distillation scale $\beta$) are explicitly listed.
- **Low Variance:** The authors report results across 5 independent random seeds. The standard deviation is extremely tight ($\pm 0.32\%$), indicating that the findings are robust and not a stochastic anomaly of specific calibration subsets.
- **Open-Source Codebase:** A complete, PyTorch-compatible codebase is open-sourced under the MIT License, which contains configuration schemas, evaluation scripts, and fine-tuned expert checkpoints, ensuring straightforward reproducibility.
