# 3. Soundness and Methodology Evaluation

This file provides a rigorous technical evaluation of the paper's mathematical formulation, experimental design choices, potential flaws, and overall reproducibility.

## Clarity of the Description
The methodology is exceptionally well-written, structured, and mathematically rigorous. 
* The transitions between the phases of the framework (pre-training, linear merging, quantization, and test-time optimization) are clear and easy to follow.
* The mathematical derivations in Section 3.1 are complete and elegant. The paper includes a step-by-step second-order Taylor series motivation and a formal proof linking weight-space curvature to coefficient-space curvature ($H_{\Lambda} = T^T H_{\theta} T$).
* The authors include a complete algorithmic pseudocode (Algorithm 1) and an exhaustive table of hyperparameters (Table 3) in the Appendix, which dramatically enhances clarity.

## Appropriateness of Methods
* **SAM for Pre-merging Geometry Control:** Using SAM to control expert loss landscape geometry is highly appropriate. The theoretical explanation (smoothing weight-space Hessian eigenvalues) perfectly motivates why SAM-trained experts possess superior intrinsic resilience to discretization noise.
* **Symmetric Uniform Per-Channel Fake Quantization:** The formulation of PTQ (Equations 9-11) is standard and widely used in industry and academia, representing a highly realistic and practical deployment setting.
* **Unsupervised Entropy Minimization under STE:** Optimizing merging coefficients $\Lambda$ at test-time using joint prediction entropy is highly appropriate for zero-labeled edge deployment. Using the Straight-Through Estimator (STE) to flow gradients back through the rounding operation is a standard and sensible choice.
* **Vision Transformer Backbone:** Selecting a Vision Transformer (`vit_tiny_patch16_224`) as the evaluation sandbox is a robust, high-difficulty stress test. ViTs are known to lack inductive biases and suffer from sharp minima, making them notoriously sensitive to quantization noise and trajectories. Showing stable improvements on ViT-Tiny is technically compelling.
* **Separation of Task-Specific Heads:** The decision to keep the classification heads in FP32 and map them individually for each task is a standard and highly appropriate design in model merging, completely avoiding any structural shape incongruence issues.

## Potential Technical Flaws and Critical Assessments

### 1. Piecewise-Constant Landscape and STE Gradient Mismatch
The authors optimize the coefficients $\Lambda$ directly over the quantized loss landscape $\mathcal{L}_{\text{entropy}}(\Lambda)$, which is piecewise constant and non-differentiable due to the discrete `round` operator. Employing STE to estimate the gradient introduces a mathematical gradient mismatch error (the true gradient is zero almost everywhere, while the STE surrogate is non-zero). 
* **Critical Assessment:** In Section 3.4 and 4.4, the authors actively acknowledge this mismatch and explain why optimization remains stable. They point to (1) the extremely low-dimensional parameter bottleneck ($\Lambda$ is only 56 parameters), which averages out individual coordinate rounding errors over large weight tensors, and (2) careful tuning of the Adam learning rate ($\eta = 10^{-3}$), which prevents local oscillations. This is a very sound and intellectually honest defense of a potentially weak mathematical assumption.

### 2. Risk of Degenerate Class/Task Collapse
Unsupervised prediction entropy minimization (conditional entropy minimization) is highly susceptible to a degenerate global minimum where the model predicts a single class with 100% confidence for all inputs. Traditional TTA methods (like TENT or SHOT) require explicit diversity regularization to prevent this collapse.
* **Critical Assessment:** The authors argue that FlatQ-Merge does not require explicit diversity regularizers because the tight coefficient search space ($\Lambda \in [0, 1]^{L \times K}$, bounded and initialized at 0.3) acts as a strong structural bottleneck, physically constraining the network to interpolate within high-quality pre-trained task manifolds. They empirically validate this hypothesis in Section 4.7 by comparing FlatQ-Merge against a high-dimensional TENT-style baseline (which indeed completely collapses to random guessing). This technical defense is highly rigorous, and the ablation study perfectly resolves the concern.

### 3. Layer-wise Coefficient Behavior
The paper utilizes independent clipping bounds $[0, 1]$ rather than a normalized convex Softmax combination. 
* **Critical Assessment:** In Section 4.5, the authors conduct a robust comparison against a Softmax baseline, demonstrating that independent clipping outperforms Softmax by **+8.20%** in 8-bit and **+3.03%** in 4-bit. They analyze the layer-wise coefficients and find that the sum of coefficients remains highly stable across layers ($\mu = 1.221, \sigma = 0.082$) and individual coefficients stay within $[0.256, 0.345]$, never actually reaching the boundaries of $0.0$ or $1.0$. This reveals that the optimization operates on a tight, sub-pixel manifold near the initial state, showing that independent clipping works by freeing gradient exploration rather than by exploiting boundary clipping. This is an outstanding and highly thorough technical analysis.

## Reproducibility
The reproducibility of the submission is **Excellent**.
1. **Hyperparameters:** Appendix A (Table 3) lists every single hyperparameter used in both expert pre-training, quantization, and test-time coefficient optimization.
2. **Procedural Flow:** Appendix B (Algorithm 1) provides a complete, clear, and mathematically precise step-by-step procedure of the entire FlatQ-Merge pipeline.
3. **Data and Splits:** The training budget (512 images per task), calibration batch size ($N=64$), and evaluation split (1,000 images per task) are clearly specified.
4. **Architectural Details:** The specific backbone architecture, layer groups ($L=14$), and task heads are fully defined.

Practitioners would have absolutely no difficulty reproducing these exact experimental results.
