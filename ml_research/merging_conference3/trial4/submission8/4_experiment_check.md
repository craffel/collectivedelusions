# Empirical Evaluation: CR-PolySACM

We evaluate the experimental design, baseline selections, quantization schemas, and ablation studies.

---

## 1. Experimental Design and Representative Setup
The empirical evaluation is highly rigorous and thoroughly designed:
- **Backbone Architecture:** Utilizing the Vision Transformer (`vit_tiny_patch16_224`) from the `timm` library is appropriate, as it contains approximately 5.7M parameters grouped into $L=14$ layer-wise modules, providing a realistic, non-trivial testbed for model merging.
- **Expert Models:** Fine-tuned on four diverse classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) with strong task-specific test-set accuracies (96.30%, 86.90%, 90.20%, and 81.30%, respectively). This represents a highly competitive, genuine multi-task setup where experts represent genuine task-specific performance ceilings.
- **Calibration Stream:** Restricting the calibration dataset to $N=64$ samples ($16$ per task) and $T=40$ optimization steps represents an extremely realistic, lightweight test-time adaptation setting suitable for edge deployments.

---

## 2. Comprehensive Baseline Selection
The author compares CR-PolySACM against six prominent baselines:
1. **Uniform Task Arithmetic (No TTA):** Blending weights uniformly with constant coefficients (0.25).
2. **AdaMerging (Unregularized TTA):** The standard layer-wise adaptive merging baseline.
3. **RegCalMerge:** Incorporating Elastic Spatial Regularization (Total Variation penalty over adjacent layers).
4. **Q-Merge:** Quantization-aware adaptive merging using Straight-Through Estimators (STE).
5. **PolyMerge:** The state-of-the-art subspace-constrained adaptive merging baseline.
6. **HessMerge:** A sharpness-aware baseline applying unconstrained Sharpness-Aware Coefficient Minimization (SACM) in the 56-dimensional space.

This comprehensive selection ensures that the benefits of both subspace constraints and local flatness optimization are thoroughly benchmarked.

---

## 3. Realism of Deployment Quantization Schemas
The paper evaluates the merged models across six diverse, hardware-relevant quantization schemas:
- FP32 (unquantized continuous weights)
- INT8 Uniform Symmetric (Tensor-wise and Channel-wise)
- INT8 Uniform Asymmetric (Tensor-wise and Channel-wise)
- INT4 Uniform Symmetric (Channel-wise)

This system-wide sweep covers the vast majority of real-world edge hardware deployment formats, ensuring the robustness claims are highly generalizable.

---

## 4. Persuasiveness of Ablation Studies
The ablation studies are highly convincing and provide direct empirical validation of the paper's central theoretical claims:
- **Ablation of $\gamma$ (Table 4):** In HessMerge, increasing the unconstrained regularization strength $\gamma$ causes a monotonic performance degradation (e.g., from 49.12% down to 48.10% in FP32). This directly validates the task-vector norm scale pathology—since unconstrained perturbations are scaled by task-vector norms, the optimizer is blind to low-norm layers like Layer 13 and overfits to the less sensitive block layers, degrading performance.
- **Ablation of $\beta$ (Table 5):** Under aggressive INT4 quantization, sweeping the clipping threshold $\beta$ reveals a clear non-monotonic trend that confirms the two predicted failure modes:
  - **Gradient Explosion Regime ($\beta \le 0.01$):** Small $\beta$ results in massive perturbation scaling multipliers (>5000x) for Layer 13, causing immediate gradient explosion and collapsing accuracy to 11.20%.
  - **Scale Blindness Regime ($\beta \ge 0.25$):** Large $\beta$ effectively turns CR-SACM back into standard unnormalized SAM, making the optimizer blind to low-norm layers and reducing accuracy to 18.15%.
  - **Optimal Trade-off:** $\beta = 0.10$ achieves the absolute highest accuracy of **19.07%**.

---

## 5. Scholarly Honesty and Transparency
The author is exceptionally upfront about two major limitations of post-hoc model merging:
- **The Expert-to-Merge Drop:** They highlight a fundamental performance drop of $-31.27\%$ between the merged model's FP32 accuracy ($57.40\%$) and the average single-task expert accuracy ($88.67\%$). They honestly explain this as a consequence of domain disconnect and interference between orthogonal task-vectors in extreme multi-domain settings, showing that resolving this gap remains an open research question.
- **Low Absolute INT4 Accuracy:** While CR-PolySACM achieves a statistically significant relative improvement (+0.97% over PolyMerge) under INT4, they honestly state that an absolute joint mean accuracy of 19.07% is practically unusable for production systems. They frame this result as a valuable scientific proof of concept rather than a recipe for immediate INT4 edge deployment.

This high level of scholarly integrity significantly enhances the credibility and scientific value of the paper.
