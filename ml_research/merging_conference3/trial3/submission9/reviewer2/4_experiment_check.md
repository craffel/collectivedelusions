# Intermediate Evaluation 4: Experimental Check

## 1. Experimental Setup and Datasets
The experimental setup uses a compact Vision Transformer (\texttt{vit\_tiny\_patch16\_224}, $5.7$M parameters) across four diverse tasks: MNIST, FashionMNIST, CIFAR-10, and SVHN.
* **The Low-Data Constraint**: The experts are fine-tuned on only 512 images per task. This is an extremely restricted data budget. 
* **Impact on Absolute Performance**: Because of this budget, the absolute accuracies are very low. For example, individual unquantized experts achieve a mean of only $64.28\%$ (whereas fully converged models on full datasets would exceed $95\%$). The merged and quantized multi-task accuracies are lower still (e.g., $30.44\%$ for the optimal 4-bit configuration). 
* **The Generalization Question**: While this setup serves as a highly tractable "sandbox" to perform extensive grid sweeps, it raises a significant question of whether the "Flatness-Robustness Synergy" and the "Over-Perturbation Threshold" generalize to large-scale models trained on full datasets. At standard scale, models have substantially more capacity, and their loss landscapes may exhibit different curvature profiles, possibly shifting or altering the observed thresholds.

## 2. Evaluation of Baselines
The baseline comparison is exceptionally thorough and comprehensive, far exceeding the typical standard for empirical merging papers. The authors evaluate:
1. **SGD Q-Merge ($\rho=0.0$)**: Verifies the benefit of flat expert pre-training over standard expert pre-training.
2. **NaiveUniform**: Verifies whether test-time optimization is even necessary when experts are flat.
3. **AdaMerging-PostQ**: Compares direct quantized optimization (via STE) against unquantized FP32 coefficient optimization followed by post-hoc quantization.
4. **Individual-Quantized**: Establishes the upper bound of unmerged performance under quantization.
5. **Advanced Baselines & Ablations**:
   * **Convex Softmax Combination**: Validates the choice of independent coefficient clipping $[0, 1]$.
   * **DARE**: Demonstrates orthogonality to state-of-the-art parameter pruning and conflict resolution.
   * **High-Dimensional TENT Adaptation**: Validates the low-dimensional structural bottleneck hypothesis.
   * **Stochastic Weight Averaging (SWA)**: Isolates the unique benefit of SAM's adversarial formulation over standard trajectory averaging.
   * **Direct Curvature Profiling**: Measures the Hessian trace empirically via weight perturbations.

The inclusion of SWA and the direct empirical measurement of weight-space flatness are outstanding additions that provide direct support for their theoretical claims.

## 3. Do the Results Support the Claims?
Within the scope of the evaluated low-data sandbox, the results strongly and consistently support the claims:
* **Synergy is Precision-Dependent**: Tables 1 and 2 clearly show that SAM pre-training has a negligible effect under 8-bit quantization (retaining $\sim 44.6\%$ accuracy) but a highly significant effect under 4-bit quantization (boosting performance from $23.00\%$ to $30.44\%$, a $+7.44\%$ absolute gain).
* **Flatness Dominates Adaptation**: The fact that NaiveUniform on flat experts ($29.03\%$) beats FlatQ-Merge on SGD experts ($23.00\%$) by $+6.03\%$ is a striking result that validates their claim about pre-merging geometry being a primary driver of success.
* **Over-Perturbation Threshold**: The abrupt drop in performance at $\rho \ge 0.1$ (collapsing to near-random $\sim 11\%$ at $\rho = 0.2$) is clearly documented. The authors' task vector cosine similarity analysis provides a satisfying and original geometric explanation: large SAM radii force different task experts to converge to the *same* wide local minima, causing a loss of specialized task-specific features (representation convergence).
* **SWA vs. SAM**: The experiment in Section 4.8 is particularly elegant, showing that while SWA-trained flat experts perform well under moderate 8-bit noise, they fail under 4-bit noise because SWA only centers the model in a smooth valley on average. In contrast, SAM's adversarial perturbation objective active forces uniform coordinate-wise robustness, which is critical against severe low-bit rounding noise.

## 4. Empirical Strengths & Weaknesses
* **Strengths**: 
  * Exceptional thoroughness of evaluations and ablations.
  * Robust statistical reporting using 3 independent random seeds with standard deviations.
  * High internal consistency between the empirical findings, the curvature sweeps, and the direct flatness measurements.
* **Weaknesses**:
  * Limited to a tiny, low-data sandbox (ViT-Tiny, 512 images). 
  * Absolute performance levels are too low for practical multi-task deployment (e.g., $30\%$ accuracy across 10-class datasets).
  * Lack of scaling experiments on larger models (e.g., LLaMA or ResNets) to back up their speculative claims about LLMs.
