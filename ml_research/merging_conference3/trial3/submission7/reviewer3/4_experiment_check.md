# 4. Experimental Evaluation Check

## Experimental Design and Setup
The empirical setup is designed to evaluate the physical granularity of merging coefficients in a controlled environment. Key elements include:
* **Model**: A custom, differentiable, coefficient-aware Vision Transformer (`MergedViTTiny`).
* **Datasets**: Four distinct classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN).
* **Calibration Budget**: $N=256$ samples per task ($1024$ samples overall), representing a highly compact stream.
* **Statistical Rigor**: 3 independent random seeds evaluated on 1000 test samples per task (4000 test samples overall), with mean and standard deviation reported for every configuration.

## Support for Core Claims
The empirical results in Table 1 (and the extended results in the Appendix) strongly support the paper's central claims:
1. **The Generalization-Granularity Curve**:
   * **Underfitting**: Level 1 (Global) exhibits severe performance deficits (23.21% for Adam, 24.84% for ES) compared to the Uniform baseline of 30.41%, indicating insufficient capacity.
   * **Intermediate Capacity**: Moving to Level 2 (Layer-wise) and Level 4 (Component-wise) improves the mean accuracy (up to 28.38% for Adam and 29.98% for ES).
   * **Overfitting**: Level 5 (Tensor-wise) unregularized configurations drop in performance (collapse of 1.47% for Adam and 0.55% for ES), illustrating transductive overfitting on the $N=256$ calibration batch.
2. **First-order vs. Zero-order Dynamics**:
   * Adam unregularized collapses to 26.91%, whereas ES unregularized remains at 29.43%.
   * The "sluggishness" hypothesis is supported: at low dimensions (Level 1, 4 params), ES successfully optimizes (and overfits) down to 24.84%. As dimensions increase to 288 parameters (Level 5), ES struggles to move far from its uniform initialization, preserving its performance near the uniform baseline.
3. **Regularization Recovery**:
   * ESR and TV show a robust stabilization effect: unregularized L5 Adam improves from 26.91% to 28.51% (+1.60%), and unregularized L5 ES improves from 29.43% to 30.17% (+0.74%).
4. **Supremacy of Static Baseline**:
   * Crucially, **not a single adaptive configuration beats the static Uniform Task Arithmetic baseline of 30.41%**. This demonstrates that the surrogate entropy loss is misaligned with classification accuracy on small budgets.

## Critical Gaps and Pragmatic Concerns
From a practitioner's point of view, there is a **significant experimental gap** regarding the quality of the "experts" being merged:
* **Very Weak Experts (Low Fidelity)**: The task-specific experts are extremely poorly converged. As seen in the table, the upper bound "Individual Experts" accuracy is only 24.93% for CIFAR-10 and 17.50% for SVHN. For 10-class datasets, these are barely better than random guessing (10%). MNIST (61.03%) and FashionMNIST (62.47%) are also far below standard levels of performance (which typically exceed 90-95%).
* **Impact of Poor Experts on Findings**: Merging models that are barely functional means the task vectors $\theta_k$ are highly noisy and unstable. This high-frequency parameter noise is highly likely to amplify transductive overfitting during test-time adaptation. In a real-world deployment, a practitioner would *never* deploy or merge experts that perform this poorly. 
* **Lack of Validation on High-Fidelity/Production Models**: While the authors openly acknowledge this as a "resource-constrained" limitation in Section 4.4, the lack of validation on even moderately well-converged experts (e.g., standard ViT-Base on ImageNet subsets) makes it difficult to know whether this "Generalization-Granularity Trade-off" is a fundamental law of model merging or simply an artifact of merging highly noisy, under-trained models.
