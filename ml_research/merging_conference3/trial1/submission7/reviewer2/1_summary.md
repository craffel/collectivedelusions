# Intermediate Evaluation: Summary of the Paper

This document provides a comprehensive summary of the paper's main topic, approach, key findings, and explicitly claimed contributions, serving as an intermediate step in the scholarly peer-review process.

---

## 1. Main Topic and Scope
The paper presents a rigorous empirical evaluation, methodology-focused sanity check, and representational analysis of the core assumption underlying state-of-the-art (SOTA) **layer-wise model merging** frameworks (such as AdaMerging and SyMerge). These frameworks claim that optimizing fine-grained, layer-by-layer, task-wise merging coefficients is critical to resolving destructive weight-space interference when combining diverse task-specific experts into a single backbone.

The paper evaluates these claims on a pre-trained **CLIP ViT-B/32** backbone (which contains 12 transformer blocks and 1 projection layer, resulting in $L=13$ discrete parameter groups) across four diverse image classification tasks: **MNIST**, **FashionMNIST**, **CIFAR-10**, and **SVHN**, using **3 independent random seeds** to ensure statistical and scientific rigor.

---

## 2. Approach and Technical Formulation
The paper formalizes layer-wise model merging and evaluates two main optimization regimes under a joint entropy minimization objective over an unlabeled calibration split ($D_{\text{val}}$ of 256 images, 64 per task):
1. **Zero-Order Adaptive 1+1 Evolution Strategy (1+1 ES):** A black-box derivative-free optimizer that proposes candidate coefficients via Gaussian mutation and accepts them based on prediction entropy reduction.
2. **First-Order Gradient Descent (Adam GD):** A white-box gradient-based optimizer using autograd to compute exact gradients and update parameters directly.

To test the physical reality and functional importance of the optimized layer-specific coefficients $\Lambda$ ($L \times K = 52$ continuous parameters), the authors construct a **Sanity-Checking and Interpretability Suite** consisting of three diagnostic control treatments:
- **Intra-Task Layer Shuffling (Shuffle Treatment):** Randomly permuting the learned coefficients across layers for each task. If layer-specificity is functionally critical, this permutation should destroy performance.
- **Task-Wise Spatial Averaging (Mean Treatment):** Collapsing the layer-wise coefficients into a single spatial average per task (reducing parameters from 52 to 4, a 92.3% reduction). If layer-wise variation is redundant, this low-parameter treatment should perform comparably or better.
- **Norm-Bounded Perturbation (Noise Treatment):** Injecting relative Gaussian noise $\gamma \in [0.05, 0.50]$ into the learned coefficients to assess the flatness of the optimization landscape.

Additionally, they perform **Representational Similarity Analysis** using linear **Centered Kernel Alignment (CKA)** at Layer 6 on CIFAR-10 to investigate how the physical weight merges translate into activation-space correspondences with the original task experts.

---

## 3. Key Findings: The Overfitting-Optimizer Paradox
The empirical results reveal a profound dialectical interaction, termed the **Overfitting-Optimizer Paradox**, which describes how the apparent validity of "layer-specificity" depends on the optimizer and task complexity:
- **For Zero-Order (1+1 ES) Optimization:** Layer-specificity is an illusion. Shuffling coefficients causes minimal decay ($85.07\% \to 83.28\%$), and Spatial Averaging (Mean Treatment) actually *improves* average accuracy to $85.21 \pm 0.11\%$ while reducing cross-seed variance. Here, Spatial Mean acts as a powerful regularizer, smoothing out high-frequency zero-order optimization noise.
- **For First-Order (Adam GD) Optimization:** Unconstrained gradient descent finds a delicate configuration that is highly sensitive to shuffling ($84.52\% \to 79.09\%$) or spatial averaging (causing a 10.35% collapse on CIFAR-10), creating an illusion of physical layer-specificity. However, this model fails to outperform the unoptimized Task Arithmetic baseline ($84.44 \pm 0.37\%$) on the unseen test set and exhibits 4x greater seed variance ($\pm 1.57\%$), revealing it as a delicate transductive overfitting artifact.
- **Extreme Landscape Flatness:** Both optimizers reside in a highly flat basin, tolerating up to 50% relative Gaussian noise on the coefficients with negligible decay.
- **Representational (CKA) Decoupling:** While Spatially Averaged models exhibit slightly higher CKA similarity to original experts, high-level kernel alignment is a poor predictor of downstream classification. Under Adam GD, the Spatially Averaged model increases CKA on CIFAR-10 ($0.9555 \to 0.9598$) but cascades into a $10.35\%$ performance collapse, showing that weight-space decision boundary integrity decouples from activation alignment.
- **Joint Entropy Task-Bias:** Standard joint entropy minimization objectives inherently sacrifice complex tasks (like SVHN) to minimize simple, low-entropy tasks (like MNIST).

---

## 4. Explicitly Claimed Contributions
The submission claims the following core contributions, supported by systematic evidence:
1. **First Rigorous Sanity-Check of Layer-Wise Merging:** Demonstrates that the layer-specificity assumption in SOTA frameworks is largely a methodological illusion or transductive overfitting artifact under standard test-time adaptation regimes.
2. **Identification of the Overfitting-Optimizer Paradox:** Rigorously maps the interaction between optimizer choices (zero-order vs. first-order) and the emergence of apparently functional but overfit layer-wise parameters.
3. **Characterization of Model Merging Landscape Flatness:** Empirically proves the extreme flatness of the coefficient optimization landscape via relative noise injection.
4. **Exposure of Representational Decoupling (CKA vs. Accuracy):** Uncovers the limitations of using activation-level CKA as a proxy for fine-grained task performance in merged models.
5. **Formulation of Joint Entropy Task-Bias:** Mathematically and empirically describes how joint entropy minimization objectives disadvantage difficult tasks.
6. **Proposed Solutions and Verification:** Proposes and validates:
   - *Scale-Normalized Weighted Joint Entropy* (Appendix E) to resolve task-bias.
   - *Proximity Regularization* (Appendix B & F) to prevent transductive overfitting and stabilize optimization.
   - *A Learning Rate Sweep Study* (Appendix A) characterizing SGD early stopping.
