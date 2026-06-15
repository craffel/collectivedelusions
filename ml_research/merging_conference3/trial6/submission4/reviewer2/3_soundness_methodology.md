# 3. Soundness and Methodology

## Clarity of the Description: Excellent and Honest
The paper is exceptionally well-written, with high mathematical clarity and a structured, logical flow. 
Rather than sweeping complex engineering realities under the rug, the authors present rigorous, closed-form mathematical derivations for every key phenomenon:
1. **Layer-Averaging Collapse Proof (Equations 5-7):** Explains exactly why and how layer-wise coefficients collapse to a single global router at deployment.
2. **Logit-Ensembling Equivalence (Equation 8):** Honestly derives why linear head-level parameter merging is mathematically identical to output-level logit ensembling.
3. **Gradient Shared Cross-Talk (Equation 10):** Explains the exact mathematical mechanism behind hard-task gradient dominance over easy tasks.
4. **Coefficient Cancellation (Equation 11):** Demonstrates the mathematical origin of heterogeneity collapse under mixed-task deployment streams.

This level of detail is rare and highly valued by practitioners, as it provides a clear, transparent window into the internal mechanics of the system rather than relying on hand-waving or black-box empirical claims.

---

## Appropriateness of Methods: Geometrically and Mathematically Sound
The proposed techniques are highly appropriate and elegant:
* **Task-Space Anchor Regularization (TSAR):** Pulling routing weights toward stable task centroids (pre-computed offline over the calibration split) is a direct, parameter-free spatial constraint that reduces optimization search space. This is highly logical and draws robust inspiration from prototypical networks in few-shot learning.
* **Low-Dimensional Unit-Sphere Projection:** Normalizing the projected coordinate onto the unit sphere stabilizes distance-based metrics, ensures scale invariance across disparate visual tasks, and makes the routing space highly tractable.
* **PCGrad for Multi-Task Gradient Conflict Resolution:** Highly appropriate for multi-task setups where tasks differ in signal-to-noise ratio. Projecting conflicting gradients onto the normal plane ensures joint multi-task parameter updates without corrupting easier task parameters.
* **Scaled Sigmoid Routing Activation:** Scaled Sigmoid bounded at $[0, 1.5]$ is a brilliant, zero-overhead solution to coefficient cancellation under heterogeneous streaming, providing necessary headroom for active experts while maintaining non-negativity.

---

## Potential Technical Flaws and Transparent Mitigations

The authors conduct a highly commendable, rigorous cycle of self-evaluation, proactively identifying potential technical concerns and addressing them empirically in the Appendix:

### 1. Uncentered PCA Projection Approximation (Section 3.1 & Appendix F.10)
* **Concern:** Forward projection (Equation 1) is applied to raw feature vectors $z(x)_b$ without test-time mean subtraction, while SVD is computed on mean-centered calibration features. This introduces a sample-dependent non-linear coordinate distortion.
* **Mitigation:** The authors honestly analyze this and prove that under their sandbox, the non-linear distortion is exceptionally small and behaves as a harmless scaling perturbation. Crucially, they justify it from a **Practitioner's** perspective: it completely eliminates the operational and memory overhead of storing and subtracting global feature means during real-time edge system deployment.

### 2. Physical ViT-Tiny Merging Limitations (Section 4.6 & Appendix H)
* **Concern:** The physical Vision Transformer validation is restricted to merging linear classification heads on a frozen backbone, which is mathematically equivalent to output-level logit ensembling, with no parameter fusion of deep internal non-linear layers (self-attention or MLP blocks).
* **Mitigation:** The authors are highly transparent and explicitly state this boundary, highlighting it as an open research direction. Furthermore, they provide a detailed technical response explaining the unique challenges of deep weight-space merging (e.g., permutation-routing coupling symmetries and non-linear coordinate coupling) and outline how TSAR can be adapted block-by-block using layer-localized anchors.

### 3. Use of Synthetic Stimuli in Physical ViT (Appendix H)
* **Concern:** The physical ViT experiment originally used synthetic 2D geometric patterns superimposed on normal noise vectors as inputs.
* **Mitigation:** To fully address any laboratory-to-deployment gaps, the authors conduct a dedicated evaluation on **raw, uncurated natural image manifolds** from MNIST and CIFAR-10 (Appendix H.1). TSAR + PCGrad outperforms Static Uniform Merging by a spectacular **+23.60% absolute accuracy margin**, confirming that the low-dimensional projection and spatial anchoring translate robustly to natural visual environments.

### 4. Low SVHN Expert Baseline Ceiling (Section 4.1 & Appendix J)
* **Concern:** The SVHN expert baseline is set to a very low 19.28% by applying high noise, which might not represent high-performing production backbones.
* **Mitigation:** The authors explain that this is a deliberate adverse stress-test environment to study router behavior in extremely noisy, low-signal-to-noise ratios. They also conduct an additional empirical evaluation under a **realistic high-accuracy SVHN expert baseline (ceiling of ~90%)** in Appendix J. The results confirm that TSAR consistently and robustly dominates, improving the Joint Mean by **+2.78%** over Static Uniform and **+3.04%** over $L_2$ regularization, proving that its behavior is decoupled from the baseline expert performance.

---

## Reproducibility: Outstanding
The reproducibility of the empirical results is outstanding:
* **Detailed Hyperparameters:** The authors outline the complete training configurations, optimizer settings, weight decays, and dataset splits (Appendix I).
* **Statistical Rigor:** All experiments are executed across **5 independent random seeds** (Seeds $\in \{10, 11, 12, 13, 14\}$) and report both mean and standard deviation.
* **Ablations and Sweeps:** The authors systematically evaluate parameter sensitivity, sample complexity, and projection stability across seeds, leaving no doubt about the statistical validity of their results.
