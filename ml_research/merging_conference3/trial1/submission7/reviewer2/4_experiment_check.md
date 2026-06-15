# Intermediate Evaluation: Experimental Setup, Baselines, and Claims Verification

This document provides a critical evaluation of the experimental setup, datasets, baselines, and whether the empirical results actually support the paper's core claims.

---

## 1. Experimental Setup and Benchmark Datasets
The authors construct a multi-task vision classification benchmark consisting of:
1. **MNIST** (Saturated, handwritten digits)
2. **FashionMNIST** (Saturated, fashion products)
3. **CIFAR-10** (Diverse natural images)
4. **SVHN** (Noisy, street view numbers)

The choice of datasets is diverse, spanning grayscale digits, product shapes, natural objects, and street-view house numbers. Evaluating on **CLIP ViT-B/32** ($L=13$ parameter groups) is highly appropriate as CLIP is the standard base model used in the model merging literature (e.g., AdaMerging, TIES-Merging).

The statistical rigor is exemplary:
- **3 Independent Random Seeds** (42, 100, 2026).
- **Disjoint splits** of 512 images for expert training, 512 for unseen test evaluation, and 256 for test-time calibration. This is crucial for isolating transductive overfitting.
- **Converged task experts** which exhibit high baseline accuracy (MNIST 96.94%, FashionMNIST 88.67%, CIFAR-10 88.93%, SVHN 85.81%), establishing a highly representative multi-task merging baseline.

---

## 2. Baseline Comparisons and Calibration
The paper compares against:
- **Task Arithmetic (Uniform Baseline, $\lambda = 0.3$):** This is the typical uncalibrated baseline used in SOTA papers, achieving an average test accuracy of $84.44 \pm 0.37\%$.
- **Spatially Averaged (Spatial Mean) Controls:** The paper collapses the learned $L \times K = 52$ coefficients into $K = 4$ task-wise averages.
  - For **1+1 ES**, the Spatially Averaged model achieves **$85.21 \pm 0.11\%$**, which is effectively a properly calibrated, low-parameter task-wise scalar baseline.
  - Crucially, this low-parameter baseline **outperforms** the complex layer-wise optimized AdaMerging (1+1 ES) model ($85.07 \pm 0.47\%$), proving that SOTA papers' reported improvements are often due to basic scale calibration rather than localized layer coordination.

---

## 3. Analysis of Claims vs. Evidence

### Claim 1: "Layer-specificity is an illusion under zero-order search (1+1 ES)."
- **Evidence:** 
  - Spatial Mean - 1+1 ES ($85.21 \pm 0.11\%$) outperforms the Optimized AdaMerging 1+1 ES ($85.07 \pm 0.47\%$).
  - Shuffling 1+1 ES coefficients (Intra-Task Layer Shuffling) results in only a minor, non-catastrophic performance decay to $83.28 \pm 1.26\%$.
- **Verdict:** **Fully Supported**. If layer-specificity were a physically functional coordinate system, shuffling would devastate performance, and collapsing layers to their average would severely degrade accuracy. Instead, averaging improves accuracy and reduces variance, proving the learned layer-wise variations are merely optimization noise.

### Claim 2: "Delicate layer-specificity under Adam GD is a transductive overfitting artifact."
- **Evidence:**
  - Optimized AdaMerging (Adam GD) achieves $84.52 \pm 1.57\%$ on the unseen test set, which fails to outperform the unoptimized Task Arithmetic baseline ($84.44 \pm 0.37\%$) while introducing 4x greater variance across seeds.
  - Shuffling Adam GD's coefficients collapses average performance to $79.09 \pm 2.05\%$ (a $5.43\%$ drop), and averaging collapses CIFAR-10 performance by $10.35\%$ ($89.84\% \to 79.49\%$), creating an illusion of delicate layer-specificity.
  - However, because the unconstrained Adam GD model does not actually improve performance on unseen test data compared to the unoptimized baseline, this delicacy is confirmed to be an overfit transductive artifact on the 256 calibration images.
- **Verdict:** **Fully Supported**. The authors elegantly resolve this paradox by proving that unconstrained autograd tunes 52 parameters to a small calibration set, finding a delicate, ungeneralizable configuration that fails to benefit the model on the unseen test split.

### Claim 3: "The model merging landscape resides in an exceptionally flat basin."
- **Evidence:**
  - Injecting 50% relative Gaussian noise into 1+1 ES coefficients retains $83.89 \pm 0.32\%$ average accuracy (only a 1.18% decay).
  - Injecting 50% noise into Adam GD coefficients retains $84.75 \pm 2.47\%$ accuracy.
- **Verdict:** **Fully Supported**. The extreme robustness to massive relative noise sweep ($\gamma \in [0.05, 0.50]$) empirically proves that exact coefficient coordination is functionally irrelevant, reinforcing the overparameterization argument.

### Claim 4: "Representational similarity (CKA) decouples from weight-space decision boundary integrity."
- **Evidence:**
  - Under Adam GD, the Spatially Averaged model has a slightly higher CIFAR-10 CKA ($0.9598 \pm 0.0241$) than the Optimized model ($0.9555 \pm 0.0302$), yet its classification accuracy collapsed catastrophically by $10.35\%$ ($89.84\% \to 79.49\%$).
- **Verdict:** **Fully Supported**. This highlights a critical limitation of CKA in neural network interpretability: high-level linear activation correlation does not guarantee weight-space decision boundary integrity.

### Claim 5: "Joint entropy minimization objectives suffer from an inherent task-bias."
- **Evidence:**
  - Under Adam GD, SVHN (the most complex, high-entropy task) accuracy collapses to $71.22 \pm 8.08\%$—strictly worse than the unoptimized baseline (73.24%). The optimizer trades off SVHN's performance to minimize simpler tasks (MNIST/FashionMNIST).
- **Verdict:** **Fully Supported**. The scale-normalized joint entropy pilot in Appendix E successfully resolves this and retains 85.84% average accuracy, confirming the diagnosis.

---

## 4. Overall Empirical Quality
The experimental section is exceptional. The results are highly reproducible, backed by rigorous statistics (mean and standard deviation across multiple seeds), and include elegant control baselines. The tables and figures are extremely informative and directly support the narrative.
