# 4. Experimental Evaluation and Empirical Support

This file presents a critical check of the paper's experimental setup, baseline coverage, dataset choices, and whether the empirical evidence rigorously and completely supports the paper's central claims.

## Evaluation of Experimental Setup

### 1. Backbone and Dataset Suitability
* **Backbone:** The authors evaluate their framework on a Vision Transformer (`vit_tiny_patch16_224`) partitioned into $L=14$ layer groups. Utilizing a Transformer is highly suitable because Transformers are notoriously sensitive to quantization rounding noise (due to outliers and representation collapse), making this a highly realistic and challenging stress-test.
* **Datasets:** The tasks consist of four diverse vision datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN. This mixture covers diverse domains (digits, clothing, natural images) and represents varying levels of classification difficulty, which is highly appropriate for evaluating model merging.
* **Data Budget:** To make exhaustive grid sweeps computationally feasible, the authors pre-train each expert on 512 images. The authors openly discuss this low-data budget as a limitation (Section 5.1, Limitation 1), noting that while absolute accuracies are lower than full-scale models (e.g., individual unquantized experts achieve ~64.28% on this budget, compared to ~95% when trained on full datasets), the relative performance improvements are statistically significant and driven by physical geometric properties that should generalize to larger scales.
* **Calibration Set and Sensitivity:** The test-time adaptation uses an unlabeled calibration batch of only $N=64$ images (16 per task). The authors include a sensitivity sweep showing that varying the calibration batch size $N \in \{16, 32, 64, 128\}$ yields statistically equivalent multi-task accuracies and highly stable coefficients. This proves that FlatQ-Merge is exceptionally data-efficient and robust to small calibration sizes.

### 2. Statistical Rigor
The paper reports all multi-task merging accuracies as **mean % $\pm$ standard deviation across 3 independent random seeds** (42, 100, 2026). This is an excellent practice that ensures the reported gains are not a fluke of random initialization or task order.

## Baseline Coverage
The paper is exceptionally strong in baseline comparisons, covering standard merging approaches, test-time adaptation variants, and structural ablations:
1. **SGD Q-Merge ($\rho=0.0$):** Standard SGD-trained experts quantized post-merging, with coefficients optimized in quantized space via STE.
2. **NaiveUniform:** SAM-trained experts merged using static uniform coefficients ($\lambda^l_k=0.3$) followed by per-channel PTQ.
3. **AdaMerging-PostQ:** Coefficients optimized in full FP32 on SAM experts, and the merged model quantized post-hoc.
4. **Individual-Quantized:** Task-specific SAM experts evaluated independently under quantization without merging (serving as an empirical performance upper bound).
5. **Convex Softmax Combination Baseline (Section 4.5):** Validates the choice of independent clipping bounds $[0, 1]$ over normalized Softmax.
6. **DARE Model Merging Baseline (Section 4.6):** Integrates FlatQ-Merge with state-of-the-art parameter pruning to see if conflict resolution affects flatness benefits.
7. **High-Dimensional TENT-Style Adaptation (Section 4.7):** Ablates the low-dimensional coefficient bottleneck against full-parameter test-time adaptation.
8. **Stochastic Weight Averaging (SWA) (Section 4.8):** Compares SAM's adversarial flatness with SWA's passive average flatness.
9. **Isotropic weight-space parameter perturbation (Section 4.9):** Directly measures weight-space curvature to empirically proxy the Hessian trace of each expert.

This exhaustive baseline coverage leaves no stone unturned and sets a very high standard for experimental validation.

## Analysis of Results and Support for Claims

The empirical results provide overwhelming support for all the paper's core claims:

### Claim 1: Precision-Dependent Flatness-Robustness Synergy
* **Claim:** Flatness improves robustness to quantization noise, but this synergy is highly precision-dependent (crucial under 4-bit, negligible under 8-bit).
* **Support:** 
  * Under 8-bit quantization (Table 1), standard SGD experts ($\rho=0.0$) achieve 44.63% (FlatQ) and 44.69% (Ada-PQ), whereas flat experts ($\rho=0.05$) achieve 44.62% and 44.58%. The delta is statistically negligible, showing that 8-bit precision preserves merging directions inherently.
  * Under 4-bit quantization (Table 2), standard SGD experts ($\rho=0.0$) struggle at 23.00% (FlatQ) and 24.16% (Ada-PQ). Enforcing flatness at $\rho=0.05$ boosts accuracies to **30.44%** (FlatQ, a **+7.44% absolute improvement**) and **30.78%** (Ada-PQ, a **+6.62% absolute improvement**). The standard deviations are tightly bounded ($\approx 2\%$), proving the gains are statistically significant and robustly support the claim.

### Claim 2: Pre-Merging Geometry Dominates Adaptation
* **Claim:** The geometric loss landscape of pre-merging experts is more critical than the sophistication of downstream test-time adaptation.
* **Support:** In 4-bit precision, merging flat experts ($\rho=0.05$) with static uniform weights (NaiveUniform) achieves **29.03%** accuracy. This completely un-adapted baseline on flat experts outperforms the highly sophisticated test-time optimized FlatQ-Merge on standard sharp experts ($\rho=0.0$, which achieves only **23.00%**). This **+6.03% absolute accuracy gain** is a stunning and decisive empirical confirmation of the claim.

### Claim 3: SWA's Passive Flatness is Insufficient for Extreme Noise
* **Claim:** SAM's adversarial perturbation objective is a necessary ingredient for 4-bit PTQ, and simple trajectory averaging (SWA) cannot substitute.
* **Support:** Section 4.8 shows that SWA experts are highly robust under 8-bit quantization (reaching **46.88%** with FlatQ, matching/exceeding SAM). However, under extreme 4-bit noise, SWA collapses to **22.62%** (FlatQ), performing on par with standard SGD experts (23.00%) and failing completely to match SAM experts (30.44%). This discrepancy perfectly confirms that SWA's passive centering is insufficient against worst-case low-bit coordinate rounding noise, whereas SAM's adversarial formulation is necessary.

### Claim 4: Physical Connection via Curvature Measurement
* **Claim:** The weight-space Hessian eigenvalues are directly suppressed by SAM, which in turn flattens the coefficient-space landscape and shields the merged network from discrete rounding errors.
* **Support:** The authors directly measure weight-space curvature (Section 4.9, Table 5) by perturbing expert parameters and measuring cross-entropy loss increase. Under a perturbation scale of $\sigma_{\text{weight}}=0.005$, SGD experts ($\rho=0.0$) increase loss by **0.157863**, while optimal SAM experts ($\rho=0.05$) increase loss by only **0.019729**—representing a massive **$8\times$ reduction in sharpness/curvature**! This direct curvature proxy perfectly correlates with the downstream 4-bit accuracy gains, providing outstanding empirical support for the second-order Taylor expansion foundation (Equation 7).

## Summary
The empirical design of this paper is exemplary. The results are presented with high statistical rigor, the baselines are exhaustive, and the claims are supported by overwhelming, highly consistent empirical evidence from multiple independent angles.
