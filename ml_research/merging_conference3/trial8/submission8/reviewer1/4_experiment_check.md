# Paper Evaluation: 4. Experimental Check

## Evaluation of Experimental Setup
The experimental setup designed by the authors is remarkably rigorous and comprehensive, setting a very high standard for empirical machine learning research:
1. **Diverse Dataset Registry:** Using a combination of four distinct, established vision datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) that exhibit disparate semantic structures and feature distributions.
2. **Competitive and Well-Tuned Baselines:** 
   - Including a **Tuned Ridge Diagonal GMM** where the L2 regularizer ($\gamma$) is dynamically selected per task via a 3-fold cross-validation scheme directly on calibration splits. This is an exceptionally strong, realistic, and highly competitive baseline.
   - Including a non-parametric **Raw Cosine** baseline, which represents a simple, zero-parameter alternative that is immune to parameter estimation variance.
3. **Statistical Soundness:** Evaluating results across **20 independent random seeds** and reporting both mean and standard deviation.
4. **Formal Significance Testing:** Performing paired t-tests and reporting exact p-values to demonstrate that the performance improvements are statistically significant.
5. **Practical Edge Constraints:** Including an end-to-end input-level corruption sweep, physical sub-task clustering ($K \in \{4, 8, 12, 16\}$), full covariance shrinkage comparisons, and emulated on-device physical profiling (latency, peak RAM, storage size) to ground the architectural claims.

---

## Critical Analysis of Claims vs. Empirical Evidence
While the empirical evaluation is exceptionally thorough, a close examination of the results reveals several nuances, gaps, and areas where the evidence only partially supports the claims:

### 1. Extremely Narrow and Fragile Region of Joint GMM Superiority
The paper claims that joint GMMs are mathematically mandatory in edge deployments to resolve overlapping task registries, where non-parametric 1D Raw Cosine thresholding collapses due to coordinate overlap. Let us analyze the empirical evidence:
- **On Disjoint Registries (Tables 3 & 4):** Simple, non-parametric Raw Cosine thresholding consistently and vastly outperforms *all* parametric GMM models under noise (e.g., at $\sigma^2=0.05$, Raw Cosine achieves $0.9040$ AUC, while SRC-DE ($M=2$) achieves $0.7648$ AUC—a major **~14% absolute AUC gap** in favor of the simple baseline).
- **On Overlapping Registries (Table 9):**
  - For $K=4$, the Full Noise-Adapted SRC-DE achieves an AUC of $0.8137$ compared to Raw Cosine's $0.5598$ or $0.7580$ (depending on configuration), which represents a valid advantage (+5.57% absolute AUC).
  - However, as soon as the registry size scales to $K \ge 8$, the Full GMM models (including SRC-DE and even the oracle Noise-Adapted variant) perform worse than the simple Raw Cosine baseline. 
    - At $K=8$, Raw Cosine achieves **$0.7634$ AUC**, while Full SRC-DE achieves $0.7480$ AUC and the oracle Noise-Adapted variant achieves $0.7821$ AUC (only a tiny +1.87% absolute benefit).
    - At $K=16$, Raw Cosine achieves **$0.7683$ AUC**, while the Full GMMs collapse to $0.6673$ AUC (with oracle Noise-Adaptation getting $0.6985$ AUC).
    - At $K=64$, Raw Cosine achieves **$0.7785$ AUC**, while the Full GMMs collapse to $0.6023$ AUC (Noise-Adapted: $0.6155$ AUC).
- **Conclusion:** The joint GMM's superiority over the simple, non-parametric Raw Cosine baseline is extremely narrow, collapsing rapidly for $K \ge 8$ due to the accumulation of representation noise across the inactive dimensions. Under realistic settings ($K \ge 8$), Raw Cosine's dimensional isolation makes it a more practical, robust, and computationally lightweight choice, even in the presence of task overlaps. The paper honest-to-goodness deconstructs this, but as a reviewer, I must emphasize that the practical necessity of deploying heavy GMM-based coordinate density estimators is highly questionable given this narrow crossover.

### 2. Empirical Utility of Independent 1D GMMs
To resolve high-dimensional scaling collapse, the authors propose an alternative systems architecture: **Independent 1D Coordinate GMMs** (Section 4.10).
- In Table 8 (disjoint registry), the 1D GMM achieves $0.9166$ AUC at $K=64$, which is high but still underperforms Raw Cosine ($0.9303$ AUC) by $1.37\%$ absolute, while being significantly more complex.
- In Table 9 (overlapping registry), the 1D GMM achieves $0.7265$ AUC at $K=4$ (worse than Raw Cosine's $0.7580$) and $0.7659$ AUC at $K=64$ (worse than Raw Cosine's $0.7785$).
- **Conclusion:** The Independent 1D GMM never actually outperforms Raw Cosine in any evaluated setting (disjoint or overlapping), while requiring on-device 1D density fitting, parameter storage, and exponential likelihood evaluations. While the authors argue that 1D GMMs offer the parametric flexibility to capture multi-modal outlier shapes, the empirical results do not demonstrate a concrete performance advantage over Raw Cosine, raising questions about its practical necessity.

### 3. Lack of Cross-Backbone and Cross-Modality Empirical Validation
The paper repeatedly asserts that the mathematical formulations of similarity coordinates and responsibility-weighted covariance shrinkage are "completely modality-agnostic and architecture-independent" (Section 1, 3, 5). It outlines qualitative generalization pathways for convolutional backbones (e.g., Global Average Pooling over middle feature maps) and LLMs (sequence pooling of early decoder layers) in Appendix A.2.
- **Gap:** There is zero actual empirical validation on convolutional backbones (like ConvNeXt) or natural language processing (NLP) experts (e.g., prompt-routing with LLaMA LoRA specialists). 
- **Conclusion:** Without concrete experimental evidence on at least one other modality or architecture, the claims of "architecture-independent" and "modality-agnostic" generalization remain speculative.

### 4. Evaluation of the Noise-Adapted Variant
The "Full SRC-DE (Noise-Adapted)" variant (which achieves the best results in Table 9 under overlapping registries) adjusts GMM covariances post-fit by adding the known representation noise variance $\sigma^2$ directly to the diagonal.
- **Gap:** This represents an oracle setup. While the authors provide a sensitivity sweep showing that overestimating the noise is safe, and propose a dynamic online noise estimator ($\hat{\sigma}^2_{\text{runtime}}$) in the Appendix, they do not show the actual end-to-end performance of SRC-DE when utilizing this dynamic online estimator on-the-fly. Providing these results would significantly strengthen the paper's claim of non-oracle feasibility.
