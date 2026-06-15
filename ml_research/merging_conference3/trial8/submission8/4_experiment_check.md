# 4. Experimental Evaluation Check

## 4.1 Rigor and Setup
The experimental evaluation in this paper is outstandingly thorough and robust:
- **Statistical Significance (20 Seeds):** Instead of standard single-run or 3-seed reporting, all empirical tables present the **mean and standard deviation across 20 independent random seeds** with matching randomized calibration subsets and noise draws. This is highly rigorous and provides a high-confidence assessment of statistical stability.
- **Symmetric Noise Injection:** The authors establish a scientifically sound protocol by applying identical representation-level noise symmetrically to both in-distribution and out-of-distribution test representations. This eliminates the "unequal noise confounder" that plagued prior evaluations (which artificially drove AUC below 0.50).
- **Comprehensive Baseline Suite:**
  1. *Raw Cosine Thresholding:* A non-parametric, robust baseline.
  2. *Unregularized Diagonal GMM (SPS-ZCA):* Standard GMM fitted without regularization.
  3. *L2-Regularized (Ridge) GMM:* Standard GMM with a static ridge hyperparameter ($\gamma=10^{-4}$).
  4. *Tuned Ridge Diagonal GMM:* A highly competitive baseline that dynamically tunes $\gamma$ per task using 3-fold cross-validation directly over the calibration splits across a five-order-of-magnitude candidate pool.

## 4.2 Key Findings and Metrics
- **Experiment 1 (Robustness to Covariate Shift):** On clean data ($\sigma^2=0.0$), SRC-DE $M=2$ achieves an outstanding AUC of **0.9599 ± 0.0048** (the highest among all mixture models). Under moderate noise ($\sigma^2=0.05$), it achieves **0.7648 ± 0.0372**, outperforming the unregularized model by **+3.57%** absolute AUC and the static Ridge model by **+2.10%** absolute AUC.
- **Experiment 2 (Sample Complexity Scaling):** Under fixed noise ($\sigma^2=0.05$), unregularized GMMs exhibit a non-monotonic, "U-shaped" performance curve for $M=2$ due to the interaction between EM component splitting and sample size. SRC-DE successfully suppresses this instability, consistently achieving the highest performance under data scarcity (e.g., **0.7784 ± 0.0400** at $N=8$ and **0.7918 ± 0.0265** at $N=16$, which represents a **+2.19%** and **+3.28%** absolute improvement over unregularized and Ridge baselines).
- **Formal Significance Audits (Paired t-tests):** The authors perform formal relational paired t-tests across the 20 seeds. The p-values are exceptionally small (e.g., $p = 5.99 \times 10^{-6}$ for SRC-DE vs. Unregularized $M=2$ at $\sigma^2=0.05$, and $p = 1.07 \times 10^{-6}$ vs. Tuned Ridge $M=2$ at $\sigma^2=0.10$), confirming that the performance gains are mathematically ironclad.
- **System-Level Impact Formulation:** Section 4.4 formally connects OOD Rejection AUC to downstream end-to-end system classification accuracy ($\mathcal{A}_{\text{sys}}$). The authors demonstrate that SRC-DE's lower False Positive Rate (45.19% vs. 51.63% for unregularized under a strict TPR=0.90 constraint) translates directly to a **+3.2% absolute gain** in downstream system accuracy (70.4% vs. 67.2% for the unregularized model). This bridges the gap between statistical metrics and real-world system utility.
- **High-Dimensional Scaling Simulation:** To show that covariance shrinkage is vital under larger registries, the authors evaluate scaling registries up to $K=64$ and show that SRC-DE consistently and globally improves OOD rejection performance (yielding up to **+4.63% absolute AUC gains**).
- **Ablation Study ($M=1$ vs. $M=2$):** Section 4.5 presents a clear ablation study of mixture components, demonstrating that increasing mixture complexity under unregularized setups exacerbates overfitting, whereas SRC-DE successfully stabilizes the multi-component boundaries.
