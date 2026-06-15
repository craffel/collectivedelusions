# Peer Review

## Summary of the Paper
The paper investigates the robustness, sample complexity, and systems-level feasibility of out-of-distribution (OOD) task rejection modules within dynamic model merging and serving frameworks on resource-constrained edge hardware. Standard approaches project early-layer hidden representations (e.g., from Layer 3 of a frozen Vision Transformer backbone) onto task centroids to form low-dimensional similarity coordinates and fit diagonal Gaussian Mixture Models (GMMs) for OOD task rejection. The paper identifies a critical "low-resource variance collapse" vulnerability when these GMMs are calibrated on small sample splits ($N \le 64$), where unregularized coordinate variance estimates collapse toward zero and trigger high false-rejection rates under representation-level covariate shift.

To address this instability, the authors propose **SRC-DE** (Shrinkage-Regularized Coordinate Density Estimation), which applies analytical, parameter-free Ledoit-Wolf-style covariance shrinkage immediately following Expectation-Maximization (EM) convergence. The shrinkage intensity is derived analytically per GMM component using soft EM responsibility weights, and regularizes local variances toward either a Global Coordinate-Wise Diagonal target (for small-scale registries) or a Spherical Diagonal target (for high-dimensional registries). Additionally, the paper exposes an "unequal noise confounder" in prior OOD evaluation pipelines and resolves a silent, critical covariance caching bug in `scikit-learn`'s `GaussianMixture` implementation.

---

## Strengths of the Paper
1. **Exceptional Empirical Honesty and Scientific Deconstruction:** The paper is highly unique in its willingness to critically audit its own parametric density models. The deconstruction of why a simple non-parametric baseline (**Raw Cosine**) vastly outperforms all multidimensional GMM models under noise in disjoint registries (due to the curse of dimensionality and monotonicity) is outstanding, high-signal, and extremely educational.
2. **Post-Fit, Zero-Overhead, Adaptive Design:** SRC-DE applies covariance shrinkage post-fit, which avoids interfering with the EM optimization loop and introduces zero training latency. Because it analytically estimates the optimal shrinkage intensity $\alpha_{\text{opt}}$ on-the-fly, it is completely parameter-free and eliminates the need for unstable on-device cross-validation tuning.
3. **Remarkably Thorough and Rigorous Evaluation:** The empirical evaluation sets a very high standard for ML research. The authors evaluate across 20 independent random seeds, provide formal paired t-tests, investigate end-to-end input-level pixel noise propagation, simulate overlapping task registries, and partition physical manifolds into hierarchical sub-tasks.
4. **Systems-Level Grounding:** Outstanding awareness of real-world edge constraints. The authors back up their claims with bare-metal ARM Cortex-M4 clock-cycle math, parameter storage sizes in Bytes, and host-emulated CPU latency and RAM utilization profiling.
5. **High-Value Bug and Confounder Discoveries:** Resolving the silent caching bug in `scikit-learn` (where manual covariance modifications are ignored during scoring because `precisions_cholesky_` is stale) and identifying the unequal-noise confounder are immediate, highly practical contributions to the community.

---

## Weaknesses and Areas for Improvement
1. **Narrow Crossover Boundary of GMM Utility:**
   The paper argues that joint GMMs are mathematically mandatory to handle overlapping task registries. However, Table 9 shows that as soon as the registry scales to $K \ge 8$, the Full GMM models (including SRC-DE and even the oracle Noise-Adapted variant) are consistently outperformed by simple non-parametric Raw Cosine thresholding. At $K=16$, Raw Cosine achieves $0.7683$ AUC while the Full GMMs collapse to $0.6673$ AUC. At $K=64$, Raw Cosine gets $0.7785$ AUC while GMMs collapse to $0.6023$ AUC. This indicates that joint GMMs have an extremely narrow and fragile region of superiority ($K \le 4$), raising questions about their practical utility over simple Raw Cosine in realistic deployment scales where $K \ge 8$.
2. **Lack of Empirical Cross-Backbone and Cross-Modality Validation:**
   The paper repeatedly claims that its mathematical formulations are "completely modality-agnostic and architecture-independent" (Section 1, 3, 5). However, there is zero empirical validation on convolutional backbones (e.g., ConvNeXt) or text-based natural language processing (NLP) experts (e.g., routing prompt activations over specialized LoRAs on LLaMA). Without at least one alternative backbone or modality empirically tested, these claims remain speculative.
3. **Oracle Dependency of the Noise-Adapted Variant:**
   The Noise-Adapted variant of SRC-DE achieves outstanding results in Table 9, but requires prior oracle knowledge of the test-time representation noise variance $\sigma^2$. While the authors present a sensitivity sweep showing that overestimating the noise acts as a safe, conservative regularizer and propose an online dynamic estimator ($\hat{\sigma}^2_{\text{runtime}}$) in the Appendix, they do not provide empirical results of the end-to-end model utilizing this non-oracle dynamic estimator on-the-fly.
4. **Kurtosis Estimation Instability under Few-Shot Calibration:**
   Calculating the optimal shrinkage intensity $\alpha_{\text{opt}}$ requires estimating the variance of the coordinate variance estimators, which relies on fourth-order central moments (kurtosis). As the authors candidly note, estimating fourth-order moments from extremely low-sample calibration splits ($N=8$ or $16$) is statistically highly unstable. The paper would be strengthened by a discussion or sensitivity sweep showing how much $\alpha_{\text{opt}}$ fluctuates across different calibration sets and its subsequent impact on task-specific boundaries.

---

## Ratings and Justifications

### Soundness: Excellent
The paper is technically and mathematically solid. The derivations are clear, and the empirical evaluation is exceptionally rigorous. The authors use 20 random seeds, perform formal paired t-tests, identify and resolve key evaluation and software bugs, and ground their findings in detailed systems constraints and physical hardware profiling.

### Presentation: Excellent
The writing is professional, precise, and highly analytical. The organization is exceptionally logical, flowing smoothly from PEFT serving challenges to deconstructing diagonal GMMs, formulating covariance shrinkage, and systematically testing and profiling the pipeline. Figure 1 and the algorithms are exceptionally clear.

### Significance: Good
The paper addresses a very important and rapidly growing challenge in edge intelligence and dynamic model serving. Although the crossover boundary analysis reveals that joint coordinate GMMs have a relatively narrow practical advantage over simple Raw Cosine under scale, the paper's deep deconstruction of this trade-off, its adaptive shrinkage solution, and its software bug fixes represent a significant contribution to the literature.

### Originality: Excellent
Adapting Ledoit-Wolf covariance shrinkage to diagonal GMM components using soft EM responsibilities is a highly novel, mathematically elegant contribution. Identifying the unequal-noise confounder and the scikit-learn Cholesky precisions bug further demonstrates significant original investigation.

---

## Overall Recommendation

**Recommendation: 5: Accept**

**Justification:** 
The submission is an exceptionally high-quality, scientifically honest, and methodologically rigorous paper that excels in both machine learning theory and edge systems engineering. It systematically deconstructs a modern serving problem, exposes massive hidden vulnerabilities and evaluation flaws, discovers software library bugs, and provides a highly practical, adaptive, training-free, and parameter-free mathematical solution. While the practical crossover boundary of joint GMMs over simple non-parametric baselines is relatively narrow, the depth of the deconstruction, the statistical significance of the results, and the systems-level validation make this a very strong and valuable contribution that should be accepted.

---

## Detailed Comments and Questions for the Authors

1. **Practical Recommendation for Edge Serving:** Given that the non-parametric Raw Cosine baseline consistently outperforms or matches GMMs (including Independent 1D GMMs and Full SRC-DE) as soon as $K \ge 8$ in both disjoint and overlapping registries, what is your direct recommendation to a systems practitioner deploying a model merging pipeline? Under what exact hardware or task registry conditions should a developer choose to deploy a GMM-based coordinate density estimator over simple, zero-overhead Cosine thresholding?
2. **Empirical Validation of the Online Dynamic Noise Estimator:** In Table 9, could you provide the empirical OOD rejection AUC results for SRC-DE when running with your proposed online dynamic noise estimator ($\hat{\sigma}^2_{\text{runtime}}$) instead of the oracle noise scale? Confirming that the dynamic, data-driven estimator can recover the oracle Noise-Adapted performance on-the-fly would significantly strengthen the practical feasibility of your approach.
3. **Variance of the Shrinkage Intensity ($\alpha_{\text{opt}}$):** Since calculating $\alpha_{\text{opt}}$ requires estimating fourth-order sample moments, which are highly unstable under extremely small calibration budgets (such as $N=8$ or $16$), did you observe high variance in the estimated $\alpha_{\text{opt}}$ across different random calibration seeds? How does this sampling variance affect the consistency and stability of the post-fit density boundaries on individual edge tasks?
4. **Cross-Modality or Cross-Architecture Proof of Concept:** Can you provide a lightweight empirical proof of concept of your coordinate-space projection and SRC-DE on either a convolutional backbone (e.g., ConvNeXt) or a text specialist prompt-router (e.g., routing specialized LoRAs on a frozen LLaMA base model) to empirically back up your modality-agnostic and architecture-independent claims?
