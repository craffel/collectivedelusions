# Peer Review

## Summary of the Paper
This paper presents a thorough methodological and empirical evaluation of out-of-distribution (OOD) task rejection within dynamic, activation-space model merging serving frameworks (such as SPS-ZCA and SABLE). These frameworks serve multiple specialized parameter-efficient fine-tuning (PEFT) adapters (e.g., LoRA) over a shared, frozen backbone. An input query's early activations are projected against pre-computed task centroids to map them into a low-dimensional similarity coordinate space. Prior work fits unregularized diagonal Gaussian Mixture Models (GMMs) on small calibration splits ($N \le 64$) during an offline phase, using log-likelihood thresholds to detect and reject OOD queries.

The authors expose a severe overfitting vulnerability in this setup, showing that diagonal GMMs experience "local variance collapse" under mild covariate shift (test-time representation noise). To stabilize these boundaries, they propose **SRC-DE** (Shrinkage-Regularized Coordinate Density Estimation), which applies analytical, parameter-free Ledoit-Wolf-style covariance shrinkage to GMM component covariance matrices immediately following expectation-maximization (EM) convergence. The paper also identifies an "unequal noise confounder" in prior evaluations, resolves a cached Cholesky precision bug in `scikit-learn` that disabled manual GMM covariance adjustments, evaluates independent 1D coordinate GMM configurations, and provides physical multi-task scaling analyses, end-to-end input image corruption audits, and emulated on-device microcontroller benchmarks.

---

## Strengths and Weaknesses

### Strengths
1. **Exceptional Methodological Rigor:** The paper is incredibly thorough and exhaustive. The authors anticipate and systematically audit almost every potential experimental or implementation confounder (such as the unequal noise setup, the `scikit-learn` Cholesky caching bug, soft Bessel's/Cochran's degrees-of-freedom corrections, overlapping task registries, and non-linear noise propagation of input-level corruptions).
2. **Clear and Statistical Deconstructions:** The text features exceptional clarity and high-quality statistical reasoning. The explanation of the GMM's U-shaped performance curve under EM component splitting and the deconstruction of why joint GMMs suffer from noise accumulation over inactive dimensions are outstanding and highly educational.
3. **Strong Systems Grounding:** The emulated on-device performance benchmarks (profiling parameter storage, calibration latency, peak RAM, single-query FPU execution cycles, and energy budgets) are superb, connecting theoretical multivariate statistics directly with physical ARM Cortex microcontroller deployment realities.

### Weaknesses
1. **Limited Conceptual Novelty:** The core algorithmic contribution is a straightforward, incremental application of standard, classical statistical tools. Linear Ledoit-Wolf covariance shrinkage is a standard, 20-year-old statistical technique. Adapting it to the diagonal parameterization of GMM component covariances as a post-fit regularization step is a useful engineering patch, but it does not represent a major conceptual leap or a paradigm-shifting algorithmic breakthrough.
2. **Fragile Core Premise:** The empirical results reveal a fundamental weakness in the joint coordinate-space GMM density routing paradigm itself. In Table 2 and Table 4, the simplest possible non-parametric baseline, **Raw Cosine similarity**, consistently and significantly outperforms all parametric GMM models (including their proposed SRC-DE) under noise (e.g., Raw Cosine achieves 0.9040 AUC vs. SRC-DE GMM's 0.7648 AUC at $N=64, \sigma^2=0.05$). Under severe noise ($\sigma^2=0.20$), Raw Cosine retains 0.7915 AUC while SRC-DE drops to 0.6059. 
3. **Catastrophic Dimensional Scaling Collapse:** Furthermore, as coordinate dimensions scale ($K \ge 16$), joint multi-dimensional GMMs collapse catastrophically due to the curse of dimensionality, as noise accumulating across the $K-1$ inactive dimensions completely buries the active routing signal. While the authors propose Independent 1D GMMs to bypass this scaling collapse, these 1D models are completely blind to semantic task overlap (reverting to the exact same vulnerabilities as Raw Cosine).
4. **Niche Application and Broad Significance:** Because the paper is hyper-focused on auditing and patching the OOD task-rejection module of a very specific, narrow activation-space PEFT serving framework (SPS-ZCA), its potential impact on the broader machine learning community is relatively limited.

---

## Soundness
**Rating: Excellent**

The paper is technically flawless and highly rigorous. The mathematical formulations are complete, clear, and statistically sound. The authors are exceptionally candid about their model's theoretical limitations (such as the violation of the fixed-target assumption in spherical shrinkage, the non-Gaussianity of bounded similarity supports, the statistical instability of fourth-moment estimators under extreme data scarcity, and ignoring EM responsibility sampling variance). The empirical audits are comprehensive, and the statistical significance paired t-tests provide ironclad mathematical proof of their local performance gains over GMM baselines.

---

## Presentation
**Rating: Excellent**

The paper is masterfully written, beautifully organized, and highly engaging. The tables and figures are clean, dense with high-quality information, and exceptionally detailed. The authors do an outstanding job of presenting complex statistical trade-offs in an intuitive and clear manner.

---

## Significance
**Rating: Fair**

While the paper's execution and hardware profiling are impressive, its broader significance is limited:
* **The Methodological Dead-End of Joint GMMs:** The paper's own empirical audits suggest that joint GMM-based coordinate density routing is a highly fragile and suboptimal paradigm. It is consistently outperformed by simple non-parametric 1D similarity thresholding under noise, and it collapses completely under dimensional scale.
* **Over-Engineering a Patch:** The proposed SRC-DE is a highly complex statistical "patch" designed to stabilize a structurally fragile framework. While it successfully improves GMM performance relative to unregularized baselines, it does not solve the fundamental limitations of coordinate-space density routing.
* **Niche Audience:** The contribution is highly specific to researchers working on activation-space dynamic model merging, which is a narrow sub-area of parameter-efficient fine-tuning serving.

---

## Originality
**Rating: Fair**

The work represents an incremental engineering modification of existing tools:
* **Derivative Mechanism:** Ledoit-Wolf shrinkage is a standard multivariate statistical estimator. Its post-fit diagonal application to GMM component covariance matrices is a straightforward adaptation rather than a novel conceptual leap.
* **Lack of Paradigm Shift:** The paper does not introduce a new dynamic model-serving model, a novel routing framework, or an original representation learning paradigm. It merely refines and audits an existing component of a prior serving model.

---

## Overall Recommendation
**Rating: 3: Weak Reject**

**Justification:**
This submission is a beautifully written, exceptionally thorough, and methodologically flawless engineering audit. The authors have done a magnificent job addressing almost every conceivable experimental confounder and statistical detail, and the systems-level physical microcontroller profiling is superb.

However, from the perspective of conceptual novelty and magnitude of contribution, the paper falls short. The core regularizer (post-fit Ledoit-Wolf shrinkage) is a standard, classical statistical mechanism applied incrementally to diagonal GMM parameters. More importantly, the empirical results expose a fundamental structural fragility in the entire joint coordinate-space GMM density routing paradigm: it is consistently and massively outperformed by the simplest non-parametric 1D similarity check (Raw Cosine) under noise, and it collapses completely under dimensional scale due to noise accumulation across inactive dimensions. 

Ultimately, the proposed SRC-DE behaves as an elegant, hyper-focused patch on a structurally fragile foundation. Given that simpler 1D thresholding schemes or hierarchical hybrid routing are fundamentally more robust and practical alternatives, the community is unlikely to build on this complex joint GMM routing path. While the paper's thoroughness is exemplary, its lack of an original, ambitious, or paradigm-shifting conceptual leap—combined with the inherent fragility of the audited framework—prevents me from recommending an acceptance.

---

## Questions and Constructive Feedback for the Authors

1. **Why not abandon joint GMMs in favor of Sparse / Hierarchical Routing?** 
   Given that the curse of dimensionality causes joint GMMs to collapse catastrophically for $K \ge 16$ (due to noise accumulation across the $K-1$ inactive dimensions), and that independent 1D GMMs are blind to semantic overlap, the most promising path forward seems to be **Hierarchical Hybrid Routing** (which you propose in Section 4.6). Why did you choose to spend so much analytical and empirical effort trying to "rescue" the fragile joint GMM framework under scale, rather than focusing on developing and fully evaluating a sparse/hierarchical routing framework that dynamically isolates active coordinates before evaluating local GMM densities?
2. **On the Instability of Fourth-Moment Estimators under Extreme Few-Shot Regimes ($N=8$):**
   You candidly acknowledge in Section 4.7 that estimating fourth-order sample moments (kurtosis) from $N=8$ or $N=16$ samples is highly unstable and leads to noisy optimal shrinkage intensity ($\alpha_{\text{opt}}$) estimates. In Table 4, under $N=8$, SRC-DE achieves $0.7784 \pm 0.0400$ AUC, while the unregularized model gets $0.7548 \pm 0.0437$ AUC. Have you considered exploring standard Bayesian priors (e.g., Normal-Inverse-Gamma) or a simple, non-parametric empirical bootstrap over the calibration split to estimate $\alpha_{\text{opt}}$ more stably in few-shot regimes, rather than relying on the analytical fourth-moment plug-in estimator?
3. **Do the Bounded similarity Space Violations Matter in Practice?**
   Since similarity coordinates reside in $[-1, 1]$, and standard GMM components are unconstrained, how much probability mass actually leaks outside $[-1, 1]$ for active tasks where means are close to $1.0$? Did you measure this leakage, and did you observe whether the log-density boundary collapse is specifically worse near the $1.0$ boundary?
