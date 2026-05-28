# Meta-Review Process and Decisions Summary

This document summarizes the meta-review process, evaluation criteria, and final decisions for the 10 paper submissions evaluated during this cycle. Exactly 3 submissions have been selected for acceptance based on their quantitative review scores, qualitative contribution strength, and scholarly integrity.

---

## 1. Overview of the Meta-Review Process

The goal of the meta-review process was to identify and accept exactly **3 out of 10 submissions** that present the most significant, rigorous, and scientifically sound contributions to the field of multi-task model merging and calibration. 

Each submission underwent peer review by 2 to 3 expert reviewers operating under distinct personas (including **The Minimalist**, **The Scholar**, and **The Empiricist**). To make the final decisions, we conducted a multi-dimensional evaluation considering:
1. **Quantitative Consensus:** The average and distribution of reviewer recommendation scores (ranging from 2: Reject to 5: Accept).
2. **Qualitative Content and Rigor:** The depth of theoretical insight, soundness of mathematical derivations, and completeness of experimental validation.
3. **Scholarly Integrity:** Proper literature contextualization, citation accuracy, and avoidance of AI-generated hallucinations or placeholder references in the bibliographies.

### Summary of Decisions and Scores

| Submission | Title | Scores | Decision | Core Reason |
| :--- | :--- | :--- | :---: | :--- |
| **Submission 3** | *A Comprehensive Empirical Study of Statistic Weighting and Test-Time Calibration...* | 5 (Accept), 4 (Weak Accept) | **ACCEPT** | Outstanding empirical honesty, exhaustive sweep of 2,250 configurations, and rigorous analysis of Synthetic Fisher collapse. |
| **Submission 7** | *Holographic Norm Scaling: Zero-Shot, Data-Free Parameter Calibration...* | 5 (Accept), 3 (Weak Reject) | **ACCEPT** | Exceptional simplicity, 100% data-free weight-space norm matching, and flawless compiler compatibility with zero runtime latency. |
| **Submission 9** | *Isotropic Parameter Resonance: Zero-Shot, Data-Free Parameter Calibration...* | 5 (Accept), 3 (Weak Reject), 2 (Reject) | **ACCEPT** | Beautiful "Unifying Attractor" theorem bridging Weight Averaging and Task Arithmetic, and elegant mathematical analysis of representation collapse. |
| **Submission 1** | *The Optimizer Confounder: SGD vs. AdamW in Model Merging...* | 3 (Weak Reject), 2 (Reject), 3 (Weak Reject) | **REJECT** | Severe bibliographic hallucinations (fabricated citations), citation compilation bugs, and minor mathematical inconsistencies. |
| **Submission 10** | *Task-Specific BatchNorm Calibration: The Minimalist Cure...* | 3 (Weak Reject), 3 (Weak Reject) | **REJECT** | Complete lack of statistical rigor (no seeds or error bars), toy-scale setup, questionable REPAIR baseline failure, and task routing constraints. |
| **Submission 5** | *Regularized BatchNorm Adaptation: Resolving Quiet Channel Pathology...* | 3 (Weak Reject), 2 (Reject), 2 (Reject) | **REJECT** | Restricted to BatchNorm architectures, redundant with standard calibration (solved in <1s), and failure of data-free variant. |
| **Submission 6** | *Preserving Expert Singular Values: Spectral Scaling and Calibration...* | 2 (Reject), 2 (Reject) | **REJECT** | Circular/trivial proofs, extremely small improvement (0.21%), and SVD reconstruction yields virtually 0% performance improvement. |
| **Submission 8** | *Spherical Parameter Merging and Selective Partitioning...* | 2 (Reject), 2 (Reject) | **REJECT** | False claims of novelty (Karcher Mean is already in MergeKit), severe internal contradictions, and lack of empirical variance decay change. |
| **Submission 2** | *Procedural Activation Proxies for Data-Free Calibration...* | 2 (Reject), 2 (Reject) | **REJECT** | Poor performance of proposed methods (worse than uncalibrated baselines), and weak experimental scale. |
| **Submission 4** | *Federated BatchNorm Calibration: A Decentralized Approach...* | 2 (Reject), 2 (Reject), 2 (Reject) | **REJECT** | Standard/incremental application of existing methods, and a "trade-off" artifact that arises from a naive, unaligned merging baseline. |

---

## 2. Detailed Meta-Reviews for Accepted Submissions

### 2.1. Submission 3: A Comprehensive Empirical Study of Statistic Weighting and Test-Time Calibration in Multi-Task Model Merging
* **Scores:** 5 (Accept), 4 (Weak Accept)
* **Meta-Recommendation:** **ACCEPT**
* **Synthesis of Reviews:**
  This paper provides a monumental, multi-dimensional empirical deconstruction of post-merge BatchNorm statistics calibration. It systematically evaluates 5 distinct statistics-merging strategies across 2,250 unique configurations, varying parameter-merging paradigms, batch sizes, test-time blending schedules, and noise levels.
* **Key Strengths:**
  1. **Validation of Simplicity:** The paper demonstrates that a dead-simple, zero-overhead **Uniform Merging** of BatchNorm statistics is an exceptionally robust baseline, performing on par with or better than complex Fisher/Variance-based weighting under test-time calibration. This is a crucial finding that steers the community away from over-engineered solutions.
  2. **Exhaustive Sweep & Rigor:** The scale of the experimental grid (2,250 configurations) is outstanding, providing an invaluable macroscopic view of statistics-merging and calibration dynamics.
  3. **Synthetic Fisher Analysis:** The theoretical and empirical deconstruction of why "Synthetic Fisher" information estimation fails at small batch sizes ($B=1$) is elegant. The authors correctly trace this failure to ReLU-induced activation sparsity and gradient skewness under out-of-distribution (OOD) Gaussian inputs, confirming this with coefficient of variation measurements (2.40 for Synthetic Fisher vs. 1.08 for Real Fisher).
  4. **Actionable Mitigation (Fisher-Soft):** The proposed "Fisher-Soft" temperature-scaling interpolation is simple, elegant, and successfully restores $B=1$ accuracy (+28.77% over raw Synthetic Fisher).
* **Key Weaknesses & Constructive Feedback:**
  1. *Lack of Statistical Rigor:* The paper fails to report error bars or standard deviations across multiple random seeds, reporting single point-estimates to four decimal places. The authors should run 3-5 random seeds to ensure the minute differences are statistically significant.
  2. *Scale & Applicability:* The evaluation is restricted to ResNet-18 on toy vision datasets (MNIST/CIFAR). The authors must discuss how their findings translate to larger, non-BatchNorm architectures (such as Transformers using LayerNorm or RMSNorm) where running statistics are absent.

---

### 2.2. Submission 7: Holographic Norm Scaling: Zero-Shot, Data-Free Parameter Calibration for Production-Ready Multi-Task Model Merging
* **Scores:** 5 (Accept), 3 (Weak Reject)
* **Meta-Recommendation:** **ACCEPT**
* **Synthesis of Reviews:**
  This submission introduces Holographic Norm Scaling (HNS), a zero-shot, data-free parameter calibration framework that operates entirely in weight space. Under the assumption that expert updates are nearly orthogonal in high dimensions, HNS matches channel-wise L2 norms directly in parameter space, recovering significant accuracy for Weight Averaging (+13.82%), TIES (+17.54%), and DARE (+18.54%).
* **Key Strengths:**
  1. **Production-Ready Design:** By operating entirely in weight space prior to inference, HNS avoids runtime forward hooks, compiles flawlessly with `torch.compile`, and adds zero runtime latency.
  2. **Data-Free & Zero-Shot:** The method operates with 100% data-free weight-space norm calculations, ensuring complete data privacy and compatibility with resource-constrained/on-device settings.
  3. **Strong Empirical Recovery:** Consistent and significant accuracy gains are demonstrated across diverse parameter-merging baselines.
  4. **Pareto-Frontier Insights:** The depth-wise partial merging analysis shows a highly practical Pareto-frontier where early layers are merged and shared, leaving only the specialized deep layers task-specific.
* **Key Weaknesses & Constructive Feedback:**
  1. *Conceptual & Terminology Obfuscation:* The "holographic" branding and analogies (e.g., "reference beams," "holographic storage") are unnecessary and obfuscate the elegant simplicity of the underlying technique (channel-wise L2 norm matching of task updates). The authors are strongly urged to simplify their terminology.
  2. *Operational Trade-off:* HNS requires task-specific parameter scaling at inference time, meaning it is not a "single merged model" capable of mixed-task batching. It is more accurately a parameter-efficient storage and deployment scheme. This trade-off must be discussed transparently.
  3. *Incomplete Frontier Evaluation:* Section 5.6 (Holographic Frontier) lacks downstream task accuracy metrics, relying solely on parameter-space cosine similarities. Without classification accuracy, the claims in Section 5.6 remain speculative.
  4. *Bibliographic Errors:* The bibliography contains multiple fabricated/placeholder arXiv IDs (e.g., `.12345`, `.22222`) and "Anonymous" citations for published preprints. The references must be thoroughly audited and corrected.

---

### 2.3. Submission 9: Isotropic Parameter Resonance: Zero-Shot, Data-Free Parameter Calibration for Multi-Task Model Merging
* **Scores:** 5 (Accept), 3 (Weak Reject), 2 (Reject)
* **Meta-Recommendation:** **ACCEPT**
* **Synthesis of Reviews:**
  This paper addresses representation collapse in model merging via Isotropic Parameter Resonance (IPR), a data-free offline parameter-space calibration framework. The authors propose Update-level IPR (U-IPR), which calculates resonance ratios from weight norms to rescale task-specific updates, proving a "Unifying Attractor" theorem that mathematically bridges Weight Averaging and Task Arithmetic in a scale-invariant manner.
* **Key Strengths:**
  1. **Elegant Mathematical Deconstruction:** The mathematical modeling of forward pass representation collapse as an exponential decay in activation variance ($ (1/K)^l $) is elegant, theoretical satisfying, and clearly exposes the locus of the pathology.
  2. **The Unifying Attractor Theorem:** The proof that U-IPR projects any linear parameter merge to the exact same mathematically optimal joint state, completely removing hyperparameter sensitivity ($\lambda$), is a highly significant and beautiful theoretical contribution.
  3. **Flawless Empirical Alignment:** The empirical results perfectly validate the "Unifying Attractor" identity, achieving *exactly* the same average accuracies across Weight Averaging, Task Arithmetic ($\lambda=0.2$), and Task Arithmetic ($\lambda=0.5$).
* **Key Weaknesses & Constructive Feedback:**
  1. *Implicit Covariance Assumption:* The variance decomposition in Equations 2 and 3 assumes that the covariance between progenitor activations ($W_{\text{init}}x$) and task updates ($T_kx$) is zero. Since task updates are optimized directly from the progenitor, this covariance is highly non-zero in practice, and its omission must be formally justified.
  2. *Theoretical Grounding for Orthogonality:* The paper relies on expert task-vector orthogonality in parameter space, but explains it only intuitively via "diverging paths in flat valleys." A formal justification of this geometric property is missing.
  3. *Failure of SA-IPR:* The proposed Subspace-Aligned IPR (SA-IPR) variant, designed to filter out orthogonal interference, strictly underperforms standard U-IPR across the board. The authors must explain why projecting onto joint subspaces degrades performance so severely.

---

## 3. Key Reasons for Rejection/Revision of Other Submissions

### 3.1. Submission 1: The Optimizer Confounder: SGD vs. AdamW in Model Merging
* **Scores:** 3 (Weak Reject), 2 (Reject), 3 (Weak Reject)
* **Core Issues:**
  1. **Critical Failures in Scholarly Integrity:** The bibliography contains fabricated, hallucinated, and severely corrupted citations. For example, it lists a paper on "DELLA-Merging" with a completely fabricated first author ("Della, S."), incorrect title, and a wrong arXiv ID (`arXiv:2410.12845`) which actually points to an unrelated clinical medicine paper. It also cites a completely hallucinated paper by "Detry, J." with an arXiv ID pointing to a physics paper. This violates basic academic standards of proper attribution.
  2. **Bizarre Formatting Errors:** Pervasive LaTeX syntax errors render multi-author citations with an ampersand before "et al." (e.g., `Jordan & et al., 2023`, `Detry & et al., 2024`).
  3. **Mathematical Inconsistencies:** Equation 6 and 7 are inconsistent regarding whether the PyTorch AdamW $\epsilon$ parameter is inside or outside the square root. Additionally, Section 3.2.1 completely omits the $- \eta \lambda W_0$ term in the absolute weight decay pull vector, compromising the trajectory derivation.

### 3.2. Submission 10: Task-Specific BatchNorm Calibration: The Minimalist Cure for Representation Collapse
* **Scores:** 3 (Weak Reject), 3 (Weak Reject)
* **Core Issues:**
  1. **Complete Lack of Statistical Rigor:** All classification accuracies are reported as single-run point estimates to four decimal places. There are no random seeds, standard deviations, or confidence intervals reported, representing a severe deficit in empirical rigor.
  2. **Toy Experimental Setup:** Confined exclusively to ResNet-18 on toy vision datasets (MNIST, Fashion-MNIST, CIFAR-10), leaving generalizability to modern transformer-based LLMs or ViTs unproven.
  3. **Questionable Baseline Failure:** The REPAIR baseline is reported as failing catastrophically (~11-12% accuracy), which directly contradicts successful evaluations of REPAIR on ResNets in the original published literature. This suggests either a flawed baseline implementation or bad hyperparameters.
  4. **Operational Constraints:** TS-BC requires task indicators and task-swapping of BatchNorm buffers at test time, which prevents processing mixed-task batches simultaneously in unified multi-task serving.

### 3.3. Submission 5: Regularized BatchNorm Adaptation: Resolving Quiet Channel Pathology
* **Scores:** 3 (Weak Reject), 2 (Reject), 2 (Reject)
* **Core Issues:**
  1. **Restricted Architectural Relevance:** The proposed RBA method only applies to convolutional architectures utilizing Batch Normalization, which is of limited interest in modern production environments dominated by Transformers (utilizing LayerNorm/RMSNorm).
  2. **Practical Redundancy:** The "Quiet Channel Pathology" is shown to be easily and completely solved by simply running standard calibration for 100 epochs (<1 second of compute), making the proposed complex RBA method practically redundant.
  3. **Empirical Failure of Data-Free Variant:** The zero-shot, data-free "Harmonic-RBA" variant degrades performance severely, performing 20% worse than the uncalibrated baseline, yet is framed in the text as a major success.

### 3.4. Submission 6: Preserving Expert Singular Values: Spectral Scaling and Calibration
* **Scores:** 2 (Reject), 2 (Reject)
* **Core Issues:**
  1. **Triviality of Proofs and Methods:** The proposed "Task-Specific BatchNorm Calibration" (TS-BNC) is presented as a novel contribution, but is actually a standard practice in activation-calibration literature (e.g., REPAIR). The mathematical proofs are largely circular or trivial.
  2. **Negligible Performance Gain:** The proposed SVD-based singular-value reconstruction method yields virtually 0% performance improvement over the uncalibrated baseline, achieving a negligible average accuracy gain of 0.21%.
  3. **Reporting Discrepancies:** Cross-table variance in reported numbers (0.75% discrepancy across tables for the exact same setting) exceeds the claimed 0.21% performance improvement, compromising reproducibility.

### 3.5. Submission 8: Spherical Parameter Merging and Selective Partitioning
* **Scores:** 2 (Reject), 2 (Reject)
* **Core Issues:**
  1. **False Claims of Novelty:** The paper claims that applying non-Euclidean Karcher means "has not been applied to parameter-space model merging." This is factually incorrect; Karcher merging is a standard method implemented in active open-source repositories like `MergeKit`.
  2. **Severe Internal Contradictions:** Section 3.1 claims absolute novelty of Karcher merging, while Section 3.5 explicitly discusses "prior non-selective implementations of spherical parameter merging... applying the spherical Karcher mean" without citing any of them.
  3. **Physical Hypothesis Disproven:** The core hypothesis that S-SKM prevents activation variance decay is directly disproven by the authors' own Table 2, which shows that S-SKM's activation variances are virtually identical to standard Weight Averaging.

### 3.6. Submission 2: Procedural Activation Proxies for Data-Free Calibration
* **Scores:** 2 (Reject), 2 (Reject)
* **Core Issues:**
  1. **Underperforming proposed methods:** The proposed "breakthrough" data-free calibration methods (PGAC and TSPC) perform significantly worse (~25-29% average accuracy) than the uncalibrated baseline (48.47%), making them counter-productive in practice.
  2. **Restricted experimental scale:** Evaluating only a tiny CNN on toy datasets limits relevance and generalizability.

### 3.7. Submission 4: Federated BatchNorm Calibration: A Decentralized Approach
* **Scores:** 2 (Reject), 2 (Reject), 2 (Reject)
* **Core Issues:**
  1. **Highly incremental contribution:** The proposed Federated BatchNorm Calibration is a highly straightforward, standard application of AdaBN/Federated averaging to model merging, lacking conceptual novelty.
  2. **Baseline Artifacts:** The identified "Representation Incoherence vs. Calibration Trade-off" is an artifact of merging unaligned neural networks on completely disjoint tasks (creating extreme parameter interference). It is not a paradigm-shifting insight, but rather a predictable consequence of a weak merging baseline.

---

## 4. Conclusion and Strategic Recommendations

The selection of **Submissions 3, 7, and 9** represents a balanced portfolio of outstanding empirical deconstructions and highly practical parameter-space calibration methods:
- **Submission 3** establishes empirical truth, proving the power of uniform statistics merging and explaining the exact mathematical/activation collapse of out-of-distribution Fisher weighting.
- **Submission 7 (HNS)** provides a direct, compiler-compatible, zero-overhead channel-wise weight scaling solution that matches the task update L2 norms in weight space.
- **Submission 9 (IPR)** establishes the "Unifying Attractor" theorem, demonstrating that isotropic parameter scaling completely removes scale-factor hyperparameter sensitivity in Task Arithmetic.

By prioritizing these papers, we maintain a high bar of scholarly integrity, solid mathematical foundations, and rigorous empirical validation for the model-merging community.
