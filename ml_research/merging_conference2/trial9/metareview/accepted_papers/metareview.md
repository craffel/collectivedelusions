# Meta-Review Report and Decisions

This document details the meta-review process, analysis, and final decisions for the 10 paper submissions evaluated for acceptance. Based on the guidelines in `metareview_plan.md` and `reviewing_criteria.md`, we have conducted a thorough review of both the numerical ratings and the qualitative review content of all submissions. 

---

## 1. Meta-Review Process and Criteria

Each of the 10 submissions was evaluated by at least one (and up to three) peer reviewers. The submissions address critical topics in machine learning, specifically focusing on multi-task model merging, post-training quantization (PTQ), sparsification, and calibration techniques.

The selection process was guided by two primary criteria:
1. **Quantitative Consensus:** Prioritizing submissions with the most positive overall recommendations (scores of 4: Weak Accept, 5: Accept, or 6: Strong Accept).
2. **Qualitative Rigor & Substance:** Evaluating the depth of the contributions, theoretical soundness, empirical completeness, and scholarly integrity (e.g., accuracy of citations and references). 

Out of the 10 submissions, **three papers** have been chosen for acceptance.

---

## 2. Executive Summary of Submissions

Below is a complete table of all 10 submissions, their corresponding reviewer scores, average scores, and our final decision:

| Submission | Paper Title / Topic | Reviewer Ratings (Scores) | Average Score | Decision |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **The Illusion of Data-Free Calibration: Deconstructing Parameter-Space Rescaling in Model Merging** | 3 (Weak Reject), 5 (Accept) | 4.00 | **ACCEPT** |
| **2** | **Is Quantization Noise in Model Merging a Parameter Scaling Issue or a Quantization Calibration Pathology?** | 5 (Accept), 3 (Weak Reject), 5 (Accept) | 4.33 | **ACCEPT** |
| **3** | Preserving Sparsity and Calibration in Model Merging (WCPR-TIES) | 2 (Reject), 2 (Reject) | 2.00 | REJECT |
| **4** | Dynamic Boundary-Weighted Parameter Alignment (D-BWPA) | 3 (Weak Reject) | 3.00 | REJECT |
| **5** | Quantization-Robust WCPR with Outlier Clamping (QR-WCPR) | 3 (Weak Reject), 3 (Weak Reject) | 3.00 | REJECT |
| **6** | Quantization-Robust Sparsity-Compensated WCPR (QR-SC-WCPR) | 3 (Weak Reject), 3 (Weak Reject) | 3.00 | REJECT |
| **7** | Data-Free Cosine Representation Variance Scaling (S-Cos-RVS) | 2 (Reject), 2 (Reject) | 2.00 | REJECT |
| **8** | Sparsity-Calibrated WCPR with Target Resampling (SC-WCPR) | 3 (Weak Reject), 2 (Reject) | 2.50 | REJECT |
| **9** | **Quantization-Constrained Optimal Transport (QCOT)** | 3 (Weak Reject), 4 (Weak Accept) | 3.50 | **ACCEPT** |
| **10** | Gaussian Moment Matching for Representation Repair (CMVA) | 2 (Reject), 2 (Reject), 2 (Reject) | 2.00 | REJECT |

---

## 3. Deep-Dive and Decisions for Accepted Submissions

### Submission 2: "Is Quantization Noise in Model Merging a Parameter Scaling Issue or a Quantization Calibration Pathology?"
*   **Reviewer Ratings:** 5 (Accept), 3 (Weak Reject), 5 (Accept)
*   **Average Score:** 4.33
*   **Strengths:** 
    *   **Rigorous Diagnostic Thesis:** The paper successfully debunks the prevailing myth that low-bit quantization collapse is an inherent weight-space representation limit of model merging. It reframes the issue as a quantization calibration pathology caused by incorrect activation normalization and the sequential calibration of shared BatchNorm (BN) running statistics.
    *   **Mathematical Rigor:** Provides an elegant mathematical proof (Appendix A.1) tracing the exponential forgetting of early tasks under sequential calibration.
    *   **Practical & Cheap Remedies:** Proposes *Task-Specific Data-Efficient BatchNorm Calibration (DE-BN)* and *Vectorized Data-Efficient Quantization Calibration (DE-QC)*, which restore over 98% of the FP32 performance under 4-bit uniform quantization in under 2 seconds.
*   **Weaknesses & Scholarly Critique:** Reviewer 2 identified severe bibliographical inaccuracies, such as citing the REPAIR paper under the wrong name (Jordan, J. instead of Keller Jordan), incorrect arXiv IDs (`arXiv:2310.00000` belongs to an unrelated paper on microservices), and other typos. Additionally, the experimental evaluation was confined to a toy-scale setting (ResNet-18 on MNIST/CIFAR-10).
*   **Decision Justification (ACCEPT):** This is the strongest paper in the batch. While the bibliography must be thoroughly cleaned up prior to camera-ready publication, the core conceptual shift and diagnostic deconstruction are highly original and represent a significant service to the model-merging community, proving that complex, slow weight-space alignments are largely redundant when activation calibration is properly applied.

### Submission 1: "The Illusion of Data-Free Calibration: Deconstructing Parameter-Space Rescaling in Model Merging"
*   **Reviewer Ratings:** 3 (Weak Reject), 5 (Accept)
*   **Average Score:** 4.00
*   **Strengths:**
    *   **Excellent Theoretical Deconstruction:** Proposes and systematically deconstructs a data-free alternative called *Channel-wise BatchNorm Variance Calibration (CBVC)*. It provides a beautiful mathematical proof (Theorem 1) of why CBVC fails (resulting in an exponential activation explosion compounding with depth, $O(s^L)$) whereas weight scaling methods (HNS/IPR) succeed.
    *   **Empirical-Theoretical Alignment:** Successfully verifies the $O(s^L)$ activation explosion empirically on ResNet-18, showing activation standard deviation exploding to 426.64 at the final block.
    *   **Robust Evaluation:** Tests under multiple precision regimes (FP32, INT8, INT4) and environmental corruptions (noise and blur), showing that its proposed *Task-Specific DE-BN* maintains 68.5% accuracy under 4-bit quantization.
*   **Weaknesses & Scholarly Critique:** Reviewer 2 raised an academic integrity concern regarding a citation (`Kim & Park, 2024`) with a fabricated arXiv ID (`arXiv:2406.12345`) that has a title nearly identical to the authors' own "proposed" method, creating a novelty attribution conflict. The paper also omitted comparisons against the foundational REPAIR baseline in its empirical tables.
*   **Decision Justification (ACCEPT):** This paper is highly complementary to Submission 2 (and indeed appears to be its clean-FP32 precursor, referred to by Submission 2 as "Paper 10"). Its theoretical contribution (Theorem 1) explaining the failure of global activation scaling vs. the success of parameter scaling is exceptionally elegant. The authors must correct the fake/faulty citation of `Kim & Park, 2024` and include REPAIR as an empirical baseline in the final version, but the overall scientific merit is outstanding.

### Submission 9: "Quantization-Constrained Optimal Transport (QCOT)"
*   **Reviewer Ratings:** 3 (Weak Reject), 4 (Weak Accept)
*   **Average Score:** 3.50
*   **Strengths:**
    *   **Elegant Closed-Form Solution:** Addresses the vulnerability of WCPR's unconstrained optimal transport maps to quantization noise (which introduce outlier parameters). Formulates this as a Wasserstein-2 distance minimization subject to an $L_\infty$-norm constraint, deriving a mathematically beautiful, closed-form solution: a clipped 1D Wasserstein barycenter (Theorem 4.1).
    *   **Deep Theoretical Foundation:** Derives analytical bounds for clipping distortion and quantization noise, establishing a qualitative "U-shaped" Pareto frontier.
    *   **Simplicity and Ease of Deployment:** The method requires no training, backpropagation, or slow optimization; it is training-free and runs in seconds, making it highly practical for production.
*   **Weaknesses & Scholarly Critique:** Reviewer 1 pointed out critical scholarly integrity issues, including multiple completely fabricated bibliography entries (e.g., citing a geostrophic fluid mechanics paper under the fake author `"Visionary, R. A."` to refer to HNS and IPR) and broken BibTeX compilation strings (e.g., `"and et al."`). The evaluation is also limited to ResNet-18 on toy classification tasks.
*   **Decision Justification (ACCEPT):** Despite the extremely sloppy and problematic bibliography, the core mathematical contributions of QCOT are exceptionally solid, elegant, and correct. The formulation of clipping as a constrained Wasserstein projection is highly original and practically useful. Correcting the hallucinated/fictitious citations is a non-negotiable requirement for publication, but the core technical quality and simplicity of the method warrant acceptance over the remaining weak-rejected and rejected papers.

---

## 4. Summary and Analysis of Rejected Submissions

### Submission 3: "Preserving Sparsity and Calibration in Model Merging (WCPR-TIES)"
*   **Reviewer Ratings:** 2 (Reject), 2 (Reject)
*   **Average Score:** 2.00
*   **Summary:** Attempts to address the "Sparsity-Calibration Dilemma" (where WCPR's optimal transport calibration destroys weight sparsity) by separating calibration from the active mask.
*   **Why Rejected:** Reviewers noted severe flaws in scholarly integrity, technical correctness, and empirical reliability. Specifically, the scale compensation is mathematically mismatched for magnitude-based pruning (TIES), the outlier clipping strategy is self-defeating due to its reliance on non-robust standard deviation, and the post-calibration masking violates core properties of Wasserstein alignment. The paper contains clear technical flaws and lacks theoretical proofs or guarantees.

### Submission 4: "Dynamic Boundary-Weighted Parameter Alignment (D-BWPA)"
*   **Reviewer Ratings:** 3 (Weak Reject)
*   **Average Score:** 3.00
*   **Summary:** Proposes a contractive boundary-weighted parameter alignment method to preserve representations in low-resource regimes.
*   **Why Rejected:** Although the technical derivations and contractive bounds (Propositions 3.3 and 3.4) are rigorous and correct, the empirical utility is severely compromised. Under corrected BatchNorm calibration (DE-BN), the proposed D-BWPA method is consistently outperformed by standard, simpler Weight Averaging and Task Arithmetic baselines.

### Submission 5: "Quantization-Robust WCPR with Outlier Clamping (QR-WCPR)"
*   **Reviewer Ratings:** 3 (Weak Reject), 3 (Weak Reject)
*   **Average Score:** 3.00
*   **Summary:** Introduces QR-WCPR, which adds median-based clipping to WCPR's 1D optimal transport, combined with DE-BN.
*   **Why Rejected:** The proposed QR-WCPR is a highly incremental, over-engineered modification of prior work. More importantly, the authors' own empirical data reveals that QR-WCPR fails to provide any significant performance gains and is consistently outperformed by basic Task Arithmetic and Weight Averaging once proper activation calibration (DE-BN) is applied.

### Submission 6: "Quantization-Robust Sparsity-Compensated WCPR (QR-SC-WCPR)"
*   **Reviewer Ratings:** 3 (Weak Reject), 3 (Weak Reject)
*   **Average Score:** 3.00
*   **Summary:** Proposes QR-SC-WCPR to resolve the Sparsity-Calibration Dilemma under physical quantization and environmental noise.
*   **Why Rejected:** The submission contains multiple critical disconnects between mathematical theory and actual empirical setups. Specifically, its proposed outlier scale-clamping mechanism has exactly zero impact on accuracy in the sparse model ablation study (clamping threshold changes have no effect, indicating the clamping is never triggered in sparse models). Furthermore, on rich activation manifolds (CIFAR-10), the proposed method degrades performance compared to standard baselines.

### Submission 7: "Data-Free Cosine Representation Variance Scaling (S-Cos-RVS)"
*   **Reviewer Ratings:** 2 (Reject), 2 (Reject)
*   **Average Score:** 2.00
*   **Summary:** Diagnoses representation collapse as variance overestimation and proposes a data-free closed-form scaling factor using weight cosine similarities.
*   **Why Rejected:** The core formulation of adaptive channel-wise scaling assumes uncorrelated, isotropic input representations, which is mathematically unsound for realistic neural networks. To compensate, the method requires tuning a highly sensitive global hyperparameter on a validation set, which invalidates its "data-free" claim. If validation data is indeed available, simple DE-BN outperforms it by over 15% with zero tuning.

### Submission 8: "Sparsity-Calibrated WCPR with Target Resampling (SC-WCPR)"
*   **Reviewer Ratings:** 3 (Weak Reject), 2 (Reject)
*   **Average Score:** 2.50
*   **Summary:** Restricts optimal transport to active parameters and performs 1D resampling to preserve the TIES/DARE sparsity masks.
*   **Why Rejected:** The scaling derivations are mathematically invalid for deterministic pruning methods (like TIES), and mapping a resampled version of the full dense weight distribution onto active indices is a major conceptual error (distorting the distribution). Empirically, the method either catastrophically degrades performance (collapsing to 56.76% under TIES vs. 71.84% for Weight Averaging) or fails to outperform simple Weight Averaging.

### Submission 10: "Gaussian Moment Matching for Representation Repair (CMVA)"
*   **Reviewer Ratings:** 2 (Reject), 2 (Reject), 2 (Reject)
*   **Average Score:** 2.00
*   **Summary:** Proposes Covariance-Matched Variance Alignment (CMVA) to repair representation collapse under Gaussian weight assumptions.
*   **Why Rejected:** The contribution is a straightforward application of classic Gaussian moment matching. Crucially, the paper completely fails to cite or discuss closely related prior work on weight scope alignment (such as Xu et al., 2024), compromising its claims of originality. Additionally, the absolute performance of the merged models is extremely poor (barely above 52% on tasks where experts get 80-90%), indicating low practical significance.

---

## 5. Conclusion

By systematically reviewing all 10 submissions, we have identified a very clear and coherent narrative within the local research cluster:
1. **The power of simple activation-space calibration (DE-BN):** Multiple papers (specifically Submissions 1, 2, 4, and 5) converge on the discovery that many previously proposed, highly complex parameter-space alignment methods are redundant once a simple, data-efficient BatchNorm calibration (DE-BN) is performed right before evaluation.
2. **The theoretical foundation of calibration:** Submission 1 and Submission 2 provide beautiful mathematical deconstructions of why global statistics-scaling fails (exponential activation explosion, Theorem 1) while task-specific local activation calibration succeeds (re-estimating actual collapsed variance).
3. **Robustness under physical constraints:** When data-free merging is absolutely required under uniform low-bit quantization, Submission 9 (QCOT) provides a beautiful, closed-form, mathematically elegant optimal transport solution with update-clipping bounds to minimize quantization noise.

Accepting **Submission 1, Submission 2, and Submission 9** forms a cohesive, high-quality cohort of accepted papers that significantly advances the community's understanding of calibration, quantization, and normalization pathologies in multi-task model merging.
