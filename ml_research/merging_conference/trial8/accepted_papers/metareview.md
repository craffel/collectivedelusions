# Meta-Review Summary and Decisions

**Conference:** Merging Conference 2026  
**Meta-Reviewer:** Autonomous Meta-Reviewing Agent  
**Date:** May 24, 2026  

---

## 1. Meta-Review Process Overview

This meta-review summarizes the evaluation and decision process for ten (10) submitted papers on the topic of **Test-Time Model Merging (TTMM)** and related test-time adaptation (TTA) paradigms. 

The objective of the meta-review process was to carefully analyze all ten submissions—specifically reviewing their technical soundness, experimental rigor, presentation quality, overall significance, and originality—and select exactly **three (3) submissions** to accept.

To make the final decisions, a rigorous comparison of both the numerical ratings and the qualitative feedback from the peer reviews was conducted. Submissions were evaluated not only on their final recommended score but also on their scientific depth, empirical completeness, theoretical grounding, and the robustness of their findings.

---

## 2. Submission Summary Table

The table below outlines the ten submissions, their paper titles, overall recommendation ratings, and a brief synthesis of their key strengths and weaknesses.

| ID | Paper Title | Recommended Rating | Synthesis of Strengths & Weaknesses |
| :---: | :--- | :---: | :--- |
| **1** | Dynamic Momentum Temporal Regularization for Robust Test-Time Model Merging | **5: Accept** | **+** Outstanding bug discovery (autograd disconnection); elegant, lightweight DMTR mechanism.<br>**-** Evaluation restricted to toy datasets (MNIST scale) and small CNNs. |
| **2** | Kronecker-Preconditioned Self-Calibrated Bayesian Test-Time Model Merging (KP-SCTS-Bayes) | **3: Weak Reject** | **+** Rigorous theoretical framing and clean Kronecker Trace preconditioning proof.<br>**-** Critical scientific contradiction in ablations (components cause negative synergy); misleading/cherry-picked reporting; toy scale. |
| **3** | AdaKL-BN: Adaptive KL-Divergence-Driven Batch Normalization Blending | **5: Accept** | **+** Highly practical $O(C)$ diagonal statistics blending; resolved critical benchmark bugs (clamping, scale disalignment).<br>**-** Evaluated only on SimpleCNN and MNIST/FashionMNIST/KMNIST. |
| **4** | Mutual Information-Guided (MI) Routing Prior and Feedback-Resistant Teacher-Regularized Test-Time Model Merging (MI-FRTR-TTMM) | **5: Accept** | **+** Discovered autograd bug; elegant information-theoretic routing (MI) to defeat OOD overconfidence; stable EMA teacher optimization.<br>**-** Toy scale; high computational/memory overhead; sensitive to small batch sizes. |
| **5** | CP-AM: Contrastive Prototypes with Angular Margin for Noise-Robust Open-World Test-Time Model Merging | **6: Strong Accept** | **+** Exceptional theoretical rigor (Propositions 3.1 & 3.2); massive empirical gains on dense domains (+6.40% / +5.15%); exhaustive sweeps; honest trade-off analysis.<br>**-** Confined to SimpleCNN on digit/fashion datasets. |
| **6** | FDF-DPA: Fully Data-Free Dynamic Prototype Adaptation with Kronecker-Trace Guided Feature Anchoring | **6: Strong Accept** | **+** Completely eliminates offline source data dependency (fully data-free); proven moment-matching Soft BN Fusion; highly novel Feature Anchoring.<br>**-** Restricted to grayscale datasets; heuristic thresholding; minor backprop overhead. |
| **7** | C-CPAL: Cosine-Normalized Contrastive Prototype Alignment for Robust Test-Time Model Merging | **4: Weak Accept** | **+** Rigorous proof of ReLU scale inflation; elegant visual schematics; clear noise sensitivity sweep.<br>**-** Toy-scale; the proposed C-CPAL optimization contributes $<1\%$ performance gain beyond SCTS-C initialization. |
| **8** | Co-acting Layer-Wise Kronecker-Fisher Adaptation for Data-Free Test-Time Model Merging (CL-KT-Fisher) | **4: Weak Accept** | **+** True data-freeness with high expressiveness; solid proofs (Lemma 3.1 & Theorem 3.2); solves "stabilizer drowning."<br>**-** Small-scale evaluation; minor notation errors; primarily integration-based novelty. |
| **9** | BK-CoMerge: Bayesian Kronecker-Preconditioned Co-acting Test-Time Model Merging for Data-Free Open-World Streams | **6: Strong Accept** | **+** Technically flawless; outstanding SOTA results (57.16% accuracy) outperforming strong baselines; elegant adaptive consensus coherence; diligent appendices.<br>**-** Toy-scale datasets. |
| **10**| Gated EMA-Proto: Robust Test-Time Model Merging with Confidence-Calibrated Prototype Updates | **5: Accept** | **+** Exceptional empirical completeness (K=5 experts, 1000-batch infinite-horizon stability); noise geometry proof; 3x faster than backprop TTA.<br>**-** Limited to MNIST-derived datasets. |

---

## 3. Accepted Submissions and Justifications

Based on a comprehensive review of the peer evaluations, **Submissions 5, 6, and 9** have been selected for acceptance. These three papers received the highest recommended score (**6: Strong Accept**) and demonstrated the most exceptional combinations of theoretical brilliance, empirical rigor, and practical significance.

### Decision 1: Accept Submission 5 (CP-AM)
* **Paper Title:** CP-AM: Contrastive Prototypes with Angular Margin for Noise-Robust Open-World Test-Time Model Merging
* **Justification:**  
  Submission 5 introduces a highly original and theoretically robust framework (CP-AM) that addresses the scale distortion vulnerability of traditional Euclidean prototype spaces under environmental noise. 
  * **Theoretical Rigor:** The paper features exceptionally strong mathematical foundations. Proposition 3.1 analytically proves how noise scales angular similarity gaps, demonstrating the mathematical necessity of establishing a training margin. Proposition 3.2 mathematically formalizes how background sparsity amplifies noise under spherical normalization, which explains the sparsity-vs-density trade-off.
  * **Empirical Strength:** On dense/textured domains (such as FashionMNIST), CP-AM achieves significant absolute improvements (+6.40% clean and +5.15% noisy) over state-of-the-art baselines.
  * **Scientific Honesty:** Rather than hiding the drop in performance on noisy MNIST, the authors highlighted it as a key scientific finding, formalized it theoretically, and proposed a concrete structural solution (Adaptive Hybrid Routing). The comprehensive ablation sweeps and flawless formatting further solidify this paper as a Strong Accept.

### Decision 2: Accept Submission 6 (FDF-DPA)
* **Paper Title:** FDF-DPA: Fully Data-Free Dynamic Prototype Adaptation with Kronecker-Trace Guided Feature Anchoring
* **Justification:**  
  Submission 6 addresses a critical, real-world limitation of existing Test-Time Model Merging frameworks: their reliance on private offline source calibration datasets to precompute class prototypes.
  * **Practical Significance:** By introducing a fully data-free, online, and unsupervised framework, FDF-DPA allows decentralized, privacy-preserving model merging on the edge, where training datasets are strictly inaccessible. It achieves an outstanding overall accuracy of 68.63% (outperforming major baselines) while maintaining real-time edge-ready latency of 134.11 ms per batch on modern hardware.
  * **Theoretical and Architectural Originality:** The paper includes a complete step-by-step proof (Appendix A) showing that their Soft BN Buffer Fusion exactly matches the moments of the Mixture-of-Gaussians activation distribution. Furthermore, the introduction of "Feature Anchoring"—which dynamically freezes highly sensitive early convolutional layers while preconditioning task-specific parameters—is a highly original and effective solution to the "feedback trap."

### Decision 3: Accept Submission 9 (BK-CoMerge)
* **Paper Title:** BK-CoMerge: Bayesian Kronecker-Preconditioned Co-acting Test-Time Model Merging for Data-Free Open-World Streams
* **Justification:**  
  Submission 9 presents a technically flawless and highly integrated framework (BK-CoMerge) that elegantly resolves the key issues of unconstrained test-time model adaptation.
  * **Empirical Superiority:** BK-CoMerge achieves a state-of-the-art overall accuracy of 57.16% on the non-stationary 5-phase target stream, outperforming highly competitive baselines (such as DF-Bayes-TTMM and KT-Fisher).
  * **Methodological Novelty:** The combination of dynamic soft Bayesian routing with SCTS, moment-matching Soft BN Buffer Fusion, trace-based Kronecker-factored sensitivity preconditioning, and Adaptive Consensus Coherence Regularization is extremely elegant. The adaptive coherence penalty dynamically scales based on local curvature, keeping the network structurally cohesive while allowing necessary layer-wise adaptation.
  * **Scholarly Rigor:** The paper contains rigorous proofs for its BN buffer fusion and Kronecker preconditioning (Propositions 1 and 2), and preemptively addresses practical concerns (latency, stabilizer sensitivities, and scaling to transformers) in a series of outstanding, diligent appendices.

---

## 4. Comparison and Exclusion of Other Highly Rated Papers

Several other submissions (specifically Submissions 1, 3, 4, and 10) received solid **5: Accept** ratings but were ultimately excluded in favor of the top three. The primary reasons for prioritizing Submissions 5, 6, and 9 over these competitive candidates are:

* **Overcoming Practical Constraints:** While Submissions 1, 4, and 10 make outstanding contributions (such as discovering the autograd bug or modeling noise geometry), they do not completely resolve the private source-data dependency as elegantly as **Submission 6 (FDF-DPA)**, which makes TTMM practically viable for strictly data-free, private edge deployments.
* **Empirical and Theoretical Edge:** While **Submission 10 (Gated EMA-Proto)** demonstrates excellent empirical completeness, **Submission 5 (CP-AM)** provides a deeper, margin-based theoretical characterization of representation geometry under noise and achieves much more substantial performance gains on complex, textured tasks.
* **Unified Architectural Harmony:** **Submission 9 (BK-CoMerge)** offers a superior, unified treatment of the parameter preconditioning and regularization problem compared to **Submission 1 (DMTR)** or **Submission 4 (MI-FRTR)**. By incorporating on-the-fly Kronecker sensitivity directly into both the preconditioning steps and the consensus coherence penalty, BK-CoMerge creates a more theoretically harmonious system that avoids representational drift without requiring redundant EMA models or manual resetting.

---

## 5. Conclusion

The standard of submissions in the Test-Time Model Merging track at Merging Conference 2026 is remarkably high. The selected papers (**Submissions 5, 6, and 9**) represent the absolute pinnacle of this cohort. They successfully push the boundaries of data-free optimization, noise-robust representation learning, and geometric adaptation, laying down robust theoretical and empirical foundations that future researchers and practitioners will undoubtedly build upon.
