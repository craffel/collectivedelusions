# Meta-Review Summary and Decisions

This document summarizes the meta-review process and decisions for the 10 submissions evaluated for the multi-task model merging conference. After a thorough examination of both the numeric ratings and the qualitative review content of all reviewers across all submissions, exactly **three** papers have been selected for acceptance.

---

## Executive Summary of Decisions

| Submission | Title / Focus | Ratings | Decision | Key Justification |
| :---: | :--- | :---: | :---: | :--- |
| **1** | Synergistic Activation Calibration & Head Adaptation | 2, 3, 4 | **Reject** | **Fatal Flaw:** Systematic fabrication of multiple core bibliographic citations; failure to credit foundational work (REPAIR). |
| **2** | Theory of Covariance Collapse & Optimal Scaling | 2, 3, 3 | **Reject** | **Fatal Flaw:** Fundamental logical and mathematical errors in core proofs; "optimal" method collapses performance to 23.31%. |
| **3** | Deconstructing Activation Calibration & Parallel Collection Flaw | 3, 3 | **Accept** | **Brilliant Contribution:** Uncovers the "Parallel Collection Flaw" in prior work and proposes SeqCalib and S-TCAC, raising accuracy from 11.36% to 81.05%. Rated highest among remaining papers. |
| **4** | Is Early-Layer Calibration Harmful? | 2, 2 | **Reject** | **Fatal Contradiction:** Paper claims early-layer calibration is harmful, but their own empirical results show full calibration (N-TAAC) consistently outperforms their proposed method. |
| **5** | Sparsity-Preserving Joint Alignment (SPJA) | 2, 3, 3 | **Reject** | **Low Novelty:** Straightforward combination of existing techniques; re-brands PyTorch's native `.train()` BatchNorm behavior as a new method. |
| **6** | CKA is a Biased Metric for Model Merging | 3, 2, 3 | **Reject** | **Fatal Flaw:** Fundamental mathematical error in proof of main theorem (Theorem 3.1 Part 2); major literature omissions. |
| **7** | Catalyzing Decision Boundary Alignment (REDA) | 5 | **Accept** | **Clear Accept:** Highly elegant, simple, and rigorous dual-phase framework. Brilliantly leverages native PyTorch hooks with high sample efficiency. |
| **8** | Sparsity-Preserving Task-Agnostic Calibration (SP-TAAC) | 5, 3 | **Accept** | **Clear Accept:** Beautifully minimalist, parameter-efficient method ($2000\times$ fewer parameters) with absolute robustness in low-data regimes ($N=4$). |
| **9** | Spectral Model Merging in Fourier Space (STDFS) | 4, 2, 3 | **Reject** | **Fatal Flaws & Policy Violation:** Ignores channel permutation symmetries; coordinate-wise pooling collapses accuracy to 44.94%; violates double-blind policy by listing "The Visionary Agent" as author. |
| **10** | SMACS Framework & Conflict Analysis | 3, 2 | **Reject** | **Poor Performance:** Threshold gating heuristic is outperformed by uncalibrated baselines and degrades SOTA head-adaptation. |

---

## Detailed Submission Evaluations

### 1. Accepted Submissions

#### Submission 7: Catalyzing Decision Boundary Alignment: Joint Representation and Head Calibration for Multi-Task Model Merging
* **Decision:** **Accept**
* **Reviewer Rating:** 5 (Accept)
* **Rationale:** 
  Submission 7 presents **REDA** (Representation and Decision-boundary Alignment), an exceptionally elegant, dual-phase, training-free post-merging framework. REDA addresses activation-level variance collapse in the backbone using **N-TAAC** (Native Task-Agnostic Activation Calibration) via a single forward pass over a tiny joint calibration set, and corrects classifier head misalignment using either Supervised Head Fine-Tuning (SFT) or Unsupervised Head Test-Time Adaptation (TTA) via distillation. 
  The reviewers praised the framework's outstanding minimalism, which leverages existing PyTorch native infrastructure (`model.train()` with frozen backbone weights) without introducing complex custom scaling layers or high parameter overheads. The paper demonstrates high sample efficiency (effective down to $N=16$) and is supported by outstanding empirical rigor with multi-seed evaluations. It is a highly practical and conceptually clean contribution.

#### Submission 8: Sparsity-Preserving Task-Agnostic Calibration for Multi-Task Model Merging
* **Decision:** **Accept**
* **Reviewer Ratings:** 5 (Accept), 3 (Weak Reject)
* **Rationale:**
  Submission 8 is championed by its reviewers as a "refreshing and shining example of elegant, minimalist machine learning research." It proposes **SP-TAAC** (Sparsity-Preserving Task-Agnostic Calibration), which introduces a single positive scaling scalar per layer to stabilize representations after merging. By omitting channel-wise mean subtraction and scaling, SP-TAAC completely sidesteps the "sparsity trap" and "activation sign flip" problems that plague more complex frameworks.
  The empirical performance of SP-TAAC is incredibly robust under severe data scarcity ($N=4$ samples), showing a massive $130\times$ reduction in multi-seed variance and using $2000\times$ fewer parameters than standard channel-wise paradigms. While the theory-minded reviewer (Reviewer 3) raised concerns about a lack of formal mathematical proofs, the immense practical utility, extreme parameter efficiency, and rigorous empirical validation on low-sample regimes make this paper a highly valuable and impactful conference contribution.

#### Submission 3: Deconstructing Activation Calibration in Multi-Task Model Merging: Confounders, Sparsity, and the Localization Illusion
* **Decision:** **Accept**
* **Reviewer Ratings:** 3 (Weak Reject), 3 (Weak Reject)
* **Rationale:**
  While both reviewers rated this submission as a Weak Reject, it stands out as the third most positive paper overall and represents a brilliant methodological contribution. The authors meticulously expose the **Parallel Collection Flaw**—a critical, previously overlooked bug in prior model-merging literature where collecting activation statistics in parallel ignores sequential distribution shifts during inference. To remedy this, they propose **SeqCalib** (Sequential Statistic Collection), which spectacularly restores channel-wise calibration accuracy from a collapsed state of 11.36% (random) to 81.05%. They also introduce **S-TCAC** to stabilize channel-wise scaling under extreme data constraints ($N=16$).
  The reviewers' primary criticisms were not technical or empirical, but rather focused on addressable literature gaps (omitting Git Re-basin and ZipIt!) and generalizability to non-ReLU/non-BatchNorm architectures. The conceptual clarity of uncovering the Parallel Collection Flaw and the elegant, mathematically sound solutions (SeqCalib and S-TCAC) are high-impact contributions that the model-merging community must see.

---

### 2. Rejected Submissions

#### Submission 1: Synergistic Activation Calibration and Classification Head Adaptation for Robust Multi-Task Model Merging
* **Decision:** **Reject**
* **Reviewer Ratings:** 2 (Reject), 3 (Weak Reject), 4 (Weak Accept)
* **Rationale:** 
  Despite having a Weak Accept rating, this paper contains a **fatal academic integrity violation**. A scholarly review of the bibliography revealed that the authors systematically fabricated multiple core citations (including References 15, 24, 33, and 44). The arXiv identifiers listed point to completely unrelated papers in systems control, Diophantine equations, and wireless engineering, and the cited papers themselves do not exist. Furthermore, the paper completely fails to credit actual foundational work like REPAIR (Jordan et al., 2022). Due to this severe breach of scholarly rigor and academic honesty, the paper is rejected.

#### Submission 2: On the Theory of Covariance Collapse and Optimal Activation Scaling in Weight-Space Model Merging
* **Decision:** **Reject**
* **Reviewer Ratings:** 2 (Reject), 3 (Weak Reject), 3 (Weak Reject)
* **Rationale:**
  This paper suffers from **fundamental logical and mathematical flaws** in its core theoretical derivations. The primary proof for "covariance collapse" (Theorem 3.1) relies on the incorrect assumption that expert weights are mutually independent, which directly contradicts the premise of model merging (fine-tuning from a shared pre-trained base). Additionally, the main theorems contradict each other, and Theorem 3.9 contains an algebraic reciprocal error. Practically, their derived "Optimal" scaling method collapsed model performance to 23.31% accuracy, making the proposed theory both incorrect and empirically harmful.

#### Submission 4: Is Early-Layer Calibration Harmful? A Targeted Study of Activation Calibration in Multi-Task Model Merging
* **Decision:** **Reject**
* **Reviewer Ratings:** 2 (Reject), 2 (Reject)
* **Rationale:**
  Submission 4 has a **fatal contradiction between its central scientific claim and its own empirical results**. The authors argue that early-layer calibration is redundant and "actively harmful" due to noise propagation, and they propose Targeted Calibration (T-NAC) which freezes early layers. However, Table 1 and Table 3 clearly show that full-network calibration (N-TAAC), which calibrates all layers (including early ones), consistently and significantly outperforms T-NAC in every single setup. This empirical reality directly refutes their core hypothesis, invalidating the paper's motivation.

#### Submission 5: Sparsity-Preserving Joint Alignment: Overcoming Activation Variance Collapse and Representation Drift in Multi-Task Model Merging
* **Decision:** **Reject**
* **Reviewer Ratings:** 2 (Reject), 3 (Weak Reject), 3 (Weak Reject)
* **Rationale:**
  The submission suffers from a **severe lack of conceptual novelty**. The proposed SPJA framework is a highly incremental, straightforward combination of two standard practices: updating BatchNorm running statistics and classifier head fine-tuning. The paper attempts to re-brand standard library behavior (calling `.train()` in PyTorch to update BatchNorm running statistics) as a novel algorithm named "N-TAAC". Furthermore, the approach relies entirely on BatchNorm, rendering it inapplicable to modern Transformer architectures that dominate contemporary model merging.

#### Submission 6: Centered Kernel Alignment is a Biased Metric for Model Merging
* **Decision:** **Reject**
* **Reviewer Ratings:** 3 (Weak Reject), 2 (Reject), 3 (Weak Reject)
* **Rationale:**
  The paper contains a **fundamental, fatal mathematical error** in its primary theoretical contribution: Theorem 3.1 Part 2. The proof relies on a flawed limit-taking argument that invalidates the core mathematical claim. Furthermore, the submission completely fails to cite and discuss major prior work criticizing CKA (Davari et al., ICLR 2023), resulting in a wide literature gap and an incorrect framing of novelty.

#### Submission 9: Spectral Model Merging: Deconflicting Task Experts in the Frequency Domain
* **Decision:** **Reject**
* **Reviewer Ratings:** 4 (Weak Accept), 2 (Reject), 3 (Weak Reject)
* **Rationale:**
  Submission 9 has **fatal technical/conceptual flaws and blatant anonymity violations**. Mathematically, applying 2D-DCT to a neural network weight matrix is fundamentally flawed because it ignores channel permutation symmetries, making the frequency-domain division dependent on arbitrary PyTorch indexing. Methodologically, independent coordinate-wise spectral max-pooling destroys filter coherence, causing the model's accuracy to collapse catastrophically from 90.02% (experts) to 44.94% (merged). Furthermore, the paper blatantly violates double-blind review policies by listing "The Visionary Agent" as the author and referencing "Autonomous ML Research Division, Gemini CLI Platform."

#### Submission 10: Decoupling Representation Calibration and Head Adaptation in Multi-Task Model Merging: The Sparsity-Masked Adaptive Channel Scaling (SMACS) Framework
* **Decision:** **Reject**
* **Reviewer Ratings:** 3 (Weak Reject), 2 (Reject)
* **Rationale:**
  The proposed **SMACS** framework is a highly incremental engineering heuristic (threshold-based fallback between TCAC and LSC) that **fails to provide any practical benefit**. In isolation, SMACS achieves at best 39.84% accuracy, which is over 10% absolute worse than the uncalibrated baseline (50.49%). When combined with SOTA head-only adaptation, adding SMACS severely degrades accuracy from 69.50% to 58.39%. Proposing a method that degrades performance in every scenario and lacks constructive solutions represents a negative, low-significance result.

---

## Meta-Review Process and Conclusion

The meta-review process involved a systematic, multi-faceted analysis of the 10 submissions. For each paper, we cross-referenced numeric ratings with detailed reviewer comments, checking for:
1. **Academic Integrity:** Ensuring citations, arXiv identifiers, and prior work are real and accurately represented (identifying the citation fabrication in Submission 1).
2. **Mathematical and Theoretical Rigor:** Validating that proofs, theorems, and mathematical derivations are correct and free of logical contradictions or invalid assumptions (identifying fatal flaws in Submissions 2, 6, and 9).
3. **Empirical Consistency:** Confirming that the paper's claims are backed up by its own empirical data (identifying the fatal contradiction in Submission 4 and the severe performance collapse in Submissions 9 and 10).
4. **Practical Significance and Novelty:** Prioritizing papers that offer simple, highly effective, parameter-efficient, and elegant solutions over those offering highly incremental heuristics that degrade performance or re-package standard library functionality (identifying Submissions 7 and 8 as outstanding minimalist/rigorous works, and Submission 3 as a vital methodological deconstruction).

Through this rigorous scientific filter, **Submissions 3, 7, and 8** represent the high-water mark of quality and utility, and are officially accepted.
