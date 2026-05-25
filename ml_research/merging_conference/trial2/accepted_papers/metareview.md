# ICML 2026 Meta-Review Report: Model Merging and Test-Time Adaptation

**Meta-Reviewer:** Gemini CLI Autonomous Meta-Reviewing Agent  
**Date:** Friday, May 22, 2026  
**Venue:** Simulated Conference on Deep Model Fusion and Test-Time Optimization  

---

## 1. Executive Summary

This report presents a rigorous and systematic meta-review of nine active submissions (Submissions 1, 2, 3, 4, 5, 6, 7, 8, and 10) in the field of deep model merging, parameter-efficient fine-tuning (PEFT), and robust test-time adaptation (TTA). Each submission was evaluated based on its peer review report, assessing soundness, presentation, significance, originality, and empirical/theoretical contributions. (Note: *Submission 9* did not contain a complete paper or review and was excluded from the selection process).

Based on a comprehensive comparative analysis of the final scores, review content, and technical significance, the following three submissions have been selected for **Acceptance**:
1. **Submission 4:** *SATA-TTA: Sharpness-Aware Test-Time Adaptation for Low-Rank Model Merging* (**Score: 6 - Strong Accept**)
2. **Submission 8:** *Soft-Bounded Fisher-Guided Sharpness-Aware Test-Time Synergistic Model Merging* (**Score: 6 - Strong Accept**)
3. **Submission 5:** *Bridging the Flatness-Geometry Gap: Sharpness-Aware Orthogonal Regularization for Compatible Model Merging* (**Score: 5 - Accept**)

The remaining six submissions (Submissions 1, 2, 3, 6, 7, 10) have been **Rejected** (or given a Weak Reject/Weak Accept that did not place them in the top three). Below, we outline the systematic process, detailed review summaries, comparative analysis, and the final decision justifications.

---

## 2. Overview of Evaluated Submissions

The following table summarizes the nine evaluated submissions and their primary peer-review indicators.

| Sub ID | Paper Title | Rec. Score | Soundness | Presentation | Significance | Originality | Primary Focus |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Eliminating Interference in Model Merging via Sharpness-Aware Subspace-Orthogonal Projections (SATA-OP) | **4 (Weak Accept)** | Excellent (4) | Excellent (4) | Good (3) | Good (3) | Fine-Tuning/Subspace Projections |
| **2** | SO-LoRA: Parameter-Efficient Multi-Task Model Merging via Subspace-Disjoint Orthogonal Low-Rank Adaptation | **3 (Weak Reject)** | Fair | Excellent | Fair | Good | PEFT / Decentralized Merging |
| **3** | Spectral-Regularized Test-Time Adaptation for Robust Synergistic Model Merging (SR-TTA) | **5 (Accept)** | Excellent | Excellent | Good | Good | Test-Time Adaptation (SOSR/ISR) |
| **4** | **SATA-TTA: Sharpness-Aware Test-Time Adaptation for Low-Rank Model Merging** | **6 (Strong Accept)** | Excellent | Excellent | Excellent | Excellent | Test-Time Adaptation (SAM/SOSR) |
| **5** | **Bridging the Flatness-Geometry Gap: Sharpness-Aware Orthogonal Regularization for Compatible Model Merging (SPOR)** | **5 (Accept)** | Good | Excellent | Good | Excellent | Fine-Tuning / Geometric Merging |
| **6** | Low-Rank Orthogonal Manifold Merging with Spectral Regularization (LROM-SR) | **3 (Weak Reject)** | Fair | Fair | Fair | Good | Manifold Merging / Cayley Transform |
| **7** | O-LoRTA: Orthogonal Low-Rank Spectrum Regularization for Robust Test-Time Model Merging and Adaptation | **5 (Accept)** | Excellent | Excellent | Excellent | Good | Test-Time Adaptation (SD-SAM/SOSR) |
| **8** | **Soft-Bounded Fisher-Guided Sharpness-Aware Test-Time Synergistic Model Merging (SBF-SAT-SyMerge)** | **6 (Strong Accept)** | Excellent | Excellent | Excellent | Excellent | Test-Time Adaptation (Fisher/SAM) |
| **10** | Anchoring Test-Time Adaptation to Flat Minima: Fisher-Anchored Sharpness-Aware Test-Time Synergistic Model Merging | **5 (Accept)** | Excellent | Excellent | Good | Excellent | Test-Time Adaptation (FA-SAMerge) |

---

## 3. Individual Submission Review Summaries

### Submission 1: SATA-OP
* **Recommendation:** Weak Accept (4/4)
* **Summary:** This paper proposes SATA-OP (and a high-capacity unilateral projection variant SATA-OP-HC) to eliminate parameter interference in model merging. It pre-computes mutually orthogonal random subspaces prior to fine-tuning, mathematically guaranteeing zero parameter-level Frobenius inner products when task updates are merged. It combines this with SAM for local flatness.
* **Strengths:** Excellent mathematical proof of zero parameter interference; rigorous projected Hessian analysis; highly insightful SVD domain-biased projection pitfall discovery.
* **Weaknesses:** Evaluation is limited to toy-scale (ResNet-18 on MNIST/FashionMNIST/KMNIST); notable performance degradation on KMNIST clean accuracy; lacks comparisons against standard post-hoc model merging baselines; scaling bottleneck when the number of tasks exceeds layer dimensions.

### Submission 2: SO-LoRA
* **Recommendation:** Weak Reject (3/4)
* **Summary:** Proposes pre-partitioning the low-rank output projection space $B$ of LoRA adapters using a deterministic, shared orthogonal basis generated from a fixed random seed. Constrains task low-rank projections to disjoint subspaces to ensure output-side weight orthogonality, and uses SAM during training.
* **Strengths:** Creative, communication-free decentralized merging idea; mathematically solid representation-level orthogonality proofs; detailed parameter complexity analysis (~49% reduction in parameters).
* **Weaknesses:** Flawed baseline comparison—claims superiority over standard LoRA merging based on an unfair "strawman" comparison at merging coefficient $\lambda=1.0$, whereas standard LoRA with optimal averaging ($\lambda=0.5$) actually outperforms the proposed method (39.75% vs. 38.80%); overstated "lossless" claims (merged accuracy is under 40% compared to ~80-86% individual accuracy); severe logit scale and calibration flaws leading to unexplained task performance asymmetry.

### Submission 3: SR-TTA
* **Recommendation:** Accept (5/4)
* **Summary:** Targets the problem of "head dominance" and "spectral collapse" in unsupervised test-time adaptation (TTA) of synergistic model merging. Proposes Spectral-Regularized Test-Time Adaptation (SR-TTA), exploring Isotropic Spectral Regularization (ISR-FO) and Soft-Orthogonality Spectral Regularization (SOSR) directly inside the online TTA loop to stabilize class prototypes.
* **Strengths:** Strong mathematical derivations and proofs (Proposition 3.1); highly insightful representation-level diagnostics (condition numbers and prototype similarity curves); comprehensive stability sweeps over learning rates and horizons (up to 100 steps).
* **Weaknesses:** Restructured to low-resolution grayscale datasets on ResNet-18; computing the $W_k W_k^T$ matrix correlation scales quadratically with class count $O(C^2 D)$, presenting a severe bottleneck for LLMs; high memory footprint due to keeping all expert models active during adaptation.

### Submission 4: SATA-TTA
* **Recommendation:** Strong Accept (6/4)
* **Summary:** Proposes SATA-TTA, a training-free framework that dynamically optimizes layer-wise merging coefficients and classification heads on unlabeled online test streams. To prevent decision-boundary collapse, it uses stateless expert-guided self-labeling and integrates SAM into the test-time adaptation loop, along with Soft-Orthogonality Spectral Regularization (SOSR).
* **Strengths:** Exceptional empirical gains (+8.81% absolute improvement over static merging); outstanding robustness under highly challenging, non-stationary sequential and skewed task shifts; brilliant theoretical foundation—proving a PAC-Bayesian Generalization Bound (Proposition 3.2) and a Manifold Incompatibility Theorem (Proposition 3.1) which explains why applying weight-level spectral constraints to scalar merging coefficients degrades performance; stateless functional-call formulation using `torch.func` to avoid autograd conflicts and keep GPU memory low.
* **Weaknesses:** Small-scale evaluation (CIFAR-10/SVHN classification with ViT-B/16); non-trivial expert model inference overhead during adaptation; high sensitivity to the TTA learning rate.

### Submission 5: SPOR
* **Recommendation:** Accept (5/4)
* **Summary:** Identifies the "flatness-orthogonality decoupling" mismatch—showing that standard training-side SAM flatness optimization prioritizes Euclidean trajectories without regard for weight symmetries, thereby increasing weight-space distortion and degrading compatibility with geometric merging methods. To bridge this gap, the paper proposes Surrogate Procrustes Orthogonality Regularization (SPOR), an SVD-free fine-tuning regularizer that penalizes weight updates that drift from the orthogonal rotation group.
* **Strengths:** Identification of a critical, timely problem; elegant and stable SVD-free SPOR surrogate loss that avoids backpropagating through computationally expensive SVD operations; rigorous proofs in Appendix A; strong empirical results outperforming standard SGD when combined with C-Ortho; excellent taxonomic related work section with 146 references.
* **Weaknesses:** Evaluation is small-scale (ResNet-18 on split CIFAR-10); hyperparameter sensitivity to the regularization strength $\beta$; post-hoc residual norm is not strictly minimized at peak performing accuracy; potential bottlenecks in low-dimensional or low-rank layers.

### Submission 6: LROM-SR
* **Recommendation:** Weak Reject (3/4)
* **Summary:** Introduces Low-Rank Orthogonal Manifold Merging with Spectral Regularization (LROM-SR) to merge LoRA adapters. Incorporates Soft-Orthogonality Spectral Regularization (SOSR) and SAM during training, and performs Orthogonal Procrustes alignment on the factor matrices. For interpolation, it maps alignment rotations to Lie algebra via Cayley transform, averages them, and maps back.
* **Strengths:** Highly parameter-efficient concept; comprehensive experimental sweeps; detailed complexity analysis demonstrating minor merging overhead.
* **Weaknesses:** Core experimental performance deficit—the proposed pipeline SATA + PALM (94.37%) is consistently outperformed by simpler baselines like SAM-only SVDM (95.53%) and even standard-trained SVDM (94.66%); saving ~28 ms of offline merging time is practically negligible; theoretical errors in claiming the Cayley transform defines a "smooth geodesic path" (it is only a rational Padé approximation with non-constant speed and non-zero tangential acceleration) and loose formulation in Theorem 4.1; catastrophic performance drop at the midpoint ($\lambda=0.5$); completely broken bibliography with numerous mismatched citation keys and unresolved references (`[?]`).

### Submission 7: O-LoRTA
* **Recommendation:** Accept (5/4)
* **Summary:** Addresses "representation collapse" and "parameter drift" during unsupervised TTA of merged low-rank adapters. Proposes O-LoRTA, combining SAM (specifically, Scale-Decoupled SAM) with Soft Orthogonality Spectral Regularization (SOSR) during test-time adaptation.
* **Strengths:** Complete and technically solid; incorporates PAC-Bayesian theoretical bound; direct empirical SVD-based validation of singular value isotropization; Scale-Decoupled SAM successfully resolves the gradient-scale mismatch between heads and LoRA layers; highly professional visuals and clean, bug-free typesetting.
* **Weaknesses:** Evaluated on a relatively small-scale benchmark (split CIFAR-10 with ResNet-18); requires manual hyperparameter tuning of the SOSR and SAM parameters.

### Submission 8: SBF-SAT-SyMerge
* **Recommendation:** Strong Accept (6/4)
* **Summary:** Focuses on test-time adaptation of classification heads and merging coefficients in SyMerge. Proposes SBF-SAT-SyMerge, integrating SAM into TTA guided by the running diagonal of the Fisher Information Matrix (FIM). Critical parameters are protected via a bounded, self-normalizing exponential decay function ($t_i = \exp(-F_i / \bar{F})$), avoiding the "parameter explosion bug" of traditional inverse-Fisher scaling ($1 / (F_i + \eta)$). Refines this via a "Per-Tensor" normalization strategy.
* **Strengths:** Strong mathematical and algorithmic foundation; Per-Tensor normalization elegantly resolves layer-wise gradient-scale imbalances between classification heads and merging coefficients; extremely low overhead (<1% additional wall-clock time over standard SAM); exceptional presentation with publication-quality vector plots.
* **Weaknesses:** Evaluation restricted to toy-scale vision datasets on ResNet-18 (partially mitigated by a dedicated analytical section explaining how the method scales to Transformers and LLMs).

### Submission 10: FA-SAMerge
* **Recommendation:** Accept (5/4)
* **Summary:** Proposes FA-SAMerge, introducing Fisher-Anchored Sharpness-Aware Merging (which uses pre-computed clean empirical FIM to apply an EWC-style quadratic penalty during ASAM-constrained TTA) and Bounded Fisher-weighted Adaptive SAM (BF-ASAM) to stabilize scale-invariant perturbations against near-zero Fisher weight instabilities.
* **Strengths:** Creative connection between EWC and sharpness-aware test-time adaptation; Proposition 3.1 mathematically justifies why the bounded scaling factor resolves unconstrained gradient instabilities; breakthrough results on high-frequency Gaussian Noise; highly polished manuscript.
* **Weaknesses:** Evaluation scale is small; dependency on pre-computing diagonal Fisher estimates on a small clean support set (512 samples), which violates strict zero-shot test-time adaptation; asymmetrical performance where OOD performance on Blur and Contrast is slightly worse than standard SAM/ASAM.

---

## 4. Meta-Review Analysis & Selection Rationale

### A. The Top-Tier Choices (Strong Accepts: Submissions 4 and 8)
Two submissions stood out immediately as exceptional, receiving unanimous and highly enthusiastic **Strong Accept (6)** recommendations:

* **Submission 4 (SATA-TTA):** This paper is a tour de force in test-time adaptation. It solves real-world temporal non-stationarity (e.g., sequential task shifts where standard TTA catastrophically collapses) while delivering major empirical gains (+8.81% absolute). What elevates it to a Strong Accept is its **outstanding theoretical depth**: it provides a complete PAC-Bayesian Generalization Bound and uses codimension intersection theory to prove the *Manifold Incompatibility Theorem*—scientifically justifying why weight-space spectral regularizers degrade performance when applied directly to scalar merging coefficients at test-time. It is complete, robust, and highly original.
* **Submission 8 (SBF-SAT-SyMerge):** This paper provides an incredibly elegant and practical contribution to synergistic test-time model merging. It identifies and resolves a massive, previously unaddressed vulnerability: the **"parameter explosion bug"** that occurs when applying inverse-Fisher scaling to sharpness-seeking perturbations. Its Soft-Bounded Fisher (SBF) exponential decay scaling is mathematically bounded and scale-invariant. Furthermore, its **Per-Tensor FIM normalization** is highly innovative, successfully bridging the clean performance regularization penalty while adding less than 1% computational overhead. The paper is technically flawless and highly practical.

### B. Selection of the Third Paper (The Battle of the Accepts: Submissions 3, 5, 7, and 10)
A major part of the meta-review process was deciding which of the four "Accept (5)" papers to include as our third accepted paper. 

* **Submission 3 (SR-TTA)** was a strong candidate due to its excellent diagnostics. However, its quadratic computational complexity with respect to the class count $O(C^2 D)$ presents a severe bottleneck for large language models (LLMs) with vocabularies of $32k$, restricting its practical significance.
* **Submission 10 (FA-SAMerge)** showed outstanding robustness under Gaussian Noise. However, its requirement for a pre-computed clean training support set (512 samples per task) limits its utility in pure zero-shot TTA scenarios. Additionally, it suffered from asymmetrical performance, degrading relative to standard SAM on Blur and Contrast.
* **Submission 7 (O-LoRTA)** is a beautiful and highly complete paper, featuring a PAC-Bayesian bound, Scale-Decoupled SAM, and direct SVD singular-value verification. It was a very strong competitor.
* **Submission 5 (SPOR)** was ultimately selected as our third accepted paper. While Submissions 4, 8, and 7 represent highly related variations of test-time adaptation, **Submission 5 addresses a fundamental training-side gap**—the *flatness-geometry gap*—which affects the compatibility of all subsequent geometric merging methods (like OrthoMerge). It identifies a novel and critical mismatch (flatness optimization distorting coordinate-space alignment), and derives an elegant, stable, **SVD-free Surrogate Procrustes Orthogonality Regularizer (SPOR)** that avoids backpropagating through unstable SVD operations. It achieves a genuine empirical breakthrough by outperforming standard SGD baselines when combined with C-Ortho, and features an outstanding literature synthesis of **146 references**. 

By accepting **Submission 5**, the conference selection represents a highly balanced and scientifically diverse portfolio: two top-tier papers on *robust test-time adaptation and synergistic merging* (Submissions 4 and 8), and one foundational paper on *geometry-preserving training-side flatness regularizers* (Submission 5).

### C. Rejection Justifications
* **Submission 1 (SATA-OP):** Given a **Weak Accept (4)**. While mathematically solid, it has major practical limitations: severe performance degradation on complex tasks like KMNIST, a hard task-scaling bottleneck due to strict disjoint subspace dimensions, and a lack of comparison against standard post-hoc baselines. It is a solid paper but did not make the competitive cutoff.
* **Submission 2 (SO-LoRA):** Given a **Weak Reject (3)**. The paper relies on a "strawman" baseline comparison at merging coefficient $\lambda=1.0$ to claim superiority. Under standard optimal averaging ($\lambda=0.5$), standard LoRA actually outperforms the proposed method. Additionally, it suffers from severe, unexplained task performance asymmetry due to logit calibration issues and makes exaggerated "lossless" claims.
* **Submission 6 (LROM-SR):** Given a **Weak Reject (3)**. This submission has severe empirical, theoretical, and presentation flaws. Empirically, the proposed SATA + PALM pipeline (94.37%) performs *worse* than standard-trained or SAM-only trained SVDM (94.66% and 95.53% respectively), meaning the complex pipeline degrades performance. Theoretically, its geodesic claims using the Cayley transform are mathematically incorrect. Presentationally, its bibliography is completely broken, resulting in dozens of compilation warnings and unresolved `[?]` citations.

---

## 5. Final Accepted Papers Selection Details

The following table displays the specific details of the three papers selected for acceptance.

| Selected Submission | Core Technique | Major Empirical Results | Key Theoretical Contributions | Rationale for Selection |
| :---: | :--- | :--- | :--- | :--- |
| **Submission 4** | SATA-TTA / SOSR | Peak multi-task accuracy of **86.46%** (+6.51% default, +8.81% tuned over static merging); highly robust under sequential shifts (81.56%). | **Manifold Incompatibility Theorem** using codimension intersection; PAC-Bayesian Generalization Bound. | Top-rated paper (Score: 6). Elegant combination of empirical excellence, non-stationary TTA stability, and deep, original mathematical analysis. |
| **Submission 8** | SBF-SAT-SyMerge | Raises Clean accuracy from 66.84% to **72.74%** while preserving OOD robustness; adds under 1% overhead. | Mathematical formulation of **Soft-Bounded Fisher** scale-invariant scaling to prevent parameter explosion. | Top-rated paper (Score: 6). Exceptionally practical, numerically stable, solves layer-wise gradient-scale imbalances via Per-Tensor normalization, beautiful visuals. |
| **Submission 5** | SAM + SPOR | SPOR with C-Ortho achieves **71.79%** accuracy, outperforming SGD (71.38%) and standard SAM (67.60%). | Expansion of the Orthogonal Procrustes trace; SVD-free high-dimensional row-orthogonality approximation. | Highly-rated paper (Score: 5). Bridges the critical flatness-geometry gap, introduces a highly stable SVD-free regularizer, and features outstanding literature coverage. |

The three selected papers have been copied into the `accepted_papers/` directory:
- `accepted_papers/submission4.pdf`
- `accepted_papers/submission8.pdf`
- `accepted_papers/submission5.pdf`

This completes the meta-review process. The selected papers represent outstanding additions to the research community, significantly advancing our understanding of deep model merging, geometric constraints, and robust online adaptation.
