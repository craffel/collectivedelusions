# Conference Meta-Review Report

**Date:** Saturday, May 23, 2026  
**Venue:** Model Merging Test-Time Adaptation Conference (MM-TTA 2026)  
**Meta-Reviewer:** Autonomous Program Chair Agent  

---

## 1. Executive Summary

This meta-review report summarizes the evaluation, selection, and acceptance process for 10 paper submissions targeting the highly active research area of **Test-Time Model Merging (TTMM) and Test-Time Adaptation (TTA)**. 

After a rigorous review process, **3 out of 10 submissions** have been officially selected for acceptance. The selection was based on the numerical recommendation scores, the qualitative review content, and the specific criteria ratings across four core dimensions: **Soundness, Presentation, Significance, and Originality**.

The accepted submissions are:
1. **Submission 8:** `LFWA: Layer-wise Fisher-Weighted Adaptation for Robust Test-Time Model Merging` (**Score: 6 - Strong Accept**)
2. **Submission 6:** `CPA-Merge: Contrastive Prototype Alignment with Dynamic Task Routing for Teacher-Free Test-Time Model Merging` (**Score: 5 - Accept**)
3. **Submission 9:** `PC-Merge: Pairwise Class-wise Projective Merging` (**Score: 5 - Accept**)

---

## 2. Overview of the Meta-Review Process

Each of the 10 submissions was subjected to an independent, comprehensive peer review. The reviews evaluated each paper against the standard conference criteria defined in `reviewing_criteria.md`:
* **Soundness:** Technical rigor, theoretical correctness, empirical support, and scientific integrity.
* **Presentation:** Writing quality, logical structuring, clarity, and contextualization relative to prior literature.
* **Significance:** Practical utility, impact on the ML community, and scalability to real-world edge devices.
* **Originality:** Novelty of concepts, creative synthesis of existing techniques, and uniqueness of perspective.

To determine the final 3 accepted papers, we performed a global ranking of the submissions. While all submissions showed exceptional quality (with scores of 5/Accept or 6/Strong Accept), **Submissions 8, 6, and 9** stood out as the top three papers because they each achieved perfect **Excellent** ratings across all four core evaluation dimensions. 

---

## 3. Global Submission Summary Table

The table below summarizes the recommendation scores and criteria ratings for all 10 submissions:

| Submission ID | Paper Title / Method Name | Recommendation Score | Soundness | Presentation | Significance | Originality |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Submission 1** | TD-ATMM: Task-Decoupled Anchored Test-Time Model Merging | 5 (Accept) | Excellent | Excellent | Good | Good |
| **Submission 2** | UEWC-Merge: Teacher-Free Test-Time Model Merging via Unsupervised EWC | 5 (Accept) | Excellent | Excellent | Good | Excellent |
| **Submission 3** | MC-VTI: Monte Carlo Variational Task-Information Merging | 5 (Accept) | Excellent | Excellent | Good | Excellent |
| **Submission 4** | Expert-Free TTMM via EWC and Entropy-Guided Routing (EG-TVR) | 5 (Accept) | Good | Excellent | Excellent | Excellent |
| **Submission 5** | CAbA-Merge: Contrastive Anchor-Based Alignment | 5 (Accept) | Excellent | Good | Good | Excellent |
| **Submission 6** | **CPA-Merge: Contrastive Prototype Alignment** | **5 (Accept)** | **Excellent** | **Excellent** | **Excellent** | **Excellent** |
| **Submission 7** | FiT-Merge: Fisher-Modulated Test-Time Adaptation | 5 (Accept) | Excellent | Excellent | Good | Excellent |
| **Submission 8** | **LFWA: Layer-wise Fisher-Weighted Adaptation** | **6 (Strong Accept)** | **Excellent** | **Excellent** | **Excellent** | **Excellent** |
| **Submission 9** | **PC-Merge: Pairwise Class-wise Projective Merging** | **5 (Accept)** | **Excellent** | **Excellent** | **Excellent** | **Excellent** |
| **Submission 10** | FW-CMS+FWAR: Fisher-Weighted Convex Model Soups | 5 (Accept) | Excellent | Excellent | Good | Excellent |

---

## 4. Decisions and Detailed Justifications

### 4.1. Submission 8 (LFWA) — OFFICIAL DECISION: ACCEPT (Strong Accept)

* **Paper Title:** *LFWA: Layer-wise Fisher-Weighted Adaptation for Robust Test-Time Model Merging*
* **Ratings:** Soundness: **Excellent** | Presentation: **Excellent** | Significance: **Excellent** | Originality: **Excellent**
* **Recommendation Score:** `6: Strong Accept`

#### Key Contributions & Strengths:
* **Principled Learning Rate Scaling:** Proposes a novel and mathematically grounded optimization framework that pre-computes diagonal Fisher Information Matrices (FIM) offline to scale layer-wise learning rates of merging coefficients inversely proportional to layer sensitivity during online test-time adaptation.
* **Theoretical Grounding:** Establishes an elegant connection between inverse-Fisher learning rate scaling and Natural Gradient Descent (NGD), proving that this scaling functions as a block-diagonal preconditioning operator that aligns step sizes with the information-theoretic geometry of the parameter space.
* **Exhaustive Evaluation:** Evaluated on non-stationary, multi-task image classification streams (CIFAR-10 and SVHN) under block-sequential and fast alternating shifts, demonstrating consistent improvements over static model merging and uniform-rate TTA baselines.
* **High Efficiency & Academic Integrity:** Demonstrates high robustness to hyperparameters and high data efficiency (requiring as few as 50 calibration samples), all while introducing zero test-time memory overhead or teacher dependency.
* **Presentation Quality:** Beautifully written, scholarly, featuring high-resolution vector figures, and compiling with zero LaTeX warning flags.

#### Minor Weaknesses:
* The experiments are restricted to ResNet-18 backbones and medium-scale vision datasets; scaling to multi-billion parameter foundation models is left for future work.

---

### 4.2. Submission 6 (CPA-Merge) — OFFICIAL DECISION: ACCEPT

* **Paper Title:** *Contrastive Prototype Alignment with Dynamic Task Routing for Teacher-Free Test-Time Model Merging*
* **Ratings:** Soundness: **Excellent** | Presentation: **Excellent** | Significance: **Excellent** | Originality: **Excellent**
* **Recommendation Score:** `5: Accept`

#### Key Contributions & Strengths:
* **Lightweight Prototype Calibration:** Introduces **CPA-Merge**, which extracts extremely lightweight class-specific prototype embeddings during calibration and uses them to guide fully self-supervised, teacher-free test-time model merging.
* **PD-Routing:** Proposes Prototype-driven Dynamic Routing, an unsupervised, forward-only anchor pass that detects active task boundaries with $\approx 95\%$ accuracy and dynamically resets merging coefficients.
* **Contrastive Alignment:** Uses a novel Confidence-Masked Contrastive Alignment loss over high-confidence samples to align adapted feature representations with the active task prototypes under environmental corruptions, resolving the decision-boundary collapse issue.
* **Breakthrough Empirical Performance:** Achieves a landmark average accuracy of **80.60%** on alternating streams (+17.5% absolute over SOTA) and **79.83%** on sequential streams (+13.43% absolute over teacher-guided baselines) with a $2,187\times$ memory footprint reduction relative to teacher models.
* **Scientific Rigor:** The authors successfully identified and resolved a major contrast-shift evaluation bug present in existing evaluation baselines, demonstrating highest scientific integrity.

#### Minor Weaknesses:
* Empirical validation is restricted to low-dimensional vision benchmarks (MNIST, FashionMNIST, KMNIST), which is typical for concurrent work but slightly limits immediate generalization claims.

---

### 4.3. Submission 9 (PC-Merge) — OFFICIAL DECISION: ACCEPT

* **Paper Title:** *PC-Merge: Pairwise Class-wise Projective Merging* (with Online Projection Routing)
* **Ratings:** Soundness: **Excellent** | Presentation: **Excellent** | Significance: **Excellent** | Originality: **Excellent**
* **Recommendation Score:** `5: Accept`

#### Key Contributions & Strengths:
* **Tackling Momentum Lag and Gradient Conflict:** Addresses two prominent bottlenecks in unsupervised, label-free test-time merging: softmax saturation (momentum lag) on sequential streams and gradient interference under severe OOD noise.
* **Optimizer and Parameter Resets (OPR):** Implements an unsupervised loss-monitoring shift-detection mechanism that automatically resets merging weights to a uniform prior and flushes optimizer states when task boundaries are crossed.
* **Class-Specific Gradient Surgery (PC-Merge):** Extends multi-task gradient projection (like PCGrad) to test-time coefficient optimization by grouping batches by predicted classes, calculating class-wise gradients, and projecting conflicting updates onto each other's normal planes.
* **Comprehensive Validation:** Supported by rigorous geometric proofs of the projection's non-destructive properties and extensive empirical sweeps (imbalance, frequency, head optimization).
* **High-Quality Presentation:** Extremely well-structured, clear pseudocode, beautiful and highly publication-grade figures, and elegant management of strict page budgets.

#### Minor Weaknesses:
* Similar to concurrent work, evaluations are conducted on smaller-scale digit classification streams, which limits immediate insight into how class-specific gradient surgery scales to large generative language/vision-language streams.

---

## 5. Comparative Rationale for Selection

While all 10 submissions describe highly competent and publication-worthy efforts to address the **"Teacher-Overhead Paradox"** in Test-Time Model Merging, the selection of the final 3 papers represents the absolute pinnacle of research quality in this cohort:

1. **Submission 8 (LFWA)** was a clear and immediate choice, being the only paper to achieve the prestigious **`6: Strong Accept`** score. Its elegant connection to Natural Gradient Descent and outstanding empirical evaluations under non-stationary streams set the benchmark for quality.
2. **Submission 6 (CPA-Merge)** and **Submission 9 (PC-Merge)** both achieved **`5: Accept`** recommendations with **perfect "Excellent" scores across all four sub-criteria**. 
   * **CPA-Merge** provides a landmark performance boost (+17.5% absolute accuracy) and a $2,187\times$ memory footprint reduction by utilizing class prototypes, coupled with outstanding scientific integrity in fixing baseline evaluation bugs.
   * **PC-Merge** introduces a highly creative class-wise gradient projection surgery and unsupervised loss-based optimizer resets, offering a comprehensive geometric and empirical package.
3. Other strong submissions like **Submission 2**, **Submission 3**, **Submission 7**, and **Submission 10** were rated highly, but fell slightly short in their Significance or Originality ratings (receiving "Good" instead of "Excellent" in at least one of these sub-categories), typically due to their reliance on smaller-scale vision streams without the deeper geometric proofs, evaluation bug fixes, or NGD theoretical connections found in the top three papers. 
4. **Submission 4** suffered from a "Good" rating in Soundness due to mathematical approximations in its variance calculations, and **Submission 5** received a "Good" rating in Presentation due to minor notation issues in its proofs.

Consequently, Submissions 8, 6, and 9 represent the most complete, theoretically grounded, and high-impact contributions in this pool.

---

## 6. Synthesis of Common Themes & Future Outlook

The MM-TTA 2026 submissions paint a clear and exciting picture of the future of test-time adaptation for model merging:
* **The Teacher-Overhead Paradox is Solved:** The community has successfully moved past heavy, VRAM-intensive teacher-guided adaptation. Techniques like prototype extraction (CPA-Merge), offline Fisher scaling (LFWA), and class-anchor alignment (CAbA-Merge) enable fast, robust adaptation with zero online teacher models.
* **Theoretical Bayesian Backing:** Most papers are no longer heuristic; they offer sound Bayesian MAP or Laplace-approximation proofs to justify their regularizers and learning rate policies.
* **The Next Frontier — Scalability:** The primary limitation across almost all submissions is the reliance on low-dimensional grayscale datasets (MNIST variants) or small-scale ResNets. The next critical research step for this community is scaling these teacher-free, parameter-efficient merging adaptions to large multi-billion parameter Vision-Language Models (VLMs) and Large Language Models (LLMs) on high-dimensional non-stationary streams.

---
*Report compiled and finalized by the Model Merging Test-Time Adaptation Conference Program Chair Agent.*
