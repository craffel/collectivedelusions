# Meta-Review Summary Report

This document outlines the meta-review process, the evaluation criteria, and the final decisions for the 10 paper submissions evaluated during this conference cycle. Based on a thorough review of the manuscript reviews and contents, exactly **3 out of 10 submissions** have been selected for acceptance.

---

## 1. Meta-Review Process & Evaluation Criteria

Each of the 10 submissions was evaluated by 2 or 3 peer reviewers. The reviewers represented distinct scholarly personas, including:
- **Minimalist:** Prized elegant, simple, and highly effective methods (guided by Occam's razor), penalizing over-engineered or complex pipelines.
- **Practitioner:** Evaluated papers on real-world utility, ease of deployment, data/compute efficiency, and scalability.
- **Critic:** Approached submissions with deep skepticism, looking for logical inconsistencies, flawed assumptions, and baseline issues.
- **Scholar:** Focused on citation accuracy, proper literature contextualization, and experimental/theoretical rigor.

The meta-review process involved aggregating the numeric recommendations, assessing the strength of the qualitative arguments, verifying the empirical soundness of the claims, and ensuring that accepted papers made genuine, reproducible, and significant contributions to the field of multi-task model merging.

---

## 2. Summary of Recommendations & Decisions

Below is the summary of the scores and the final decisions for all 10 submissions. Recommendation scores range from **1 (Strong Reject)** to **6 (Strong Accept)**.

| Submission | Reviewer 1 Score | Reviewer 2 Score | Reviewer 3 Score | Average Score | Final Decision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Submission 3** | — | 6 (Strong Accept) | 3 (Weak Reject) | **4.50** | **ACCEPT** |
| **Submission 10** | 5 (Accept) | 3 (Weak Reject) | — | **4.00** | **ACCEPT** |
| **Submission 9** | 2 (Reject) | 5 (Accept) | — | **3.50** | **ACCEPT** |
| **Submission 5** | 3 (Weak Reject) | 2 (Reject) | 4 (Weak Accept) | **3.00** | REJECT |
| **Submission 6** | 3 (Weak Reject) | 3 (Weak Reject) | 3 (Weak Reject) | **3.00** | REJECT |
| **Submission 4** | 2 (Reject) | 4 (Weak Accept) | 2 (Reject) | **2.67** | REJECT |
| **Submission 7** | 2 (Reject) | 4 (Weak Accept) | 2 (Reject) | **2.67** | REJECT |
| **Submission 1** | 2 (Reject) | — | 3 (Weak Reject) | **2.50** | REJECT |
| **Submission 8** | 3 (Weak Reject) | 2 (Reject) | 2 (Reject) | **2.33** | REJECT |
| **Submission 2** | 2 (Reject) | 2 (Reject) | 2 (Reject) | **2.00** | REJECT |

---

## 3. Meta-Review and Rationale for Accepted Submissions

### 1. Submission 3: *Deconstructing Activation Calibration: Task-Agnostic Alignment for Multi-Task Model Merging*
* **Decision:** **ACCEPT**
* **Reviewer Recommendations:** Reviewer 2: **6 (Strong Accept)** | Reviewer 3: **3 (Weak Reject)** (Average: **4.50**)
* **Summary of Work:** The paper addresses the issue of "variance collapse" in multi-task model merging. It critically deconstructs the state-of-the-art Task-Conditional Activation Calibration (TCAC) method, revealing that forcing task-specific original affine parameters onto a merged, blended feature space causes a geometric representation mismatch and causes the original expert running statistics to cancel out. The authors propose **Native Task-Agnostic Activation Calibration (N-TAAC)**, which uses a joint calibration dataset from all tasks, sets the BatchNorm momentum to 1.0 in training mode with frozen weights, and performs a single forward pass to overwrite native running statistics.
* **Review Feedback & Discussion:** 
  - *Strengths:* Reviewers praised the method's extreme simplicity (a 5-line native PyTorch forward loop), high practical appeal (produces a single, static model with zero test-time overhead or task-ID dependencies), and stellar empirical results (+30% absolute accuracy improvement over the complex TCAC baseline).
  - *Weaknesses:* Reviewer 3 (Practitioner) noted that N-TAAC is mathematically identical to a standard "BatchNorm Reset" baseline, meaning the paper overstates its novelty as a "newly proposed" framework. The bibliography also contained a fabricated citation for the REPAIR paper, and the theoretical proofs relied on unrealistic i.i.d. assumptions for expert weight matrices.
* **Meta-Review Rationale:** Despite the scholarly attribution and citation errors (which must be corrected before publication), this paper represents an outstanding contribution. It successfully applies Occam's razor to model merging, demonstrating that a highly practical, zero-overhead baseline (BatchNorm Reset) can completely dominate complex, dynamic, and hook-based methods. It acts as a powerful course correction for the community, making it highly worthy of acceptance.

---

### 2. Submission 10: *Deconstructing Head-Only Adaptation in Multi-Task Model Merging: Is Representation Calibration Over-Engineered?*
* **Decision:** **ACCEPT**
* **Reviewer Recommendations:** Reviewer 1: **5 (Accept)** | Reviewer 2: **3 (Weak Reject)** (Average: **4.00**)
* **Summary of Work:** This submission presents a critical deconstruction of representation calibration (REPAIR/TCAC) and test-time coefficient optimization (AdaMerging/SyMerge). Using Centered Kernel Alignment (CKA), the authors show that linear weight merging does not destroy the backbone's intermediate representations (CKA > 0.95 in early/middle layers), and that interference is highly localized to the final feature block. They mathematically prove that unconstrained test-time coefficient optimization is unstable (due to runaway negative gradients), and that clamped optimization is inactive (the clamping paradox). They propose **Head-only Adaptation (SFT or TTA)** on a small calibration set, adjusting less than 0.1% of parameters to recover performance.
* **Review Feedback & Discussion:** 
  - *Strengths:* Reviewers appreciated the conceptual elegance and the critical, skeptical perspective. The CKA analysis was deemed highly illuminating, and head-only adaptation proved extremely data-efficient, stable, and easy to implement.
  - *Weaknesses:* Reviewer 2 (Scholar) pointed out that the paper completely failed to cite **T3A** (Iwasawa & Matsuo, NeurIPS 2021), a seminal paper on test-time classifier adjustment under domain shift, which overstates the novelty of the proposed head adaptation. Additionally, the "clamping paradox" proof was criticized as being artificially engineered ($\lambda_{\max} = \lambda_0$), and the activation explosion proof ignored Batch Normalization (which cancels weight scaling).
* **Meta-Review Rationale:** This paper provides a highly valuable sanity check for the model merging community. The empirical finding that backbone representations remain mostly intact after weight merging, and that simple classification head adjustment outperforms complex activation calibration, is of high practical significance. While the theoretical proofs contain some self-serving assumptions and there is a major literature omission (T3A), the empirical and CKA-based contributions are rock-solid and highly instructive for future research.

---

### 3. Submission 9: *Less is More: Preserving ReLU Sparsity via Layer-wise Scaling-only Activation Calibration in Multi-Task Model Merging*
* **Decision:** **ACCEPT**
* **Reviewer Recommendations:** Reviewer 1: **2 (Reject)** | Reviewer 2: **5 (Accept)** (Average: **3.50**)
* **Summary of Work:** The paper focuses on intermediate activation variance collapse during multi-task model merging. It identifies "the sparsity trap" in channel-wise methods like TCAC, showing that estimating channel-wise statistics on small calibration sets leads to division-by-zero or extreme scaling of noise in ReLU networks at test time. The authors propose **Layer-wise Scaling-only Calibration (LSC)** and **Threshold-gated Selective Calibration (TSC)**, which use a single scaling parameter per layer, reducing calibration parameters by over 1300x.
* **Review Feedback & Discussion:** 
  - *Strengths:* Reviewer 2 (Minimalist) lauded the paper's simplicity, parameter efficiency, and the theoretical deconstruction of "the sparsity trap." The discussion in Appendix A analyzing CNNs vs. PLMs under weight pruning was also highly praised.
  - *Weaknesses:* Reviewer 1 (Critic) raised a fatal logical contradiction: the paper claims to preserve ReLU sparsity/non-negativity, but the empirical implementation is applied *Pre-ReLU* where there is no sparsity/non-negativity. Post-ReLU, the method performs worse than the uncalibrated baseline. Furthermore, the method requires knowing task identity during inference, creating an "existential paradox" since routing inputs to separate experts (which gets 84.2% accuracy) is vastly superior to the merged model (which gets 45.03% or 51.63% accuracy).
* **Meta-Review Rationale:** Submission 9 is accepted as the third paper because its average recommendation is significantly higher than the remaining pool. It offers a highly parameter-efficient alternative to channel-wise calibration and a rigorous mathematical exposition of "the sparsity trap." Although the "task identity paradox" and the pre- vs. post-ReLU implementation contradiction are serious conceptual weaknesses, they are transparently documented and provide fertile ground for community discussion. Compared to other 3.0-rated submissions, the conceptual and analytical depth of this paper is superior.

---

## 4. Summary of Decisions for Rejected Submissions

The remaining 7 submissions were rejected due to a combination of conceptual flaws, weak baselines, over-engineered complexity, and limited significance:

- **Submission 5 (Score: 3.00 - Reject/Weak Reject/Weak Accept):** Proposes multivariate covariance calibration (M-CAC/R-MCAC). While theoretically elegant, the empirical evaluation is exceptionally weak: the experts are severely undertrained (CIFAR-10 accuracy of only 66.66% on ResNet-18), there is a complete lack of statistical analysis (no seeds or error bars), and there is a major presentation contradiction in the Figure 1 caption (M-CAC collapses to random guessing in Table 1, but Figure 1 claims it dominates).
- **Submission 6 (Score: 3.00 - Three Weak Rejects):** Addresses a sequential activation dependency bug in prior hook-based codes and proposes TTBN/STTBN (no-shot calibration). However, the method is strictly limited to BatchNorm-based networks (architecturally obsolete for LLMs/ViTs), highly fragile in production due to test-time batch size/momentum sensitivity, and suffers from a massive performance gap (8.7% absolute drop compared to experts).
- **Submission 4 (Score: 2.67 - Two Rejects / One Weak Accept):** Introduces "Holographic CDMA Merging" using weight-space multiplexing. Rerejected due to extreme over-engineering, which destroys the latency and simplicity advantages of model merging. It compares unfairly to traditional merging baselines, completely omits PEFT/LoRA baselines, and is evaluated only on substandard expert models.
- **Submission 7 (Score: 2.67 - Two Rejects / One Weak Accept):** Addresses hardware-aware selective activation calibration via Sequential Variance-Collapse Selection (SVCS). Rejected because SVCS is mathematically flawed and consistently **underperforms** a simple, zero-overhead **Random Selection** baseline across almost all configurations, making the core proposed algorithm practically useless. It also contained a fabricated citation of the REPAIR paper.
- **Submission 1 (Score: 2.50 - Reject / Weak Reject):** Evaluates Activation Overlap Search (AOS) on ResNet-18. Plagued by major technical contradictions, statistically implausible results (reporting exactly 0.00% standard deviations across multiple seeds), a mathematically flawed single-coefficient formulation, and an outdated/low-scale evaluation.
- **Submission 8 (Score: 2.33 - Two Rejects / One Weak Reject):** Proposes a data-free representation calibration framework. Undermined by major conceptual inconsistencies, circular proofs of core theorems, unconvincing empirical results (where it is outperformed by a naive linear baseline), and systemic bibliography fabrication.
- **Submission 2 (Score: 2.00 - Three Rejects):** Proposes block-wise activation ensemble (CPOS) using hypercomplex math. Rejected because it is not actually "model merging" but a parallel ensemble requiring $O(N)$ parameters and $O(N)$ compute. Crucially, a simple task-routing baseline is $2.4\times$ faster, uses half the memory, and outperforms CPOS by $19.97\%$ absolute accuracy, rendering the proposed method practically useless.
