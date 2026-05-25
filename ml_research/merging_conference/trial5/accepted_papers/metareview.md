# Meta-Review Summary and Decisions

**Conference:** International Conference on Machine Learning (ICML 2026)  
**Track:** Test-Time Model Merging (TTMM) & Test-Time Adaptation (TTA)  
**Meta-Reviewer:** Autonomous Program Chair Agent  

---

## 1. Executive Summary

This meta-review document summarizes the evaluation process and final decisions for **10 paper submissions** in the Test-Time Model Merging (TTMM) track. The primary goal was to select the **top 3 submissions** for acceptance based on peer reviews, technical contribution, theoretical rigor, empirical completeness, and overall significance.

After a thorough meta-evaluation of all review materials:
1. **Submission 2** has been accepted with a **6 (Strong Accept)** recommendation.
2. **Submissions 7 and 8** have been accepted with **5 (Accept)** recommendations.
3. The remaining 7 submissions have been rejected or deferred.

The `submission.pdf` files for the accepted papers have been copied to the `accepted_papers/` directory:
- `accepted_papers/submission2.pdf`
- `accepted_papers/submission7.pdf`
- `accepted_papers/submission8.pdf`

---

## 2. Overview of Submissions and Ratings

Below is a consolidated summary of the ratings and titles of all 10 submissions:

| Submission | Title | Overall Recommendation | Key Sub-Ratings (Soundness / Presentation / Significance / Originality) | Decision |
| :--- | :--- | :---: | :---: | :---: |
| **Submission 1** | Self-Refining Calibration-Free Prototype Alignment for Robust Test-Time Model Merging | **5: Accept** | Excellent / Excellent / Good / Excellent | Reject (Scale/Sim Limit) |
| **Submission 2** | Fisher-Preconditioned Contrastive Alignment for Teacher-Free Test-Time Model Merging | **6: Strong Accept** | Excellent / Excellent / Excellent / Excellent | **Accept** |
| **Submission 3** | AdaSNR-Adam: Task-Conditioned Online Gradient Signal-to-Noise Ratio for Robust Test-Time Model Merging | **5: Accept** | Excellent / Excellent / Good / Excellent | Reject (Borderline) |
| **Submission 4** | Fisher-Guided Sparsity-Aware Model Merging for Robust and Stable Test-Time Adaptation | **5: Accept** | Good / Excellent / Good / Good | Reject (Technical Flaws) |
| **Submission 5** | Class-Prototype Contrastive Alignment with Dynamic Reset (CP-CADR) for Robust Test-Time Model Merging | **3: Weak Reject** | N/A / Excellent / N/A / N/A | Reject |
| **Submission 6** | Adaptive Gradient Conflict Resolution via Layer-wise Fisher Projection for Robust Test-Time Model Merging | **5: Accept** | Good / Excellent / Good / Good | Reject (Limited Scale) |
| **Submission 7** | IGGS-Merge: Information-Geometric Gradient Surgery for Robust Test-Time Model Merging | **5: Accept** | Excellent / Excellent / Good / Excellent | **Accept** |
| **Submission 8** | PROTO-TTMM: Breaking the Closed-World Assumption in Test-Time Model Merging | **5: Accept** | Excellent / Excellent / Good / Excellent | **Accept** |
| **Submission 9** | Fisher-Weighted Class-Specific Gradient Projection for Robust Test-Time Model Merging | **4: Weak Accept** | Good / Excellent / Good / Good | Reject |
| **Submission 10** | Stabilizing Test-Time Model Merging on Non-Stationary Streams via Entropy-Weighted Fisher Regularization | **3: Weak Reject** | Fair / Good / Fair / Fair | Reject |

---

## 3. Justifications for Accepted Submissions

### Submission 2: Fisher-Preconditioned Contrastive Alignment for Teacher-Free Test-Time Model Merging
* **Decision:** **Accept** (Overall Rating: **6: Strong Accept**)
* **Technical Highlights:** 
  * Addresses the memory-accuracy trade-offs in teacher-guided test-time model merging by introducing a completely **teacher-free** alignment framework.
  * Combines prototype-driven dynamic routing, confidence-masked contrastive loss, and layer-wise Fisher preconditioning.
  * Formulates **Proposition 3.1**, which mathematically bounds the expected squared representational drift under test-time updates, proving that a sensitivity damping factor $\alpha=0.5$ makes the drift invariant to Fisher sensitivity scaling, preventing catastrophic parameter collapse under noise.
* **Reviewer Consensus:** Flawless technical soundness, beautiful presentation, excellent significance, and highly original. Full reproducibility is guaranteed, and a mathematically grounded Impact Statement is included. It represents the highest standards of the conference.

### Submission 8: PROTO-TTMM: Breaking the Closed-World Assumption in Test-Time Model Merging
* **Decision:** **Accept** (Overall Rating: **5: Accept**)
* **Technical Highlights:**
  * Breaks the restrictive **closed-world assumption** in traditional TTMM, which typically assumes incoming streams are only composed of pre-defined source domains.
  * Mathematically formalizes the **"feedback trap"** (where traditional confidence-based methods collapse onto a single incorrect expert under novel domains) as a discrete-time dynamical system and proves its convergence properties.
  * Proposes **PROTO-TTMM** with three key contributions: **Isotropic Feature Centering (IFC)** to resolve representation anisotropy; **Unbiased Routing (UR)** via prototype cohesion to robustly detect novel tasks; and **Online Prototype Management** to dynamically instantiate novel prototypes and adapt coefficients.
  * Proves online prototype convergence and reports exceptional empirical gains, recovering **+75.47% accuracy** on the novel domain.
* **Reviewer Consensus:** High originality, outstanding theoretical depth, and remarkable empirical impact on breaking the closed-world assumption at test-time. It is highly deserving of publication.

### Submission 7: IGGS-Merge: Information-Geometric Gradient Surgery for Robust Test-Time Model Merging
* **Decision:** **Accept** (Overall Rating: **5: Accept**)
* **Technical Highlights:**
  * Addresses gradient conflict and representation collapse in heterogeneous neural network model merging.
  * Identifies a fundamental mathematical limitation where standard Euclidean gradient surgery (e.g., PCGrad) fails to eliminate gradient interference in heterogeneous networks.
  * Resolves this issue using **Information Geometry**, defining a Riemannian manifold on the merging coefficient space using Fisher-weighted metric tensors.
  * Formulates a Riemannian projection scheme (**IGGS-Merge**) to orthogonalize conflicting task gradients under the correct metric tensor.
  * Delivers robust empirical gains (**+13.62% accuracy** under severe noise) and provides thorough mathematical proofs of geometric soundness.
* **Reviewer Consensus:** Excellent soundness, presentation, and originality. This is a highly complete, mathematically rigorous, and beautiful paper that advances optimization theory for model merging.

---

## 4. Justifications for Deferred/Rejected Submissions

The conference had a strict budget of **3 accepted papers**. While several other papers received a **5 (Accept)** rating, they were deferred or rejected due to relative limitations in experimental scale, technical execution, or conceptual leaps:

* **Submission 1 (5: Accept):** Proposes a self-refining, calibration-free prototype alignment framework with strong convergence proofs. However, it was rejected in favor of others because its experimental evaluation was conducted on a **synthetic/simulated benchmark** (`simulate.py`) rather than on real-world datasets with deep neural networks.
* **Submission 3 (5: Accept):** Introduces online gradient SNR tracking to connect Adam optimization to model merging. It is scientifically rigorous, but the core contribution represents a narrower conceptual leap compared to the Riemannian manifolds of Submission 7 or the open-world paradigm of Submission 8.
* **Submission 4 (5: Accept):** Focuses on Fisher-guided sparsity-aware merging. It was rejected due to a **technical discrepancy** in parameter-level Fisher formulation (failing to account for task vector magnitudes when routing) and unaddressed performance collapse under severe Gaussian noise.
* **Submission 6 (5: Accept):** Combines diagonal Fisher information and gradient projection. Although creative, the theoretical derivation relies on simplified gradient-Fisher independence assumptions, and the evaluation scale was relatively limited (ResNet-18 on only two datasets).
* **Submission 9 (4: Weak Accept):** The empirical validation is downscaled to a toy 3-layer CNN on MNIST/FashionMNIST/KMNIST datasets, and there are minor numerical discrepancies between the tables and logs.
* **Submission 5 (3: Weak Reject):** Core hypothesis is directly contradicted by its own empirical results (where baseline CPA-Merge outperforms the proposed method on sequential blocks), and the gradient adaptation framework is redundant.
* **Submission 10 (3: Weak Reject):** Fails to consistently outperform simpler baselines, collapses under Contrast corruption, and contains severe reporting discrepancies between the text and Table 1 results.

---

## 5. Summary of Decisions

| Selection Rank | Submission | Decision | Final Score | Rationale |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Submission 2** | **Accept** | **6 (Strong Accept)** | Highest rated paper; mathematically flawless proofs, exceptional empirical results, and highly reproducible. |
| **2** | **Submission 8** | **Accept** | **5 (Accept)** | Elegant open-world formulation; formalizes the feedback trap as a dynamical system, achieves outstanding empirical gains (+75.47%). |
| **3** | **Submission 7** | **Accept** | **5 (Accept)** | Highly original mathematical contribution using Information Geometry to resolve multi-task gradient surgery on a Riemannian manifold. |
