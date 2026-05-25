# Meta-Review Summary Report

**Date:** May 23, 2026  
**Conference:** International Conference on Machine Learning (ICML 2026)  
**Topic Area:** Test-Time Model Merging (TTMM) & Test-Time Adaptation (TTA)  
**Meta-Reviewer:** Gemini CLI Autonomous Agent  

---

## 1. Executive Summary
This report summarizes the meta-review process and decisions for 10 paper submissions targeting the highly active domain of **Test-Time Model Merging (TTMM)**. TTMM is an emerging paradigm aimed at dynamically fusing task-specific expert neural networks into a single multi-task model during deployment to adapt to streaming, non-stationary test data. 

Following a rigorous, multi-dimensional review of the manuscripts, experimental protocols, theoretical proofs, and peer review reports, **three submissions** have been selected for acceptance:
1.  **Submission 3 (Strong Accept - Score: 6):** *Information-Geometric Gradient Surgery for Open-World Test-Time Model Merging* (IGGS-OW)
2.  **Submission 10 (Accept - Score: 5):** *Dynamic Routing and Test-Time Fisher Information for Robust Riemannian Model Merging* (DR-Fisher)
3.  **Submission 6 (Accept - Score: 5):** *Fisher-Preconditioned Contrastive Alignment for Open-World Test-Time Model Merging* (FP-OW)

The accepted papers represent substantial theoretical and empirical breakthroughs in making TTMM fully unsupervised, highly robust to non-stationary streams and severe covariate shifts, and computationally viable for resource-constrained edge devices.

---

## 2. Meta-Review Process and Evaluation Criteria
Each of the 10 submissions was evaluated across four primary dimensions established by the conference guidelines:
*   **Soundness (Technical Rigor):** The mathematical formulation, theoretical correctness, experimental setup, baseline appropriateness, and reproducible seeding.
*   **Presentation (Clarity):** The layout of tables and figures, the flow of the narrative, consistent notation, and adherence to style guides.
*   **Significance (Impact):** Practical utility, real-world relevance, scalability of backbones/datasets, and potential to shape future research.
*   **Originality (Novelty):** The uniqueness of the proposed combinations of information geometry, gradient optimization, and representation alignment.

Special attention was paid to identifying **conceptual contributions** (such as identifying previously overlooked pipelines or optimization failures) and **computational feasibility at test-time** (where high-overhead backpropagation or multi-pass gradient surgery undermines the edge-friendly motivation of TTMM).

---

## 3. Comprehensive Overview of Submissions

| Submission ID | Title | Key Framework | Score | Decision | Key Strengths / Critical Weaknesses |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **S1** | Information-Geometric Open-World Test-Time Model Merging | IG-PROTO-TTMM | 5 | Reject | **Strengths:** Beautiful proofs, excellent trajectory plots, fast execution.<br>**Weaknesses:** Small-scale benchmarks, slightly lower significance than FP-OW. |
| **S2** | Fisher-Weighted Prototype Alignment for Robust Teacher-Free Test-Time Model Merging | FWPA | 5 | Reject | **Strengths:** Elegant bridge between parameter curvature and alignment.<br>**Weaknesses:** Limited toy-scale (MNIST) experiments, high hyperparameter sensitivity ($\beta$). |
| **S3** | **Information-Geometric Gradient Surgery for Open-World Test-Time Model Merging** | **IGGS-OW** | **6** | **Accept** | **Strengths:** Flawless presentation, outstanding empirical results (+22% novel accuracy), elegant Proposition 4.1. Only Strong Accept. |
| **S4** | Riemannian Gradient Surgery with Contrastive Orthogonal Projection for Stable Test-Time Model Merging | RGS-COP | 4 | Reject | **Strengths:** Strong theoretical proofs, detailed resource/latency profiling.<br>**Weaknesses:** Mathematical mischaracterization of "Orthogonal Projection" (it is a scalar multiplier); negligible empirical improvements over simpler baselines. |
| **S5** | Calibration-Free Test-Time Model Merging via Active Online Fisher Preconditioning | AO-TTMM | 4 | Reject | **Strengths:** Dynamic on-the-fly online Fisher estimation via entropy gradients.<br>**Weaknesses:** Prohibitive computational overhead (full backpropagation at every step); degraded performance under extreme noise. |
| **S6** | **Fisher-Preconditioned Contrastive Alignment for Open-World Test-Time Model Merging** | **FP-OW** | **5** | **Accept** | **Strengths:** Clean sweep of "Excellent" across Soundness, Presentation, Significance, and Originality. Massive clean-stream boost (+9% absolute), flawless formatting. |
| **S7** | Temporal Momentum Surgery for Stable and Adaptive Test-Time Model Merging | TMS | 5 | Reject | **Strengths:** Identifies and resolves momentum lag at task boundaries.<br>**Weaknesses:** Restrictive evaluation scale, high reliance on shared loss basin assumption. |
| **S8** | IGGS-PROTO: Information-Geometric Gradient Surgery for Open-World Test-Time Model Merging | IGGS-PROTO | 5 | Reject | **Strengths:** Bypasses feedback trap with 100% Novelty Detection Rate.<br>**Weaknesses:** Discrepancy between math formulation (appendix uses inverse Fisher) and actual implementation (code uses uniform gradient). |
| **S9** | FOGS-Merge: Information-Geometric Gradient Surgery on Contrastive Alignment Gradients for Robust Test-Time Model Merging | FOGS-Merge | 5 | Reject | **Strengths:** Massive accuracy leaps (+37.76% overall), clever confidence gating.<br>**Weaknesses:** Extremely high computational latency (requires up to 10 separate backward passes per batch), making it impractical for edge devices. |
| **S10** | **Dynamic Routing and Test-Time Fisher Information for Robust Riemannian Model Merging** | **DR-Fisher** | **5** | **Accept** | **Strengths:** Landmark diagnostic finding (BN buffer omission in previous TTMM pipelines); spectacular empirical gains (+58% absolute); fully data-free and unsupervised. |

---

## 4. In-Depth Justification for Accepted Submissions

### 1. Submission 3: IGGS-OW (Strong Accept - Score: 6)
*   **Title:** *Information-Geometric Gradient Surgery for Open-World Test-Time Model Merging*
*   **Rationale for Acceptance:**
    Submission 3 is the top-performing paper in this pool, receiving a unanimous and enthusiastic **Strong Accept**. The authors identify and diagnose two fundamental limitations of contemporary open-world model merging: the *Representational Space Alignment Mismatch* (where offline prototypes are computed from static experts while test features come from active merged models, leading to a 68.3% False Positive Rate) and the *Feedback Loop Trap* (where decayed coefficients freeze learning). 
    
    They resolve both elegantly via **IGGS-OW** using:
    1.  A unified static space precomputation that reduces the False Positive Rate to 0%.
    2.  An Information-Geometric Riemannian Space preconditioned by joint diagonal Fisher Information.
    3.  Entropy-guided Riemannian adaptation on the probability simplex.
    
    The theoretical foundation is extraordinarily strong: **Proposition 4.1** provides a formal guarantee that preconditioning suppresses representational drift in highly sensitive layers. Empirically, the method achieves a massive **+22.92%** absolute improvement on the novel domain (reaching 92.24% accuracy) with perfect novelty routing. It is technically flawless, beautifully presented, and highly reproducible.

### 2. Submission 10: DR-Fisher (Accept - Score: 5)
*   **Title:** *Dynamic Routing and Test-Time Fisher Information for Robust Riemannian Model Merging*
*   **Rationale for Acceptance:**
    Submission 10 makes a **landmark diagnostic contribution** to the model merging literature. The authors reveal that prior TTMM frameworks completely omitted Batch Normalization (BN) running buffers (running mean and variance) during weight fusion, forcing the merged model to use stale pre-trained running statistics. By introducing **differentiable weight and BN buffer merging**, they resolve this activation mismatch, elevating the baseline static merging accuracy by over 10% absolute.
    
    Furthermore, they address the "private training data dependency" of traditional Fisher-based methods by proposing **TT-Fisher**, which estimates diagonal parameter sensitivities directly on the incoming unsupervised test stream via pseudo-labels. Combined with **Entropy-Based Expert Routing (EBER)**, DR-Fisher delivers monumental performance gains of **up to +58% absolute accuracy** on clean and corrupted alternating streams, maintaining under 50ms per batch on a single CPU. It represents a major leap in making TTMM fully unsupervised, data-free, and practically deployable on edge devices.

### 3. Submission 6: FP-OW (Accept - Score: 5)
*   **Title:** *Fisher-Preconditioned Contrastive Alignment for Open-World Test-Time Model Merging*
*   **Rationale for Acceptance:**
    Submission 6 stands out for its exceptional completeness and polish, earning a clean sweep of **"Excellent"** in Soundness, Presentation, Significance, and Originality. The paper tackles uniform learning rate optimization bottlenecks by scaling adaptation steps inversely with average diagonal Fisher Information. 
    
    The paper is highly praised for:
    1.  **Proposition 3.1**, which rigorously bounds the expected squared representational drift of layer-wise activations by $O(\bar{F}_w^{-1})$.
    2.  An outstanding empirical evaluation spanning 6 diverse baselines across both clean and corrupted non-stationary streams.
    3.  Resolving historical limitations by properly centering offline prototypes to achieve a 0% False Positive Rate on known tasks and integrating Test-Time BatchNorm Adaptation (AdaBN) to reduce false positives under corruption from 100% to 3.33%.
    
    The manuscript is beautifully organized, exceptionally thorough (with 55 well-contextualized references), and features extremely polished results tables. Its combination of theoretical depth and highly verified empirical gains makes it a clear accept.

---

## 5. Summary of Key Excluded Submissions
*   **Submission 9 (FOGS-Merge - Score: 5):** While FOGS-Merge reports spectacular empirical gains (+37% clean accuracy), it introduces a critical design flaw: computing class-specific contrastive gradients requires grouping batches by predicted classes and performing separate backpropagation passes for each present class. In standard classification benchmarks with up to 10 classes, this demands up to **10 separate backward passes through the entire deep network backbone per batch**, resulting in a 10x latency increase that completely violates the low-resource, edge-friendly assumptions of TTMM.
*   **Submission 8 (IGGS-PROTO - Score: 5):** This paper is highly rigorous but was rejected in favor of others due to a critical discrepancy between the Appendix (which describes parameter updates preconditioned by inverse Fisher metric) and the actual implementation (which uses standard un-preconditioned gradient descent on raw logits).
*   **Submission 4 (RGS-COP - Score: 4):** Despite high polish, this submission mischaracterizes its core contribution: "Contrastive Orthogonal Projection" is mathematically formulated as a scalar multiplier rather than a projection matrix. Thus, it only scales the gradient without projecting it orthogonal to any manifold. Additionally, its empirical gains over much simpler baselines were extremely marginal (<0.05%).
*   **Submission 5 (AO-TTMM - Score: 4):** This calibration-free approach estimates Fisher Information on-the-fly, but requires performing a backward pass of the entropy loss with respect to *all* parameters of the model (millions of weights) at every single step, introducing a prohibitive computational burden for edge devices.

---

## 6. Conclusion
The meta-review process successfully filtered out submissions with implementation-math discrepancies (S8), hidden computational bottlenecks (S9, S5), or loose mathematical definitions (S4). The chosen papers—**Submission 3, Submission 10, and Submission 6**—provide a balanced and formidable cohort of accepted papers that advance the boundaries of Test-Time Model Merging with high mathematical rigor, magnificent empirical gains, and true real-world edge feasibility.
