# Meta-Review Decisions and Process Summary

## 1. Overview of the Meta-Review Process

As Meta-Reviewers, our goal was to conduct a rigorous, principled, and comprehensive evaluation of **10 paper submissions** in the domain of model merging, test-time adaptation (TTA), and parameter-space fusion. The review panel assessed each submission on multiple criteria: mathematical soundness, empirical significance, clarity of presentation, originality, and practical relevance to edge deployment and real-world robustness.

To select the **top 3 submissions to accept** out of the 10 candidates, we adopted a two-tiered decision process:
1. **Quantitative Filtering:** We parsed and compiled the numerical recommendation scores from all 29 individual reviews to calculate average scores and rank-order the submissions.
2. **Qualitative Content Auditing:** We conducted a deep qualitative analysis of the review texts, focusing on the core scientific contributions, the severity of the identified weaknesses, methodological robustness, and the level of reviewer consensus. 

Through this process, we identified a highly competitive group of top-performing papers and resolved a three-way tie for the final acceptance slot based on practical impact, empirical validity, and reviewer unanimity.

---

## 2. Quantitative Summary of Submissions

Below is the complete score matrix for all 10 submissions. Individual scores are mapped to the standard peer-review rating scale (1: Strong Reject, 2: Reject, 3: Weak Reject, 4: Weak Accept, 5: Accept, 6: Strong Accept).

| Submission | Reviewer 1 Score | Reviewer 2 Score | Reviewer 3 Score | Consensus / Average Score | Primary Topic / Methodology | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Submission 1** | 6 (Strong Accept) | 4 (Weak Accept) | 6 (Strong Accept) | **5.33** | Robustness audit of Quantization-Aware Model Merging | **ACCEPT** |
| **Submission 2** | 5 (Accept) | 5 (Accept) | 5 (Accept) | **5.00** | Offline Few-Shot Validation Tuning (OFS-Tune) | **ACCEPT** |
| **Submission 3** | 4 (Weak Accept) | 3 (Weak Reject) | 3 (Weak Reject) | **3.33** | Continuous parameterization heuristics | Reject |
| **Submission 4** | 5 (Accept) | 5 (Accept) | 6 (Strong Accept) | **5.33** | ZipMerge Failure Post-Mortem & SVD PEFT Alignment | **ACCEPT** |
| **Submission 5** | 3 (Weak Reject) | 3 (Weak Reject) | 3 (Weak Reject) | **3.00** | SRAM-efficient Q-PolyMerge | Reject |
| **Submission 6** | 3 (Weak Reject) | 6 (Strong Accept) | N/A | **4.50** | Curvature-aware model merging & subspace projection | Reject |
| **Submission 7** | 4 (Weak Accept) | 4 (Weak Accept) | 4 (Weak Accept) | **4.00** | Generalization-Granularity Trade-off analysis | Reject |
| **Submission 8** | 4 (Weak Accept) | 6 (Strong Accept) | 5 (Accept) | **5.00** | PAC-Bayes continuous GP model merging (GP-BayesMerge) | Reject |
| **Submission 9** | 5 (Accept) | 3 (Weak Reject) | 6 (Strong Accept) | **4.67** | Pre-merging flatness-robustness geometry optimization | Reject |
| **Submission 10**| 4 (Weak Accept) | 6 (Strong Accept) | 5 (Accept) | **5.00** | ChebyMerge with Controllable Spectral Decay (CSD) | Reject |

---

## 3. Justification for the Accepted Submissions

The following three papers were selected for acceptance based on their exceptional average scores, deep conceptual originality, and outstanding practical or methodological contributions:

### 1. Submission 1 (Consensus Average: 5.33)
*   **Key Contribution:** This submission conducts an exceptionally rigorous, independent robustness audit of quantization-aware model merging. It exposes a major, previously unstudied vulnerability: **Quantization-Operator Overfitting**. The authors demonstrate that continuous layer coefficients optimized under simulated rounding (such as with a Straight-Through Estimator, or STE) overfit intensely to the simulated operator and collapse catastrophically when evaluated on slightly different target hardware operators (the "Cross-Schema Generalization Gap").
*   **Strengths & Reviewer Consensus:** Reviewers 1 and 3 are highly enthusiastic, describing the paper as "technically flawless", "paradigm-shifting", and "an outstanding conceptual leap." The paper stands out because of its deep methodological skepticism, refuting the foundational premise that direct quantization-aware search is optimal, and proving instead that full-precision search followed by post-hoc quantization consistently generalizes better. The authors constructively propose solutions like pre-discretization landscape smoothing and formalize a Hybrid Optimization Pipeline (Algorithm 1) to bridge the operator gap. It represents a major contribution that will establish more honest, hardware-realistic evaluation standards in the field.

### 2. Submission 4 (Consensus Average: 5.33)
*   **Key Contribution:** This paper presents a rare and outstanding example of scientific honesty: a thorough "failure post-mortem" of the authors' own co-optimization framework (ZipMerge). The authors systematically deconstruct ZipMerge's concurrent pruning and coefficient search, revealing that a simpler, decoupled baseline (Prune-then-Merge) is structurally superior under severe domain shift. Furthermore, they resolve coordinate mismatches in parameter space by proposing **Orthogonal Procrustes SVD Alignment** for PEFT adapters.
*   **Strengths & Reviewer Consensus:** The reviewers praised the paper's transparency, exceptional writing, and high practical utility. By rotating separately learned adapter coordinate spaces *post-hoc* before averaging, they demonstrate a massive **+16.45%** absolute improvement in joint task accuracy with zero calibration data and sub-millisecond CPU overhead. The paper also includes extensive physical hardware execution profiling on ARM Cortex CPUs, showing a **1.89× physical speedup** on mobile processors. This combination of scientific integrity, elegant mathematics, and tangible hardware profiling made it a unanimous favorite among reviewers.

### 3. Submission 2 (Consensus Average: 5.00)
*   **Key Contribution:** This paper introduces **Offline Few-Shot Validation Tuning (OFS-Tune)** as a robust, static, and zero-overhead alternative to backpropagation-heavy online Test-Time Adaptation (TTA) in model merging. It optimizes merging coefficients offline using as few as 5 to 10 validation samples per task, and conceptualizes the **"Overfitting-Optimizer Paradox"**—proving that low-dimensional parameterizations (e.g., polynomial curves over depth) act as crucial analytical filters that prevent memorization of validation noise.
*   **Strengths & Reviewer Consensus:** While there is a three-way quantitative tie at 5.00 between Submissions 2, 8, and 10, **Submission 2 was selected for the final spot due to its outstanding practical utility and unified consensus (5, 5, 5)**. In production environments, backpropagation-heavy online TTA is highly risky due to latency, computational overhead, and the threat of representational collapse under noisy, non-i.i.d. deployment streams. OFS-Tune delivers a static merged model requiring **zero runtime modification and zero test-time compute**, whilst achieving equal or superior performance to online TTA. It exhibits complete immunity to target distribution shifts, temporal bursts, and extreme label noise. Backed by both a highly calibrated continuous simulation (30 random seeds) and a physical PyTorch deep CNN validation on real image datasets, this work represents an essential, highly robust methodological course correction for the community.

---

## 4. Deconstruction of Non-Accepted High-Scoring Submissions

To ensure absolute transparency and fairness, we carefully deconstructed the other high-performing submissions that reached a 5.00 average but were not selected, detailing why they were ultimately passed over:

### 1. Submission 8 (Consensus Average: 5.00)
*   **Methodology:** Framed test-time model merging as a Bayesian inference problem (GP-BayesMerge), deriving a continuous Gaussian Process (GP) prior over depth from first-principles PAC-Bayes theory, coupled with an online activation CKA Kronecker prior to capture task correlations.
*   **Reasons for Non-Acceptance:** Although theoretically elegant, Reviewer 1 pointed out critical empirical limitations. Specifically, the authors **omitted the strongest direct baselines (RegCalMerge and PolyMerge)** from their main physical weight-merging experiments. Furthermore, the simulation was shown to have an inherent design bias that exaggerated baseline failure. Finally, running $K$ distinct expert networks at test-time to estimate the online CKA task-correlation matrix introduces an unquantified and substantial runtime/latency overhead, making its deployment on resource-constrained edge devices questionable. Given these empirical gaps and the lack of validation for their unsupervised CCV tuning algorithm, Submission 2 was favored for its practical, zero-overhead offline alternative.

### 2. Submission 10 (Consensus Average: 5.00)
*   **Methodology:** Modeled continuous model merging as a spectral approximation problem under an orthogonal Chebyshev basis (ChebyMerge) with Controllable Spectral Decay (CSD) to resolve monomial ill-conditioning and prevent overfitting.
*   **Reasons for Non-Acceptance:** Reviewer 1 highlighted that the core algorithmic change (transitioning from monomials to Chebyshev polynomials) is a standard basis change from classical numerical analysis building directly on PolyMerge (2024), making the algorithmic novelty somewhat incremental. More importantly, physical validation on CLIP ViT-B/32 revealed a significant practical limitation: **all adaptive merging methods (including ChebyMerge-CSD) underperformed the static, non-adaptive Task Arithmetic baseline under short test streams**. This suggests that unsupervised online test-time adaptation remains a highly fragile open challenge. By contrast, Submission 2's OFS-Tune demonstrated immediate practical utility by consistently outperforming static baselines with absolute stability.

### 3. Submission 9 (Consensus Average: 4.67)
*   **Methodology:** Explored pre-merging weight-space geometry (flatness via SAM/SWA) to improve low-precision edge deployment under post-training quantization.
*   **Reasons for Non-Acceptance:** While the paper has high empirical merit, Reviewer 2 identified a **critical theoretical flaw** in the core mathematical foundations (Section 3.1). The proof contains a major logical gap by treating training-time task-specific supervised Hessians at expert points as equivalent to test-time joint unsupervised entropy Hessians at merged quantized points. Additionally, relying on local Taylor expansions for non-infinitesimal 4-bit rounding noise is technically loose. Because these theoretical assertions are central to the paper's methodology and framing, the submission was deemed not ready in its current form.

---

## 5. Conclusions

The meta-review process successfully identified three outstanding works—**Submissions 1, 2, and 4**—that represent the pinnacle of scientific quality, rigorous validation, and immediate practical utility in the model-merging literature. 

- **Submission 1** changes how the community evaluates low-bit ensembling by exposing the Cross-Schema Generalization Gap.
- **Submission 4** sets an exemplary standard for empirical honesty and delivers a mathematically elegant, lightweight SVD alignment mechanism.
- **Submission 2** offers a highly robust, zero-overhead offline validation-tuning alternative that bridges the gap between theoretical optimization and secure, real-world edge deployment.

These selections are supported by strong reviewer consensus, robust experimental foundations, and high potential for community impact.
