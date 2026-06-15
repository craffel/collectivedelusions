# Meta-Review Decisions and Process Report

This meta-review report summarizes the evaluation process, reviewing criteria, and final selection decisions for the 10 submissions evaluated for the conference. Out of the 10 submissions, exactly 3 have been selected for acceptance.

---

## 1. Executive Summary & Selection Process

The program committee conducted a thorough, multi-phase peer-review process. Ten papers were submitted, each receiving three reviews (with the exception of Submission 2, which received two), for a total of 29 individual peer reviews. 

The primary objectives of the meta-review process were:
1. **Scientific Integrity & Soundness:** Ensuring all mathematical derivations are correct and empirical claims are supported by rigorous, multi-seed, statistically significant evaluations.
2. **Impact & Practical Significance:** Prioritizing papers that solve genuine, real-world bottlenecks in parameter-space model merging and edge-ML deployment rather than incremental variants.
3. **Conceptual Originality:** Championing works that challenge established dogma, apply Occam's razor, and avoid unnecessary complexity or misleading metaphors.
4. **Scholarly Rigor:** Demanding transparent and honest reporting, high-quality literature positioning, and clear attribution of prior ideas.

Following these principles, the final recommendations are:
* **Accepted Submissions (3):** Submission 5, Submission 4, Submission 7
* **Rejected Submissions (7):** Submission 1, Submission 2, Submission 3, Submission 6, Submission 8, Submission 9, Submission 10

---

## 2. Comprehensive Analysis of Accepted Submissions

### 1. Submission 5 (Consensus Accept)
* **Title:** *Demystifying Test-Time Dynamic Model Merging: Vectorization Collapse, Batch-Average Confounders, and the Power of Proper Priors*
* **Reviewer Ratings:** 6: Strong Accept | 5: Accept | 6: Strong Accept (Average: **5.67**)
* **Synthesis of Contributions:**
  The paper performs a vital deconstruction of the test-time dynamic model-merging literature. Instead of proposing increasingly complex, wave-inspired, or multi-parameter routing networks, the author(s) expose critical failure modes in standard routing architectures, such as **Vectorization Collapse** and the **Batch-Average Smoothing Confounder**. They prove that a mathematically simpler, classical prior-based baseline can outperform complex state-of-the-art dynamic routers.
* **Justification for Acceptance:**
  This paper represents a masterclass in scientific deconstruction, advocacy for simplicity, and empirical rigor. Backed by parallel 10-seed sweeps, exhaustive ablation analyses, and systems-level latency profiling on real classification experts, the paper's claims are absolutely bulletproof. All reviewers praised its high originality, outstanding intellectual honesty, and transparent framing, making it an easy consensus accept with the highest priority.

### 2. Submission 4 (Consensus Accept)
* **Title:** *Task-Space Anchor Regularization: A Rigorous Empirical Solution to Low-Data Overfitting in Dynamic Model Merging* (Running Title: *TSAR*)
* **Reviewer Ratings:** 6: Strong Accept | 5: Accept | 4: Weak Accept (Average: **5.00**)
* **Synthesis of Contributions:**
  The paper addresses a severe optimization bottleneck in dynamic model merging where lightweight routers overfit to local sampling noise under extreme calibration split scarcity (e.g., $|D_{cal}| \le 64$ samples), leading to representation-space collapse and a failure to generalize to OOD tasks. To resolve this, the authors propose **Task-Space Anchor Regularization (TSAR)**, which computes stable, task-specific representation centroids (anchors) and incorporates a quadratic distance penalty on a low-dimensional unit sphere. It also integrates **Projecting Conflicting Gradients (PCGrad)** to resolve gradient cross-talk.
* **Justification for Acceptance:**
  TSAR provides an elegant, geometrically grounded, and highly practical solution requiring only **20 trainable parameters**. The empirical evaluation is outstanding—sweeping over 5 random seeds across diverse baselines, resolving heterogeneity collapse under mixed-task streams via scaled Sigmoids, and validating on physical Vision Transformers (+13.90% to +23.60% accuracy gains). A single-layer global router variant ($L=1$) successfully bypasses the PCGrad complexity bottleneck for 20-task scalability. This represents a solid, highly complete, and deployment-ready contribution.

### 3. Submission 7 (Selected Accept)
* **Title:** *Micro-Batch Homogenization & Parameter-Free Subspace Routing* (Running Title: *PFSR + MBH*)
* **Reviewer Ratings:** 6: Strong Accept | 6: Strong Accept | 2: Reject (Average: **4.67**)
* **Synthesis of Contributions:**
  This paper introduces a zero-shot, completely non-parametric dynamic model-merging framework under the PEFT/LoRA paradigm. It proposes **Parameter-Free Subspace Routing (PFSR)**, which projects high-dimensional penultimate-layer features onto a low-dimensional task coordinate subspace using the cosine similarity against frozen expert classification weights. Gating coefficients are computed via temperature-scaled Softmax, completely eliminating trainable routing parameters. It also proposes **Micro-Batch Homogenization (MBH)** to schedule and homogenize mixed-task serving streams on-the-fly, resolving "heterogeneity collapse" at the data-serving level rather than the weight-parameter level.
* **Justification for Acceptance:**
  Although highly controversial due to a single "2: Reject" review, the paper's theoretical, mathematical, and systems contributions are of the absolute highest caliber, prompting two "6: Strong Accept" ratings. Reviewer 2 critiqued the simulation-only scale of the real-world benchmarks (DomainNet and LLaMA-7B), but the paper explicitly and honestly discloses this constraint, validating the core mechanics on high-fidelity manifolds. The theoretical deconstruction is outstanding—providing a flawless first-order Taylor expansion and Jacobian analysis to mathematically prove **Layer-Averaging Collapse** in multi-layer routers. The scholarly positioning and statistical rigor (Class-Size Scaling Calibration modeling random Gaussian similarities) are exemplary. By applying Occam's razor to strip out routing parameters, it enables instantaneous expert registration/retirement with $O(1)$ SGMV GPU execution. The peak reviews are incredibly enthusiastic, and the theoretical merits heavily outweigh the empirical sandbox constraints.

---

## 3. Analysis of Non-Accepted Contenders (Runner-Ups)

### 1. Submission 3
* **Title:** *Deconstructing Capacity and Generalization in Dynamic Model Merging: The Block-wise Weight-Sharing Router*
* **Reviewer Ratings:** 6: Strong Accept | 5: Accept | 3: Weak Reject (Average: **4.67**)
* **Why it was not accepted:**
  While Submission 3 achieved the same average score as Submission 7, a deep comparison of the review content revealed fatal flaws in Submission 3's core theoretical and empirical claims. Reviewer 2 (Weak Reject) pointed out that the core mathematical derivation of Expected Ruggedness (Equation 10) contains a significant algebraic omission that assumes constant expected routing decisions across sequential layers—an assumption that is physically and theoretically incorrect. Furthermore, in physical sequential ensembling (Table 4), the unshared physical router ($M=1$) actually **outperformed** the block-shared router ($M=3$) by 2.78% absolute accuracy under stable task streams. This result directly contradicts the paper's central thesis that block-sharing is universally superior. Finally, because the physical experiments were restricted to a toy 3-layer MLP expert backbone, intermediate block-sharing sizes (the optimal "sweet spot") were never physically validated. These theoretical and empirical discrepancies severely undermine the scientific validity of the submission, leading to its rejection in favor of Submission 7.

### 2. Submission 8
* **Title:** *Hybrid-Router: Partitioning Model Merging Layer-wise for Resource-Efficient Deployment*
* **Reviewer Ratings:** 5: Accept | 4: Weak Accept | 4: Weak Accept (Average: **4.33**)
* **Why it was not accepted:**
  Submission 8 is a highly practical, systems-aware paper that proposes a layer-wise partitioning scheme (Hybrid-Router) to statically merge early layers and dynamically ensemble final layers, reducing active VRAM footprint and latency by over 71%. It also proposes Dynamic Batch Filtering to handle batch heterogeneity. While technically solid and presenting strong engineering value, the paper's contribution is heavily incremental and lacks high conceptual novelty. Its quantitative ViT evaluations and the "Overfitting-Optimizer Paradox" are evaluated strictly within a synthetic sandbox, while physical validation is restricted to a toy 25k-parameter SimpleCNN where the paradox was not replicated. Lacking the high conceptual novelty, rigorous mathematical proofs, and glowing peak enthusiasm of the top 3 accepted papers, it was not selected.

---

## 4. Summary of Rejected Submissions

The remaining submissions were rejected due to severe technical flaws, logical contradictions, data inconsistencies, or scholarly integrity issues:

* **Submission 1 (EHPB):** (6, 2, 3 | Average: 3.67)  
  Proposes a highly complex holographic parameter binding scheme. It was rejected due to a complete lack of statistical error bars/significance tests, confinement to a synthetic sandbox, and poor empirical performance where the core method was heavily dominated by simple static averaging (+26.9% absolute gap).
* **Submission 2:** (3, 3 | Average: 3.00)  
  A clear consensus weak reject. The proposed dynamic routing baseline was weak, the experimental evaluation was limited, and the overall contribution was deemed highly incremental.
* **Submission 6:** (3, 2, 2 | Average: 2.33)  
  Suffered from severe data reporting discrepancies and a breach of scientific integrity. Almost all numeric claims in the main narrative (Abstract/Intro) and Table 2 (Ablation) were completely different from the actual results presented in Table 1 and recorded in `results.json`. Additionally, the proposed complex PAC-Bayesian regularizer underperformed a simple unconstrained baseline across all settings.
* **Submission 9 (CAM-Router):** (2, 2, 2 | Average: 2.00)  
  A consensus strong reject due to critical technical, statistical, and scholarly integrity flaws. The bibliography contained multiple fabricated reference listings with fake/hallucinated author names. The Abstract systematically inflated accuracy claims. Methodologically, the DHG mechanism introduced stateful temporal dependencies that break the standard i.i.d. assumptions for production serving. Widespread numeric contradictions existed between Tables 4 and 6.
* **Submission 10 (TCPR):** (2, 3, 3 | Average: 2.67)  
  Rejected due to a severe self-contradiction and narrative mismatch. The front matter framed the proposed Task-Correlation Prior Regularization (TCPR) as a highly successful regularizer. However, the experimental tables and back matter demonstrated that TCPR was an empirical failure that degraded performance relative to unregularized Sigmoidal routing. The paper was self-refuting and required a complete re-framing before publication.

---

## 5. Final Acceptance Recommendations

The program committee highly recommends the acceptance of the following three outstanding papers, which advance the state-of-the-art in parameter-space model merging with exceptional empirical rigor, elegant mathematical formulations, and systems-level co-design:

1. **Submission 5** — *Demystifying Test-Time Dynamic Model Merging: Vectorization Collapse, Batch-Average Confounders, and the Power of Proper Priors*
2. **Submission 4** — *Task-Space Anchor Regularization: A Rigorous Empirical Solution to Low-Data Overfitting in Dynamic Model Merging*
3. **Submission 7** — *Micro-Batch Homogenization & Parameter-Free Subspace Routing*
