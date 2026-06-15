# Meta-Review Summary Report

This meta-review summary outlines the selection process, scoring, and qualitative evaluations for ten submissions on dynamic model merging, routing, and test-time modular ensembling. Out of the ten submissions, exactly **three** papers have been selected for acceptance.

---

## 1. Meta-Review Methodology and Overview

Our meta-review process evaluated submissions across both quantitative metrics (numerical reviewer scores) and qualitative criteria (empirical robustness, theoretical depth, and systems feasibility). A primary goal of test-time dynamic model merging is to deliver real-world performance benefits (both accuracy gains and serving efficiencies) while maintaining clean, scalable, and mathematically sound architectures. 

To conduct a rigorous meta-review, we mapped recommendations to standard numeric scores:
* **6: Strong Accept**
* **5: Accept**
* **4: Weak Accept**
* **3: Weak Reject**
* **2: Reject**
* **1: Strong Reject**

Beyond numerical averages, we carefully evaluated:
1. **The "Reality Gap":** Whether the proposed framework is validated only on idealized, toy-scale synthetic simulators (sandboxes) or demonstrates physical wall-clock speedups and accuracy gains on actual deep neural networks (e.g., Vision Transformers, GPT-2, LLaMA) on physical GPU/CPU hardware.
2. **Computational and Serving Feasibility:** Many parameter-space dynamic merging frameworks introduce complex micro-batch scheduling and weight-materialization overheads that are too slow for real-time inference, rendering them dead-on-arrival for practitioners.
3. **Scientific Integrity and Completeness:** We verified that mathematical formulations are complete, reproducible (with no missing SVD or orthogonalization equations), and that empirical results are statistically rigorous (reporting standard deviations/seeds rather than suspicious, noise-free, perfect scores).

---

## 2. Recommendation Summary Table

Below is the summary table of all ten submissions, listing individual reviewer recommendations and computed average scores:

| Submission | Reviewer 1 Recommendation | Reviewer 2 Recommendation | Reviewer 3 Recommendation | Average Score | Decision |
|:---|:---|:---|:---|:---:|:---|
| **Submission 1** | 2: Reject | 5: Accept | 3: Weak Reject | 3.33 | **Reject** |
| **Submission 2** | 5: Accept | 3: Weak Reject | 4: Weak Accept | 4.00 | **Reject** (Reality Gap / MBH Bottleneck) |
| **Submission 3** | 6: Strong Accept | 3: Weak Reject | 3: Weak Reject | 4.00 | **Reject** (Over-engineered / Outperformed by 5-NN) |
| **Submission 4** | 3: Weak Reject | 5: Accept | 6: Strong Accept | **4.67** | **ACCEPT** (Top Candidate) |
| **Submission 5** | 3: Weak Reject | 3: Weak Reject | 6: Strong Accept | 4.00 | **Reject** (Missing Math / Suspicious Perfect Data) |
| **Submission 6** | 3: Weak Reject | 5: Accept | 3: Weak Reject | 3.67 | **Reject** |
| **Submission 7** | 5: Accept | 3: Weak Reject | 3: Weak Reject | 3.67 | **Reject** |
| **Submission 8** | 3: Weak Reject | 3: Weak Reject | 3: Weak Reject | 3.00 | **Reject** |
| **Submission 9** | 5: Accept | 4: Weak Accept | 4: Weak Accept | **4.33** | **ACCEPT** (Top Candidate) |
| **Submission 10**| 4: Weak Accept | 5: Accept | 3: Weak Reject | **4.00** | **ACCEPT** (Top 4.00 Candidate - Pi 4 Verification) |

---

## 3. Detailed Decisions for Accepted Submissions

### Submission 4: Parameter-Free Task-Space Projection (PFSR)
* **Average Score:** 4.67 (Scores: 3, 5, 6)
* **Decision:** **ACCEPT**
* **Summary:** This paper applies Occam's razor to strip away complex, over-parameterized routing architectures, introducing a training-free and data-free closed-form linear projection (PFSR) that extracts task-space centroids using SVD on the classification heads of frozen specialists. It further investigates Löwdin-Orthogonalized Task-Space Projection (OTSP) as a symmetric basis alternative.
* **Justification for Acceptance:** Submission 4 is the highest-rated paper in the cohort and represents a mathematically outstanding and intellectually honest contribution. It stands out for its deep theoretical characterizations of ensembling pathologies, such as proving the equivalence of OTSP and PFSR under symmetric conditions, and explaining the "Noise Amplification/Spillover Penalty" under asymmetric task manifolds. The paper was praised by Reviewer 3 as "setting a gold standard for theoretical depth and self-critical analysis in model ensembling." While Reviewer 1 requested more extensive downstream evaluations, the mathematical rigor and foundational value to the Mixture of Experts (MoE) literature make this a clear accept.

### Submission 9: SABLE (Sample-wise Activation Blending of Low-Rank Experts)
* **Average Score:** 4.33 (Scores: 5, 4, 4)
* **Decision:** **ACCEPT**
* **Summary:** SABLE addresses the systems bottleneck of "heterogeneity collapse" in test-time model merging (where mixed-task streaming batches force weight-averaging, degrading specialized expert accuracy). By shifting ensembling from parameter space to activation space via the distributive property of matrix multiplication, SABLE achieves per-sample activation blending, rendering it completely immune (0.00% collapse) to batch heterogeneity without stateful scheduler bloat.
* **Justification for Acceptance:** SABLE is an exceptionally solid, highly complete Systems-ML contribution. It provides exemplary physical verification on high-end hardware, demonstrating a **6.8$\times$ wall-clock latency speedup** and **36.4% memory savings** on an actual NVIDIA A100 GPU serving engine. Reviewers strongly commended its stateless network-level solution that bypasses stateful scheduling pipelines (like MBH) and its rigorous documentation of its own boundaries (e.g., the Representational Blurring Paradox). SABLE represents a solid, highly practical contribution that the multi-tenant serving community is highly likely to build on.

### Submission 10: SPS-ZCA (Single-Pass Sample-Wise Routing with Zero-Shot Centroid Alignment)
* **Average Score:** 4.00 (Scores: 4, 5, 3)
* **Decision:** **ACCEPT**
* **Summary:** This paper introduces SPS-ZCA, a training-free dynamic model-merging framework designed for edge CPUs. It executes base models and expert adapters in a single parallel pass, blending activations sample-wise on-the-fly, while routing inputs in the early-stage representation space (Layer 3 CLS tokens) using pre-computed unsupervised centroids.
* **Justification for Acceptance:** Among the multiple papers tied with an average score of 4.00, SPS-ZCA stands out due to its exceptional engineering polish and its complete success in bridging the "reality gap." Rather than limiting its evaluation to simulated sandboxes or high-end GPUs, the authors compiled their framework as ONNX Runtime CustomOps and benchmarked it on physical Raspberry Pi 4 CPU hardware—demonstrating a verified **3.91$\times$ physical speedup**. It also includes robust GMM Coordinate Density Estimators to reject out-of-distribution queries before activation blending. While Reviewer 3 sought more theoretical proofs of sequential non-linear blending stability, the outstanding physical hardware verification on real Vision Transformers and GPT-2 models makes this an incredibly strong, deployable Systems-ML contribution.

---

## 4. Analysis of High-Scoring Rejected Submissions (Average 4.00)

Three other papers achieved an average recommendation of 4.00 but were ultimately rejected in favor of Submission 10 due to critical practical, empirical, or methodological flaws identified during the review process:

### Submission 2: Information-Geometric Subspace Routing (FIOSR)
* **Average Score:** 4.00 (Scores: 5, 3, 4)
* **Decision:** **REJECT**
* **Major Flaws Identified:**
  1. **The "Reality Gap" on Real Backbones:** While FIOSR achieves a high **+8.56%** accuracy improvement in an idealized synthetic sandbox, this improvement collapses to a highly modest and practically insignificant **+1.33%** when evaluated on physical ResNet-18 features. Real-world activations exhibit dense, non-axis-aligned coordinate correlations that diagonal Fisher coordinate warping fails to capture.
  2. **MBH Scheduling Redundancy:** To handle batch heterogeneity, FIOSR relies on Micro-Batch Homogenization (MBH), which partitions streaming batches and executes up to $K$ sequential forward passes. This is computationally equivalent to simply running the original unmerged specialized experts sequentially, completely undermining the primary motivation of model merging.
  3. **High Inference Latency:** Scaling the information-geometric warping to capture off-diagonal correlations in high-dimensional representations ($d \ge 1024$) requires eigenvalue decompositions with a prohibitive $O(d^3)$ computational complexity at inference time.

### Submission 3: Bayesian Dynamic Routing (GP-DR)
* **Average Score:** 4.00 (Scores: 6, 3, 3)
* **Decision:** **REJECT**
* **Major Flaws Identified:**
  1. **Over-engineering and Lack of Edge:** GP-DR uses Gaussian Process Regression (GPR) to estimate epistemic routing uncertainty. However, both Reviewer 2 and Reviewer 3 noted that GP-DR is unnecessarily over-engineered and computationally heavy.
  2. **Outperformed by Simple Baselines:** GP-DR underperforms simpler, pre-existing static/linear methods (such as PFSR) under representational coupling.
  3. **Outperformed by Classical Non-Parametric Baselines:** More critically, the empirical evaluations demonstrate that a simple, classic **5-NN Euclidean distance** metric consistently outperforms the posterior variance of the GPR under representational overlap. Introducing high-dimensional Bayesian machinery is therefore scientifically unjustified when simple distance metrics provide superior OOD fallback triggers.

### Submission 5: Sample-wise Activation Blending (PFAB)
* **Average Score:** 4.00 (Scores: 3, 3, 6)
* **Decision:** **REJECT**
* **Major Flaws Identified:**
  1. **Missing Mathematical Formulations:** A core part of the methodology—specifically, the SVD-based parameter orthogonalization—is completely missing its equations and mathematical derivations in the text. This is a severe presentation and completeness gap that renders the method non-reproducible.
  2. **Suspiciously Idealized, Noise-Free Empirical Results:** In both synthetic sandbox and DomainNet (ViT-B/16) pilots, the paper reports that the proposed PFAB matches the Expert Oracle ceiling *perfectly* down to two-decimal precision (e.g., exactly 78.80% on DomainNet and exactly 81.50% in the sandbox) with zero routing classification errors. As highlighted by Reviewer 2, achieving 100% correct zero-parameter cosine similarity routing of penultimate representations across noisy, real-world visual domains is mathematically and empirically implausible, strongly suggesting that the pilots were highly idealized or artificially isolated.
  3. **Lack of Statistical Rigor:** The paper reports absolutely zero standard deviations, confidence intervals, or seed details for any of its accuracy or latency benchmarks.
  4. **Performance Collapse on Real-World Data:** The single-pass ELC pathway suffers a catastrophic **36.30% absolute accuracy collapse** (dropping from 78.80% to 42.50%) on real DomainNet, indicating that early-layer pre-computed centroids are highly fragile under real style/covariate shifts.

---

## 5. Conclusion

The selection of **Submission 4**, **Submission 9**, and **Submission 10** represents the optimal subset of papers for this conference. By pairing theoretical brilliance (Submission 4) and high-end GPU streaming latency breakthroughs (Submission 9) with exceptionally verified edge-hardware CPU performance (Submission 10), this selection guarantees a balanced, high-impact, and scientifically rigorous representation of modern dynamic model merging research. 

Other candidates, despite sharing identical average scores of 4.00, were rejected due to severe practical limitations (Submission 2), over-engineered/underperforming frameworks (Submission 3), or critical presentation gaps and suspiciously idealized empirical reporting (Submission 5).
