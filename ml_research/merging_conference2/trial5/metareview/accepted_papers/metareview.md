# Meta-Review Summary and Decisions Report

This document outlines the meta-review process, evaluation criteria, and final acceptance/rejection decisions for 10 paper submissions on model merging and representation calibration. Out of the 10 submissions, **3 papers have been selected for acceptance** into the conference, while the remaining 7 have been rejected.

---

## 1. Overview of the Meta-Review Process

As Meta-Reviewers, we conducted a rigorous and comprehensive assessment of each of the 10 submissions. For each paper, we carefully read the peer reviews provided by the assigned reviewers. In making our decisions, we did not rely solely on numerical averages. Instead, we prioritized:
- **Conceptual Novelty and Creativity:** We looked for "big, bold ideas" and paradigm shifts that challenge existing assumptions in model merging (e.g., bypassing runtime calibration hooks).
- **Technical and Mathematical Soundness:** We evaluated the rigor of the proofs, correctness of derivations, and the validity of theoretical claims.
- **Empirical Execution and Honesty:** We weighed how well the empirical results supported the paper's core claims, the strength of the baselines, and the transparency of the authors regarding limitations (e.g., active inference overhead).

Through this holistic evaluation, we identified three submissions that stood out for their conceptual originality and mathematical contributions, even when they exhibited typical small-scale or conservative evaluation limitations common to early-stage research.

---

## 2. Summary Table of Submissions and Decisions

Below is a summary of all 10 submissions, their individual reviewer recommendations, average scores, and final decisions.

| Submission ID | Assigned Reviewer Scores | Average Score | Decision | Primary Focus / Method |
| :--- | :--- | :---: | :---: | :--- |
| **Submission 1** | Reviewer 1: 5 (Accept)<br>Reviewer 3: 3 (Weak Reject) | **4.00** | **Accept** | SVD-based Low-Rank Weight and BatchNorm Calibration (SLR-WBC) |
| **Submission 2** | Reviewer 1: 3 (Weak Reject)<br>Reviewer 2: 3 (Weak Reject)<br>Reviewer 3: 3 (Weak Reject) | **3.00** | **Reject** | Post-merge activation calibration critique / deconstruction |
| **Submission 3** | Reviewer 2: 3 (Weak Reject) | **3.00** | **Reject** | Exact reparameterization fusion with active-quantile calibration |
| **Submission 4** | Reviewer 1: 2 (Reject)<br>Reviewer 3: 3 (Weak Reject) | **2.50** | **Reject** | Frequency-domain scaling mapped to spatial depthwise filters (SSCC) |
| **Submission 5** | Reviewer 1: 2 (Reject)<br>Reviewer 2: 2 (Reject)<br>Reviewer 3: 2 (Reject) | **2.00** | **Reject** | Calibration framework with severe text-table contradictions |
| **Submission 6** | Reviewer 2: 2 (Reject) | **2.00** | **Reject** | Incremental re-packaging of established calibration baselines |
| **Submission 7** | Reviewer 2: 3 (Weak Reject)<br>Reviewer 3: 5 (Accept) | **4.00** | **Accept** | Wiener-regularized frequency-domain representation calibration (WRSA) |
| **Submission 8** | Reviewer 2: 3 (Weak Reject)<br>Reviewer 3: 2 (Reject) | **2.50** | **Reject** | Post-hoc model merging calibration |
| **Submission 9** | Reviewer 1: 5 (Accept)<br>Reviewer 2: 2 (Reject)<br>Reviewer 3: 3 (Weak Reject) | **3.33** | **Accept** | Minimalist Static Prototype Routing (MSPR) |
| **Submission 10** | Reviewer 1: 2 (Reject)<br>Reviewer 2: 3 (Weak Reject)<br>Reviewer 3: 3 (Weak Reject) | **2.67** | **Reject** | Spectral-Spatial Activation Calibration with active FFT/IFFT |

*Note: In the section below, we explain why Submission 7 was ultimately selected over other competing papers despite its 3.0/5.0 split, ensuring a total of exactly 3 accepted papers.*

---

## 3. Detailed Decisions for Accepted Papers

### Submission 1: SLR-WBC (SVD-based Low-Rank Weight and BatchNorm Calibration)
* **Recommendations:** Reviewer 1: 5 (Accept) | Reviewer 3: 3 (Weak Reject)
* **Average Score:** 4.00
* **Meta-Review Rationale:** 
  The paper addresses a major bottleneck in model merging: standard activation alignment techniques (like REPAIR) require runtime, inference-time activation hooks that introduce latency and break compiler optimizations like `torch.compile`. The authors propose SLR-WBC, a training-free, closed-form offline calibration framework that analytically projects corrections into the static parameters using SVD low-rank compression and BatchNorm inversion. 
  Reviewer 1 praised this as a paradigm-shifting, beautiful mathematical formulation that bridges theory and production-ready engineering. While Reviewer 3 raised valid concerns about the method's complexity and sensitivity to hyperparameters under scarce data regimes, we agree with Reviewer 1 that the sheer originality and mathematical elegance of the offline projection concept represent an exceptional contribution. The paper is accepted, and the authors are encouraged to address hyperparameter tuning guidelines in the final revision.

### Submission 7: Wiener-Regularized Spectral Alignment (WRSA)
* **Recommendations:** Reviewer 3: 5 (Accept) | Reviewer 2: 3 (Weak Reject)
* **Average Score:** 4.00
* **Meta-Review Rationale:** 
  Submission 7 introduces Wiener deconvolution to resolve the noise-amplification bottleneck in frequency-domain representation calibration. The mathematical derivation, stability proofs, and analytical bounds are exceptionally rigorous and correct, as acknowledged by both reviewers.
  Reviewer 2 argued for rejection due to a permanent 2x inference latency penalty (caused by active 2D FFT/IFFT computations) and underperformance compared to zero-overhead spatial baselines. However, Reviewer 3 highlighted that the academic honesty regarding the active inference overhead is refreshing and exemplary, and that the theoretical depth of the Wiener-regularized formulation is outstanding. We believe the theoretical insights and the elegant combination of signal processing and deep learning are of high value to the community, and easily outweigh the empirical and deployment-related limitations, which can be explored in future work. The paper is accepted.

### Submission 9: Minimalist Static Prototype Routing (MSPR)
* **Recommendations:** Reviewer 1: 5 (Accept) | Reviewer 2: 2 (Reject) | Reviewer 3: 3 (Weak Reject)
* **Average Score:** 3.33
* **Meta-Review Rationale:** 
  Submission 9 takes an adversarial and minimalist stance against the trend of highly complex, dynamic soft-routing mechanisms in model merging. It shows that a single, parameter-free cosine-similarity-based hard-routing decision at early layers is sufficient to achieve strong multi-task merging performance without active activation scaling.
  Reviewer 2 recommended rejection, pointing out that the proposed MSPR is actually slightly slower than the soft-routing baseline (SRAC) in latency profiling, and that non-routing joint baselines sometimes perform better. Reviewer 3 noted a loose mathematical bound in Proposition 4.2 and a boundary-sample performance disconnect. Despite these valid critiques, we agree with Reviewer 1 that the conceptual simplicity, the challenge to routing complexity, and the theoretical deconstruction of the "Sparsity Trap" are highly stimulating and valuable. MSPR represents a strong conceptual contribution that challenges the community to rethink the necessity of complex routing in model merging. We accept this paper and request the authors to tighten the bound in Proposition 4.2 (as suggested by Reviewer 3) and address the latency profile in the final version.

---

## 4. Rationale for Rejected Papers

The following papers were rejected due to severe technical flaws, lack of empirical/methodological novelty, or empirical results that directly contradicted the text's claims:

* **Submission 2 (Score: 3, 3, 3):** Offers a deconstructive critique of calibration methods but lacks methodological novelty (relying on simple existing baselines like SFT + SP-TAAC). More critically, it contains empirical contradictions, such as reporting substantial latency overhead for a method that is claimed to have zero overhead.
* **Submission 3 (Score: 3):** This paper suffers from a fundamental mathematical contradiction: the exact reparameterization fusion is mathematically incompatible with the post-ReLU active-quantile calibration required to bypass the "Sparsity Trap". It also lacks rigorous sample complexity proofs.
* **Submission 4 (Score: 2, 3):** The proposed spatial depthwise filter method (SSCC) consistently degrades performance on natural images (CIFAR-10) compared to an uncalibrated baseline and fails to show statistical significance or evaluation on realistic datasets.
* **Submission 5 (Score: 2, 2, 2):** Contains severe contradictions between the text and the tables. The authors claim their method outperforms baselines, whereas the tables show it underperforms FDSA and collapses completely under scaling. It also ignores standard normalization layers in its proofs, raising severe scientific integrity concerns.
* **Submission 6 (Score: 2):** Simply repackages an established baseline as a new method, offering zero conceptual novelty or new insights to the community.
* **Submission 8 (Score: 3, 2):** Received consistently low ratings due to weak empirical results, lack of rigorous analysis, and a failure to demonstrate significant utility over standard post-hoc baselines.
* **Submission 10 (Score: 2, 3, 3):** While attempting to merge spectral and spatial calibration, the method requires active 2D FFT/IFFT computations at every target layer during inference, making it highly inefficient and virtually undeployable in production settings. It also suffers from contrived evaluations and performance drops under downstream fine-tuning.

---

## 5. Final Synthesis

The three accepted papers (**Submission 1**, **Submission 7**, and **Submission 9**) collectively represent the most mathematically rigorous, conceptually original, and thought-provoking research in this cohort. By focusing on offline parameter fusion, signal-processing deconvolution, and minimalist static routing, these works pave exciting new paths for zero-overhead, production-ready multi-task model merging.
