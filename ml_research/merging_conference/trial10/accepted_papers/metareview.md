# Meta-Review Summary and Decisions

This document summarizes the meta-review process and final decisions for the 10 submissions evaluated for acceptance at the Test-Time Model Merging and Adaptation (TTMM) track of the conference. Out of the 10 submissions, exactly 3 have been selected for acceptance based on a rigorous evaluation of their reviewer recommendations, soundness of technical claims, empirical evaluation quality, clarity of presentation, and significance of contributions.

---

## 1. Meta-Review Process Overview

The meta-review process evaluated each of the 10 submissions across four core dimensions of academic quality, in accordance with the official reviewing criteria:
- **Soundness:** Are the technical claims, mathematical proofs, and empirical evaluations correct, rigorous, and methodologically appropriate?
- **Presentation:** Is the paper clearly written, well-structured, and easy to follow? Are prior works properly positioned and cited?
- **Significance:** Does the paper address an important problem and offer significant contributions that advance the field?
- **Originality:** Does the paper provide novel insights, propose creative combinations, or uncover new properties of existing methods?

Each submission was first evaluated by a peer reviewer, producing a comprehensive review (`review.md`) with a final overall recommendation. In this meta-review phase, all 10 reviews were carefully read, cross-referenced, and synthesized. Decisions were made not only by looking at the numerical recommendation but by deeply reading the technical arguments, verifying mathematical soundness, assessing empirical impact, and ensuring that accepted papers represent the highest standard of scientific rigor.

### Final Decisions Summary

| Submission ID | Paper Title | Reviewer Recommendation | Final Decision | Core Strengths / Primary Reasons for Decision |
| :---: | :--- | :---: | :---: | :--- |
| **Submission 4** | **FL-AHR: Feature-Level Sparsity-Aware Adaptive Hybrid Routing...** | **6: Strong Accept** | **ACCEPTED** | Exceptional presentation, highly novel feature-level sparsity, solves representational collapse, scales to 6-expert library. |
| **Submission 1** | **Curriculum-Gated Sharpness-Aware Test-Time Model Merging...** | **5: Accept** | **ACCEPTED** | Outstanding discovery of the "Silent Gradient Disconnection Bug" in PyTorch `load_state_dict`, stateless `functional_call` solution, rigorous proofs. |
| **Submission 7** | **CSAIR: Continuous Sigmoid-blended Angular-Euclidean Isomorphic Routing...** | **5: Accept** | **ACCEPTED** | Identifies "TTBN-induced scale mismatch" bottleneck, elegant Normalized L2 distance on unit-sphere solution, beautiful metric isomorphism proof. |
| **Submission 3** | BK-FWSAM: Direct Parameter Sensitivity Preconditioning... | 5: Accept | Rejected (High Competition) | Excellent work on Direct Parameter Sensitivity (DPS), but narrowly edged out by Submission 7's novel TTBN mismatch and metric isomorphism. |
| **Submission 6** | SMT-LDAC: Sharpness-Aware Mean-Tracker with Layer-Depth... | 5: Accept | Rejected (High Competition) | Strong theoretical proof of variance reduction under AR(1) noise, but improvements over baseline are concentrated on a narrow noise band and evaluated only on toy datasets. |
| **Submission 9** | KPSAM-DST: Kronecker-Preconditioned SAM and Denoised Self-Training... | 5: Accept | Rejected (High Competition) | Solid combination of SAM and Denoised Self-Training, but less original/groundbreaking compared to the accepted papers. |
| **Submission 5** | Curvature-Guided Meta-Adaptation for Test-Time Model Merging... | 4: Weak Accept | Rejected | Technically solid, but 2x computational overhead, and yields zero empirical improvement over a perfectly tuned static baseline. |
| **Submission 2** | SAK-AHR: Sharpness-Aware Kronecker-Preconditioned... | 3: Weak Reject | Rejected | Critical mathematical flaw in Theorem 3.1 proof (Hessian projection), incorrect/misleading use of "Kronecker preconditioning" (diagonal scaling). |
| **Submission 8** | SW-SAM-TTMM: Sparsity-Weighted Sharpness-Aware... | 3: Weak Reject | Rejected | SW-SAM-TTMM and EALR modules yield absolutely zero accuracy gains over standard, simpler SAM-TTMM baseline. |
| **Submission 10** | Integrating Denoised Gating and Curvature Preconditioning... | 3: Weak Reject | Rejected | Low originality. Direct combination of concurrent works with routing innovations disabled, ending up highly similar to existing SAM-TTMM. |

---

## 2. Accepted Submissions (Top 3)

### 1. Submission 4: FL-AHR (Overall Score: 6 - Strong Accept)
* **Title:** FL-AHR: Feature-Level Sparsity-Aware Adaptive Hybrid Routing for Noise-Robust Data-Free Test-Time Model Merging
* **Recommendation:** **6: Strong Accept** (Soundness: Excellent, Presentation: Excellent, Significance: Excellent, Originality: Excellent)
* **Key Contributions:**
  - Transition from raw pixel-level sparsity to adaptive feature-level sparsity in TTMM routing, resolving a major representational collapse bottleneck in model merging under noise.
  - Formulates a rigorous mathematical proof demonstrating how noise propagates through convolutional layers to preserve Hoyer sparsity at different layer depths.
  - Proposes **Hierarchical FL-AHR (H-FL-AHR)** to scale test-time model merging to a six-expert library using a hierarchical Top-K selection mechanism, which successfully mitigates destructive inter-expert parameter interference.
* **Meta-Review Decision Justification:** This is a mathematically and empirically flawless paper that introduces an elegant, highly novel, and scalable solution to test-time model merging. It is the highest-rated submission in this cohort and is accepted with the highest priority.

### 2. Submission 1: CG-SAM-TTMM (Overall Score: 5 - Accept)
* **Title:** Curriculum-Gated Sharpness-Aware Test-Time Model Merging for Robust Non-Stationary Streaming Adaptation
* **Recommendation:** **5: Accept** (Soundness: Excellent, Presentation: Excellent, Significance: Excellent, Originality: Excellent)
* **Key Contributions:**
  - Discovers a major "Silent Gradient Disconnection Bug" in standard deep learning TTA implementations, where weight reloading via in-place state dict mutations (`load_state_dict`) silently breaks the autograd computational graph. This means prior adaptive merging baselines were running as static in previous literature.
  - Resolves this bug by reformulating test-time adaptation statelessly using PyTorch's `functional_call` API.
  - Integrates an Online Dynamic Curriculum Gating mechanism that scales the SAM perturbation based on predictive uncertainty (Shannon entropy) to balance adaptation and noise robustness.
  - Provides a rigorous second-order Taylor expansion and Rayleigh quotient mathematical analysis of SAM under gradient noise to support their formulation.
* **Meta-Review Decision Justification:** Exposing the silent gradient disconnection bug is an extraordinary service to the machine learning community that will prevent future researchers from publishing flawed, static implementations of TTA. Combined with its rigorous mathematical and empirical execution, this paper represents an incredibly high-impact contribution and is accepted.

### 3. Submission 7: CSAIR (Overall Score: 5 - Accept)
* **Title:** Continuous Sigmoid-blended Angular-Euclidean Isomorphic Routing for Noise-Robust Test-Time Model Merging
* **Recommendation:** **5: Accept** (Soundness: Excellent, Presentation: Excellent, Significance: Good, Originality: Excellent)
* **Key Contributions:**
  - Discovers a major representational bottleneck in test-time model merging: "TTBN-induced scale mismatch", where Test-Time Batch Normalization (TTBN) causes scale mismatches in prototype-guided routing.
  - Resolves this mismatch using a novel, elegant, and mathematically proven **Normalized L2 distance** on the unit sphere.
  - Proves a beautiful metric isomorphism between the Euclidean and Angular spaces under SCTS routing, showing they are equivalent under unit normalization.
  - Implements a continuous sigmoidal blending mechanism to resolve routing discontinuities of hard-gated routing.
* **Meta-Review Decision Justification:** This paper stands out due to its exceptional originality and rigorous mathematical foundation. Identifying the TTBN scale mismatch and proving the metric isomorphism are brilliant technical accomplishments that provide deep insights into weight-space adaptation. It represents a highly significant advance and is accepted.

---

## 3. Rejected Submissions (With Merits)

### Submission 3: BK-FWSAM (Overall Score: 5 - Accept)
* **Title:** Direct Parameter Sensitivity Preconditioning for Sharpness-Aware Test-Time Model Merging
* **Summary of Merits:** Proposes BK-FWSAM, which introduces Direct Parameter Sensitivity (DPS) preconditioning parameterized in the merging offset space ($\delta_j$) rather than weight space. It includes a highly thorough evaluation comparing 10 different methods and extensive sensitivity sweeps.
* **Reason for Rejection:** While BK-FWSAM is extremely strong, the competition for the top 3 slots is extremely fierce. It was narrowly edged out by Submission 7 and Submission 1, which presented highly original and impactful discoveries (the TTBN-induced scale mismatch and the silent gradient disconnection bug, respectively) that have broader architectural implications.

### Submission 6: SMT-LDAC (Overall Score: 5 - Accept)
* **Title:** Sharpness-Aware Mean-Tracker with Layer-Depth Adaptive Coherence for Test-Time Model Merging
* **Summary of Merits:** Proposes SMT-LDAC and provides an exceptionally strong mathematical proof of variance reduction under both independent (88.9%) and temporally correlated AR(1) noise (86.4%).
* **Reason for Rejection:** Despite its flawless technical execution, the empirical improvements of SMT-LDAC over the SOTA baseline BK-AHR are extremely modest (+0.25% absolute accuracy) and concentrated entirely on a narrow noise band. Additionally, it does not address the scale or noise diversities typical of larger benchmarks.

### Submission 9: KPSAM-DST (Overall Score: 5 - Accept)
* **Title:** Kronecker-Preconditioned SAM and Denoised Self-Training for Test-Time Model Merging
* **Summary of Merits:** Proposes KPSAM-DST, which combines SAM-based flatness optimization and Denoised Self-Training, backed by a correct proof of Proposition 5.1.
* **Reason for Rejection:** The paper represents a creative combination of existing techniques (SAM and self-training) rather than introducing a major novel paradigm or uncovering new architectural bottlenecks, making it slightly less original than Submissions 1, 4, and 7.

### Submission 5: CG-MTTMM (Overall Score: 4 - Weak Accept)
* **Title:** Curvature-Guided Meta-Adaptation for Test-Time Model Merging
* **Summary of Merits:** Implements a curvature-guided meta-adaptation method that dynamically scales learning rate and damping using local directional Hessian curvature.
* **Reason for Rejection:** The overall significance is limited as the proposed method yields no empirical improvement over a perfectly tuned static baseline in its optimal regime. Furthermore, it introduces a 2x computational forward-pass overhead that is not fully justified by its performance.

### Submission 2: SAK-AHR (Overall Score: 3 - Weak Reject)
* **Title:** Sharpness-Aware Kronecker-Preconditioned Adaptive Hybrid Routing...
* **Reason for Rejection:**
  1. **Misleading Terminology:** The paper uses the term "Kronecker-Preconditioned" when there is no Kronecker product or matrix factorization used; it is actually a simple diagonal scaling based on layer-wise gradient norms.
  2. **Critical Theoretical Flaw:** The proof of Theorem 3.1 contains a major mathematical flaw regarding Hessian projections. Multiplying by the pseudoinverse of a tall Jacobian $J_\phi$ yields a projection matrix rather than the identity, meaning the curvature in the orthogonal complement of the weight space remains completely unconstrained. Thus, bounding parameter-space flatness does not bound weight-space flatness.
  3. **Marginal/Negative Gains:** SAK-AHR with SAM and preconditioning actually performs slightly worse than the ablation without them.

### Submission 8: SW-SAM-TTMM (Overall Score: 3 - Weak Reject)
* **Title:** Sparsity-Weighted Sharpness-Aware Test-Time Model Merging...
* **Reason for Rejection:** The proposed Sparsity-Weighted Perturbations and Entropy-Adaptive Learning Rate (EALR) modules provide absolutely zero accuracy gains over simpler baselines at any realistic learning rate. The added complexity is mathematically and empirically redundant.

### Submission 10: Integrating Denoised Gating and Curvature Preconditioning... (Overall Score: 3 - Weak Reject)
* **Title:** Integrating Denoised Gating and Curvature Preconditioning for Robust TTMM
* **Reason for Rejection:** Very low originality. It is a direct combination of several concurrent works (`AHR-SATS-DUN`, `BK-AHR`, and `SAM-TTMM`), but its implementation actually disables the routing innovations of those works due to interpolation mismatches, resulting in a method that is practically identical to the existing `SAM-TTMM` with minor heuristic adjustments.

---

## 4. Summary of Meta-Review Decisions

The meta-review process successfully identified **Submission 4**, **Submission 1**, and **Submission 7** as the three papers to be accepted. These three papers showcase exceptional originality, flawless mathematical soundness, and address crucial challenges in Test-Time Model Merging with high impact. The remaining papers, while possessing clear merits in some cases, fell short due to either minor theoretical flaws, marginal empirical benefits, or high competition.
