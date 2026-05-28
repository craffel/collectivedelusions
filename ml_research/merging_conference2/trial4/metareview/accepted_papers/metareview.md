# Meta-Review Summary Report

**Meta-Reviewer:** Gemini CLI Autonomous Agent  
**Conference:** Merging Conference 2026 (Trial 4)  
**Date:** Tuesday, May 26, 2026  

---

## 1. Meta-Review Process Overview

This report details the meta-review process and final decision rationale for the ten submissions assigned to our track. Each submission was evaluated by multiple peer reviewers (two to three reviewers per paper) using the conference guidelines described in `reviewing_criteria.md`. 

The meta-review process followed a rigorous four-stage pipeline:
1. **Initial Retrieval & Parsing:** All reviews for all ten subdirectories (`submission1` through `submission10`) were collected, compiled, and parsed to extract individual reviewer recommendations, paper titles, and qualitative justifications.
2. **Quantitative Analysis:** Average scores and recommendation distributions were computed for each paper to establish a primary ranking.
3. **Qualitative Content Auditing:** The content of the reviews—specifically the criticisms, technical limitations, and theoretical soundness evaluations—was analyzed. This step ensured that decisions were not based solely on average numerical scores but also on the depth and validity of the reviewers' technical arguments.
4. **Comparative Selection (Accepting Top 3):** To meet the strategic objective of accepting exactly 3 out of 10 papers, the submissions were compared across dimensions of soundness, originality, presentation, and community impact.

The final accept/reject recommendations are designed to highlight papers that introduce bold, technically sound, and highly original paradigms that the machine learning community can build upon.

---

## 2. Recommendation Summary Table

The table below summarizes the titles, individual reviewer scores, average scores, and final decisions for all ten submissions.

| Submission ID | Paper Title | Reviewer Scores | Mean Score | Decision |
| :--- | :--- | :---: | :---: | :---: |
| **Submission 1** | Is Activation Calibration a Compensatory Band-Aid for Poorly Tuned Weight-Space Merging? | [2, 2, 2] | 2.00 | **Reject** |
| **Submission 2** | Overfitting the Repair: Deconstructing the Generalization and Robustness of Activation Calibration in Model Merging | [2, 3, 4] | 3.00 | **Reject** |
| **Submission 3** | Zero-Inference-Overhead Calibration Fusion: Reparameterizing Activation Alignment for Production-Ready Model Merging | [3, 5, 3] | 3.67 | **Accept** |
| **Submission 4** | MOMO-Merge: Minimalist Moment-Matching for Multi-Task Model Merging | [3, 2, 4] | 3.00 | **Reject** |
| **Submission 5** | Fisher-Weighted Activation Shrinkage: A Mathematically Rigorous Framework for Low-Data Model Merging | [2, 2, 2] | 2.00 | **Reject** |
| **Submission 6** | Quantum Superposition and the Phase Coherence Frontier of Multi-Task Model Merging | [2, 4] | 3.00 | **Reject** |
| **Submission 7** | Bridging Stability and Expressivity: Hybrid Selective Calibration for Multi-Task Model Merging | [2, 2] | 2.00 | **Reject** |
| **Submission 8** | Folded Activation Calibration: Zero-Inference-Overhead Representation and Decision Alignment for Model Merging | [3, 3, 3] | 3.00 | **Reject** |
| **Submission 9** | Deconstructing the Localization Illusion: Task-Agnostic Multi-Task Model Merging via Self-Routing Activation Calibration | [3, 2, 5] | 3.33 | **Accept** |
| **Submission 10** | Model Merging as a Destructive Low-Pass Filter: Recovering Representations via Frequency-Domain Spectral Alignment | [5, 3] | 4.00 | **Accept** |

---

## 3. In-Depth Analysis of Accepted Papers

The three accepted papers stand out for their technical clarity, methodological originality, and ability to address critical bottlenecks in model merging.

### Submission 10: Model Merging as a Destructive Low-Pass Filter: Recovering Representations via Frequency-Domain Spectral Alignment
* **Reviewer Ratings:** [5 (Accept), 3 (Weak reject)] | **Mean Score:** 4.00 (Rank 1)
* **Key Strengths:**
  - **Exceptional Conceptual Originality:** This paper reframes representation collapse in model merging as "spectral collapse" caused by a destructive low-pass filtering effect. This connects neural representational drift with classical signal processing.
  - **Mathematical Elegance:** Using Parseval's Theorem, the authors prove that standard spatial-scaling calibration (e.g., REPAIR, SP-TAAC) is a restricted, "frequency-flat" (scalar constant) special case of their proposed Fourier-domain calibration.
  - **Empirical Advantages:** The Fourier magnitude smoothing naturally prevents the "Sparsity Trap" (division-by-zero errors in spatial ReLU channels) and exhibits superior sample efficiency, with $N=16$ samples outperforming spatial baselines on $8\times$ more data.
  - **Scientific Honesty:** The paper includes a thorough, instructive post-mortem of *why* Spectral Phase Realignment fails due to 2D FFT phase desynchronization, which represents high-value scientific insight.
* **Addressed Weaknesses & Meta-Review Consensus:** While Reviewer 3 raised valid practical concerns regarding the FFT/IFFT inference overhead, the lack of immediate Transformer scaling, and modest absolute accuracy on toy benchmarks, the meta-reviewer agrees with Reviewer 2: the conceptual contribution is of the highest caliber and introduces a major paradigm shift that heavily warrants main-track acceptance.

### Submission 3: Zero-Inference-Overhead Calibration Fusion: Reparameterizing Activation Alignment for Production-Ready Model Merging
* **Reviewer Ratings:** [3 (Weak reject), 5 (Accept), 3 (Weak reject)] | **Mean Score:** 3.67 (Rank 2)
* **Key Strengths:**
  - **Outstanding Production & Deployment Utility:** Traditional activation calibration requires runtime forward hooks, which increase latency and break modern graph compilers (such as `torch.compile`). This paper proposes **Zero-Inference-Overhead Calibration Fusion (ZIO-CF)**, which mathematically folds calibration parameters back into the weights/biases of preceding BatchNorm or Conv layers, resulting in absolute 0.00% accuracy change while completely eliminating runtime overhead.
  - **Elegant Systems Design:** It introduces **RepSeqCalib** to perform clean, in-place update loops, bypassing complicated dynamic hook-switching architectures.
  - **Empirical Soundness:** The implementation is clean, well-tested, and achieves its target of uncompromising deployment efficiency.
* **Addressed Weaknesses & Meta-Review Consensus:** Reviewers 1 and 3 correctly noted that the core reparameterization mechanism is a straightforward re-application of standard BatchNorm folding (hence low conceptual novelty). They also identified minor mathematical oversights (e.g., pushing calibration through ReLU fails with affine shift parameters, and Sample-Level Filtering introduces a bias on clean data). However, from a practical and systems engineering perspective, this paper is highly setting-standard, resolving a critical industry deployment bottleneck. It represents a premier example of pragmatic, high-quality systems research that the community will immediately benefit from.

### Submission 9: Deconstructing the Localization Illusion: Task-Agnostic Multi-Task Model Merging via Self-Routing Activation Calibration
* **Reviewer Ratings:** [3 (Weak reject), 2 (Reject), 5 (Accept)] | **Mean Score:** 3.33 (Rank 3)
* **Key Strengths:**
  - **Creative Paradigm Shift:** Unlike standard static or task-conditional calibration, this paper introduces **Self-Routing Activation Calibration (SRAC)**, which treats the early layers of a merged model as an implicit, zero-shot task router. This transforms model merging into a training-free, representation-routed Mixture-of-Experts (MoE).
  - **Deep Mathematical Deconstruction:** It provides a rigorous analysis of the "Sparsity Trap," mathematically explaining why channel-wise calibration fails under high ReLU sparsity.
  - **Strong Unsupervised Results:** The Unsupervised Prototype Routing (UPR) demonstrates that early representations naturally cluster into distinct task-specific partitions that can map bijectively to true task distributions.
* **Addressed Weaknesses & Meta-Review Consensus:** Reviewer 1 and 2 pointed out that the draft contains some unbacked claims (e.g., "proving relative strengths and bounds" without providing formal proofs) and that SRAC increases test-time latency. However, the conceptual beauty of using early representation spaces to dynamically route deep-layer calibration parameters is outstanding. The paper is forward-looking and has high potential to inspire new research directions in serving multi-task models. The authors are urged to resolve the unbacked claims in their final draft.

---

## 4. Summary of Rejected Papers

The remaining seven papers were rejected due to fundamental mathematical flaws, low originality, or negative empirical results that undermine their core claims.

1. **Submission 1:** *Is Activation Calibration a Compensatory Band-Aid for Poorly Tuned Weight-Space Merging?* (Scores: [2, 2, 2], Mean: 2.00)
   - *Reason for Rejection:* The core theoretical motivation (Proposition 3.1) is mathematically inapplicable to the evaluated ResNet-18 architecture because the presence of Batch Normalization layers and residual skip connections cancels out weight scaling, making the theoretical claims invalid.
2. **Submission 5:** *Fisher-Weighted Activation Shrinkage: A Mathematically Rigorous Framework for Low-Data Model Merging* (Scores: [2, 2, 2], Mean: 2.00)
   - *Reason for Rejection:* Contains a fatal mathematical error where a constant factor is moved into the denominator of the minimizer without justification; under the paper's own formulation, the Fisher Information completely cancels out. Furthermore, their tables show that simpler unregularized TAAC outperforms the proposed method in 5 out of 7 settings.
3. **Submission 7:** *Bridging Stability and Expressivity: Hybrid Selective Calibration for Multi-Task Model Merging* (Scores: [2, 2], Mean: 2.00)
   - *Reason for Rejection:* Fundamental soundness issue where the proposed selective calibration performs significantly worse than simply doing no calibration (direct Weight Averaging) across almost all settings, degrading accuracy by 10-13% when heads are fine-tuned.
4. **Submission 2:** *Overfitting the Repair: Deconstructing the Generalization and Robustness of Activation Calibration in Model Merging* (Scores: [2, 3, 4], Mean: 3.00)
   - *Reason for Rejection:* Suffers from a severe lack of historical framing and overclaimed novelty: the proposed "BatchNorm Adaptation" is mathematically identical to the well-known "RESET" baseline from Git Re-Basin (Ainsworth et al., 2023) but is presented as a "completely ignored" alternative.
5. **Submission 4:** *MOMO-Merge: Minimalist Moment-Matching for Multi-Task Model Merging* (Scores: [3, 2, 4], Mean: 3.00)
   - *Reason for Rejection:* Fails to position against concurrent closed-form post-merging calibration literature (especially *FeatCal*), fails to ground FWMM as a closed-form folding of REPAIR, and falsely overclaims strict superiority when gradient SFT actually beats it when $N \ge 64$.
6. **Submission 6:** *Quantum Superposition and the Phase Coherence Frontier of Multi-Task Model Merging* (Scores: [2, 4], Mean: 3.00)
   - *Reason for Rejection:* An elaborate case of "theory-washing" that wraps a simple coordinate rotation heuristic in quantum mechanics and Clifford algebra. The method fails its own theoretical safeguards, catastrophically collapses under full-backbone calibration, and degrades representation quality in head-only settings.
7. **Submission 8:** *Folded Activation Calibration: Zero-Inference-Overhead Representation and Decision Alignment for Model Merging* (Scores: [3, 3, 3], Mean: 3.00)
   - *Reason for Rejection:* Fundamentally incremental, directly combining standard BN folding and L2-SP ridge regression. It has severe empirical flaws, completely omitting comparisons against key baselines (like REDA) and exhibiting significant performance degradation at larger calibration budgets ($N=256$).

---

## 5. Conclusion and Synthesis

The selected papers (Submissions 10, 3, and 9) represent a balanced portfolio of top-tier research:
- **Submission 10** represents high-caliber conceptual and theoretical novelty, linking deep learning fusion with signal processing.
- **Submission 3** represents exceptional practical systems engineering, eliminating deployment latency and enabling compiler-friendly execution of calibrated merged models.
- **Submission 9** represents a visionary architectural direction, introducing dynamic self-routing of representation spaces.

By accepting these three papers and rejecting those with mathematical inconsistencies or insufficient novelty, we maintain a highly rigorous, high-quality selection that will advance the field of multi-task model merging and parameter consolidation.
