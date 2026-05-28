# Area Chair Meta-Review Report and Decisions

**Conference:** International Conference on Machine Learning (ICML 2026)  
**Track:** Deep Model Fusion and Edge Deployment Robustness  
**Date:** May 28, 2026  

---

## 1. Executive Summary

Model merging (or model fusion) has emerged as a high-impact, resource-efficient paradigm for combining task-specific expert neural networks without retraining or parameter overhead. However, deploying these merged models under realistic edge-computing scenarios introduces severe vulnerabilities, most notably "representation collapse" (the exponential decay of activation scales), post-training quantization (PTQ) noise, and environmental out-of-distribution (OOD) corruption. 

In this meta-review round, we evaluated **10 paper submissions** investigating these challenges. The submissions explore diverse methodologies including curved hyperbolic parameter geometry, optimal transport (Wasserstein alignment), hyperdimensional holographic binding, post-training quantization, structured pruning, and dynamic test-time/inference-time self-calibration.

Following a rigorous multi-perspective review process incorporating the insights of expert reviewers with specialized personas—including **Novelty Seekers**, **Minimalists**, **Practitioners**, **Theorists**, and **Empiricists**—we have selected **3 out of 10 submissions for acceptance**. 

### Summary of Decisions
*   **Accepted Submissions (3):** Submission 2, Submission 8, Submission 10
*   **Rejected Submissions (7):** Submission 1, Submission 3, Submission 4, Submission 5, Submission 6, Submission 7, Submission 9

---

## 2. Meta-Review Methodology

The Area Chair (AC) conducted a systematic, content-driven synthesis of all reviews, balancing numerical recommendations with qualitative arguments. Each paper was evaluated on four core dimensions: **Soundness**, **Presentation**, **Significance**, and **Originality**, in accordance with the conference guidelines.

Particular attention was paid to:
1.  **Conceptual Originality vs. Incremental Heuristics:** Prioritizing bold paradigm-shifting insights over minor, hyperparameter-heavy adjustments to existing baselines.
2.  **Empirical Rigor vs. Toy Benchmarks:** Ensuring claims are backed by honest, multi-seed evaluation and compared against strong, uncalibrated baselines.
3.  **Practical Utility and Deployability:** Balancing theoretical elegance with practical constraints, such as compatibility with modern Transformer architectures (LayerNorm/RMSNorm) and edge-device hardware (latency, memory, write-protected flash).
4.  **Scientific and Scholarly Integrity:** Strictly penalizing severe double-blind violations and fabricated/hallucinated literature citations.

---

## 3. Detailed Evaluation of Accepted Papers

### Submission 2: "Non-Euclidean Model Merging: Geometric Preservation of Representation Scale on Hyperbolic Manifolds"
*   **Reviewer Recommendations:** Reviewer 1 (Novelty Seeker): **6 (Strong Accept)** | Reviewer 3 (Practitioner): **3 (Weak Reject)**
*   **Average Score:** 4.50 (Highest in cohort)
*   **AC Synthesis & Decision: ACCEPT (Strong Champion)**
*   **Contributions:** The paper proposes merging expert neural parameters in negative-curvature hyperbolic space (the Lorentz/Hyperboloid manifold $\mathbb{H}^d$) rather than standard flat Euclidean space to mitigate representation collapse. More importantly, it uncovers and mathematically deconstructs **"The Floating-Point Illusion"**: proving that the success of hyperbolic merging is not due to negative curvature itself, but rather a subtle float32 rounding artifact on the Minkowski light-cone that acts as an implicit scale-preserver. Under double precision (FP64), the effect vanishes, and the model collapses back to standard weight averaging.
*   **Strengths:** 
    *   **Exceptional Conceptual Novelty:** A brilliant, paradigm-shifting connection between curved parameter manifolds and ML hardware precision boundaries.
    *   **Rigorous Mathematical-Empirical Alignment:** Watertight proof (Theorem 4.1) and empirical validation of the FP32 vs. FP64 dichotomy.
    *   **Exhaustive Taxonomy:** Formulates and deconstructs spherical positive-curvature merging and curvature-adaptive merging (CAHM).
*   **Weaknesses & Refinement Guidance:** 
    *   *Practical Performance Gap:* FP32 Hyperbolic Merging (51.25%) is outperformed by flat-space calibration baselines like Holographic Norm Scaling (HNS, 61.40%). 
    *   *Hardware/Precision Fragility:* The method relies on FP32-specific rounding, which may behave differently or require scale tuning in industry-standard formats like BF16, FP16, or INT8.
    *   *Scale:* Evaluation is limited to ResNet-18 on toy datasets (MNIST/CIFAR).
*   **Acceptance Rationale:** While the Practitioner rightly points out the performance gap and hardware fragility, the Novelty Seeker's championing is extremely compelling. The discovery of the "Floating-Point Illusion" is a spectacular, mind-bending contribution that reframes numerical precision from a passive constraint to an active geometric mechanism. This paper is highly influential and will inspire significant future research at the intersection of non-Euclidean parameter geometry and hardware compiler arithmetic.

---

### Submission 8: "Mitigating Calibration Variance in Data-Efficient BatchNorm Calibration for Quantized Model Merging"
*   **Reviewer Recommendations:** Reviewer 1 (Theory): **3 (Weak Reject)** | Reviewer 2 (Theory): **3 (Weak Reject)** | Reviewer 3 (Minimalist): **5 (Accept)**
*   **Average Score:** 3.67
*   **AC Synthesis & Decision: ACCEPT**
*   **Contributions:** This paper exposes the **"calibration variance pathology"** of data-efficient BatchNorm calibration (DE-BN), showing that using small support sets (e.g., 32 samples) to restore quantized merged models leads to high accuracy variance (e.g., 3.19% std under 4-bit quantization). To resolve this, it proposes two training-free, forward-only protocols: Robust Multi-Seed BatchNorm Calibration (RMS-BC) and **Data-Efficient Momentum BatchNorm Calibration (DEM-BC)**, which sequentializes calibration by smoothly adapting running moments with a small momentum $\beta$ over $S$ sequential forward passes.
*   **Strengths:**
    *   **Minimalist Engineering Elegance:** DEM-BC requires zero training or backward passes, integrating seamlessly into PyTorch via a single-line modification of the `.momentum` attribute.
    *   **Exceptional Practical Utility:** Under 4-bit quantization, DEM-BC improves average accuracy by **+11.28%** absolute and reduces standard deviation from 3.19% to 0.03% (a 99.0% reduction in variance).
    *   **High Empirical Discipline:** Uses 10 random seeds to rigorously track and report variance.
*   **Weaknesses & Refinement Guidance:**
    *   *Theoretical Grounding:* Reviewers noted mathematical formulation inaccuracies in the statistical pooling of RMS-BC (Eq. 5) and potential systematic biases in DEM-BC.
    *   *Scope:* The method is restricted to BatchNorm-based architectures, and the introduction's mentions of LLaMA and BERT (which use LayerNorm) are conceptually misleading.
*   **Acceptance Rationale:** The AC strongly aligns with the Minimalist reviewer. While the theory-minded reviewers raised valid concerns regarding statistical pooling proofs, the practical impact and elegance of DEM-BC are undeniable. Resolving severe performance volatility in low-bit edge deployments with a zero-cost, training-free, single-line configuration change is a triumph of practical machine learning engineering. The authors must correct the RMS-BC mathematical formulations and remove the misleading LLaMA/BERT references in their final camera-ready revision.

---

### Submission 10: "Inference-Time Self-Calibration: Rethinking Activation Calibration and Out-of-Distribution Robustness in Quantized Model Merging"
*   **Reviewer Recommendations:** Reviewer 1 (Minimalist): **3 (Weak Reject)** | Reviewer 2 (Minimalist): **5 (Accept)** | Reviewer 3 (Theory/Empiricist): **2 (Reject)**
*   **Average Score:** 3.33
*   **AC Synthesis & Decision: ACCEPT**
*   **Contributions:** The paper proposes **Inference-Time Self-Calibration (ITSC)** to resolve representation collapse in extreme 4-bit post-training quantization and out-of-distribution (OOD) environmental shifts. Instead of relying on static, pre-curated offline calibration datasets, the model dynamically self-calibrates its BatchNorm running statistics on-the-fly over incoming unlabeled inference streams. To resolve slow-adaptation or bias on short streams, they introduce **Adaptive Momentum Scheduling (AMS)** ($m_k = \max(m_{\text{inf}}, 1/k)$), which computes a cumulative moving average on early batches before transitioning to standard exponential moving averages.
*   **Strengths:**
    *   **Elegant and Practical Paradigm Shift:** Moves model merging from fragile offline static calibration to highly robust dynamic, zero-data inference-time self-calibration.
    *   **Spectacular Empirical Performance:** Under 4-bit quantization, ITSC-AMS yields a **+10.39%** clean accuracy gain over static DE-BN, and massive OOD gains of **+36.02%** under Defocus Blur and **+32.75%** under Gaussian Noise.
    *   **Dynamic AMS Scheduler:** A neat, parameter-free scheduling mechanism that wipes out initial stream transient bias instantly.
*   **Weaknesses & Refinement Guidance:**
    *   *Redundant "Prior Warm-starting" (ITSC-PW) Variant:* The proposed ITSC-PW-AMS variant is mathematically redundant. Because the momentum at the first batch is $m_1 = 1.0$, the warm-started initial stats are multiplied by zero and completely overwritten. Table results confirm identical performance.
    *   *Self-Aggrandizing Language:* The manuscript repeatedly uses overly dramatic, non-objective phrasing (e.g., "adopt the perspective of the Visionary", "our visionary extensions").
    *   *Scale:* Evaluation is on ResNet-18 on MNIST/CIFAR.
*   **Acceptance Rationale:** The AC champions this paper due to its immense practical significance. Test-time self-calibration represents a major paradigm shift for robust edge deployment, achieving spectacular, robust performance gains under severe quantization and environmental noise. The mathematical redundancy of the "Prior Warm-starting" extension must be removed in the final camera-ready version, presenting only the clean, elegant ITSC-AMS model. Additionally, the theatrical "Visionary" terminology must be completely stripped to maintain objective, scholarly rigor.

---

## 4. Detailed Evaluation of Rejected Papers

### Submission 1: "Pragmatic Model Merging: Mixed-Distribution BatchNorm Calibration for Noisy and Quantized Edge Deployment"
*   **Reviewer Recommendations:** Reviewer 1 (Novelty Seeker): **2 (Reject)** | Reviewer 2 (Scientific Rigor): **2 (Reject)** | Reviewer 3 (Minimalist): **4 (Weak Accept)**
*   **Average Score:** 2.67
*   **Core Flaws & Rejection Reasons:** 
    *   *Incremental Conceptual Novelty:* The proposed Mixed-Distribution BatchNorm Calibration (MD-BN) simply inserts standard noise and blur-augmented samples into a single-pass calibration batch. This is a highly well-established heuristic in test-time adaptation literature and lacks original conceptual leaps.
    *   *Underperforming the Uncalibrated Baseline:* Table 1 reveals a severe soundness flaw. Under Gaussian noise, the simple **None (Uncalibrated)** baseline achieves **49.82%** (FP32) and **49.87%** (INT8), whereas the proposed **MD-BN (Mixed-Ours)** achieves only **44.15%** (FP32) and **45.21%** (INT8). Thus, doing absolutely no calibration performs significantly better than their method under noise.
    *   *Unexplained Baselines & Overstated Claims:* The paper reports a "Multi-Ours" baseline but completely fails to define or explain it in the text. It also claims to "heal" noise fragility when it performs worse than the uncalibrated starting point.

---

### Submission 3: "Activation-Driven Synaptic Resonance: A Dynamic Bio-Inspired Paradigm for Quantization-Robust Model Merging"
*   **Reviewer Recommendations:** Reviewer 2 (Empiricist): **2 (Reject)**
*   **Average Score:** 2.00
*   **Core Flaws & Rejection Reasons:**
    *   *Conceptual Mischaracterization:* The paper is advertised as model merging, but it actually keeps all task-specific expert models intact and runs them in parallel, selecting active sub-networks. This leads to $\mathcal{O}(K)$ inference latency and memory complexity. It is a parallel ensemble, not a true parameter-space merge, defeating the primary efficiency goal of model merging.
    *   *Numerical Sloppiness:* There are massive, unacceptable discrepancies between the metrics cited in the body of the text and those reported in Table 1.
    *   *Weak Evaluation & Baseline Omissions:* Restricts experiments to ResNet-18 on toy datasets. It completely omits comparison against standard model merging baselines (Task Arithmetic, TIES, DARE) and has zero statistical validation (no random seeds, error bars, or standard deviations).

---

### Submission 4: "The Pragmatic Way to Merge Models: Quantile-Based Weight Clipping and Data-Efficient Calibration Solve Low-Bit Quantization Collapse"
*   **Reviewer Recommendations:** Reviewer 1 (Theory): **2 (Reject)** | Reviewer 2 (Empiricist): **2 (Reject)** | Reviewer 3 (Minimalist): **2 (Reject)**
*   **Average Score:** 2.00
*   **Core Flaws & Rejection Reasons:**
    *   *Discrepancy Between Narrative and Data:* The paper repeatedly claims that the proposed Quantile Weight Clipping (QWC) matches or outperforms Quantile Co-Quantization Tuning (QCOT). However, their own Table 1 shows QWC performing up to 10-14% *worse* than QCOT across all settings. This represents a severe scientific overstatement.
    *   *Lack of Originality:* The proposed components (quantile clipping, weight standardization, and BatchNorm calibration) are straightforward, hyperparameter-heavy combinations of standard, existing PTQ and model merging heuristics.
    *   *Theoretical Flaws:* The theoretical justification in Section 4 contains a fundamental mathematical error: claiming that weight standardization preserves activation bounds, which ignores the highly non-linear activation functions (ReLUs) and task-specific scaling dynamics.

---

### Submission 5: "Quantization-Constrained Sliced Wasserstein: Multi-Dimensional Optimal Transport for Robust Model Merging"
*   **Reviewer Recommendations:** Reviewer 1 (Theory): **2 (Reject)** | Reviewer 2 (Theory): **2 (Reject)** | Reviewer 3 (Empiricist): **3 (Weak Reject)**
*   **Average Score:** 2.33
*   **Core Flaws & Rejection Reasons:**
    *   *Fatal Theoretical Flaw:* The core mathematical formulation has a severe error: the linear reconstruction step via the pseudo-inverse violates the infinity-norm constraint that the paper claims to strictly satisfy. This mathematical inconsistency completely invalidates the Rademacher generalization bounds (Theorem 4.6).
    *   *Empirical Contradiction:* The primary empirical claim that QCSW outperforms existing baselines is contradicted by the paper's own main results table (Table 1), where QCSW underperforms much simpler 1D baselines (such as WCPR/QCOT) on average across multiple tasks and quantization levels.
    *   *High Computational Complexity:* The method requires calculating the pseudo-inverse and iterative projection of weight matrices, scaling as $\mathcal{O}(D_{\text{in}}^3)$ per layer. This is computationally prohibitive for larger models.

---

### Submission 6: "Holographic Synaptic Alignment"
*   **Reviewer Recommendations:** Reviewer 3 (Theory): **3 (Weak Reject)**
*   **Average Score:** 3.00
*   **Core Flaws & Rejection Reasons:**
    *   *Mathematical Flaw in Theorem 4.1:* The proof of the unbiasedness of the retrieved quantization noise relies on a false independence assumption between the quantization error and the random orthogonal phase keys. A simple mathematical counterexample proves that they are dependent, making the retrieved noise fundamentally biased.
    *   *Information-Theoretic "Hand-Waving" on DE-BN:* The claim that DE-BN "absorbs" holographic crosstalk noise is conceptually flawed. Normalizing the variance prevents activation explosion but keeps the Signal-to-Noise Ratio (SNR) at $\mathcal{O}(1/K)$, leading to massive mutual information loss.
    *   *Simulated Capacity Sweep:* The capacity sweep up to $K=20$ is simulated using random Gaussian noise instead of actual trained task experts, failing to prove that the hologram can store and retrieve 20 meaningful tasks.
    *   *Impractical Routing:* The unsupervised model-based routing (EGBR-HSA) requires evaluating all $K$ phase keys, scaling inference latency as $\mathcal{O}(K)$, which completely defeats the efficiency of model merging.

---

### Submission 7: "Activation-Aware Structured Channel Pruning for Quantized Merged Models"
*   **Reviewer Recommendations:** Reviewer 1 (Empiricist): **3 (Weak Reject)** | Reviewer 2 (Practitioner): **3 (Weak Reject)** | Reviewer 3 (Theory/Practitioner): **3 (Weak Reject)**
*   **Average Score:** 3.00
*   **Core Flaws & Rejection Reasons:**
    *   *System-Level Deployment Contradiction:* The paper is framed as a highly practical solution for extreme microcontrollers. However, the proposed serving-time algorithm (Dynamic ACP, Algorithm 1) requires online BatchNorm calibration, weight modification, and weight quantization during serving. This is completely unrealistic for extreme edge microcontrollers due to write-protected Flash memory, tiny SRAM capacity, and intolerable cold-start latency spikes (668.4 ms).
    *   *The "Model Merging Storage Paradox":* If the pruning is performed offline, the storage-saving benefits of model merging are lost since separate pruned models must be stored.
    *   *Underperforming Simple Static Baselines:* In standard FP32 and INT8 precisions, a simple static baseline (Weight L1 pruning) consistently outperforms the proposed dynamic method.

---

### Submission 9: "Unified Mixture Calibration: Demystifying the Role of Activation Scale in Task-Agnostic Model Merging"
*   **Reviewer Recommendations:** Reviewer 1 (Empiricist): **2 (Reject)** | Reviewer 2 (Scholarship/Integrity): **2 (Reject)** | Reviewer 3 (Empiricist): **2 (Reject)**
*   **Average Score:** 2.00
*   **Core Flaws & Rejection Reasons (Disqualifying):**
    *   **Severe Breach of Scientific Integrity:** Reviewer 2 discovered multiple **completely fabricated citations** in the literature review (including `Visionary et al., 2026`, `Leclaire et al., 2024/2026`, and `Authors, 2026`). Citing non-existent papers represents a fatal ethical breach that warrants immediate rejection.
    *   *Double-Blind Review Violation:* Citing a paper under review as literal "Authors, 2026" is a severe violation of double-blind guidelines.
    *   *Severe Empirical Anomalies:* The paper exhibits a massive empirical bug where the "Uncalibrated" baseline magically gains up to **+28%** absolute accuracy when subjected to lossy 4-bit/8-bit quantization or random Gaussian input noise compared to its full-precision (FP32) performance. This indicates a critical bug in the evaluation pipeline.
    *   *Poor Performance:* The proposed CA-UC method consistently underperforms standard "Naive Mixed Calibration" across all settings.

---

## 5. Conclusion & Future Directions

The submissions in this cohort highlight a crucial evolutionary transition in model merging research: moving from static, hyperparameter-heavy, and offline Euclidean heuristics to more dynamic, hardware-aware, and geometric paradigms. 

The three accepted papers provide highly distinct but complementary contributions:
1.  **Submission 2 (Hyperbolic Geometry):** Forces the community to look beyond standard flat Euclidean coordinates, providing a fascinating cautionary tale about how hardware precision limits can act as active geometric mechanisms.
2.  **Submission 8 (Momentum Calibration):** Exposes the hidden volatility of data-efficient calibration and resolves it with a clean, single-line momentum update.
3.  **Submission 10 (Inference-Time Self-Calibration):** Transitions the entire paradigm from offline static pre-calibration to dynamic, data-free on-the-fly self-calibration on incoming test streams.

**Future Directions:** A promising and high-impact avenue of future research would be the synthesis of these three paradigms: exploring how inference-time self-calibration and momentum-based tracking can be mapped onto non-Euclidean parameter manifolds, and scaling these dynamic self-calibration principles from BatchNorm-based convolutional backbones to LayerNorm/RMSNorm-based Transformer architectures (LLMs and ViTs) dominating modern artificial intelligence.
