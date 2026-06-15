# Meta-Review Summary Report and Decisions

This document summarizes the meta-review process and decisions for the 10 paper submissions evaluated during this cycle. The goal of this process was to systematically review 10 deep learning model-merging and quantization paper submissions and select exactly **3 submissions** for acceptance based on the content, rigor, and scores of their peer reviews.

---

## 1. Overview of the Meta-Review Process

We analyzed 10 subdirectories (`submission1` through `submission10`). Each subdirectory contained the paper submission itself and three expert reviews (with the exception of `submission7`, which contained two expert reviews). In total, **29 detailed reviews** were parsed and evaluated.

For each submission, we:
1. Extracted all quantitative ratings (Soundness, Presentation, Significance, Originality, and Overall Recommendation).
2. Read the reviewer justifications, identifying key strengths and critical technical, empirical, or scholastic weaknesses.
3. Computed the consensus recommendations and assessed whether the papers' proposed contributions were robust, well-supported, and valuable to the machine learning community.
4. Balanced borderline consensus against technical depth, preferring elegant deconstructive insights and simple, highly practical designs over over-engineered, hyper-complex, or poorly performing heuristics.

---

## 2. Comprehensive Analysis of All Submissions

Below is the structured analysis and meta-review decision for each of the 10 submissions.

### Submission 1: "PhaseMerge: Fourier Phase Interference for Noise-Cancelling Model Merging"
*   **Consensus Ratings:** 
    *   Reviewer 1: **2: Reject** (Soundness: Fair, Presentation: Excellent, Significance: Poor, Originality: Good)
    *   Reviewer 2: **3: Weak Reject** (Soundness: Fair, Presentation: Excellent, Significance: Poor, Originality: Good)
    *   Reviewer 3: **3: Weak Reject** (Soundness: Fair, Presentation: Excellent, Significance: Fair, Originality: Excellent)
*   **Average Score:** 2.67 / 5 (Strong Reject / Weak Reject)
*   **Strengths:** Highly creative and interdisciplinary concept of viewing weight-space merging as wave interference in the Fourier frequency domain; mathematically rigorous; exceptional writing and presentation; zero inference latency.
*   **Weaknesses:** 
    1. *Theoretical Mismatch:* Applying 2D FFT to dense, permutation-invariant layer matrices lacks physical justification, as dense weights have no natural spatial indices.
    2. *Lack of Competitiveness:* The proposed PhaseMerge framework is consistently and significantly outperformed by the simpler, real-space **PolyMerge** baseline (by 5.17% to 6.00% absolute accuracy).
    3. *Toy Evaluation:* Evaluated on an obsolete 5.7M parameter ViT patch model with test sets of only 100 samples.
    4. *Optimization Instability:* Degrades in performance and spikes in variance when provided with larger calibration streams ($M=32$), contradicting core claims.
*   **Decision:** **REJECT** (Unanimous negative consensus due to theoretical mismatch and weak empirical utility).

---

### Submission 2: "OmniMerge: Robust Model Merging across Heterogeneous Quantization Standards"
*   **Consensus Ratings:**
    *   Reviewer 1: **4: Weak Accept** (Soundness: Good, Presentation: Excellent, Significance: Good, Originality: Good)
    *   Reviewer 2: **3: Weak Reject** (Soundness: Fair, Presentation: Excellent, Significance: Fair, Originality: Good)
    *   Reviewer 3: **3: Weak Reject** (Soundness: Fair, Presentation: Fair, Significance: Fair, Originality: Fair)
*   **Average Score:** 3.33 / 5 (Weak Reject)
*   **Strengths:** Addresses a highly practical edge-AI bottleneck (PTQ compiler schema mismatch across heterogeneous hardware accelerators); zero-overhead, metadata-free, and training-free.
*   **Weaknesses:**
    1. *Lack of Statistical Rigor:* Reports all results as single-run accuracies with no standard deviations or multiple seeds, despite multiple layers of stochasticity (Stochastic Operator Sampling + scale noise injection).
    2. *Redundant Complexity:* The proposed "OmniMerge" framework suffers from redundant complexity, as a simpler noise-perturbation variant (SZNP Only) is performatively superior on average while being much simpler.
    3. *Toy Scale:* Evaluation uses a 5.7M parameter ViT-Tiny model and extremely under-trained task experts fine-tuned on only 256 training images, leading to expert accuracies barely above random guessing.
*   **Decision:** **REJECT** (Mixed-to-negative consensus, lack of statistical rigor, and performative inferiority compared to its own simpler ablated variants).

---

### Submission 3: "Re-Quantization Silence: Multi-Axial Re-Quantization Auditing (RQA) and Mitigations"
*   **Consensus Ratings:**
    *   Reviewer 1: **5: Accept** (Soundness: Excellent, Presentation: Excellent, Significance: Good, Originality: Excellent)
    *   Reviewer 2: **2: Reject** (Soundness: Poor, Presentation: Excellent, Significance: Poor, Originality: Fair)
    *   Reviewer 3: **3: Weak Reject** (Soundness: Fair, Presentation: Fair, Significance: Fair, Originality: Fair)
*   **Average Score:** 3.33 / 5 (Weak Reject / Mixed)
*   **Strengths:** Exceptional transparency in deconstructing its own proposed mitigations (SAWS and QA-ACS) and documenting their failure modes; highly elegant individual unmerged expert control experiment to decouple task interference from quantization noise.
*   **Weaknesses:**
    1. *Over-Claiming on a Non-Issue:* The paper frames "Re-Quantization Silence" as a universal, catastrophic bottleneck. However, the authors' own analysis reveals that under industry-standard per-channel configurations, naive re-quantization is nearly lossless (~1.8% drop under 4-bit). The collapse is strictly localized to per-tensor configurations, which are rarely used in practice.
    2. *Ineffectiveness of Proposed Mitigations:* Under the only regime where collapse is catastrophic (per-tensor), both proposed mitigations (SAWS and QA-ACS) fail to provide protection and perform worse than the naive baseline, or suffer from unsupervised entropy collapse.
    3. *Critical Methodological Flaws:* The reported "Double Quantization Noise" of 30.40% in INT8 is anomalously high, serving as a self-inflicted artifact of using a highly naive, unclipped symmetric quantizer.
*   **Decision:** **REJECT** (Mixed/negative reviews with a hard Reject. The core problem is highly localized to an unrepresentative quantization setup, and the proposed mitigations are ineffective or redundant).

---

### Submission 4: "SpectralMerge: Frequency-Domain Regularization for Weight-Space Model Merging"
*   **Consensus Ratings:**
    *   Reviewer 1: **3: Weak Reject** (Soundness: Fair, Presentation: Good, Significance: Fair, Originality: Fair)
    *   Reviewer 2: **3: Weak Reject** (Soundness: Fair, Presentation: Good, Significance: Fair, Originality: Fair)
    *   Reviewer 3: **5: Accept** (Soundness: Good, Presentation: Excellent, Significance: Good, Originality: Excellent)
*   **Average Score:** 3.67 / 5 (Weak Reject / Mixed)
*   **Strengths:** Elegant signal-processing analogy regularizing layer trajectories; mathematical analysis regarding boundary conditions and even-symmetry derivatives is solid.
*   **Weaknesses:**
    1. *Self-Inflicted Empirical Collapse:* The "PEFT-Induced Step-Function Discontinuity" that causes SpectralMerge-LP to collapse to 29.00% on ResNet-18 is a self-inflicted artifact of optimizing merging coefficients on 13 completely frozen layers where task vectors are zero.
    2. *Complexity with Low Return:* A simple DC-only baseline outperforms SpectralMerge under Online TTA, making the complex DCT-based optimization framework mathematically redundant.
    3. *Toy scale:* Evaluated on binary CIFAR splits with weak experts and no error bars on tiny validation sets ($M \in [10, 15]$).
*   **Decision:** **REJECT** (Mixed reviews with negative lean; the mathematical complexity is not justified by empirical performance compared to simpler baselines).

---

### Submission 5: "SG-TA: Sigmoid-Gated Soft Masking for Spatial Model Merging"
*   **Consensus Ratings:**
    *   Reviewer 1: **4: Weak Accept** (Soundness: Excellent, Presentation: Excellent, Significance: Good, Originality: Good)
    *   Reviewer 2: **3: Weak Reject** (Soundness: Fair, Presentation: Excellent, Significance: Fair, Originality: Good)
    *   Reviewer 3: **3: Weak Reject** (Soundness: Fair, Presentation: Excellent, Significance: Fair, Originality: Fair)
*   **Average Score:** 3.33 / 5 (Weak Reject)
*   **Strengths:** Outstanding empirical rigor (evaluating over 5 random seeds with standard deviations); rigorous baseline comparison (including unpruned layer-wise scaling, Fisher-Weighted Averaging, and a trained Joint MTL upper bound); highly detailed ablations and honest discussion of performance limitations.
*   **Weaknesses:**
    1. *Lack of Theoretical Grounding:* The paper is predominantly heuristic and lacks rigorous theoretical grounding, mathematical proofs, or formal guarantees. Key qualitative assumptions (e.g., Orthogonal Noise Hypothesis, surrogacy to diagonal Fisher) are unproven.
    2. *Highly Incremental:* The proposed SG-TA is a minor simplification of TIES-Merging (pure magnitude pruning without sign consensus) or a straightforward global extension of Decoupled Prune-then-Merge. The resulting performance improvement over TIES-Merging ($+0.76\%$) is not statistically significant.
    3. *Toy scale:* Confined to a 5.7M parameter ViT on simple image datasets.
*   **Decision:** **REJECT** (Mixed-to-negative consensus; the paper is highly heuristic and incremental, offering statistically insignificant performance gains over existing methods).

---

### Submission 6: "Sparse Task Arithmetic: Deconstructing the Redundancy of Sign Resolution in Model Merging"
*   **Consensus Ratings:**
    *   Reviewer 1: **4: Weak Accept** (Soundness: Good, Presentation: Good, Significance: Good, Originality: Good)
    *   Reviewer 2: **4: Weak Accept** (Soundness: Good, Presentation: Excellent, Significance: Good, Originality: Good)
    *   Reviewer 3: **3: Weak Reject** (Soundness: Fair, Presentation: Excellent, Significance: Good, Originality: Good)
*   **Consensus Score:** 3.67 / 5 (Consensus Positive / Mixed)
*   **Strengths:**
    1.  *Conceptual Parsimony (Occam's Razor):* Outstanding deconstructive critique of modern over-engineered sparse model-merging techniques (TIES, DARE). It proves that coordinate-wise sign voting, dominant sign election, and zeroing out conflicting updates are entirely redundant, and that weight-space denoising (removing low-magnitude fine-tuning noise) is the primary driver of successful model merging.
    2.  *Symmetric Evaluation Protocol:* Excellent empirical rigor. The authors conduct complete hyperparameter sweeps over the scaling coefficient $\lambda \in [0.1, 1.0]$ across **all** baselines to eliminate tuning bias, setting an exemplary standard.
    3.  *Insightful Overlap Analysis:* Theoretically and empirically demonstrates that coordinate-wise mask overlap across tasks is extremely rare ($3.1\%$ to $4.3\%$ at $s=20\%$), mathematically rendering sign-voting moot for over 96% of coordinates.
    4.  *Engineering Simplicity:* Can be implemented in just three core lines of PyTorch code, drastically reducing engineering debt, serving latency, and implementation fragility in production systems.
*   **Weaknesses:**
    1.  *Mathematical Error in Expected Energy:* Reviewer 3 points out a mathematical error in Section 3.1 regarding the expected energy of sparse vectors under magnitude pruning (which assumes random pruning instead of deterministic magnitude-based tail selection). This invalidates the R-STA scaling factor but does not impact the main performant variant, **Tuned STA**.
    2.  *Literature Omission:* Missed citing "Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic" (He et al., TMLR 2024), which must be addressed by citing and contextualizing.
    3.  *Toy-Scale Vision Setup:* Evaluated on a 4-task vision classification benchmark using an 86M parameter ViT-B-32 backbone, lacking immediate LLM evaluations.
*   **Decision:** **ACCEPT** (A highly refreshing and influential deconstructive "course correction" paper. The strengths of conceptual parsimony, rigorous symmetric baseline evaluation, and massive engineering simplicity heavily outweigh the minor literature omission and localized theoretical R-STA scaling error).

---

### Submission 7: "SuiteMerge: Systematic Methodological Audit of Modern Adaptive Model Merging"
*   **Consensus Ratings:**
    *   Reviewer 2: **5: Accept** (Soundness: Excellent, Presentation: Excellent, Significance: Good, Originality: Good)
    *   Reviewer 3: **5: Accept** (Soundness: Excellent, Presentation: Excellent, Significance: Excellent, Originality: Excellent)
*   **Consensus Score:** 5.00 / 5 (Unanimous Accept)
*   **Strengths:**
    1.  *Critical Discovery of Task Suite Bias:* Exposes a major and previously unreported benchmark bias in model merging, where algorithms are validated on exactly one combination of four simple datasets, masking critical local failures.
    2.  *Rigorous Evaluation & Transparency:* Evaluates five distinct multi-task suites of varying domain distances across **30 independent random seeds**.
    3.  *Offline Few-Shot Validation Tuning (OFS-Tune):* Proposes a simple, regularized offline alternative using polynomial trajectory constraints that consistently matches or exceeds online Test-Time Adaptation (TTA) methods while completely bypassing on-device backpropagation latency, stream non-stationarity, and the "privilege trap" of needing oracle task labels.
    4.  *Physical Validation:* Demonstrates and verifies the simulated results on an independent physical weight-space CNN.
*   **Weaknesses:** Physical weight-space validation is restricted to a small-scale CNN. Nelder-Mead solver computational scaling is noted as a potential bottleneck for multi-billion parameter LLMs, though the authors propose coordinate gradients (OFS-Adam) to address this.
*   **Decision:** **ACCEPT** (An exceptionally strong, methodologically flawless, and highly practical paper. It exposes major benchmark biases, provides a clean and robust alternative, and sets a new standard for statistical rigor in the field).

---

### Submission 8: "CR-PolySACM: Clipping-Regularized Sharpness-Aware Subspace Model Merging"
*   **Consensus Ratings:**
    *   Reviewer 1: **3: Weak Reject** (Soundness: Fair, Presentation: Excellent, Significance: Fair, Originality: Good)
    *   Reviewer 2: **3: Weak Reject** (Soundness: Good, Presentation: Excellent, Significance: Fair, Originality: Good)
    *   Reviewer 3: **6: Strong Accept** (Soundness: Excellent, Presentation: Excellent, Significance: Excellent, Originality: Excellent)
*   **Average Score:** 4.00 / 5 (Weak Reject / Mixed)
*   **Strengths:** Exceptionally rigorous mathematical derivations and quadratic noise decomposition; continuous polynomial subspace parameterization (PolyMerge component) is elegant, simple, and effective; outstanding scientific candor and transparency regarding limitations.
*   **Weaknesses:**
    1.  *Negative Utility of Primary Contribution:* While the baseline PolyMerge is highly effective, the primary proposed contribution—the addition of the **CR-SACM flatness optimization loop**—introduces significant algorithmic complexity (requiring double the forward-backward passes, clipping thresholds, and layer-wise norm calculations) but **consistently degrades performance** across all practical deployment formats (FP32 and all four INT8 formats) compared to the simpler PolyMerge baseline.
    2.  *Artificial Creation of Scale Pathology:* The "task-vector norm scale pathology" that CR-SACM is designed to solve is an artifact of the authors' choice to perturb intermediate layer-wise coefficients instead of perturbing the actual 12 polynomial coefficients directly. Perturbing the actual optimization variables directly via standard parameter-space SAM would have completely bypassed the scale pathology, a baseline the authors omit.
    3.  *Unviable Regime gains:* CR-SACM only provides a marginal $+0.97\%$ benefit under extreme, broken INT4 quantization, where absolute accuracy is too poor (19.07%) to be practically usable.
*   **Decision:** **REJECT** (Despite an extremely enthusiastic champion rating, the consensus is negative because the primary proposed technical contribution is highly complex, over-engineered, and actively degrades representation quality in all standard deployment formats).

---

### Submission 9: "Exclusive Parameter Merging: Coherence-Preserved Multi-Task Model Fusion"
*   **Consensus Ratings:**
    *   Reviewer 1: **3: Weak Reject** (Soundness: Fair, Presentation: Fair, Significance: Fair, Originality: Fair)
    *   Reviewer 2: **2: Reject** (Soundness: Poor, Presentation: Excellent, Significance: Poor, Originality: Fair)
    *   Reviewer 3: **2: Reject** (Soundness: Poor, Presentation: Excellent, Significance: Poor, Originality: Fair)
*   **Average Score:** 2.33 / 5 (Strong Reject)
*   **Strengths:** Outstanding writing, organization, and mathematical clarity; exemplary scientific honesty and transparency regarding the absolute performance gap; extensive sensitivity analyses.
*   **Weaknesses:**
    1.  *Fundamental Mathematical Scale Mismatch:* Soft-EPA makes routing decisions in standardized task vector space but applies updates in unstandardized space. This inconsistent scale coupling allows low-variance tasks (like MNIST) to dominate coordinates of high-variance tasks (like SVHN), causing SVHN/CIFAR-10 accuracy to collapse catastrophically (CIFAR-10 collapses from 75.83% to 36.98%).
    2.  *Destructive Topography Scrambling:* Routing coordinates independently at the individual scalar level destroys the joint covariance structure and representation manifolds of dense neural network matrices.
    3.  *Biased Evaluation Protocol:* Restricts continuous baselines to a zero-order (1+1)-ES optimizer mismatch, forcing them to fail.
    4.  *Lack of Scale Preservation:* Fails to scale active coordinates under high sparsity ($p=0.8$), leading to activation magnitude decay and complete performance collapse compared to DARE.
*   **Decision:** **REJECT** (Unanimous negative consensus due to mathematically inconsistent scale mismatches, topography scrambling, and biased baseline evaluations).

---

### Submission 10: "Quantum Wavefunction Superposition Merging (QWS-Merge)"
*   **Consensus Ratings:**
    *   Reviewer 1: **5: Accept** (Soundness: Good, Presentation: Excellent, Significance: Good, Originality: Excellent)
    *   Reviewer 2: **3: Weak Reject** (Soundness: Fair, Presentation: Excellent, Significance: Fair, Originality: Good)
    *   Reviewer 3: **5: Accept** (Soundness: Good, Presentation: Excellent, Significance: Excellent, Originality: Excellent)
*   **Consensus Score:** 4.33 / 5 (Strong Consensus Accept)
*   **Strengths:**
    1.  *Outstanding Conceptual Originality:* Challenges the traditional static weight-averaging paradigm of model merging. Instead of interpolating static coordinates, it models parameters as continuous, complex-valued wavefunctions and applies a physics-grounded quantum wave-coherence ensembling formulation.
    2.  *Strong Empirical Regularization:* Demonstrates exceptional regularization properties that prevent the parameter-space collapse and transductive overfitting of continuous routing baselines under high multi-task domain conflict.
    3.  *Excellent Writing and Clarity:* The mathematical derivations of wavefunction scaling and coherence tracking are elegant, clean, and highly structured.
    4.  *Scientific Honesty:* Transparent and scientifically candid in documenting and analyzing its own limitations (specifically the batch-dependent inference cost and the low-sample calibration variance).
*   **Weaknesses:**
    *   Requires batch-dependent inference computations in its full formulation, introducing runtime latency.
    *   Lacks extensive error bars on its tiny 64-sample calibration set.
*   **Decision:** **ACCEPT** (A highly innovative, conceptually ambitious, and original paper that brings a fresh wave-coherence perspective to model ensembling. The high novelty, theoretical depth, and strong empirical regularization make it a standout candidate).

---

## 3. Comparative Synthesis and Decision Rationale

To accept exactly **3 out of 10 submissions**, we conducted a comparative analysis of the leading candidates:

| Submission ID | Title / Topic | Recommendation Profile | Average Score | Decision | Key Comparative Rationale |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **7** | **SuiteMerge** | Accept, Accept | **5.00 / 5** | **ACCEPT** | Unanimous top-rated paper. Exposes massive benchmark biases and proposes a robust, highly practical offline polynomial tuning method (OFS-Tune) with zero runtime latency. |
| **10** | **QWS-Merge** | Accept, Weak Reject, Accept | **4.33 / 5** | **ACCEPT** | Strong consensus positive. Exceptionally novel quantum wavefunction/wave-coherence ensembling formulation that provides powerful representation regularization. |
| **6** | **Sparse Task Arithmetic (STA)** | Weak Accept, Weak Accept, Weak Reject | **3.67 / 5** | **ACCEPT** | Consensus positive. Elegant application of Occam's Razor, proving complex sign-consensus heuristics are redundant. Provides massive engineering simplicity (3 lines of PyTorch). Wins over Submission 8 due to positive utility. |
| **8** | **CR-PolySACM** | Weak Reject, Weak Reject, Strong Accept | **4.00 / 5** | **REJECT** | Despite a 6 (Strong Accept) champion, the primary contribution (CR-SACM) is highly complex and *actively degrades* performance in all usable formats (FP32/INT8) compared to its baseline. |
| **4** | **SpectralMerge** | Weak Reject, Weak Reject, Accept | **3.67 / 5** | **REJECT** | Proposes frequency trajectory constraints but suffers from self-inflicted optimization collapse and is outperformed by a simple DC baseline. |

### Why Submission 6 (STA) was selected over Submission 8 (CR-PolySACM):
*   **Performative Utility:** In Submission 8, the proposed CR-SACM loop consistently *decreases* performance on all standard formats (FP32, INT8) and only helps under INT4, which is non-functional anyway (19% accuracy). In contrast, Submission 6's Tuned STA matches or outperforms complex state-of-the-art baselines like TIES and DARE.
*   **Complexity vs. Simplicity:** Submission 8 introduces significant optimization complexity (multi-step gradients, norm clipping, and boundary clamping) to solve a self-inflicted scale pathology. Submission 6 does the exact opposite: it simplifies the ensembling process, replacing highly complex sign-voting pipelines with a clean 3-line magnitude-pruning loop.
*   **Broad Consensus:** Submission 6 represents a broad, positive consensus (two Weak Accepts and one Weak Reject), whereas Submission 8 suffers from a severe divergence (two Weak Rejects pointing out fatal flaws of over-engineering, and one highly positive champion). Broad consensus is preferred under conference guidelines when the criticisms of over-engineering remain un-addressed.

---

## 4. Conclusion

The accepted cohort represents three outstanding and highly complementary contributions to the model-merging literature:
1.  **Submission 7 (SuiteMerge)** establishes essential benchmarking standards by auditing task suite bias and introducing a robust, offline polynomial tuning paradigm (OFS-Tune).
2.  **Submission 10 (QWS-Merge)** explores the conceptual frontier of weight space, using quantum wave mechanics to regularize multi-task representations.
3.  **Submission 6 (STA)** acts as a vital methodological correction, trimming away redundant sign-consensus heuristics and establishing a minimalist, magnitude-pruned baseline for future research.

These papers are fully documented, and their complete `submission/` directories have been successfully preserved under the `accepted_papers/` workspace directory.
