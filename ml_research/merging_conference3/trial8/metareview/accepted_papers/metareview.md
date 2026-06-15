# Meta-Review Report: Model Merging and Edge Serving Conference (MMES 2026)

## 1. Overview of the Meta-Review Process
This report summarizes the meta-review process and final decisions for 10 paper submissions evaluated for MMES 2026. Each submission was audited through multiple peer reviews detailing strengths, weaknesses, technical soundness, originality, significance, presentation quality, and scholarly integrity. 

Our mandate was to select **exactly three** out of the ten submissions for acceptance. 

To ensure a highly rigorous, objective, and impact-maximizing selection, our meta-reviewing process followed these systematic steps:
1. **Compilation of Reviewer Scores and Ratings:** Extracted and analyzed all reviewers' numerical recommendations and soundness ratings across all 28 available reviews.
2. **Deconstruction of Qualitative Merits:** Read the detailed feedback to evaluate the core mathematical models, experimental methodologies, real-world applicability, and academic rigor.
3. **Audit of Empirical and Scholarly Integrity:** Checked for major scholarly reporting flaws, missing baseline citations, contradictory quantitative reporting, overblown claims, and toy-only evaluation limits.
4. **Outcome Ranking and Selection Justification:** Ranked all papers by average rating and qualitative soundness, then conducted a comparative analysis to decide on the top three accepted submissions and justify the rejection of other high-scoring papers.

---

## 2. Summary and Tally of the 10 Submissions

### Submission 1: HyperMerge (Hyperbolic Space Activation Routing and Fusion)
* **Core Concept:** Proposes a non-Euclidean framework (HyperMerge) that projects Low-Rank Adaptation (LoRA) activation-space updates into hyperbolic space (Poincaré Ball/Klein space) to resolve "representation crowding" in flat Euclidean spaces, using Beltrami-Klein Einstein midpoints to resolve Möbius ordering bias.
* **Reviewer Ratings:**
  * Reviewer 1: **3: Weak Reject** (Soundness: Fair)
  * Reviewer 2: **3: Weak Reject** (Soundness: Fair)
  * Reviewer 3: **2: Reject** (Soundness: Fair)
  * **Average Score: 2.67 / 6.00**
* **Synthesis of Feedback:**
  * *Strengths:* Mathematically elegant and sophisticated algebraic formulation; beautiful writing and visual presentation.
  * *Weaknesses:* Exhibits a fundamental conceptual contradiction. LoRA updates are inherently small-norm displacement vectors ($\|E\|_2 \ll 1$), meaning they map near the origin where hyperbolic space is locally flat (Euclidean), making the NEG volume growth near the boundary inactive. Mathematically, it operates in a virtually linear regime, which explains why the simple flat Euclidean baseline (SABLE) consistently **outperforms** HyperMerge in both standard (84.03% vs 83.40%) and crowded/overlapping (77.98% vs 76.62%) sandbox regimes. Additionally, severe numerical inconsistencies were identified between Table 1 and Table 2, and the evaluation is limited to a synthetic sandbox.
* **Meta-Review Decision:** **Reject**

---

### Submission 2: Q-SPS (Quantized Single-Pass Activation-Space Dynamic Blending)
* **Core Concept:** Addresses edge deployment of PEFT adapters on resource-constrained CPUs. Quantizes expert LoRAs to low-bit integers (INT8/INT4), executes additions in pure integer precision within a single parallel forward pass, introduces post-hoc Quantization-Aware Scale Calibration (QASC) to prevent degradation, and applies conditional gating (CG-Q-SPS) to bypass low-weight experts.
* **Reviewer Ratings:**
  * Reviewer 1: **5: Accept** (Soundness: Good)
  * Reviewer 2: **6: Strong Accept** (Soundness: Excellent)
  * Reviewer 3: **3: Weak Reject** (Soundness: Good)
  * **Average Score: 4.67 / 6.00**
* **Synthesis of Feedback:**
  * *Strengths:* Exemplary hardware-aware systems-ML co-design (exploits physical register limits, cache structures, and Neon intrinsics); outstanding intellectual honesty and theoretical de-entangling analysis (proving explicit basis orthogonalization is mathematically redundant/detrimental under noise due to noise spillover); slashes memory footprints by 87.5% and achieves a 3.97$\times$ speedup while recovering 99.5% unquantized accuracy.
  * *Weaknesses:* Evaluation is simulated (inside the Isolated Coordinate Sandbox) rather than deployed on a physical edge CPU, though the simulation is highly high-fidelity and calibrated against Broadcom specifications.
* **Meta-Review Decision:** **Accept** (Top 3 Selection)

---

### Submission 3: SA-QAB (Quantized Scale-Aware Activation Blending)
* **Core Concept:** Investigates fixed-point microcontroller deployment of low-rank activations, using integer-only approximations and fixed-point bit-shifting to bypass expensive operations, coupled with post-merge scale recovery factors.
* **Reviewer Ratings:**
  * Reviewer 1: **3: Weak Reject** (Soundness: Not found easily / Fair)
  * Reviewer 2: **3: Weak Reject** (Soundness: Fair)
  * Reviewer 3: **5: Accept** (Soundness: Good)
  * **Average Score: 3.67 / 6.00**
* **Synthesis of Feedback:**
  * *Strengths:* Solid engineering adaptation for physical microcontrollers; good empirical results in recovering from representation collapse.
  * *Weaknesses:* Logical contradictions in advertising the framework as calibration-free when it relies on clean FP16 calibration streams; limited theoretical novelty compared to existing quantization methods.
* **Meta-Review Decision:** **Reject**

---

### Submission 4: PAC-ZCA (Certifiable Dynamic Model Merging)
* **Core Concept:** Bridges statistical learning theory and modular deep learning by deriving PAC-Bayes generalization bounds for dynamic model ensembling, establishing a "certifiable serving" paradigm with provable out-of-sample risk guarantees. Introduces spherical feature normalization to resolve high-dimensional SVD overfitting.
* **Reviewer Ratings:**
  * Reviewer 1: **3: Weak Reject** (Soundness: Fair)
  * Reviewer 2: **6: Strong Accept** (Soundness: Excellent)
  * Reviewer 3: **5: Accept** (Soundness: Excellent)
  * **Average Score: 4.67 / 6.00**
* **Synthesis of Feedback:**
  * *Strengths:* Outstanding theoretical foundation bringing PAC-Bayes generalization and risk bounds to active model merging; exceptional empirical rigor evaluating over 5 random seeds with standard deviations, comparing against 8 comprehensive baselines; excellent disclosure of the SVD overfitting collapse.
  * *Weaknesses:* Minor criticisms regarding over-engineering from Reviewer 1, though the other reviewers strongly praise its mathematical rigor and completeness.
* **Meta-Review Decision:** **Accept** (Top 3 Selection)

---

### Submission 5: PEAR (Single-Pass Layer Adaptive PEFT)
* **Core Concept:** Proposes a minimalist dynamic activation ensembling framework that achieves state-of-the-art performance with zero trainable parameters and flat $O(1)$ latency, introducing the Early-Layer Routing Compromise and ELFT to resolve the Color Routing Paradox.
* **Reviewer Ratings:**
  * Reviewer 1: **2: Reject** (Soundness: Not found / Fair)
  * Reviewer 2: **5: Accept** (Soundness: Excellent)
  * Reviewer 3: **3: Weak Reject** (Soundness: Fair)
  * **Average Score: 3.33 / 6.00**
* **Synthesis of Feedback:**
  * *Strengths:* Highly practical systems-aligned perspective; double-tier evaluation combining a controlled synthetic sandbox with end-to-end real ViT pipeline on images.
  * *Weaknesses:* Suffers from a fundamental logical contradiction (the Early-Layer Routing Trilemma); statistically under-powered real-world evaluation on very narrow task sets.
* **Meta-Review Decision:** **Reject**

---

### Submission 6: LSPR (Closed-Form Linear Algebra Routing / Subspace Projection Routing)
* **Core Concept:** Proposes a dynamic ensembling method based on closed-form linear algebra (QR decomposition and orthogonal projection) rather than complex statistical density pipelines, accompanied by Split-Rank LoRA and Warm Alignment.
* **Reviewer Ratings:**
  * Reviewer 1: **6: Strong Accept** (Soundness: Excellent)
  * Reviewer 2: **4: Weak Accept** (Soundness: Good)
  * Reviewer 3: **3: Weak Reject** (Soundness: Fair)
  * **Average Score: 4.33 / 6.00**
* **Synthesis of Feedback:**
  * *Strengths:* Outstanding conceptual elegance advocating for Occam's razor; mathematically clean and transparent.
  * *Weaknesses:* Major empirical validation gap. The entire framework is evaluated strictly within a low-dimensional, synthetic, single-layer sandbox with zero real-world datasets or models. It lacks statistical reporting (no error bars, standard deviations, or seeds), and shows a massive performance drop (~20% absolute regression) under Post-Hoc Warm Alignment.
* **Meta-Review Decision:** **Reject** (Outperformed by Submissions 2, 4, and 7)

---

### Submission 7: ChemMerge (Biochemistry Reaction Cascades for Routing)
* **Core Concept:** Models representation flow through neural layers as a continuous-time multi-component chemical reactor governed by non-equilibrium reaction kinetics. Derives step-size bounds for explicit Euler updates, implements an exact analytical Exponential Integrator solver to guarantee stable concentrations in $[0, 1]$, and proves continuous-time global convergence.
* **Reviewer Ratings:**
  * Reviewer 1: **6: Strong Accept** (Soundness: Excellent)
  * Reviewer 2: **4: Weak Accept** (Soundness: Good)
  * Reviewer 3: **5: Accept** (Soundness: Excellent)
  * **Average Score: 5.00 / 6.00**
* **Synthesis of Feedback:**
  * *Strengths:* Outstanding, paradigm-shifting originality that rethinks routing as a continuous physical process; rigorous and technically flawless mathematical proofs; proves a profound DSP-biochemistry duality showing the continuous equations are equivalent to an adaptive EMA filter; exemplary scientific transparency (prominent disclosure box clarifying simulated vs. routing-only environments).
  * *Weaknesses:* Minor theoretical inconsistencies regarding concentration conservation and operating under an oscillatory discretization regime under default step size, which are easily correctable.
* **Meta-Review Decision:** **Accept** (Top 3 Selection)

---

### Submission 8: SRC-DE (Diagonal GMM Covariance Shrinkage)
* **Core Concept:** Proposes applying Ledoit-Wolf covariance shrinkage to diagonal GMM parameters using soft EM responsibilities to improve coordinate-space density routing.
* **Reviewer Ratings:**
  * Reviewer 1: **5: Accept** (Soundness: Excellent)
  * Reviewer 2: **3: Weak Reject** (Soundness: Excellent)
  * **Average Score: 4.00 / 6.00** (Only 2 reviewers)
* **Synthesis of Feedback:**
  * *Strengths:* Flawless empirical audits, meticulous statistical significance testing, and deep systems profiling on microcontrollers.
  * *Weaknesses:* Limited conceptual novelty (applying classical Ledoit-Wolf shrinkage to GMM parameters is highly incremental). More critically, the empirical results expose structural fragility in the joint coordinate GMM density routing paradigm: it is consistently and massively outperformed by simpler, zero-overhead 1D similarity thresholding (Raw Cosine) under scale.
* **Meta-Review Decision:** **Reject**

---

### Submission 9: Centroid-Gated Entropy Routing (CG-EER)
* **Core Concept:** Explores completely calibration-free, zero-shot test-time model merging and online centroid tracking.
* **Reviewer Ratings:**
  * Reviewer 1: **3: Weak Reject** (Soundness: Fair)
  * Reviewer 2: **5: Accept** (Soundness: Excellent)
  * Reviewer 3: **3: Weak Reject** (Soundness: Poor/Fair)
  * **Average Score: 3.67 / 6.00**
* **Synthesis of Feedback:**
  * *Strengths:* Exceptional academic honesty in disclosing and analyzing failure cases; valuable systems-level profiling and latency mitigation (Amortized Pseudo-Labeling).
  * *Weaknesses:* Fully unsupervised methods (EPL-OCA and EER) completely collapse on real ResNet-18 features due to uncalibrated OOD overconfidence (Entropy Calibration Discrepancy) and self-referential corruption. The only viable pipeline (CG-EER) is semi-supervised and requires pre-computed offline centroids, violating and invalidating the paper's core calibration-free thesis. It lacks mathematical proofs and convergence analyses.
* **Meta-Review Decision:** **Reject**

---

### Submission 10: ESM-LVC (Self-Organizing Lotka-Volterra Ecosystem)
* **Core Concept:** Models model ensembling as a self-organizing dynamic biological ecosystem, implementing Lotka-Volterra competition-cooperation differential equations directly into the forward pass of a transformer block.
* **Reviewer Ratings:**
  * Reviewer 2: **5: Accept** (Soundness: Excellent)
  * Reviewer 3: **3: Weak Reject** (Soundness: Good)
  * **Average Score: 4.00 / 6.00** (Only 2 reviewers)
* **Synthesis of Feedback:**
  * *Strengths:* Highly creative conceptual bridge to mathematical ecology; rigorous trajectory boundedness proofs (Theorem 3.1); physical model verification on actual ViT tokens.
  * *Weaknesses:* Severe scholarly reporting deficiency. The paper compares its method extensively against SABLE and SPS-ZCA in almost every table and figure, yet **completely omits any bibliographic entry or citation** for these two core baselines. This major scholarly omission makes it impossible for peers to verify the baseline definitions, trace their origins, or validate the paper's claims.
* **Meta-Review Decision:** **Reject** (Scholarly integrity issues and lower ranking compared to accepted papers)

---

## 3. Final Decisions and Selection Justification

The three papers selected for acceptance are **Submission 7 (ChemMerge)**, **Submission 2 (Q-SPS)**, and **Submission 4 (PAC-ZCA)**. 

### Why These Three Papers Were Accepted:
1. **Submission 7 (ChemMerge) [Avg: 5.00]:** This is the highest-rated paper of the conference. It represents an incredibly creative, paradigm-shifting leap that moves away from standard, memory-less sequence routing. It is mathematically complete, provides rigorous continuous-to-discrete stability proofs, and sets a stellar benchmark for scientific honesty and reporting.
2. **Submission 2 (Q-SPS) [Avg: 4.67]:** This paper represents a masterclass in hardware-systems and machine learning co-design. It provides a massive, practical 87.5% memory footprint reduction and 3.97$\times$ speedup for serving multiple LoRA experts on low-power CPUs. Its theoretical de-entangling analysis is highly elegant, and its simulation environment is rigorously calibrated.
3. **Submission 4 (PAC-ZCA) [Avg: 4.67]:** This submission stands out for its exceptional mathematical rigor. By deriving generalization bounds using PAC-Bayes theory for active model merging, it provides provable, out-of-sample risk guarantees ("certifiable serving"). This theoretical depth is paired with a flawless empirical protocol (evaluating 5 seeds across 8 baselines with standard deviations).

### Justification for Rejecting Sibling Contenders:
* **Why not Submission 6 (LSPR) [Avg: 4.33]?:** Although LSPR advocates for a highly elegant return to first-principles linear algebra (Occam's razor), it is severely limited by its empirical validation. Unlike Submissions 2, 4, and 7 (which evaluate or calibrate on real-world networks/manifolds), LSPR is evaluated **entirely** inside a low-dimensional, toy synthetic single-layer sandbox. It also suffers from a catastrophic 20% performance regression in post-hoc warm alignment and completely lacks basic statistical reporting (error bars, seeds).
* **Why not Submission 10 (ESM-LVC) [Avg: 4.00]?:** ESM-LVC represents an outstanding, highly original concept. However, as a matter of academic integrity and scholarly completeness, it cannot be accepted. The authors extensively compare their work against two core baselines (SABLE, SPS-ZCA) across almost every figure and table, yet completely omit any bibliographic reference or citation for them. This severe omission prevents peers from verifying the baseline definitions, which violates fundamental scientific publishing standards.
* **Why not Submission 8 (SRC-DE) [Avg: 4.00]?:** While the empirical audits and paired t-tests are flawless, the core contribution is a straightforward application of classical Ledoit-Wolf shrinkage to diagonal GMM parameters. Furthermore, its empirical findings expose a structural fragility in the entire joint GMM density routing paradigm under scale, showing it is consistently outperformed by simple, zero-overhead Raw Cosine checks.

In conclusion, Submissions 2, 4, and 7 represent the absolute pinnacle of originality, mathematical rigor, empirical validation, and potential for community impact, making them the clear selection for MMES 2026.
