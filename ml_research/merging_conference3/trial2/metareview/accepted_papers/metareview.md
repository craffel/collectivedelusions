# Conference Meta-Review and Selection Report

This report outlines the meta-review process and decision rationales for 10 submissions evaluated under a test-time model merging and network compression paradigm using a pre-trained CLIP ViT-B/32 or ViT-Tiny visual backbone. Based on the rigorous evaluation of both the quantitative scores and the qualitative contents of the peer reviews, three submissions have been selected for acceptance.

---

## 1. Process Overview and Selection Criteria

The selection process was conducted with rigorous attention to both raw scores and, more importantly, the technical soundness, empirical depth, and practical significance of the submissions as documented by the peer reviewers. Each submission was evaluated by three independent reviewers. 

### Selection Criteria:
1. **Technical Soundness and Integrity:** Mathematical correctness, logical consistency, and empirical support for central claims.
2. **Empirical Rigor:** The scale of evaluations, inclusion of baseline comparisons, reporting of statistical significance (standard deviations across multiple seeds), and thoroughness of ablation studies.
3. **Conceptual Novelty and Insights:** The ability to move the community forward by exposing fundamental phenomena, providing deconstructive insights, or introducing innovative diagnostic tools.
4. **Practical Significance and Utility:** Real-world deployment viability, execution costs (latency, memory footprint during calibration), and trade-offs between performance and complexity.

---

## 2. Summary of Submissions and Decisions

Below is the summary of all 10 submissions, including their individual reviewer recommendations, average scores, and final decisions. Recommendations are mapped to numerical scores for objective aggregation:
* **Strong Accept:** 6
* **Accept:** 5
* **Weak Accept:** 4
* **Weak Reject:** 3
* **Reject:** 2
* **Strong Reject:** 1

| Submission | Paper Title | R1 | R2 | R3 | Average Score | Decision |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **RegCalMerge: Calibration-Aware Test-Time Model Merging** | 4 | 4 | 4 | **4.00** | **Accept** |
| **2** | Spectral Model Merging via Singular Value Slicing (SVS) | 3 | 2 | 5 | **3.33** | Reject |
| **3** | **PolyMerge / SplineMerge: Robust Constrained TTA Merging** | 5 | 3 | 5 | **4.33** | **Accept** |
| **4** | Activation-Guided Channel-Wise Gating | 3 | 2 | 5 | **3.33** | Reject |
| **5** | Norm-Equalized Task Arithmetic (NETA) | 4 | 4 | 4 | **4.00** | Reject |
| **6** | **Quantization-Aware Model Merging (Q-Merge)** | 4 | 5 | 5 | **4.67** | **Accept** |
| **7** | ThermoMerge: Temperature-Based Model Merging | 3 | 2 | 2 | **2.33** | Reject |
| **8** | Norm-Preserved Budgeted Task-Vector Pruning (NP-BTVP) | 5 | 3 | 3 | **3.67** | Reject |
| **9** | Barycentric Proximity-Anchored Merging (BPAM) | 4 | 3 | 5 | **4.00** | Reject |
| **10** | Layer-Wise AdaMerging and the Spatial Averaging Paradox | 2 | 5 | 3 | **3.33** | Reject |

---

## 3. Detailed Rationales for Accepted Submissions

### Submission 6: Quantization-Aware Model Merging (Q-Merge)
* **Average Score:** 4.67 (R1: 4 - Weak Accept, R2: 5 - Accept, R3: 5 - Accept)
* **Rationale for Acceptance:**
  Q-Merge is the highest-rated submission in this cohort and represents an exceptionally strong engineering contribution to edge-device model deployment. It is the first framework to formulate and solve model merging directly under the quantization operator, optimizing layer-wise merging coefficients under non-differentiable 8-bit (INT8) and 4-bit (INT4) representations.
  * **Strengths:** 
    1. *High Pragmatic Utility:* Extremely lightweight, converging in seconds. Blending coefficients are discarded post-calibration, resulting in zero latency or parameter overhead during inference.
    2. *Empirical Excellence:* Achieves nearly lossless 8-bit merging and highly viable 4-bit merging. Under 8-bit, Q-Merge (74.30%) even outperforms the unquantized uniform baseline (71.88%).
    3. *Exemplary Controls:* The authors isolated the optimizer confounding factor by comparing zero-order (1+1 ES) and first-order (Adam with STE) methods, proving the clear superiority and 2.7x lower seed-to-seed variance of the STE-based gradient pipeline.
  * **Weaknesses:** Conceptual novelty is slightly incremental (synthesizing STE and AdaMerging), and evaluations are restricted to a toy-scale visual backbone (ViT-Tiny, 5.7M parameters). However, the systems-level thoroughness (dynamic activation quantization, scale discretization sensitivity, integration with AdaRound) outweighs these limitations.

### Submission 3: PolyMerge / SplineMerge: Robust Constrained TTA Merging
* **Average Score:** 4.33 (R1: 5 - Accept, R2: 3 - Weak Reject, R3: 5 - Accept)
* **Rationale for Acceptance:**
  This paper addresses a fundamental vulnerability in test-time adaptive model merging: the **Overfitting-Optimizer Paradox**, where unconstrained layer-wise optimization of coefficients on small, unlabeled target streams leads to transductive overfitting and jagged, physically unrealistic trajectories. The paper proposes continuous polynomial (PolyMerge) and spline (SplineMerge) constraints to project the optimization into a low-dimensional subspace.
  * **Strengths:**
    1. *Exceptional Empirical Rigor:* Simulations are executed across **30 independent random seeds** with detailed standard deviations and paired t-tests confirming high statistical significance ($p < 10^{-12}$).
    2. *Diagnostic and Theoretical Depth:* Exposes transductive overfitting elegantly and provides rigorous mathematical proofs linking the subspace constraints to flatter local loss basins and robust generalization.
    3. *Discrete Tuning Advantage:* Replaces continuous hyperparameter tuning (impossible at test time due to the lack of labeled validation sets) with a robust discrete architectural selection (polynomial degree $d$).
  * **Weaknesses:** Global polynomials suffer from an underfitting bottleneck on pre-trained CLIP weights due to highly heterogeneous layer sensitivities. However, the authors successfully mitigated this by introducing SplineMerge (Piecewise Constant), which recovers performance to 96.00% accuracy. The paper's conceptual contribution is profound and highly durable.

### Submission 1: RegCalMerge: Calibration-Aware Test-Time Model Merging
* **Average Score:** 4.00 (R1: 4 - Weak Accept, R2: 4 - Weak Accept, R3: 4 - Weak Accept)
* **Rationale for Acceptance:**
  RegCalMerge is selected as our third accepted paper over other submissions with a 4.00 average (Submissions 5 and 9) due to its exceptional diagnostic contributions, scientific rigor, and superior empirical performance. The paper targets two critical adaptive model-merging anomalies: transductive overfitting (Overfitting-Optimizer Paradox) and **Sacrificial Task Bias** (where simple tasks dominate joint gradients, degrading complex tasks like SVHN). It introduces CalMerge (using Class-Capacity Normalization and Scale-Normalized Entropy Weighting) and Elastic Spatial Regularization (ESR).
  * **Strengths:**
    1. *Exposing Production Vulnerabilities:* Exposing Sacrificial Task Bias is of high practical significance, providing a crucial warning to practitioners against blindly minimizing joint prediction entropy.
    2. *Outstanding SOTA Performance:* CalMerge achieves state-of-the-art Joint Mean accuracy (61.82%) and elevates the hardest task (SVHN) to 32.03%—the highest recorded performance in the literature.
    3. *Innovative Diagnostics:* The introduction of the "spatial shuffling diagnostic" (shuffling optimized layer-wise coefficients) is an exceptionally creative and convincing empirical check that proves layer-wise parameter drift behavior.
  * **Weaknesses:** Evaluations are limited to smaller visual datasets (MNIST, CIFAR-10, SVHN). However, its clean mathematical formulation, dense 2D hyperparameter sweeps ($\beta \times \gamma$), and immense scientific honesty make it a highly valuable and complete contribution.

---

## 4. Detailed Rationales for Rejected Submissions (High-Score Contenders)

### Submission 5: Norm-Equalized Task Arithmetic (NETA)
* **Average Score:** 4.00 (R1: 4 - Weak Accept, R2: 4 - Weak Accept, R3: 4 - Weak Accept)
* **Rationale for Rejection:**
  While NETA is a clean, training-free, and mathematically sound approach designed to solve "task dominance" by equalizing layer-wise Frobenius norms, it has severe limitations that make it strictly inferior to standard baselines:
  1. *Underperforms Simple Baselines:* NETA's overall multi-task average accuracy (87.17%) actually **underperforms standard, zero-compute Task Arithmetic (87.76%) and DARE (87.78%)**.
  2. *Severe Degradation on Hard Tasks:* On the most challenging dataset (SVHN), NETA's performance drops significantly by **-3.12%** compared to standard Task Arithmetic (from 80.14% to 77.02%). In real-world, high-stakes deployments, degrading capabilities on the hardest domain to achieve artificial "fairness" across simpler, already high-performing tasks is unacceptable.
  3. *Limited Practical Significance:* Layer-wise AdaMerging outperforms NETA by a massive $+3.72\%$ absolute margin (90.89% vs 87.17%). A practitioner would easily favor AdaMerging's performance gains over NETA.
  4. *Heuristics and Generalizability:* It relies on manual, architecture-specific grouping heuristics (e.g., grouping positional/class embeddings with the first Transformer block) to prevent positional distortions, limiting its automated generalizability.
  
  Consequently, NETA is declined in favor of RegCalMerge (Submission 1), which actively improves performance (SOTA Joint Mean and highest SVHN accuracy) and provides much deeper diagnostic insights.

### Submission 9: Barycentric Proximity-Anchored Merging (BPAM)
* **Average Score:** 4.00 (R1: 4 - Weak Accept, R2: 3 - Weak Reject, R3: 5 - Accept)
* **Rationale for Rejection:**
  BPAM is a deconstructive, parameter-frugal adaptive merging baseline that optimizes exactly $K$ global task-wise scalars. Despite its strengths in scientific honesty, it was rejected due to several fatal methodological and practical flaws highlighted by Reviewer 2:
  1. *Fatal Missing Baseline (The Zero-Shot CLIP Baseline):* The entire paper is built around a "0-Weight Performance Mystery"—the supposed mystery of how MNIST and SVHN maintain high performance when assigned coefficients of exactly $0.0000$ and the base model is suppressed. However, the authors **completely omitted the zero-shot base CLIP ViT-B/32 accuracies**. Untouched pre-trained CLIP already achieves robust zero-shot classification on MNIST and SVHN. This "mystery" is simply the default capability of the pre-trained base model, representing a severe methodological oversight.
  2. *Proposed Constraint Hurts Performance:* BPAM's core mathematical contribution—the Convex Barycentric Simplex Projection—actually **degrades performance compared to simple Unconstrained Scaling** by up to 2.30% absolute. The authors provide absolutely no empirical evidence proving that "activation collapse" (the justification for their constraint) ever occurs.
  3. *Strictly Dominated by Static Baselines:* BPAM-Full (75.22% average accuracy, requiring 14.2 minutes of GPU calibration) is strictly outperformed by static **TIES-Merging + Head Tuning (78.50%)** which requires 0.0 minutes of runtime.
  4. *Severe Calibration Overhead:* To run, BPAM requires feeding each batch through all $K$ expert teachers concurrently, requiring **9 parallel foundation models in GPU memory** during calibration, which is highly self-defeating.
  
  Due to these major flaws, BPAM does not meet the standards for acceptance.

---

## 5. Brief Rationales for Other Rejected Submissions (Lower-Score Cohort)

* **Submission 2 (Spectral Model Merging via SVS):** (Average: 3.33) Operates in the spectral domain (SVD on task vectors). While mathematically elegant, it has subtle theoretical gaps in its scale-invariance proofs (e.g., omitting the bias term in LayerNorm/L2 cancellation, and assuming linear homogeneity inside residual connections). More critically, standard training details and hyperparameters required for reproducibility are completely omitted from the main text and there is no appendix.
* **Submission 4 (Activation-Guided Channel-Wise Gating):** (Average: 3.33) Evaluates channel-wise softmax gating. It suffers from a "utility paradox" where the channel-wise gating mechanism is shown in ablation studies to be completely redundant and perform identically to simple, uniform static compositions of layers. It also has highly suspicious numerical invariance in Table 2 (reporting identical accuracies to three decimal places across highly distinct calibration inputs, including random noise and zero tensors), pointing strongly to a division-by-zero bug.
* **Submission 7 (ThermoMerge):** (Average: 2.33) Evaluates temperature-based merging. It is restricted to ResNet-18 on toy datasets, failing to prove utility on larger foundation models, and suffers from poor technical soundness.
* **Submission 8 (Norm-Preserved Budgeted Task-Vector Pruning):** (Average: 3.67) Introduces post-hoc weight sparsification (NP-BTVP). While technically solid and highly honest, its evaluations are restricted to a small visual backbone and tiny classification splits. It received two Weak Rejects due to the limited complexity and scope of the evaluation regime compared to SOTA.
* **Submission 10 (Layer-Wise AdaMerging / Spatial Averaging):** (Average: 3.33) Identifies the Spatial Averaging Paradox but has severe limitations. It receives a Reject score (2) from Reviewer 1 due to the uncalibrated prediction entropy bottleneck and destructive weight-space interference on harder tasks, which their proposed remedies fail to restore.

---

## 6. Conclusion

By selecting **Submission 6 (Q-Merge)**, **Submission 3 (PolyMerge / SplineMerge)**, and **Submission 1 (RegCalMerge)**, the final accepted papers represent a highly balanced, scientifically rigorous, and high-impact selection:
1. Q-Merge advances network compression and model merging under discrete representation boundaries (systems deployment focus).
2. PolyMerge/SplineMerge introduces profound conceptual, diagnostic, and mathematical insights regarding optimization under transductive overfitting (theoretical focus).
3. RegCalMerge exposes sacrificial task bias, provides creative diagnostics, and sets new state-of-the-art performance boundaries on complex task blending (empirical and diagnostic focus).

All three accepted papers have had their corresponding `submission/` contents successfully migrated to the conference's accepted repository under `accepted_papers/submissionN/`.
