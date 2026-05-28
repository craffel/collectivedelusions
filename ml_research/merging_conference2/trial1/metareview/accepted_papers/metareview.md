# Meta-Review Decisions and Process Report

This meta-review report details the decisions, criteria, and selection process for the 10 paper submissions evaluated for conference publication. Based on a thorough analysis of both numerical ratings and detailed review content, we have selected exactly **3 out of the 10 submissions** to accept.

---

## 1. Executive Summary and Decisions Table

The following table summarizes the 10 submissions, including their titles, reviewer scores, calculated average scores, and final decisions (Accept/Reject).

| Submission | Paper Title | Reviewer Ratings | Average Score | Decision |
| :--- | :--- | :--- | :---: | :---: |
| **Submission 9** | *Deconstructing Test-Time Model Merging: Is Joint Optimization a Methodological Illusion?* | 6 (Strong Accept)<br>5 (Accept)<br>4 (Weak Accept) | **5.00** | **ACCEPT** |
| **Submission 5** | *Pragmatic Multi-Task Model Merging via Task-Conditional Activation Calibration* | 6 (Strong Accept)<br>3 (Weak Reject) | **4.50** | **ACCEPT** |
| **Submission 1** | *Demystifying Orthogonal Model Merging: Is Manifold Geometry Doing the Heavy Lifting?* | 5 (Accept)<br>3 (Weak Reject) | **4.00** | **ACCEPT** |
| **Submission 2** | *Wasserstein Spectral Alignment: A Principled Optimal Transport Framework for Model Merging* | 5 (Accept)<br>4 (Weak Accept)<br>2 (Reject) | **3.67** | **REJECT** |
| **Submission 6** | *[Model Merging / Fine-Tuning Evaluation and Benchmarking]* | 4 (Weak Accept)<br>3 (Weak Reject)<br>2 (Reject) | **3.00** | **REJECT** |
| **Submission 4** | *[Unspecified / Model Merging Variant]* | 3 (Weak Reject)<br>3 (Weak Reject)<br>2 (Reject) | **2.67** | **REJECT** |
| **Submission 3** | *ThermoMerge: Thermodynamic and Quantum Statistical Subspace Alignments for Multi-Task Model Merging* | 3 (Weak Reject)<br>2 (Reject) | **2.50** | **REJECT** |
| **Submission 8** | *[Unspecified / Model Merging Variant]* | 3 (Weak Reject)<br>2 (Reject)<br>2 (Reject) | **2.33** | **REJECT** |
| **Submission 7** | *SA-TTA: Sharpness-Aware Test-Time Adaptation for Robust Model Merging* | 2 (Reject)<br>2 (Reject) | **2.00** | **REJECT** |
| **Submission 10**| *Submission and Formatting Instructions for International Conference on Machine Learning (ICML 2026)* | 1 (Strong Reject)<br>1 (Strong Reject)<br>1 (Strong Reject) | **1.00** | **REJECT** |

---

## 2. Meta-Reviewing Process and Selection Methodology

Our selection process followed a structured **Research -> Strategy -> Execution** cycle, strictly adhering to the conference's criteria for **soundness, presentation, significance, and originality**:

1. **Content Over Scores:** We prioritized the substance, correctness, and scalability of the papers over superficial scores. For example, while mathematical complexity can be attractive, we critically evaluated whether it represented a genuine contribution or "flashy math trap" over-engineering.
2. **Deconstruction and Simplification:** We highly valued papers that championed scientific hygiene and simplicity. Demystifying complex, cubic-complexity methods (like OrthoMerge or joint test-time optimization) by proving that simple, properly-scaled Euclidean baselines achieve identical or superior results was deemed of exceptional significance to the ML community.
3. **Resolving Conflicting Reviews:**
   - For **Submission 5**, the *Weak Reject* was primarily concerned with BatchNorm dependency. However, the *Strong Accept* reviewer argued that the conceptual leap of *representation-level repair* (TCAC) was a major paradigm shift that outweighed the limited scope. We sided with the *Strong Accept* because extending TCAC to LayerNorm/RMSNorm is a promising future direction that the community will build on.
   - For **Submission 1**, the *Weak Reject* focused heavily on bibliographic and citation hygiene (mangled references, phantom citations). However, both reviewers agreed that the core scientific deconstruction and mathematical proofs were outstanding. Because bibliographic errors are easily remediable during camera-ready preparation, we accepted this paper.
   - For **Submission 2**, despite having an average score of 3.67 and a score of 5, the *Reject* reviewer raised **fatal scientific, experimental, and theoretical contradictions** (described below) that completely undermined the paper's soundness. Thus, we rejected it in favor of the clean, rigorous deconstruction in Submission 1.

---

## 3. Detailed Meta-Reviews for Accepted Submissions

### Meta-Review: Submission 9
**Title:** *Deconstructing Test-Time Model Merging: Is Joint Optimization a Methodological Illusion?*  
**Recommendation:** **ACCEPT (Strong)**  
**Summary of Decisions and Rationale:**  
Submission 9 provides a highly original, timely, and brave critique of the popular "test-time adaptive model merging" paradigm (e.g., SyMerge, AdaMerging). It challenges the core assumption that these methods achieve "synergistic joint learning" by optimizing merging coefficients ($\lambda$) and classification heads ($W$) simultaneously on unlabeled test streams.
- **Key Strengths:**
  1. *The Runaway Gradient Theorem (Proposition 3.1):* The authors mathematically prove and empirically demonstrate that under standard cross-entropy distillation losses, coefficient gradients are consistently negative, driving them to grow monotonically and cause activation explosions unless manually clamped.
  2. *The Clamping Paradox:* They show that when clamping is applied to stabilize the optimization, the coefficients remain stuck at the boundaries, rendering the optimization inactive.
  3. *The Synergy Illusion:* A simple "Head-only TTA" baseline (with frozen coefficients) recovers over 95% of the total adaptation gains, showing that head adaptation is doing all the heavy lifting and joint optimization is a methodological illusion.
- **Area of Improvement:** To make the paper absolutely bulletproof, the authors should include a small-scale experiment on a Transformer-based system (e.g., merging LoRA adapters of a small LLM) to demonstrate that the runaway gradient theorem remains architecture-agnostic at scale.
- **Conclusion:** An outstanding, paradigm-shifting paper that will prevent the community from chasing a mirage and wasting immense computational resources.

### Meta-Review: Submission 5
**Title:** *Pragmatic Multi-Task Model Merging via Task-Conditional Activation Calibration*  
**Recommendation:** **ACCEPT**  
**Summary of Decisions and Rationale:**  
Submission 5 introduces **Task-Conditional Activation Calibration (TCAC)**, which shifts the model merging paradigm from parameter-space interventions to dynamic, representation-level repair during inference. Under weight averaging, multi-task features suffer from activation variance collapse. TCAC heals this by keeping lightweight, task-conditional calibration parameters (<64 KB per task) and swapping them at runtime ($0.38$ ms).
- **Key Strengths:**
  1. *From Averaged to Task-Conditional Targets:* Standard activation calibration (like REPAIR) averages statistics across models, which is semantically flawed for multi-task settings with distinct domains. TCAC is the first to propose task-conditional calibration.
  2. *Orthogonality and Generality:* TCAC is applied on top of existing SOTA methods (TIES, DARE), yielding massive, universal boosts (+25.10% for TIES, +37.81% for DARE).
  3. *Fascinating Out-of-Domain Transferability:* The authors show that calibrating MNIST using completely out-of-domain CIFAR-10 images recovers a massive amount of performance, indicating that the parameters capture weight-space-induced geometric distortions of the feature space.
- **Area of Improvement:** The method is currently restricted to BatchNorm layers. The authors must address or discuss concrete paths for extending TCAC to LayerNorm/RMSNorm (typical in ViTs and LLMs) where normalization happens along the channel dimension.
- **Conclusion:** A highly innovative paper with great practical and theoretical impact. The training-free, zero-backpropagation nature of TCAC makes it incredibly valuable for real-world edge deployments.

### Meta-Review: Submission 1
**Title:** *Demystifying Orthogonal Model Merging: Is Manifold Geometry Doing the Heavy Lifting?*  
**Recommendation:** **ACCEPT**  
**Summary of Decisions and Rationale:**  
Submission 1 is a vital corrective for the field, exposing a "Flashy Math Trap" in model merging. It deconstructs **Orthogonal Model Merging (OrthoMerge)**, which performs weight merging on the Riemannian manifold of the orthogonal group. The authors prove that the manifold mathematics is a red herring, and that simple properly-scaled Euclidean baselines easily outperform it.
- **Key Strengths:**
  1. *Exposing the Structural Contradiction:* The authors show that in OrthoMerge's "conflict-aware" strategy, the orthogonal component has an extremely small norm, and the standard Euclidean residual carries almost all update information.
  2. *Decoupled Magnitude-Corrected (DMC) Merging:* They propose DMC-Merge, a simple Euclidean counterpart that performs conflict-aware magnitude correction without SVD or Lie algebra mappings, beating OrthoMerge by over 21 percentage points.
  3. *Elegant Mathematical Proofs:* Theorem 1 proves that diagonal feature scaling updates are orthogonal to the tangent space of the orthogonal group, explaining why OrthoMerge's projection discards up to 94% of task vector norms. Theorem 2 mathematically establishes that spectrum-flattening degrades the Signal-to-Noise Ratio (SNR) of weight updates.
- **Required Revisions for Camera-Ready:** The paper has severe scholarly/bibliographic hygiene issues that the authors must fix prior to publication:
  - Remove all 51 uncited "phantom" references.
  - Remove completely irrelevant, programmatically-appended non-ML bibliography entries (e.g., Tidey, 2018 on transgender family intimacies; Zhang & Wei, 2025 on Shanghai urban resettlement; Ran et al., 2022 on hydrology runoff forecasting).
  - Fix all split/mangled BibTeX entries.
  - Cite the DARE baseline (Yu et al., 2024) and correct the SAIM misattribution.
- **Conclusion:** A brilliant and essential critical contribution that advocates for simplicity and rigorous evaluation. It is accepted subject to the mandatory bibliographic cleanup.

---

## 4. Detailed Meta-Reviews for Key Rejected Submissions

### Meta-Review: Submission 2
**Title:** *Wasserstein Spectral Alignment: A Principled Optimal Transport Framework for Model Merging*  
**Recommendation:** **REJECT**  
**Summary of Decisions and Rationale:**  
While Submission 2 presents an elegant Bures-Wasserstein optimal transport framework to address spectral collapse in model merging, a rigorous analysis of the review content reveals **fatal conceptual, computational, and theoretical flaws**:
1. *Fatal Conceptual Flaw in Experimental Setup:* The authors fine-tune and merge models on three disjoint datasets (CIFAR-10, SVHN, Fashion-MNIST) with a *shared classification head*. This means they add together projection weights for completely unrelated classes (e.g., airplane, digit 0, T-shirt). This is a logical impossibility and explains why absolute performance of all merged models is abysmal (WSA peaks at 48.33% accuracy, whereas individual experts average 88.06%). 
2. *Extreme Computational Unscalability:* WSA requires solving a matrix fixed-point equation involving multiple matrix square roots (SVDs/eigendecompositions) at every iteration for every layer. For a modest modern model like LLaMA-7B ($d_{\min}=4096$), running 15–20 iterations of cubic-complexity $O(T K d^3)$ operations across all layers is computationally prohibitive, completely defeating the purpose of fast, training-free model merging.
3. *Fundamental Mathematical Contradiction in Theorem 4.5:* The authors claim that an upper bound on loss deviance (controlled by expected Frobenius parameter distance) justifies WSA's covariance alignment. However, this expected Frobenius distance is a strictly convex quadratic function uniquely minimized by the arithmetic mean (Task Arithmetic). Thus, from the perspective of Theorem 4.5, WSA is guaranteed to have a *looser (worse)* bound than the very baseline it critiques.
4. *Extreme Hyperparameter Sensitivity:* Despite claiming to be "hyperparameter-free", WSA is catastrophically sensitive to the regularization parameter $\epsilon$ (dropping 29.12% absolute accuracy when changing $\epsilon$ from $10^{-8}$ to $10^{-4}$ in a non-monotonic fashion).

Given these deep scientific and methodological contradictions, the paper is not ready for publication.

### Other Rejected Submissions (Submissions 3, 4, 6, 7, 8, 10)
- **Submission 3** (*ThermoMerge*): Proposes a complex thermodynamic analogy (Fermi-Dirac and Bose-Einstein statistics) for subspace alignment. While theoretically novel, it suffers from the same cubic-complexity SVD bottleneck as OrthoMerge, and is rejected due to a severe lack of empirical validation, missing multi-seed runs, and lack of transformer scale evaluations (Average score: 2.50).
- **Submissions 4, 6, 8:** These are standard incremental merging papers that suffer from poor baseline tuning, low absolute performance, or weak evaluation on small scale models (Averages: 2.67, 3.00, 2.33 respectively).
- **Submission 7** (*SA-TTA*): Suffers from severe optimization instabilities and lacks thorough evaluations, receiving unanimous rejections (Average: 2.00).
- **Submission 10:** This is the official ICML formatting template submitted in error by the authors. It contains zero original research or machine learning content, and is unanimously strongly rejected (Average: 1.00).
