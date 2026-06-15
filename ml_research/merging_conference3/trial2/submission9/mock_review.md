# Mock Peer Review

**Paper Title**: Barycentric Proximity-Anchored Merging: A Critical, Deconstructive Audit of Parameter-Frugal Test-Time Model Merging  
**Author**: Julian Vance (University of Oxford)  
**Target Venue**: ICML 2026  

---

## Meta-Evaluation Ratings

* **Overall Recommendation**: **4: Weak Accept** (A technically solid paper that provides extremely timely, deconstructive insights into the test-time model merging literature, though with some empirical and methodological gaps that should be addressed before publication.)
* **Confidence**: **4: High** (I have thoroughly analyzed the mathematical formulation, the baseline comparisons, the experimental details, and the underlying codebase.)
* **Soundness**: **3: Good** (The mathematical proofs and scale preservation boundaries are correct. The ray-scaling projection is well-justified. However, joint optimization is highly imbalanced, and there is a significant memory bottleneck.)
* **Presentation**: **3: Good** (Excellently written and structured. The narrative is engaging and intellectually honest. However, it is held back by a complete lack of visual diagrams/figures and deprecated LaTeX commands.)
* **Significance**: **4: Excellent** (Highly significant. It serves as an active "sanity check" on the rapidly increasing architectural complexity of test-time model merging, mapping the trade-offs of parameter frugality and establishing crucial baseline boundaries.)
* **Originality**: **3: Good** (While individual components of the proposed BPAM baseline are simple, the deconstructive evaluation paradigm, the CKA representation sharing analysis, and the honest auditing of their own regularizer represent a highly original analytical contribution.)

---

## 1. Summary of the Submission
The paper presents a critical, deconstructive audit of **test-time adaptive model merging**, exploring the physical boundaries of parameter frugality. To facilitate this audit, the author introduces **Barycentric Proximity-Anchored Merging (BPAM)**, a minimalist baseline that reduces the trainable parameters to exactly $K$ global task-wise scalars (where $K = 8$ is the number of expert tasks). BPAM incorporates a convex simplex projection to mathematically preserve weight scale and bounds parameter norms, and utilizes a closed-form Mean-Field Proximity Penalty to stabilize optimization. 

Evaluating BPAM on the standard 8-task image classification benchmark using CLIP ViT-B/32, the paper reveals several core insights:
1. Restricting adaptation to a single bottleneck layer (BPAM-Restricted) collapses performance to 51.38%, proving that whole-model parameter blending is necessary.
2. In extremely low-parameter regimes (8 global weight scalars), weight-space optimization alone lacks the degrees of freedom to resolve fine-grained parameter conflicts, forcing downstream classifier head adaptation to become the primary driver of performance gains (lifting accuracy from 69.21% to 75.22%).
3. A fascinating "0-Weight Performance Mystery" is resolved using Centered Kernel Alignment (CKA) representation analysis, proving that specialized experts (like SVHN/MNIST) can maintain high performance even when completely pruned from the weights because their features are robustly shared across other experts in the compact fine-tuning basin.

---

## 2. Main Strengths

* **Exceptional Analytical Transparency and Intellectual Honesty**: The paper stands out for its rigorous and honest scientific deconstruction. Rather than claiming their proposed regularizer (Mean-Field Proximity Penalty) is universally essential, the author openly admits that it is empirically redundant under standard data regimes (due to the 8-parameter space being intrinsically immune to transductive overfitting). They then design a specialized extreme low-data experiment (5 samples per class) to empirically isolate the exact boundary where the penalty becomes critical.
* **Symmetric and Rigorous Evaluation Design**: The experimental layout is divided symmetrically into *Part A: Frozen Classification Heads* and *Part B: Active Classification Head Adaptation*. This decoupling is a major strength. It forces a clear distinction between genuine weight-space representation alignment and downstream classifier probe tuning, exposing a major blind spot where higher-capacity adaptive methods may overfit decision boundaries.
* **Fascinating Qualitative Insights on Weight-Space Geometry**: The resolution of the "0-weight performance mystery" using Linear CKA similarities is outstanding. It provides concrete empirical evidence that fine-tuned expert models starting from the same base reside in highly compact, shared loss basins, allowing visual representations (such as digits) to be robustly reconstructed from other experts (like traffic signs) even when the specialized parameters themselves are mathematically discarded.
* **Strong Theoretical and Physical Safeguards**: The mathematical proof bounding the Frobenius norm of the merged weights using the triangle inequality is rigorous and correct. Choosing a ray-scaling ($L_1$-normalization) projection over an exact orthogonal simplex projection is backed by a highly logical, domain-specific justification: preserving the directional ratios of collaborative expert contributions rather than pushing parameters to zero.

---

## 3. Major Weaknesses / Critical Flaws

### Weakness 1: Severe Optimization Imbalance in Joint Co-adaptation (BPAM-Full)
In BPAM-Full, the 8 global task-merging scalars $\Lambda$ and the 388,096 classification head parameters $H$ are optimized concurrently using the exact same Adam optimizer and uniform learning rate ($\eta = 10^{-3}$) without any specialized scheduling. This creates a severe optimization imbalance. Because the classification head parameters outnumber the weight-space scalars by **nearly five orders of magnitude**, their loss landscape curvature and gradient scales differ dramatically. The high-capacity classification heads will rapidly dominate the co-adaptation process, overfitting prediction boundaries before the 8 global weight scalars can converge to their optimal multi-task coordinates.
* **Omission**: While the author mentions in Section 4.4 that they extended the codebase to support asymmetric scheduling ($\eta_{\text{head}} < \eta_{\text{weight}}$) and observed "preliminary" convergence improvements, **the paper fails to provide any quantitative ablation results** or comparative tables verifying this claim. Leaving this key optimization fix un-ablated is a major empirical gap.

### Weakness 2: Peak Memory Bottleneck and the "Expert Leak"
The teacher-guided test-time adaptation objective requires computing the KL-divergence between the merged model predictions and the predictions of each individual expert teacher. During the adaptation/calibration phase, this requires the unlabeled images to be passed through **all $K = 8$ specialized expert networks in parallel** to generate teacher pseudo-labels.
* **Scalability Bottleneck**: This "Expert Leak" severely limits the practical scalability of test-time adaptation. If a system merges 50 or 100 experts, the GPU must host 51 or 101 models simultaneously during calibration, which is completely infeasible for standard hardware or edge nodes.
* **Omission**: The paper does not evaluate a **teacher-free adaptation baseline** (e.g., using entropy minimization like TENT or self-training directly on the merged model's representations) to see if adaptation can be achieved without loading all expert models, or to measure the exact performance drop associated with eliminating the expert teachers.

### Weakness 3: Critical Scientific Gaps in Experimental Rigor
The experimental section exhibits two major scientific gaps that hold it back from standard publication tier:
1. **Omission of Individual Task Results for Static Baselines (Table 1, Part A)**: In the frozen-head setting, the individual task-wise accuracies (for SUN397, Cars, etc.) are completely omitted (marked as `--`) for the two fundamental static baselines: **Task Arithmetic** and **TIES-Merging**. Only their average accuracies are reported. Since these task accuracies are well-known and standard, omitting them prevents a complete side-by-side comparison.
2. **Complete Absence of Error Bars and Statistical Significance Testing**: Test-time adaptation is performed on small local splits (such as 20% splits or the extreme 5-samples split). All tables report point estimates without any standard deviations, confidence intervals, or statistical significance indicators across multiple random seeds. Given that several performance margins are extremely slim (e.g., BPAM-Static 69.21% vs. Task Arithmetic 69.10%, or BPAM-Full 75.22% vs. Task Arithmetic+Head Tuning 74.80%), it is impossible to determine whether these marginal improvements are statistically significant or merely random noise.

---

## 4. Detailed Actionable Constructive Feedback

1. **Complete Table 1, Part A**: Retrieve and report the individual task accuracies for the standard **Task Arithmetic** and **TIES-Merging** baselines under strictly frozen classification heads to ensure the table is complete and fully comparable.
2. **Add Multi-Seed Evaluations with Error Bars**: Run the test-time adaptation experiments across at least 3 to 5 random seeds (especially for the 5-sample extreme low-data regime and the standard calibration splits). Report standard deviations (e.g., $75.22 \pm 0.35\%$) and perform standard t-tests to verify if BPAM's marginal improvements over Task Arithmetic are statistically significant.
3. **Include Quantitative Results for Asymmetric Scheduling**: In Section 4.4, provide a table or line plot comparing the performance of **Uniform Co-adaptation (Default)** vs. **Asymmetric Co-adaptation** across a range of head learning rates ($\eta_{\text{head}} \in \{10^{-3}, 10^{-4}, 10^{-5}\}$). This will transform your conceptual explanation into a verified empirical solution.
4. **Evaluate a Teacher-Free Adaptation Baseline**: To address the "Expert Leak" scalability bottleneck, implement and evaluate a simple teacher-free adaptation baseline (e.g., test-time entropy minimization on the merged model's outputs). Compare its performance against the teacher-guided variant to measure the exact trade-off between memory footprint and multi-task accuracy.
5. **Add Visual Conceptual Figures**:
   - **Algorithmic Diagram**: Introduce a schematic diagram illustrating the convex barycentric simplex (e.g., a triangle for 3 task experts), showing the pre-trained base model, the update direction, the ray-scaling projection back to the simplex boundary, and the Mean-Field Proximity Penalty acting as a spring pulling back towards the centroid.
   - **CKA Similarity Heatmap**: In the discussion of the "0-Weight Performance Mystery" (Section 4.5), visualize the Linear CKA similarities as a 2D heatmap matrix (comparing the base model, individual experts, and the merged model) to make the representation sharing argument visually striking.

---

## 5. Minor Issues / Typos / Formatting Suggestions

* **Deprecated Font Commands**: Replace all occurrences of deprecated, old-style LaTeX font commands (such as `{\it ...}` and `{\bf ...}`) with modern, standard LaTeX equivalents (`\textit{...}` and `\textbf{...}`) throughout the codebase to ensure modern compiler compliance and clean rendering.
* **Spelling**: In `reviewing_criteria.md`, there is a minor typo: "...glease use sparingly" (should be "please use sparingly").
* **Section Reference**: Ensure that table caption placement is strictly above tables, as currently rendered in the LaTeX, which is excellent.
