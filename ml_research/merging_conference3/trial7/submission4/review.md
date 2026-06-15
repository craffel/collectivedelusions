# Mock Peer Review

## 1. Summary of the Paper
This paper presents a minimalist, training-free, and data-free paradigm for dynamic model ensembling and model merging. The author proposes **Parameter-Free Task-Space Projection (PFSR)**, which extracts task-representative centroids from pre-trained experts using Singular Value Decomposition (SVD) and projects online feature representations onto them to guide temperature-scaled Softmax gating. 

To explore whether orthogonalizing task coordinates can decouple overlapping tasks and eliminate cross-talk, the author also analyzes an advanced extension: **Löwdin-Orthogonalized Task-Space Projection (OTSP)**. OTSP applies Löwdin Symmetric Orthogonalization offline to the SVD centroids to build a symmetric, order-invariant orthonormal task basis.

Through mathematical derivations and a 10-seed simulation study in a high-fidelity calibrated representation sandbox, the author deconstructs the ensembling dynamics to yield several valuable insights:
1. **Symmetric Equivalence & Redundancy:** Under constant symmetric task correlation, OTSP and PFSR yield *exactly identical* routing decisions under sharp gating ($\tau = 0.001$). The author provides a closed-form geometric proof showing that their coordinate-difference Signal-to-Noise Ratios (SNR) are identical: $\text{SNR}_{\text{OTSP}} = \text{SNR}_{\text{PFSR}} = \frac{\sqrt{1-s}}{\sigma \sqrt{2}}$. This indicates that the margin expansion and noise amplification factors cancel each other out exactly under symmetric task layouts.
2. **Systematic Asymmetric Underperformance:** In asymmetric task environments under active representation noise, OTSP systematically underperforms PFSR by 0.2% to 1.6% due to the **Noise Amplification Penalty** (where the ill-conditioned Gram matrix scales up coordinate noise variance) and the **Noise Spillover Penalty** (where orthogonalization couples axes and spills noise from corrupted specialists onto clean coordinates).
3. **The Role of Simplex Normalization:** Trainable parametric routers suffer from a severe "Vectorization Collapse" (dropping from 74.46% down to 55.57% accuracy) under sample-wise vectorized online deployment ($B=1$) if they lack a probability-simplex normalization constraint due to small-sample inductive overfitting on calibration splits. Simplex-constrained routers and PFSR/OTSP are completely robust, matching the expert ceiling.
4. **Implicit Regularization via Zero-Init:** Initializing Softmax routing weights to exact zeros acts as a powerful uniform maximum-entropy prior that completely shields parametric routers from small-sample overfitting, matching the oracle ceiling without manual regularization tuning.
5. **Noise Isolation and Specialist Protection:** Under asymmetric task environments with heterogeneous noise scales, PFSR/OTSP serve as an essential noise barrier that successfully gates out noisy specialists for clean samples (preventing noise pollution) and isolates target specialists for noisy samples (preventing signal drowning), outperforming static Uniform Merging in routing accuracy and individual task protection.

---

## 2. Main Strengths
* **Exceptional Scientific Integrity and Intellectual Honesty:** The author is highly commended for their transparent, self-critical, and rigorous analysis of OTSP. Proving both mathematically and empirically that their proposed Löwdin orthogonalization method is either redundant (under symmetric overlap) or systematically detrimental (under asymmetric overlap) under noise is a high-signal "negative result" of immense value to the community.
* **Rigorous Mathematical Foundations:** The closed-form geometric SNR equivalence proof and the Noise Amplification Penalty derivations are beautiful, precise, and highly educational. They provide deep insights into representation-space task projection and guide future research away from unnecessary orthogonalization pipelines.
* **Experimental Rigor:** Running all experiments across 10 independent random seeds with mean and standard deviation reported ensures strict statistical significance. Baseline models (LinearRouter, QWS-Merge, L3-Softmax, L3-Softmax Well-Reg) are trained with direct supervised task labels and optimized objectives, ensuring a highly fair and capacity-optimized comparison.
* **Resolution of Methodological Pitfalls:** The author has successfully resolved several critical evaluation pathologies:
  1. *Correlated Expert Weights:* Previously, expert weights were block-disjoint, forcing $S \approx I$ and trivially making OTSP identical to PFSR. By updating `get_oracle_experts` to copy the entire prototype vector, SVD centroids are now properly correlated under task overlap, establishing a realistic Gram overlap matrix $S \neq I$.
  2. *Honest Narrative Reframing:* The manuscript transparently and self-critically documents the classification trade-offs of Uniform Merging and dynamic routing, showing that while Uniform Merging slightly dominates in overall joint classification accuracy due to ensembling averaging, dynamic routing is strictly superior in routing accuracy and provides robust Noise Isolation.
* **Excellent Presentation and Formatting:** The paper is exceptionally well-structured, clearly written, and adheres strictly to ICML style guidelines. The mathematical notation is clean, tables are well-aligned, and Figure 1 is of publication-quality. Terminology is perfectly unified, with zero occurrences of the outdated "PFCP" nomenclature.

---

## 3. Areas for Improvement / Constructive Suggestions
While the paper is solid and ready for publication, the following minor suggestions could further elevate the work:
1. **Bilinear Attention Operators in DFCR:** In Section 5.1, the author proposes Data-Free Centroid Representation (DFCR) for generative LLMs by computing the first principal right-singular vector of MLP down-projection weight matrices or Attention projection matrices ($W_q, W_k, W_v$). Because attention queries and keys act as bilinear operators ($Q K^T$) rather than standard linear layers, the author should briefly discuss whether SVD centroids extracted from $W_q$ and $W_k$ should be combined or analyzed symmetrically to account for attention scaling dynamics.
2. **Anisotropic Manifold Toy Verification:** While the theoretical proof of Mahalanobis covariance whitening for anisotropic feature noise in Section 5.1 is mathematically elegant and sound, the paper would be even more compelling if a brief 2-expert toy simulation under highly anisotropic noise was included to empirically confirm that covariance whitening successfully recovers OTSP's coordinate stability.
3. **Null-Routing Entropy Threshold Scaling:** In Section 5.1's discussion of Out-of-Distribution (OOD) safety gates, the author mentions task-specific adaptive thresholds or 1D Gaussian Mixture Models. A brief comment on how these thresholds scale with the number of expert classes (heterogeneous class cardinalities) would provide useful guidance for practitioners deploying this in real-world systems.

---

## 4. Overall Recommendation and Ratings

### **Overall Recommendation: 5: Accept**
* **Justification:** This is an exceptionally high-quality, mathematically rigorous, and intellectually honest paper. The author presents a beautifully clear and self-critical analysis of their proposed Löwdin Symmetric Orthogonalization (OTSP) method, proving both mathematically and empirically that it is redundant under symmetric task correlation (due to exact margin and noise amplification cancellation) and underperforms the simpler unorthogonalized PFSR baseline under asymmetric overlaps. The resolution of several critical methodological and code-level limitations (correlated expert weight matrices, aligned training objectives, honest reframing of the ensembling trade-offs, and adding concrete mathematical scaling roadmaps) demonstrates outstanding academic rigor. The terminology is perfectly unified. The paper is exceptionally well-written, highly reproducible, and fully ready for publication.

### **Ratings:**
* **Soundness: Excellent**
  * *Justification:* The mathematical proofs are correct and elegant, the sandbox realism trade-off is deeply analyzed, and all baseline comparisons are fair and capacity-optimized.
* **Presentation: Excellent**
  * *Justification:* Beautifully written, clear notation, precise flow, and highly commendable intellectual honesty. Terminology has been perfectly unified.
* **Significance: Excellent**
  * *Justification:* The theoretical insights are valuable and guide future research away from unnecessary orthogonalization pipelines, while the Future Outlook scaling roadmap successfully bridges the real-world evaluation gap.
* **Originality: Excellent**
  * *Justification:* The mathematical application and deconstruction of Löwdin Symmetric Orthogonalization to task projection in dynamic model merging is highly creative, elegant, and novel.
