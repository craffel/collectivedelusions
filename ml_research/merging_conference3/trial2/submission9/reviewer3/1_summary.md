# 1. Summary of the Paper

## Main Topic
The paper presents a critical, deconstructive audit of test-time adaptive model merging, aiming to explore the mathematical boundaries of parameter frugality. Test-time adaptive model merging has recently evolved toward complex, high-capacity frameworks, such as overparameterized normalizing flows (e.g., 2.6M parameters in FoldMerge) or high-dimensional layer-wise adapters (e.g., AdaMerging or SyMerge). This paper steps back and investigates if such high capacity is strictly necessary and where the performance gains in test-time model merging actually originate.

## Proposed Approach
To investigate these boundaries, the authors introduce a minimalist, low-parameter framework called **Barycentric Proximity-Anchored Merging (BPAM)**. BPAM operates on the following principles:
1. **Convex Barycentric Simplex Projection:** Restricts the merging coefficients to a convex barycentric simplex, ensuring the merged weights lie within the convex hull of the base model and expert weights, mathematically bounding the weight norms to prevent scale distortion or activation collapse.
2. **Mean-Field Proximity Regularization:** A soft, closed-form $\ell_2$ penalty that pulls task-specific coefficients toward a uniform barycentric centroid to stabilize optimization and prevent transductive overfitting.
3. **Teacher-Guided Adaptation:** Optimizes coefficients using joint KL-divergence between the merged model predictions and the individual frozen expert teachers on unlabeled test streams.

The authors evaluate three configurations of BPAM across an 8-task image classification benchmark using CLIP ViT-B/32:
- **BPAM-Restricted:** Merges only the visual projection layer (`model.visual.proj`) with exactly 8 task-wise parameters and strictly frozen classification heads.
- **BPAM-Static:** Merges the entire image encoder using 8 global task-wise parameters (broadcast layer-wise) and strictly frozen classification heads.
- **BPAM-Full:** Merges the entire image encoder using 8 task scalars and concurrently adapts all task-specific linear classification heads (388K classifier parameters).

---

## Key Findings and Quantitative Evidence
1. **Whole-Model vs. Localized Merging:** BPAM-Restricted achieves an average accuracy of only **51.38%**, representing a catastrophic drop compared to standard baselines. This proves that localized, single-layer parameter blending is highly insufficient, and whole-model blending is mathematically necessary to preserve fine-tuned representations.
2. **Frozen-Head Performance Upper Bound:** BPAM-Static achieves **69.21%** average accuracy, matching the performance of standard linear Task Arithmetic (69.10%). This is achieved with only 8 parameters total on strictly frozen classifier heads.
3. **The Role of Simplex Constraints (Ablation):** An ablation of the convex simplex projection ("Unconstrained Scaling") yields **71.51%** average accuracy, which is $+2.30\%$ absolute higher than default BPAM-Static. This suggests that while the convex simplex acts as a critical scale-preserving safeguard, it slightly limits optimization expressivity in extremely low-parameter regimes.
4. **Classification Head Domination:** BPAM-Full (co-adapting weight coefficients and linear classifiers) achieves **75.22%** average accuracy (a $+6.01\%$ gain over BPAM-Static). This reveals that for extremely low-parameter regimes, downstream decision-boundary adaptation (classification head tuning) drives the vast majority of the optimization gains, rather than weight-space alignment.
5. **Comparison to Head-Tuning Baselines:** BPAM-Full barely beats Task Arithmetic + Head Tuning (**74.80%**) by $+0.42\%$ and underperforms TIES-Merging + Head Tuning (**78.50%**) by $-3.28\%$. This indicates that applying decision-boundary tuning on top of a strong, conflict-resolved static weight-space model (like TIES-Merging) is strictly superior to joint optimization under severe weight-space parameter constraints (such as BPAM's 8 global task scalars).
6. **Immunity to Transductive Overfitting:** Using a 20% calibration split and 80% unseen test split, the unregularized model ($\beta = 0$) achieves **69.30%** unseen accuracy versus **69.29%** for the regularized model ($\beta = 0.01$). This shows that the 8-parameter search space is intrinsically immune to transductive overfitting, making the Mean-Field Proximity Penalty practically redundant for the 8-scalar regime, though it is shown to be crucial under extreme low-data scenarios (5 samples per class) where unregularized optimization drifts wildly.
7. **0-Weight Expert Performance & CKA Analysis:** SVHN and MNIST experts converge to exactly $0.0000$ coefficient weight, yet achieve **78.15%** and **88.09%** accuracy respectively under BPAM-Static. Centered Kernel Alignment (CKA) similarity checks demonstrate that the merged weight representation space contains reconstructed representations for SVHN (CKA of 0.1372 vs 0.0632 for base) and MNIST (CKA of 0.5000 vs 0.3754 for base), driven by representation sharing from other experts (such as numerical shapes in GTSRB).

---

## Explicitly Claimed Contributions
1. **A Critical Deconstructive Audit:** Explores the physical and mathematical boundaries of parameter-frugal test-time model merging under two under-discussed phenomena: whole-model blending and parameter constraint boundaries.
2. **BPAM Framework:** Introduces an elegant, low-parameter framework restricted to exactly $K$ global task-wise scalars, achieving a $99.99\%$ parameter footprint reduction compared to FoldMerge and a $99.3\%$ reduction compared to AdaMerging.
3. **Convex Simplex and Proximity Formulations:** Introduces a scale-preserving convex simplex projection to prevent activation collapse and a Mean-Field Proximity Penalty to prevent overfitting on local calibration streams.
4. **Regime Mapping:** Systematically maps test-time model merging across frozen-head and active-head regimes, identifying the exact threshold where layer-wise and architectural degrees of freedom become necessary to resolve cross-task weight conflicts.
