# Summary of the Paper

## 1. Title and Metadata
- **Title**: Barycentric Proximity-Anchored Merging: A Critical, Deconstructive Audit of Parameter-Frugal Test-Time Model Merging
- **Author**: Julian Vance (University of Oxford)
- **Target Venue**: ICML 2026
- **Reviewer Focus**: Mock Reviewer (Focusing on identifying weaknesses, flaws in reasoning, missing baselines, unclear methodology, and weak theoretical/empirical justification)

---

## 2. Executive Summary and Problem Statement
The paper addresses the emerging paradigm of **test-time adaptive model merging**—where fine-tuned, task-specific expert weights are merged using coefficients optimized on unlabeled test-time data streams. 

The author argues that recent state-of-the-art approaches (such as AdaMerging, SyMerge, and FoldMerge) introduce substantial architectural complexity and parameter over-allocation. For example, FoldMerge trains a 4-layer normalizing flow network with 2.6 million parameters to learn a non-linear weight-space diffeomorphism, while AdaMerging and SyMerge use layer-wise adapters or high-capacity low-rank adapters.

Through a minimalist lens, the paper presents a **deconstructive audit** to explore the fundamental boundaries of parameter frugality. To do this, the author introduces a minimalist baseline named **Barycentric Proximity-Anchored Merging (BPAM)**, which uses exactly $K$ global scalar coefficients (where $K = 8$ is the number of expert tasks).

---

## 3. Proposed Method: BPAM
BPAM combines three main minimalist techniques:
1. **Convex Barycentric Simplex Projection**: The weight fusion is restricted to a convex barycentric simplex (coefficients are non-negative and sum to $\leq 1.0$), ensuring that the merged weights lie within the convex hull of the pre-trained and expert weights. This is designed to preserve activation scale and prevent scale distortion without extra parameters.
2. **Mean-Field Proximity Regularization**: An $\ell_2$ penalty pulls the task-specific coefficients towards the uniform centroid $\frac{1}{K+1}$, aiming to stabilize optimization and prevent transductive overfitting on small test-time splits.
3. **Teacher-Guided Test-Time Adaptation**: Light test-time optimization that updates the $K$ coefficients on unlabeled target streams by minimizing the joint KL-divergence between the merged model predictions and individual expert teacher predictions.

### Spatial Configurations Evaluated:
- **BPAM-Restricted**: Merges only the visual projection layer (`model.visual.proj`) of the image encoder (8 parameters, frozen heads).
- **BPAM-Static**: Merges the entire image encoder using the same 8 task coefficients (8 parameters, frozen heads).
- **BPAM-Full**: Merges the entire image encoder (8 task coefficients) and concurrently optimizes all 8 task-specific linear classifiers (classification heads, totaling 388K parameters).

---

## 4. Key Experimental Results
The method is evaluated on the standard 8-task image classification benchmark using CLIP ViT-B/32:
- **BPAM-Restricted** achieves **51.38%** average accuracy, collapsing performance and demonstrating that single-layer localized merging is insufficient.
- **BPAM-Static** achieves **69.21%** average accuracy, matching linear Task Arithmetic (69.10%) but underperforming compared to static **TIES-Merging** (72.90%) and adaptive SOTA like **AdaMerging** (83.17%), **SyMerge** (83.56%), and **FoldMerge** (83.56%) under frozen heads.
- **BPAM-Full** achieves **75.22%** average accuracy under joint optimization (head + weight), barely outperforming **Task Arithmetic + Head Tuning** (74.80%) and underperforming compared to **TIES-Merging + Head Tuning** (78.50%) and **SyMerge + Head Tuning** (89.74%).
- **Ablation Studies**: Shows that the Mean-Field Proximity Penalty is empirically redundant on standard test-time adaptation splits ($\beta = 0$ achieves 69.30% unseen accuracy, while $\beta = 10^{-2}$ achieves 69.29% unseen accuracy), but helps stabilize optimization when calibration samples are extremely scarce (5 samples per class).

---

## 5. Main Conceptual Takeaways of the Audit
- **Weight-space constraints**: Global task-wise scaling (only $K$ scalars) is too mathematically constrained to resolve fine-grained parameter conflicts, which is why higher-capacity layer-wise methods (like AdaMerging) perform significantly better.
- **Classification head adaptation**: Under extremely parameter-frugal regimes (8 weight-space parameters), downstream classifier head adaptation drives almost all test-time gains, shifting the optimization burden from weight-space alignment to decision boundary adaptation.
- **The 0-weight performance mystery**: Explains how the model performs highly on MNIST (88.09%) and SVHN (78.15%) even when their merging coefficients are optimized to exactly $0.0000$, pointing to a highly compact shared weight-space basin and cross-task representation sharing (e.g., GTSRB numerical traffic signs helping digit classification).
