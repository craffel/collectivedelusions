# Comprehensive Summary: "Barycentric Proximity-Anchored Merging: A Critical, Deconstructive Audit of Parameter-Frugal Test-Time Model Merging"

## 1. Main Topic and Scope
This paper investigates the limits of "parameter frugality" in test-time adaptive model merging. Model merging aims to combine multiple specialized task-expert neural networks (fine-tuned from a shared pre-trained base model) into a single unified network without retraining. Test-time adaptive model merging seeks to optimize the merging parameters on unlabeled test streams. The authors attempt to deconstruct this paradigm by proposing **Barycentric Proximity-Anchored Merging (BPAM)**, a minimalist framework that scales down the number of optimized weight-space parameters to exactly $K$ global task scalars (where $K$ is the number of expert tasks, i.e., $K=8$ for an 8-task benchmark). The scope is limited to a CLIP ViT-B/32 image encoder evaluated across an 8-task image classification benchmark.

## 2. Technical Approach
The proposed BPAM framework rests on three primary mathematical and structural components:
*   **Convex Barycentric Simplex Projection:** Restricts the task-merging coefficients $\lambda_k$ to a convex simplex ($\lambda_k \geq 0$ and $\sum_{k=1}^K \lambda_k \leq 1.0$), forcing the merged weight matrices to lie within the convex hull of the base and expert weights. This is claimed to prevent activation scale distortion and preserve weight norms without requiring overparameterized normalizing flows (like FoldMerge). If the coefficients exceed the simplex boundary during Adam updates, they are projected back using a ray-scaling ($L_1$-normalization) scheme.
*   **Mean-Field Proximity Regularization:** A soft closed-form $\ell_2$ penalty that pulls the individual task coefficients toward a uniform barycentric centroid ($\frac{1}{K+1}$), which assigns equal weight to the base model and each expert. This is intended to stabilize the optimization landscape and prevent transductive overfitting on small test-time calibration streams.
*   **Teacher-Guided Adaptation:** Optimizes the $K$ global task scalars (and optionally the classification heads) by minimizing the joint KL-divergence between the merged model's predictions and the individual frozen expert teacher models' predictions on unlabeled target calibration streams.

The authors evaluate three configurations of BPAM to explore spatial and parameter constraints:
1.  **BPAM-Restricted:** Optimizes exactly 8 parameters only on a single bottleneck layer (`model.visual.proj`) while freezing all other 157 layers and all classification heads.
2.  **BPAM-Static:** Optimizes exactly 8 parameters globally across the entire image encoder while freezing all classification heads.
3.  **BPAM-Full:** Optimizes 8 global task-scalars on the image encoder and concurrently tunes all 8 task-specific linear classifiers (classification heads), totaling 388,104 parameters.

## 3. Key Findings
*   **The Inefficacy of Bottlenecking:** Restricting parameter-space blending to a single bottleneck layer (BPAM-Restricted) collapses performance to **51.38%** average accuracy (down from ~69.10% for baseline Task Arithmetic), demonstrating that whole-model weight-space blending is necessary.
*   **Minimalist Base Performance:** BPAM-Static (8 global parameters, frozen classification heads) achieves **69.21%** average accuracy, which is virtually identical to the static, zero-compute Task Arithmetic baseline (69.10%).
*   **The Dominance of Head Tuning:** BPAM-Full achieves **75.22%** average accuracy, representing a +6.01% gain over BPAM-Static. This indicates that classification head adaptation, rather than weight-space optimization, is the primary driver of performance in low-parameter adaptive regimes.
*   **Redundancy of the Proximity Penalty:** The ablation study shows that setting the regularization coefficient $\beta = 0.0$ (unregularized) vs. $\beta = 10^{-2}$ yields almost identical results (69.30% vs. 69.29% unseen test accuracy), demonstrating that the proximity penalty is empirically redundant under standard settings.
*   **The "0-Weight" Phenomenon:** Under frozen heads, the MNIST and SVHN merging coefficients converge to exactly $0.0000$ (completely suppressing their expert weights), yet the model still achieves 88.09% accuracy on MNIST and 78.15% on SVHN. The authors attribute this to compact weight basins and representation sharing across the other fine-tuned experts (e.g., GTSRB).

## 4. Claimed Contributions and Accompanying Evidence
*   **Contribution 1:** A critical, deconstructive audit exploring the physical boundaries of parameter-frugal test-time adaptive merging.
    *   *Evidence:* Analysis of BPAM-Restricted vs. BPAM-Static vs. BPAM-Full (Table 1), showing that whole-model blending is necessary and that head adaptation dominates low-parameter regimes.
*   **Contribution 2:** Barycentric Proximity-Anchored Merging (BPAM), a minimalist 8-parameter adaptive baseline.
    *   *Evidence:* Implementation of the optimization pipeline and evaluation on the 8-task CLIP ViT-B/32 benchmark.
*   **Contribution 3:** Formulations of a scale-preserving convex simplex constraint and a Mean-Field Proximity Penalty.
    *   *Evidence:* Mathematical derivations and a split-test ablation study (Table 4) showing their stability and behavior under standard and extreme low-data calibration regimes (Table 5).
*   **Contribution 4:** Mapping the exact thresholds where layer-wise and architectural degrees of freedom are needed.
    *   *Evidence:* Comparisons to SOTA high-capacity methods like AdaMerging, SyMerge, and FoldMerge (Table 1), showing a ~14% performance gap under frozen heads that can only be resolved by higher parameter capacity.
