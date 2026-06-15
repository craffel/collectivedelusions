# Peer Review

**Submission Title:** Barycentric Proximity-Anchored Merging: A Critical, Deconstructive Audit of Parameter-Frugal Test-Time Model Merging

---

## 1. Summary of the Paper
This paper presents a critical, deconstructive audit of test-time adaptive model merging, focusing on the boundary between parameter frugality and performance. The authors introduce **Barycentric Proximity-Anchored Merging (BPAM)**, a minimalist test-time adaptation framework that uses only $K$ global task-wise scalars (where $K=8$ is the number of expert tasks). BPAM incorporates two primary mathematical components:
1. **Convex Barycentric Simplex Projection:** Restricts merging coefficients to a convex barycentric simplex to prevent activation scale distortion and preserve weight norms in a closed-form, parameter-free manner.
2. **Mean-Field Proximity Regularization:** A closed-form $\ell_2$ penalty that anchors individual task coefficients toward a uniform barycentric centroid to stabilize optimization and prevent transductive overfitting.

The authors evaluate BPAM across three distinct configurations on an 8-task image classification benchmark using CLIP ViT-B/32:
- **BPAM-Restricted:** Merging is restricted to a single projection layer (`model.visual.proj`) with frozen heads (8 parameters total), achieving 51.38% average accuracy.
- **BPAM-Static:** Whole-model encoder merging with frozen heads (8 parameters total), achieving 69.21% average accuracy, which matches Task Arithmetic (69.10%).
- **BPAM-Full:** Whole-model encoder merging with concurrent classification head adaptation (8 task scalars + 388K classifier parameters), achieving 75.22% average accuracy.

The paper is positioned not as a state-of-the-art competitor, but as a diagnostic boundary probe baseline to map where layer-wise degrees of freedom (like in AdaMerging) or high-capacity mapping models (like FoldMerge/SyMerge) become essential for weight-space blending.

---

## 2. Strengths and Weaknesses

### Strengths
- **Rigorous Symmetric Evaluation (Part A vs. Part B):** The separation of evaluations into frozen classification heads (Part A) and active classification head adaptation (Part B) is highly commendable. It clearly distinguishes between performance gains driven by actual weight-space blending and those driven by decision-boundary adaptation.
- **Exceptional Analysis of the "0-Weight Performance Mystery":** Section 4.5 is a major highlight. The investigation into why the model performs highly on tasks with zero weight-space coefficients (MNIST and SVHN) is supported by solid empirical evidence using Centered Kernel Alignment (CKA) similarity scores. This provides valuable scientific insight into representation sharing.
- **Transparent and Constructive Tone:** The paper is exceptionally honest and deconstructive. Rather than trying to artificially claim superiority over high-capacity adaptive baselines, the authors transparently outline BPAM's performance limits and identify where joint co-adaptation becomes bottlenecked.
- **Thorough Stress Tests:** The inclusion of extreme low-data calibration (5 samples per class) and hyperparameter sensitivity analyses ($\beta$) provides a strong empirical foundation for when the proposed Proximity Penalty becomes active and beneficial.
- **Presentation and Clarity:** The manuscript is exceptionally well-structured, mathematically clear, and properly positions itself relative to concurrent deconstructive literature (such as the SAIM Audit and Layer-wise Model Merging Sanity Check).

### Weaknesses
- **Lack of Statistical Significance and Variance Reporting:**
  - The empirical tables (Table 1, Table 3, and Table 4) do not report standard deviations, confidence intervals, or run details across multiple random seeds/splits.
  - Test-time adaptation is highly sensitive to the specific samples in the calibration stream, especially in the extreme low-data regime (5 samples per class in Table 4). Without reporting mean and variance across 3-5 random seeds, it is impossible to evaluate whether small differences (such as BPAM-Static's +0.11% gain over Task Arithmetic in Table 1, or the minor differences between $\beta$ values in Table 4) are statistically significant or merely statistical noise.
- **Incomplete Rationale for the Convex Simplex Constraint:**
  - The authors claim that the convex simplex projection is critical to prevent "activation scale distortion or activation collapse" caused by unconstrained task vector scaling.
  - However, in Table 1, "Unconstrained Scaling" actually outperforms BPAM-Static by a significant margin of **+2.30%** absolute (71.51% vs. 69.21%), and "Unconstrained + Head Tuning" outperforms BPAM-Full by **+1.90%** (77.12% vs. 75.22%).
  - Critically, the authors provide no empirical evidence (such as weight norm trajectories, activation distributions, or attention map statistics) showing that unconstrained scaling actually experiences scale distortion or collapse under the evaluated conditions. Without such evidence, it is difficult to justify why a practitioner should accept a $\sim$2% performance penalty to use the constrained BPAM instead of simple, unconstrained scaling.
- **Conceptual Inconsistency regarding Ray-Scaling Sparsification:**
  - In Section 3.5 (Step 4), the authors argue that standard Euclidean orthogonal projection onto the simplex is undesirable because of its strong sparsification effect, which pushes multiple coordinates to exactly zero. They justify their ray-scaling projection by claiming it avoids this hard sparsification and preserves collaborative contributions.
  - However, in Section 4.5, they report that the converged coefficients for SVHN ($\lambda_5$) and MNIST ($\lambda_6$) are **exactly 0.0000**. This empirical result directly contradicts their theoretical motivation for preferring ray-scaling, as hard sparsification still occurs.
- **Underspecified Joint Co-adaptation Analysis:**
  - The authors discuss how the massive parameter scale imbalance (8 scalars vs. 388K head parameters) can cause the classification heads to dominate the co-adaptation process in BPAM-Full. They mention that their codebase was extended to support separate learning rates (`--head-lr`).
  - However, the paper does not provide any quantitative results demonstrating how varying the head learning rate ($\eta_{head}$) affects performance or whether it successfully bridges the performance gap with higher-capacity baselines.

---

## 3. Ratings

### Soundness: Good
The mathematical formulation is clear and correct, and the evaluation is symmetric. However, the soundness rating is capped at "Good" due to the lack of error bars/variance reporting, the missing empirical proof of activation collapse for unconstrained scaling, and the conceptual contradiction regarding ray-scaling sparsification.

### Presentation: Excellent
The paper is beautifully written, easy to follow, and exceptionally transparent. The related work is highly contextualized, and the mathematical notation is consistent and complete.

### Significance: Good
The paper's significance is solid. While it does not introduce a new SOTA method, it functions as a highly valuable diagnostic boundary probe that clarifies the exact threshold where layer-wise degrees of freedom become necessary. The CKA-based representation analysis also contributes meaningfully to our understanding of multi-task weight spaces.

### Originality: Good
The paper offers a novel "deconstructive audit" framework applied to the extreme parameter limits of test-time model merging. While the individual mathematical components (convex combinations, simplex projection) are standard, their synthesis and diagnostic evaluation represent a creative and insightful contribution.

---

## 4. Overall Recommendation
**Recommendation: 4: Weak accept**

**Justification:**
This paper is a highly polished, scientifically honest, and educational contribution to the model-merging literature. It successfully maps the exact boundaries where global task-wise scaling becomes too constrained to resolve parameter conflicts, requiring layer-wise adapters or auxiliary classification head tuning. The analysis of the "0-weight performance mystery" using CKA similarities is outstanding. 

However, the paper has notable empirical gaps—specifically, the lack of statistical error bars, the absence of empirical evidence demonstrating activation collapse in unconstrained scaling (which makes the proposed simplex constraint's performance penalty difficult to justify), and a conceptual contradiction on whether ray-scaling avoids hard sparsification. If the authors address these empirical weaknesses during the discussion phase, this paper would make a strong candidate for acceptance.

---

## 5. Detailed Comments and Questions for the Authors

1. **Statistical Rigor:** Test-time adaptation is heavily dependent on the specific calibration samples. Could you please run your experiments (Table 1, Table 3, and Table 4) across at least 3 to 5 random seeds/splits and report the mean and standard deviations? This is particularly crucial for the extreme low-data calibration (Table 4) and to confirm if the +0.11% gain of BPAM-Static over Task Arithmetic is statistically significant.
2. **Empirical Evidence of Activation Collapse:** You argue that your convex simplex constraint is a critical safeguard against activation scale distortion and collapse. Since "Unconstrained Scaling" outperforms BPAM-Static by +2.30% absolute, can you provide concrete empirical evidence (such as activation norm plots or maximum weight norms over epochs) showing that unconstrained scaling actually degrades activation distributions under some conditions?
3. **Ray-Scaling Sparsification Contradiction:** In Section 3.5, you state that ray-scaling is preferred over Euclidean orthogonal projection because it avoids hard sparsification. Yet, in Section 4.5, your converged coefficients for SVHN ($\lambda_5$) and MNIST ($\lambda_6$) are exactly 0.0000. How do you reconcile this contradiction? Does your optimization loop contain an explicit non-negativity clamp ($\lambda_k \leftarrow \max(0, \lambda_k)$) prior to ray-scaling that causes this hard pruning?
4. **Asymmetric Co-adaptation Results:** You discuss the optimization imbalance in BPAM-Full and mention that your framework supports separate learning rates (`--head-lr`). Could you provide a brief table or discussion of the performance sensitivity when varying the ratio between the head learning rate ($\eta_{head}$) and weight learning rate ($\eta_{weight}$)?
5. **Baselines in Part B:** In Table 1, what are the exact adapter parameter counts for "SyMerge + Head Tuning" and "FoldMerge + Head Tuning"? Please replace the "High" placeholder with explicit parameter counts for completeness.
