# Revision Plan: Addressing Mock Review Feedback

We are revising our draft to resolve all weaknesses highlighted by the Mock Reviewer (Reviewer 2, score 3). Below is our prioritized plan:

## 1. Align Mathematical Formulation and Implementation (Critical Flaw 1 & Weakness 1)
- **Mathematical Dimension Correction:** Update Equation 14 to show that the diagonal scaling matrix $D_l$ is multiplied on the right side of the task vector sum: $(\sum \lambda_t \Delta W_{t, l}) D_l$. This resolves the dimension conflict where $W$ is $d_{\text{out}} \times d_{\text{in}}$ and $D_l$ is $d_{\text{in}} \times d_{\text{in}}$.
- **Reconstruction Objective Clarification:** Correct Equation 13 to formally state the end-to-end representation reconstruction loss on the final latent embeddings ($f_{\text{FP32}}(X)$ vs. $f_{\text{hybrid}}(X)$) instead of intermediate layer-wise activations.
- **Scientific Honesty and Justification:** Add a discussion in Section 3.3 explaining that for compact models (such as ViT-B-32 with 86M parameters), end-to-end feature-reconstruction is highly effective as it directly preserves semantic similarity in downstream representation space. Explain how it can be scaled block-wise for large models (LLMs) to avoid memory overhead.

## 2. Eliminate Confounding Variables in Table 1 & Table 2 (Critical Flaw 2)
- **New Baseline Addition:** Formally introduce the "FP32 Merged Bound (Optimized $\lambda$)" baseline alongside the "FP32 Merged Bound (Uniform, $\lambda_t = 0.5$)" baseline.
- **Scientific Clarification:** Explain that the optimized FP32 baseline achieves 95.12% accuracy on average (MNIST 99.04%, SVHN 91.20%).
- **Re-evaluate Claims:** Reframe the INT8 QP-Merge results. Show that QP-Merge INT8 (95.08%) is virtually lossless compared to the *optimized* FP32 baseline (95.12%, a drop of only 0.04%), while naive INT8 quantization achieves 94.93% (matching uniform FP32). Remove any scientifically unsubstantiated claims about quantization acting as a "regularizer", attributing the accuracy gain correctly to coefficient optimization.

## 3. Scale-Up and Edge deployment Discussions (Critical Flaw 3 & Presentation Weaknesses)
- **SOTA PTQ Comparisons:** Discuss how advanced single-model PTQ methods (e.g., AWQ, SmoothQuant, AdaRound) are "merging-blind" and do not address task interference or weight shifts in multi-task parameters.
- **Outlier Densities & Multi-task Scaling:** Add a paragraph in Section 5 (Discussion) analyzing the scaling properties of QP-Merge. Discuss how the union of outliers scales as more tasks are added (e.g., $T=8$ or $10$), and how density-accuracy-latency trade-offs can be managed pragmatically.
- **Edge Acceleration Details:** Provide deeper analysis of standard hardware execution layouts (INT4 dense Tensor Cores + SpMM sparse multiplier) and compile storage ratios to substantiate the real-world edge execution claims.
- **Calibration Sensitivity Discussion:** Mention the robust convergence characteristics of QE-Calib over random calibration draws and the effect of calibration size $M$ on generalizability.
