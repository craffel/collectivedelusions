# Novelty and Positioning Check

## Conceptual Novelty and Delta
The core idea of PG-Merge is to apply a dynamic sparse gradient mask to test-time model merging coefficients. Instead of optimizing all $L \times K$ layer-wise coefficients (56 parameters in the current setup), the optimizer only updates the top-$p\%$ with the largest absolute gradient magnitudes on each batch.
*   **Novelty Characterization:** While gradient pruning and sorting based on absolute magnitudes are well-established techniques in standard supervised training, communication-efficient distributed optimization, and parameter-efficient fine-tuning (PEFT), the paper's main novelty lies in **repurposing this mechanism as an analytical regularizer** to resolve the Overfitting-Optimizer Paradox in test-time model fusion. This is a clever and highly practical connection.
*   **The Delta from Prior SOTA:** Unlike RegCalMerge, which adds auxiliary spatial penalties (Elastic Spatial Regularization) with delicate weight-scaling hyperparameters, PG-Merge does not modify the loss objective. Unlike PolyMerge, which restricts coefficients to a low-degree polynomial trajectory over network depth, PG-Merge operates without imposing rigid global geometric priors. The delta is substantial: it strips away the mathematical scaffolds of prior work to demonstrate that simple coordinate restriction is highly performant.

## Literature Positioning
The related work section places PG-Merge clearly in the context of:
1.  **Static Model Merging:** (Task Arithmetic, Model Soups, TIES-merging).
2.  **Active/Test-Time Model Merging:** (AdaMerging, RegCalMerge, PolyMerge, QWS-Merge).
3.  **Test-Time Adaptation (TTA):** (Tent, etc.).
4.  **Gradient Pruning & Sparse Optimization:** (LoRA, MECTA, etc.).

### Strengths in Positioning
*   **The Rigid Subspace Critique:** The paper does a remarkable job of positioning itself against PolyMerge. It points out that PolyMerge's rigid polynomial constraint fails when merging properly converged experts, where layer-wise gradient dynamics are complex and non-linear. This critique is heavily backed by the revised experiments where PolyMerge collapses on MNIST (13.48%) and yields a poor joint average of 46.97%.
*   **Vindication of Occam's Razor:** By demonstrating that PG-Merge ($p=0.05$) outperforms RegCalMerge (62.70% vs. 62.35%) without using Class-Capacity Normalization or Elastic Spatial Regularization, the paper makes a strong, minimalist case that current regularizers are unnecessarily complex.

### Areas for Improvement / Discussion Gaps
*   **The Sparsity Ratio ($p$) as a Hyperparameter:** The paper claims PG-Merge is "non-parametric" and "adds zero hyperparameter bloat." However, the sparsity ratio $p$ is a critical hyperparameter. While it is more intuitive than the penalty weight $\lambda$ in RegCalMerge, it still must be chosen. In an online test-time scenario, we do not have access to validation labels to tune $p$. The authors should discuss how a practitioner can select or adapt $p$ on-the-fly without ground-truth labels.
*   **Task-wise vs. Layer-wise Sparsity:** Since the mask is applied globally across the $L \times K$ coefficient tensor, certain layers or tasks might be disproportionately frozen. Discussing whether a per-task or per-layer sparsity constraint would be beneficial could strengthen the conceptual positioning.
