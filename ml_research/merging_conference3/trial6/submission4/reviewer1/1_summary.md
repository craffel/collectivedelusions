# 1. Summary

## Main Topic and Motivation
The paper addresses a critical optimization challenge in **dynamic model merging**, where lightweight routing layers (e.g., L3-Router, QWS-Merge) are calibrated on extremely data-scarce splits (e.g., $B_{cal} \le 64$ samples) to dynamically scale task-merging coefficients on the fly. The authors expose a previously undocumented vulnerability: unconstrained routing parameters overfit aggressively to the local noise of calibration samples under severe scarcity, leading to representation-space collapse and failure to generalize to out-of-distribution (OOD) tasks.

## Proposed Approach
To mitigate this overfitting, the authors introduce **Task-Space Anchor Regularization (TSAR)**. The core idea is to anchor the layer-wise routing weights to the pre-computed centroids (anchors) of pre-trained expert representations in a low-dimensional projection space.
- **Low-Dimensional Space:** High-dimensional pooled features ($D=192$) are projected into a low-dimensional subspace ($d = K = 4$) via PCA or Random Gaussian projection and normalized onto the unit sphere.
- **Task Feature Anchors:** Task centroids are computed as the average of the low-dimensional projected features for each task over the calibration split.
- **TSAR Penalty:** A quadratic distance penalty pulls layer-wise routing weights toward their corresponding task-space centroids during optimization, scaled by a hyperparameter $\lambda_{anchor}$.
- **PCGrad Integration:** To resolve multi-task gradient cross-talk and hard-task dominance, the authors integrate **Projecting Conflicting Gradients (PCGrad)** during router calibration.

## Key Findings and Claims
1. **Exposure of Low-Data Overfitting:** Unregularized dynamic routers suffer from catastrophic representation-space collapse when calibrated on data-sparse splits ($B_{cal} \le 64$).
2. **Superior Performance of TSAR:** Under homogeneous batch deployment ($B_{cal}=64$), the L3-Linear + TSAR router achieves a Joint Mean accuracy of 54.10% (and 57.06% when combined with PCGrad), outperforming standard $L_2$-regularized linear routing (44.72%) and QWS-Merge (39.88%).
3. **Failure of Complex Formulations:** The highly complex, "quantum-inspired" SOTA method (QWS-Merge) is shown to perform worse than static uniform merging due to its highly non-convex, non-monotonic optimization landscape.
4. **Layer-Averaging Collapse:** The authors mathematically prove that layer-wise dynamic routing coefficients averaged across layers collapse deployment-time representation capacity to a single-layer global router, rendering multi-layer routers mathematically redundant.
5. **Mitigation of Heterogeneity Collapse:** In heterogeneous mixed-task streams, unconstrained dynamic routers suffer from "heterogeneity collapse" due to batch averaging. This is resolved by using non-negative scaled Sigmoid activations, which achieve 50.80% joint accuracy.
6. **Physical Model Merging:** The authors report that TSAR + PCGrad outperforms Static Uniform Merging by +13.90% on merging classification heads of a real pre-trained Vision Transformer (ViT-Tiny), and +23.60% on raw natural images.

## Explicitly Claimed Contributions (with Evidence)
- **Exposure of low-data overfitting:** Demonstrated by comparing unconstrained Global Linear Router (23.20%) vs. regularized baselines in Table 1.
- **Formulation of TSAR:** Described mathematically in Section 3 and validated empirically in Section 4.2.
- **Empirical performance gains:** Detailed in Table 1, showing that TSAR + PCGrad achieves 57.06% Joint Mean accuracy.
- **Detailed audits (Sensitivity, Complexity, Stream):** Validated in Tables 2, 3, and 4, showing robustness to $\lambda_{anchor}$ and scaling dynamics with $B_{cal}$ and stream types.
- **Massive-Scale Scalability and Physical Validation:** Validated in Section 4.6, Appendix B, and Appendix F, showing successful scaling to $K=20$ tasks and head-level merging on a real ViT.
