# Evaluation Task 2: Novelty and Literature Delta

## Key Novel Aspects
The paper introduces several highly novel elements to the field of test-time model merging and dynamic routing:
1. **Dynamic Information-Geometric Curvature for Routing**: Applying the diagonal empirical Fisher Information Matrix (dFIM) as a local coordinate-warping metric tensor *dynamically at test-time* to replace flat Euclidean/cosine similarity. This represents a novel synthesis of information geometry and dynamic weight ensembling.
2. **Smoothed and Power-Scaled Fisher Regularization**: Formulating a mathematically regularized inverse-variance metric ($\tilde{F}$) that is robust under extreme data scarcity (microscopic calibration splits, e.g., $N_c = 16$).
3. **Class-Size Scaling Calibration (CSC) and Micro-Batch Homogenization (MBH)**: Developing extreme-value-theory normalization and batch-partitioning mechanics specifically designed to stabilize parameter-free ensembling under asymmetric class sizes and sequential streams ($B=1$).

## The 'Delta' from Prior Work
The paper positions itself very clearly and systematically relative to existing literature:
- **Static Model Merging vs. Dynamic Routing**: Methods like Model Soups, Task Arithmetic, TIES-Merging, and DARE are inherently static and non-adaptive. They merge weights offline and cannot handle heterogeneous, mixed-domain streams dynamically. FIOSR operates dynamically at test-time on a sample-by-sample basis.
- **Parametric Test-Time Routers vs. Parameter-Free Routing**: Frameworks like FoldMerge, QWS-Merge, and L3-Softmax optimize parameters on a calibration split at test-time. This paper exposes that these methods are fundamentally flawed under extreme data scarcity (overfitting / "Dynamic Routing Paradox") and single-sample sequential streams (variance collapse / "Vectorization Collapse"). FIOSR is completely training-free and parameter-free, resolving both collapse modes.
- **Parameter-Free Subspace Routing (PFSR) vs. FIOSR**: PFSR (Euclid & Hilbert, 2026) bypassed optimization using unweighted cosine similarity, but this implicitly assumes a flat, isotropic weight space. FIOSR's delta is treating the space as a Riemannian manifold warped by Fisher Information, suppressing task-irrelevant noise dimensions.
- **Fisher Information in Deep Learning**: While Kirkpatrick et al. (Elastic Weight Consolidation) and Matena & Raffel (Fisher Merging) use Fisher Information, they do so for static offline ensembling or continual learning. This is the first work to use dFIM as an online, dynamic projection warping metric at test-time.

## Characterization of Novelty
The novelty of this work is **significant**. 
It is not merely an incremental tweak of an existing model merging heuristic; rather, it introduces a robust information-geometric foundation (Riemannian manifold warped by Fisher Information) to solve critical, well-documented engineering failure modes (few-shot overfitting and sequential stream instability) in dynamic model routing. Furthermore, the combination of CSC and MBH demonstrates a deep understanding of the practical and statistical constraints of deploying these models in low-latency real-world environments.
