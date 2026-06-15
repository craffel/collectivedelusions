# 2. Novelty and Originality Check

## Originality of the Concepts
The core concepts in the paper represent a highly original combination of **information geometry**, **Riemannian manifold theory**, and **modular test-time parameter ensembling**:
1. **Riemannian Manifold Framing:** Framing the neural network representation and weight space as a Riemannian manifold rather than a flat Euclidean space is mathematically principled. While the Fisher Information Matrix (FIM) has been used in deep learning for regularized training (like Elastic Weight Consolidation) and natural gradient descent, its application as an **on-the-fly analytical coordinate-warping local metric tensor** for test-time subspace routing is highly creative and novel.
2. **Category-Error-Free Formulation:** The authors resolve a common conceptual confusion in representation-space metric filtering by demonstrating that under a conditional Gaussian assumption, the dFIM corresponds exactly to the inverse coordinate noise variance ($F_{j} \propto 1/\sigma_{j}^2$). This elegant connection provides a solid theoretical justification for using representation-derived variances to warp weight-space similarities.
3. **Micro-Batch Homogenization (MBH) and Class-Size Scaling Calibration (CSC):** 
   - **CSC** elegantly uses extreme value theory of Gaussian random variables projected onto a sphere to resolve the statistical maximum bias under asymmetric class vocabulary dimensions.
   - **MBH** provides a novel, unsupervised, stream-level batch partitioning protocol that allows sample-wise parameter-free routing to remain computationally efficient ($G \le K$ forward passes) while resolving the **Vectorization Collapse** and **Heterogeneity Collapse** issues that plague standard dynamic merging.

## Differentiating from Prior Work
The paper positions itself very clearly against several closely related prior works:
- **Parameter-Free Subspace Routing (PFSR):** PFSR computes raw, unweighted cosine similarity between representations and frozen class prototype vectors. By using standard cosine similarity, PFSR implicitly assumes a flat Euclidean space. FIOSR differs by constructing a local Riemannian metric $\mathbf{g} = \text{diag}(\tilde{F})$ that warps this space, successfully suppressing noisy coordinates (which have low Fisher sensitivity) and magnifying clean coordinates. The paper isolates the causal benefit of this warping, showing that FIOSR outperforms the PFSR baseline by a massive **+8.56%** absolute accuracy under homogeneous batching.
- **Parametric Routers (Linear Router, QWS-Merge, L3-Softmax):** Parametric routers optimize projection weights on the calibration split. Under few-shot regimes (e.g., $N=64$), they suffer from severe overfitting (the Dynamic Routing Paradox), collapsing to uniform performance. FIOSR, being completely parameter-free and training-free, requires zero test-time optimization and completely bypasses this overfitting failure mode.

## Novelty Rating: Excellent
The paper provides a refreshing, highly rigorous mathematical framing of test-time model merging. Instead of introducing another empirical heuristic, the authors ground their ensembling rules in information-geometric and statistical principles. The theoretical justifications and novel formulations (smoothed dFIM regularizer, CSC, MBH) demonstrate significant conceptual advancement.
