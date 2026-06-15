# Intermediate Review Evaluation: Novelty Check (2_novelty_check.md)

This document provides an assessment of the key novel aspects of the paper, the 'delta' from prior work, and a characterization of its overall novelty (ranging from incremental to significant).

---

## 1. Delta from Prior Work

The paper carefully positions its contribution relative to three distinct bodies of literature:

### A. Weight Ensembling and Task Arithmetic
* **Prior Work:** Traditional weight ensembling methods (e.g., Simple Weight Interpolation, Task Arithmetic, TIES-Merging, and DARE) are static and non-adaptive. They compute a single merged set of weights that cannot adapt dynamically to varying task compositions during test-time inference.
* **Delta:** FIOSR is a **dynamic, test-time** merging framework that computes sample-specific routing coefficients on the fly to guide parameter ensembling based on the characteristics of each incoming test sample.

### B. Dynamic Routing and Mixture-of-Experts
* **Prior Work:** State-of-the-art dynamic model merging methods (e.g., FoldMerge, QWS-Merge, and L3-Softmax) optimize routing coefficients at test-time over a calibration split using parametric routing networks. However, these parametric routers are highly susceptible to the *Dynamic Routing Paradox* (few-shot overfitting) and *Vectorization Collapse* (instability under single-sample sequential streams).
* **Delta:** FIOSR is completely **parameter-free and training-free**. It completely bypasses test-time optimization, making it entirely immune to overfitting and Vectorization Collapse.

### C. Parameter-Free Subspace Routing (PFSR)
* **Prior Work:** PFSR (2026) sidesteps the optimization bottleneck by using unweighted cosine similarity to project representations onto frozen classification weight anchors. However, unweighted cosine similarity implicitly assumes a flat, isotropic Euclidean weight space ($\mathbf{g} = \mathbf{I}$).
* **Delta:** FIOSR rejects the flat Euclidean assumption. It treats the parameter representation space as a ** Riemannian manifold** where local coordinate sensitivities are warped using an analytical metric tensor derived from the diagonal empirical Fisher Information Matrix (dFIM).

### D. Information Geometry in Deep Learning
* **Prior Work:** Classic methods like Elastic Weight Consolidation (EWC) use diagonal empirical Fisher Information Matrix (dFIM) offline to prevent catastrophic forgetting. Static ensembling methods like Fisher Merging use FIM to combine weights offline.
* **Delta:** FIOSR is the **first work** to utilize Fisher Information **dynamically at test-time** as a coordinate-warping metric tensor for parameter-free subspace routing, unifying information-geometric curvature with dynamic test-time model ensembling.

---

## 2. Characterization of Novelty

We characterize the novelty of this paper as **Significant and Conceptually Elegant**. It is not merely an incremental patch or combination of existing heuristics; rather, it introduces a clean, mathematically rigorous perspective that reframes parameter-free ensembling through the lens of Riemannian manifold theory.

### Why the Novelty is Significant:
1. **Theoretic-Empirical Synergy:** The paper does not just propose a coordinate weighting heuristic; it mathematically derives that the diagonal Fisher Information under a conditional Gaussian assumption corresponds *exactly* to the inverse coordinate variance ($F_j = 1/\sigma_j^2$). This provides a category-error-free information-geometric justification for coordinate warping.
2. **Dual-Space Mapping:** The paper addresses a potential conceptual gap—applying representation-derived metrics to warp classification weights—by proving a dual-space relationship. Under regularized cross-entropy training, classifier weights act as dual vectors that align closely with representation centroids, bounding finite-sample directional misalignment as $O_p(1/\sqrt{N_c})$.
3. **Rigorous Statistical Safeguards:** Instead of adjusting scores heuristically, the Class-Size Scaling Calibration (CSC) uses extreme value theory of Gaussian random variables projected onto a sphere to divide raw similarities by $\sqrt{2\log C_k / d}$, correcting for maximum bias under asymmetric vocabulary dimensions.
4. **Closing the Loop on Non-Gaussianity:** The appendix includes a rigorous derivation of Fisher Information under non-Gaussian rectified (ReLU) activations, proving that the inverse-variance relationship $F_j \propto 1/\sigma_j^2$ dominates coordinate sensitivity even under extreme noise and non-negative sparsity.
5. **Practical Real-world Stress Testing:** The authors did not rest on coordinate-aligned synthetic noise. They evaluated the framework under rotated, correlated noise, demonstrating the limitations of standard diagonal Fisher and proving the viability of on-the-fly Covariance EVD shrinkage alignment (**FIOSR-Online**). They also validated the framework on actual physical networks (ResNet-18) with real images (MNIST, FashionMNIST, SVHN) using global pre-calibration mean-centering, which successfully addresses the translation bias.

### Summary:
FIOSR represents a substantial leap forward in the test-time model merging literature, replacing over-parameterized optimization routines and flat geometric assumptions with a mathematically rigorous, training-free, and highly stable information-geometric coordinate-warping projection.
