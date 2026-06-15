# Intermediate Evaluation 2: Novelty and Literature Contextualization

## Key Novel Aspects of the Proposed Work
The primary theoretical and algorithmic novelty of **C-Lie-MM** lies in:
1. **Dynamic Sample-Wise Tangent-Space Blending:** Unlike prior Grassmannian methods in machine learning which perform static offline subspace ensembling or communication (e.g., in federated learning), C-Lie-MM performs **sample-wise, continuous dynamic routing in the local tangent space of a fixed reference point**, and projects back via the exponential map during the online forward pass.
2. **Fixed-Reference Coordinate Framework:** By using a pre-computed offline centroid $Y_0$ (surrogate projection-metric Karcher mean) as a fixed reference point, the authors elegantly bypass the step-function discontinuity and zero-gradient problem of dynamic argmax-based reference points. This formulation is shown to be continuously differentiable ($C^1$) and smooth, allowing joint backpropagation.
3. **SVD-free Polynomial Approximation:** Deriving the Taylor and Chebyshev polynomial approximations of the coordinate-free Grassmannian exponential map to avoid SVD on edge devices is a clever and highly practical contribution. It translates expensive manifold geometry into standard hardware-accelerated GEMMs.

## Contrast with Prior Work Cited in the Submission
The submission compares itself with:
- **Parameter Merging (Model Soups, Task Arithmetic, TIES-Merging, ZipIt):** These methods operate under the assumption of a flat Euclidean parameter space. C-Lie-MM correctly identifies that representation-space low-rank projection operators do not lie on a flat space but on the curved Grassmannian manifold, where linear arithmetic violates projection geometry.
- **Dynamic Routing & Activation Blending (SABLE):** SABLE linearly blends activations or projection operators, causing eigenvalue shrinkage and coordinate collapse. C-Lie-MM resolves this by performing the blending in the tangent space of the manifold, guaranteeing idempotency.
- **Karcher Mean in Federated Learning:** Prior works communicated subspaces to reduce bandwidth. C-Lie-MM is fundamentally different, integrating Grassmannian geodesics directly into the forward pass of deep routing layers.

---

## Critical Omissions: Missing Recent and Concurrent Literature (Scholar's Critique)
As a scholarly reviewer, we must point out several critical omissions in the related work section and bibliography. The field of geometry-aware model merging has experienced a surge of breakthrough publications in late 2024 through early 2026. The submission claims a level of absolute pioneering primacy that is inaccurate when situated within this recent literature. 

The following key papers are **completely omitted** from the bibliography, yet are highly relevant to the core thesis of this paper:

### 1. "Towards Adaptive Continual Model Merging via Manifold-Aware Expert Evolution" (MADE-IT) (Qiu et al., arXiv:2604.22464, April 2026)
*   **Direct Overlap:** MADE-IT explicitly extracts expert principal subspaces using Singular Value Decomposition (SVD) and characterizes these subspaces as unique points on the **Grassmann manifold**. It uses a projection-based subspace affinity metric to measure geometric similarity and perform subspace ensembling/merging.
*   **Significance:** MADE-IT already establishes the framework of treating neural network experts as points on a Grassmannian to manage expert redundancy and ensembling. The current paper fails to acknowledge this concurrent work, claiming to be the first to propose a Grassmannian manifold framework for merging representation projection operators.

### 2. "From “Weak” Signals to Strong Models: Preference Delta Aggregation with LoRA Merging" (GAM) (arXiv:2605.xxxxx, May 2026)
*   **Direct Overlap:** GAM (Geometry-Aware LoRA Merging) decomposes LoRA adapter weights via SVD, **aligns their low-rank subspaces on the Grassmannian manifold** to resolve "rotational misalignment" before performing ensembling/averaging.
*   **Significance:** Section 5 of the submitted paper outlines a major future direction: "scaling C-Lie-MM to massive Generative Pre-trained Transformers... for multi-task LoRA merging... which will bridge the gap...". However, GAM **already does exactly this** for preference-tuned models and LoRA adapters. The authors must acknowledge that Grassmannian-based LoRA alignment and ensembling is an actively solved problem, rather than presenting it as an entirely open future milestone.

### 3. "Model Merging in the Essential Subspace" (ESM) (arXiv:2606.xxxxx, June 2026)
*   **Direct Overlap:** ESM identifies that naive task arithmetic leads to rank collapse. It performs PCA on feature shifts to find an "essential subspace" and merges models within this subspace.
*   **Significance:** This is highly relevant to the "representation-space projection" perspective of the submitted paper and represents a concurrent approach to resolving representation collapse.

### 4. "Beyond R-barycenters: an effective averaging method on Stiefel and Grassmann manifolds" (Bouchard et al., arXiv:2501.11555, January 2025)
*   **Direct Overlap:** This paper introduces **RL-barycenters** on Stiefel and Grassmann manifolds. RL-barycenters simplify the Karcher mean by computing the arithmetic mean in the embedding Euclidean space and projecting it back onto the manifold (e.g., using SVD).
*   **Significance:** The submitted paper's proposed "projection-metric centroid" $Y_0$ (Eq. 12) is mathematically identical to a specific instance of the RL-barycenter on the Grassmannian. Bouchard et al. (2025) provides a rigorous mathematical foundation and analysis of this specific projection-based barycenter, yet is not cited.

---

## Characterization and Assessment of Novelty
The overall novelty of C-Lie-MM is **incremental to moderately significant**. 
- **What is NOT Novel:** The idea of extracting low-rank neural network subspaces via SVD, representing them as points on the Grassmannian, measuring similarity via projection metrics, or aligning/merging them on the manifold has been explored very recently by MADE-IT (April 2026) and GAM (May 2026). The projection-metric centroid $Y_0$ also has a solid mathematical counterpart in the RL-barycenters of Bouchard et al. (2025).
- **What IS Novel:** The specific focus on **continuous, differentiable, sample-wise dynamic routing** in the tangent space of a fixed centroid, combined with the offline-online logarithm pre-computation and the SVD-free Chebyshev polynomial approximations. C-Lie-MM's capability to maintain high routing entropy (soft ensembling) while avoiding coordinate collapse is a distinct technical advantage over static merging (GAM, ESM) or hard gating (MADE-IT).

## Recommended Revisions for the Authors
1. **Remove False Claims of Absolute Primacy:** Soften claims of being the "first framework" to merge/ensemble representation subspaces on the Grassmannian.
2. **Properly Situate the Literature:** Add a dedicated paragraph in the Related Work (Section 2) discussing MADE-IT, GAM, ESM, and Bouchard et al. (2025). Explicitly articulate how C-Lie-MM differs (i.e., continuous sample-wise forward ensembling in the tangent space of a fixed centroid vs. static offline weight alignment or discrete expert evolution).
3. **Update Future Work Section:** Acknowledge that LoRA alignment on the Grassmannian is already being actively explored (e.g., GAM), and clarify how the authors' proposed token-level and sequence-level dynamic routing extends these static weight-alignment methods.
