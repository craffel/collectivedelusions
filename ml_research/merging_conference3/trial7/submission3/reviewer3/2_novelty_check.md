# 2. Novelty Check and Delta Analysis

## Key Novel Aspects
The paper introduces two primary novel components to the post-hoc dynamic model merging literature:
1. **Gaussian Process Dynamic Routing (GP-DR):** Applying non-parametric Gaussian Process regression (GPR) to map representational features to model-merging routing weights. It models discrete one-hot task targets with a continuous Gaussian likelihood to yield a closed-form analytical posterior mean (for routing) and posterior variance (for uncertainty estimation).
2. **Micro-Batch Homogenization (MBH):** A buffer-level partitioning scheme that groups heterogeneous incoming streaming inputs into homogeneous micro-batches before they enter the modular neural network, specifically targeting "vectorization collapse" under batch execution.

## Delta from Prior Work
The proposed method is closely related to and builds directly upon several recent frameworks, with the following key differences:

* **Delta from PFSR (Parameter-Free Subspace Routing):**
  * GP-DR utilizes the exact same coordinate subspace projection as PFSR (measuring maximum cosine similarity to pre-computed class prototypes or centroids).
  * However, PFSR applies a sharp softmax with a low temperature directly to these similarity coordinates. GP-DR instead feeds these coordinates as inputs to independent GP priors with an RBF (or other) kernel, deriving analytical posterior mean blending weights.
  * Unlike PFSR, which has no mechanism for uncertainty quantification and is "blindly confident," GP-DR introduces a closed-form posterior variance $\sigma^2(\psi_*)$ to trigger an out-of-distribution (OOD) rejection fallback to a flat uniform prior.
  * GP-DR also introduces theoretical proofs of sum-to-one consistency and global/localized Lipschitz-smoothness bounds, which PFSR lacks.

* **Delta from Parametric Routers (TSAR, L3-Linear, L3-Softmax):**
  * Parametric routers train small neural networks (such as linear projections or MLPs) via gradient descent on small validation/calibration splits.
  * GP-DR bypasses training entirely. It treats the calibration set as static spatial landmarks on the representation manifold, resolving optimal routing weights in a single, training-free matrix pass.

* **Delta from Standard Batch Streaming:**
  * Standard streaming pipelines pass heterogeneous batches directly through the network backbone. Under task diversity, this causes representation-averaging and routing collapse.
  * MBH intercepts the streaming batch buffer, partitions the batch based on predicted task index argmaxes, and forwards task-homogeneous micro-batches independently, isolating representation spaces.

## Characterization of Novelty
The novelty of this submission is mixed and can be characterized as follows:

* **Incremental Modeling on Prior Subspace Work:** 
  The core representation-space projection of GP-DR is identical to PFSR. Placing a standard continuous Gaussian Process regression layer on top of this pre-existing coordinate space is an incremental modeling step. The mathematical machinery of GPR (kernels, Cholesky solvers, marginal likelihoods) is standard and well-established in spatial statistics and classical machine learning. Wrapping a pre-existing non-parametric router (PFSR) in GPR equations does not represent a fundamental paradigm shift.

* **"Negative Novelty" of GPR Complexity:**
  From a perspective that values elegant simplicity, the introduction of GPR is a highly complex way to solve a problem that simpler heuristics address more effectively. Specifically:
  1. The GP-DR posterior mean shrinks predictions toward the uniform prior, which allows irrelevant classification heads to compete, dragging in-distribution accuracy down below PFSR (e.g., $72.40\%$ vs. $77.60\%$).
  2. The GPR posterior variance is mathematically and empirically shown to suffer from "unit-sphere variance collapse," rendering it blind to realistic unit-sphere noise.
  3. The paper's own experiments (Table 8) show that **simple distance-based heuristics (like 5-Nearest Neighbor distance) substantially outperform GP-DR's posterior variance by a massive margin under representational overlap and coupling.**
  Therefore, the elaborate GPR mathematical framework (Cholesky solvers, non-negative variance clamping, diagonal jitter, Lipschitz proofs, and alternative complex kernels) represents unnecessary architectural complexity and mathematical obfuscation. A simpler and more effective design would combine PFSR's simple routing with a k-NN distance check for OOD fallback, completely bypassing GPR.

* **Pragmatic systems-level Novelty (MBH):**
  The identification of "vectorization collapse" under heterogeneous streams is highly insightful, and the proposed MBH is a pragmatic, systems-level contribution. However, partitioning the batch and executing sequential micro-batch forward passes is a brute-force approach. It fractures the parallel vectorized execution graph, introducing a substantial wall-clock latency penalty ($1.75\times$ on CPU, and up to $3.2\times$ on GPU). While functionally novel, it adds significant operational complexity compared to finding a parallel, single-pass mathematical regularizer that prevents representation collapse without splitting the batch.
