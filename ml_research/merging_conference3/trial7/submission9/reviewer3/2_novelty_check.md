# Evaluation Component 2: Novelty and Delta Assessment

## 1. Key Novel Aspects of the SABLE Framework
The primary conceptual contribution of SABLE is the shift from **parameter-space ensembling** to **activation-space ensembling** inside the deep network's forward pass to resolve streaming heterogeneity. 

While parameter-space model merging (such as task arithmetic or PFSR) typically averages parameters over a batch dimension to execute a single forward pass, SABLE ensembles the activations on-the-fly. The key technical mechanisms introduced to make this feasible and scalable include:
1. **Sample-wise Activation Blending:** Formulating the linear combination of low-rank adapter updates dynamically per-sample using the distributive property.
2. **Layer-Dependent Hybrid-Rank Selection Protocol:** A unique architecture where hidden layers are compressed using strict low-rank adapters ($r \le 8$) while the final classification/projection layer is kept full-precision, resolving capacity bottlenecks.
3. **Refined Zero-Data Centroids:** Constructing task-level centroids by L2-normalizing pre-trained expert classification weights before taking their row-mean, preventing vector cancellation in a completely data-free manner.
4. **Early-Layer and Mid-Layer (Late Adaptation) Routing Options:** Structural variants designed to balance the trade-off between representational alignment and multi-layer task adaptation.

---

## 2. The "Delta" from Prior Work
We can systematically compare SABLE to existing paradigms along several axes:

* **Delta from PFSR (Parameter-Free Subspace Routing) [pfsr]:**
  * *PFSR:* Computes sample-wise routing coefficients but is forced to average them over the batch dimension ($\bar{\alpha}$) to perform a single weight-space merge. This causes severe *heterogeneity collapse* as batch size $B$ increases.
  * *SABLE:* Computes sample-wise routing coefficients but applies them in *activation space* ($Y_b = X_b W_{\text{base}} + \sum_k \alpha_{k, b} X_b A_k B_k$). This retains sample-level resolution, achieving 0.00% collapse regardless of batch size.
* **Delta from MBH (Micro-Batch Homogenization) [mbh]:**
  * *MBH:* A stateful systems-level scheduling layer that dynamically buffers, sorts, and partitions heterogeneous queries into homogeneous micro-batches to save weight-space merging from collapse.
  * *SABLE:* A stateless network-level ensembling algebra. It completely strips away temporal queues, buffers, sorting algorithms, and stateful serving dependencies.
* **Delta from MoE-Adapters [moeadapters] and LoraHub [huang2023lorahub]:**
  * *LoraHub:* Uses gradient-free search to find a static weight combination; it is non-adaptive at test-time and requires target-task calibration splits.
  * *MoE-Adapters:* Trains parametric gating networks, requiring a heavy multi-task training phase, and is highly sensitive to out-of-distribution (OOD) inputs.
  * *SABLE:* Completely non-parametric and calibration-free, utilizing frozen classification heads or pre-trained semantic embedders to route dynamically in a single-pass without training.

---

## 3. Characterization of Novelty
From a theoretical perspective, the novelty of SABLE is **primarily incremental and algebraic**, rather than fundamentally conceptual or mathematically transformative:

1. **Applied Algebraic Rearrangement:** The core mathematical foundation relies on the basic distributive property of linear algebra ($X(W_{\text{base}} + \sum_k \alpha_k A_k B_k) = X W_{\text{base}} + \sum_k \alpha_k (X A_k) B_k$). This identity is a standard property of matrix multiplication. The novelty lies not in the mathematics itself, but in its creative application to circumvent batch-level parameter averaging in deep learning serving.
2. **Heuristic Centroid Construction:** The "Refined Zero-Data Centroids" method is a neat geometric heuristic (L2-normalizing and averaging expert classification weights). However, it lacks a rigorous, mathematically derived probabilistic or statistical framework that formally bounds the approximation error between these weight-derived centroids and the true activation-space manifolds.
3. **Engineering-Driven Architecture:** The *Layer-Dependent Hybrid-Rank* strategy and *Top-$M$ Expert Pruning* are highly practical engineering solutions to real-world deployment challenges (capacity bottlenecks and CUDA kernel launch overhead). They are directly analogous to existing sparse MoE routing mechanisms (e.g., Top-k token gating) and do not represent a new class of deep learning theory or optimization.

**Conclusion on Novelty:** While SABLE is highly innovative as an engineering solution that elegantly solves a severe systems-level problem (heterogeneity collapse) at the network level, its theoretical and mathematical contribution is modest. It is an excellent example of applying simple, clean linear algebra to rearrange computations, but it does not introduce new theoretical paradigms, convergence proofs, or statistical guarantees.
