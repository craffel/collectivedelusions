# Evaluation Task 2: Novelty and Delta from Prior Work

## 1. Characterization of Novelty
The submission exhibits a high degree of **conceptual and mathematical originality**. Rather than presenting an incremental variation of existing linear weight-averaging techniques (such as SLERP, TIES-Merging, or Git Re-Basin), the paper introduces an entirely new paradigm at the intersection of deep neural network model merging and **Vector Symbolic Architectures (VSA) / Hyperdimensional Computing (HDC)**. 

We characterize this novelty as **highly significant and conceptual**. The paper does not merely propose a minor heuristic; it reframes weight space as a holographic associative memory. This is a foundational departure from traditional post-hoc merging frameworks.

---

## 2. Key Novel Aspects and "Delta" from Prior Work

The paper's key novel contributions and their specific deltas from the state of the art are:

### A. Bridging Hyperdimensional Computing and Deep Parameter Spaces
- **Delta:** Classical VSAs and HDC frameworks (e.g., Plate’s Holographic Reduced Representations, Kanerva’s Hyperdimensional Computing) have been used primarily for symbolic cognitive modeling, distributed associative representation of features, and vector-based database lookups. 
- **Novelty:** EHPB extends these high-dimensional binding and superposition operators from 1D feature vectors to 2D neural network parameter matrices, enabling the superposition of multiple full expert networks into a single parameter substrate.

### B. Exposing and Formalizing "Heterogeneity Collapse"
- **Delta:** While the deep learning community is aware that keeping multiple expert models in memory is expensive and that dynamic routing can suffer from throughput bottlenecks, the specific runtime/hardware-level phenomenon of ensembling coefficients averaging out across a heterogeneous batch has not been formally characterized or named.
- **Novelty:** This paper explicitly exposes and defines *heterogeneity collapse* under streaming mixed-task workloads. It builds an index-shuffled evaluation framework to benchmark this failure mode, demonstrating that standard framework runtimes flatten expert specialization under batched deployment.

### C. The Post-Hoc Model Ensembling Trilemma
- **Delta:** Prior papers typically evaluate merging methods on a simple two-dimensional scale: accuracy vs. parameters.
- **Novelty:** The authors formulate a new theoretical framework: the **Post-Hoc Model Ensembling Trilemma**. This trade-off space is spanned by three mutually competing desiderata:
  1. *Dynamic Adaptability (D)* (sample-wise gating $\alpha(x)$)
  2. *Resource Efficiency (R)* ($O(P)$ active memory footprint independent of $K$)
  3. *Weight Integrity (W)* (preservation of expert coordinates, $\Xi = 0$)
  
  The paper elegantly demonstrates that existing methods only satisfy two out of three:
  - *Static Merging* (Uniform, TIES) satisfies **R + W** (no noise, $O(P)$ size, but static).
  - *Direct Sample-wise Routing* satisfies **D + W** (dynamic, no noise, but $O(KP)$ memory).
  - *EHPB* satisfies **D + R** (dynamic, $O(P)$ memory, but sacrifices Weight Integrity by introducing reconstruction noise $\Xi$).

### D. Deconstruction of the Coordinate Isolation Confounder
- **Delta:** Most papers introducing a new method gloss over its limitations or present highly selective empirical tables. 
- **Novelty:** This paper systematically deconstructs *why* EHPB's reconstruction error remains scale-invariant across hidden dimensions ($D \in [64, 2048]$). It identifies the **Coordinate Isolation Confounder**—showing that element-wise Hadamard binding isolates coordinates and prevents the central-limit-averaging noise decay ($O(1/\sqrt{D})$) that classical VSAs enjoy. It provides an empirical proof-of-concept demonstrating that a transition to circular convolution operators indeed restores the noise decay in associative retrieval, offering a clear roadmap for future hyperdimensional weight-space ensembling.

### E. Specialized Mitigation Frameworks
- **Delta:** Existing noise cleanup methods typically operate on inputs or outputs.
- **Novelty:** The authors design and evaluate three custom-tailored post-hoc mitigations to combat weight superposition noise:
  - *Residual-EHPB* (with an innovative hardware-friendly *Structured Row-wise Residual-EHPB* variant).
  - *Continuous Cleanup Networks (CCN)* (layer-wise linear/bottleneck MLP activation-space denoisers).
  - *ReLU Bias Correction* (post-activation running subtraction or learnable coordinate-wise scale/shift).
