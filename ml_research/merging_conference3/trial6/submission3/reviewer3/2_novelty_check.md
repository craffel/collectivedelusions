# Novelty Check

## Key Novel Aspects of the Work
1. **Block-wise Weight Sharing in Parameter Space:** Instead of learning independent routing parameters for every single layer (unshared) or forcing a single global router, the paper proposes a uniform block grouping of size $M$, sharing routing parameters inside each block. This is a novel, highly efficient compromise that balances capacity and generalization.
2. **Deconstruction and Formalization of "Coefficient Ruggedness":** The paper provides a mathematical formalization of layer-to-layer coefficient variations as *coefficient ruggedness*, defining a general expected ruggedness model that incorporates depth-dependent variance scales and adjacent block correlations.
3. **Physical Sequential Model-Merging Evaluation:** While previous works relied on virtual-layer weight-space ensembling sandboxes (where coefficients are averaged over the layer dimension before applying them), this paper designs and evaluates a true sequential propagation setup where weights are physically blended at each layer during runtime forward propagation. This setup is a major step toward practical deep network deployment.
4. **Learnable Task Scaling Ceiling ($\lambda_{max}$):** Allowing the task vector scaling ceiling $\lambda_{max}$ to be learned end-to-end via gradient descent, which is shown to be highly stable and superior to static manual tuning.
5. **Practical GPU-Level Implementation Recipe and Pilot Demonstration:** Providing a concrete implementation guide and actual latency profiling for BWS-Router on physical Vision Transformers.

---

## The "Delta" from Prior Work
- **Delta from L3-Router (Vance & Carter, 2025):** L3-Router learns independent routing networks for each layer of the backbone. BWS-Router modifies this by sharing weights across layer blocks of size $M$. This reduces the parameter count from $L \cdot K \cdot (d + 1)$ to $(L / M) \cdot K \cdot (d + 1)$—a 66.7% reduction for $M=3$ and a 91.7% reduction for $M=12$. Furthermore, it is shown that BWS-Router prevents sequential feature distortion, which unshared L3-Router suffers from under data-scarce calibration (e.g. SVHN collapse).
- **Delta from QWS-Merge (Schrödinger et al., 2024):** QWS-Merge uses wave-superposition routing with non-monotonic trigonometric equations. BWS-Router shows that QWS-Merge has high complexity and exhibits severe optimization instability across random seeds. BWS-Router replaces this with regularized classical linear projections combined with bounded Sigmoidal or Softmax gating, which is much more stable and simple.
- **Delta from BC-Router (Vance et al., 2025):** BC-Router analyzed Softmax gating for zero-sum bottlenecks. BWS-Router provides a systematic evaluation and selection rule between Softmax and Sigmoidal gating across closed-world (classification) and open-world (OOD deactivation/multi-task mixing) scenarios.
- **Delta from Static Merging (e.g., TIES, Task Arithmetic):** Static merging combines weights once at compile time. BWS-Router performs input-dependent parameter blending at runtime, which resolves severe weight-space semantic conflicts under which static uniform merging completely collapses (recovering up to ~80% accuracy vs. 23.5% for static uniform).

---

## Characterization of Novelty
The novelty of this paper can be characterized as **significant and highly grounded**.
- Rather than proposing a completely new mathematical metaphor (such as QWS-Merge's quantum wavefunction analogy), the authors focus on a rigorous deconstruction of existing frameworks, identifying clear failure modes (such as overfitting on small calibration splits, cascading representation drift, and parameter excess).
- The solution—block-wise weight sharing—is elegant, highly intuitive, and deeply practical. It addresses the overparameterization of unshared routing while retaining functional specialization.
- The transition from virtual-layer sandboxes to physical sequential weight blending represents a highly significant delta, as it moves the model-merging literature from stylized simulations to realistic deep sequential propagation environments.
- The extremely thorough nature of the empirical sweeps (over block sizes, activations, regularization, bias initialization, PCA dimension, non-linear kernels, and expert scaling) elevates the work beyond a simple incremental tweak, making it a comprehensive manual/guide for practitioners.
- Thus, the novelty is not merely "incremental weight-sharing," but a foundational, empirical deconstruction and stabilization of sequential dynamic weight-space ensembling.
