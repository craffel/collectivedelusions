# 1. Summary of the Paper

## Main Topic and Problem Statement
The paper addresses the challenge of **dynamic test-time model ensembling** across heterogeneous, non-stationary task streams. When a model serves sequential queries that exhibit temporal coherence (e.g., domain shifts or user task transitions), stateful routing is required to balance stability (filtering out query-level representation noise) with plasticity (rapidly adapting under task switches). 

Prior stateful ensembling routers (e.g., Momentum-Merge, ChemMerge) operate in unconstrained flat Euclidean spaces ($\mathbb{R}^K$) and project their states onto the probability simplex post-hoc via Softmax normalization. The authors argue that this unconstrained-to-constrained mismatch causes:
1. **Representational Lag (Hysteresis):** Accumulation of flat-space inertia, which must "unwind" before the Softmax output reflects a task transition.
2. **Geometric Scale Mismatches:** Euclidean interpolation between probability vectors wiggles off the natural probability manifold, altering the norm and scale of intermediate feature activations.
3. **High-Frequency Jitter:** Attempts to mitigate lag by increasing responsiveness introduce severe high-frequency oscillations in routing decisions.

## Proposed Approach: Unitary Geodesic Routing (UGR)
To resolve these limitations, the paper introduces **Unitary Geodesic Routing (UGR)**, a non-Euclidean geometric approach that models ensembling states directly on the curved $(K-1)$-dimensional unit hypersphere $\mathbb{S}^{K-1} \subset \mathbb{R}^K$. 

Key technical components of UGR include:
1. **Born Simplex Projection:** Instead of using a Softmax layer, UGR maps the spherical state vector $\mathbf{s}_t^{(l)}$ coordinate-wise to the probability simplex using the square-root homeomorphism from Information Geometry: $\alpha_{k, t}^{(l)} = (s_{k, t}^{(l)})^2$. This natively satisfies the simplex constraints with zero geometric distortion or scale-mismatch artifacts.
2. **Closed-Form Geodesic Updates:** State updates are modeled as continuous transitions along the shortest curved geodesic path (great-circle) using a closed-form, Rodrigues-like Spherical Linear Interpolation (Slerp) operator, bypassing expensive numerical ODE solvers or matrix exponentials.
3. **Torque-Driven Adaptive Agility:** The geodesic step size is dynamically scaled in proportion to the "representational torque" (the angular distance $\phi = \arccos(\mathbf{s}^T \mathbf{w})$ between the current state and incoming target). If the stream is stable, the torque vanishes, locking the weights and suppressing jitter. Under a sudden task transition, the torque explodes, instantly accelerating the rotation to eliminate representational lag.
4. **Spatial-Temporal Geodesic Coupling:** To enforce temporal coherence across consecutive queries, the final layer's state of the previous query ($\mathbf{s}_{t-1}^{(L)}$) is propagated to initialize the first adapted layer's boundary condition of the current query ($\mathbf{s}_t^{(L_{\text{frozen}})}$).

## Key Findings and Empirical Results
The authors evaluate UGR in a 14-layer synthetic Analytical Coordinate Sandbox (ICS) across 10 independent random seeds, and on a real-world multi-task text classification task (using TF-IDF features from the `20newsgroups` dataset) across 5 independent seeds. 

* **Synthetic Sandbox (Table 1):** 
  - Standard UGR achieves **75.08%** Joint Mean Accuracy, outperforming the SOTA continuous biochemical kinetics baseline (ChemMerge Reset) by **+5.43%** absolute margin.
  - Standard UGR reduces intra-query routing jitter ($L \ge 5$) to **19.51 $\times 10^{-4}$**, which is a **2.10$\times$ reduction** compared to ChemMerge.
  - Under a realistic block-structured stream (50-sample blocks), UGR's overall routing jitter drops to **11.63 $\times 10^{-4}$** while maintaining a joint classification accuracy of **75.17%**.
* **Real-World Text Classification (Table 2):**
  - Standard UGR achieves **92.25%** Joint Mean Accuracy, outperforming Coupled Momentum-Merge by **+4.13%** and Coupled ChemMerge by a massive **+21.60%** absolute margin.
  - Standard UGR slashes routing jitter to **3.68 $\times 10^{-4}$** (a **1.63$\times$ reduction** over Coupled Momentum-Merge).
  - The fully Softmax-free variant (using ReLU and $L_1$-normalization on similarity scores) slashes jitter even further to a pristine **1.50 $\times 10^{-4}$** while achieving **87.40%** accuracy.
* **Serving Latency and Throughput (Table 3):**
  - Timing benchmarks on an Intel Xeon CPU reveal that UGR adds less than **0.07 ms** per query over the stateless baseline, achieving **2052.7 QPS** (and **2295.3 QPS** for the Softmax-Free target variant), completely bypassing the latency bottlenecks of virtual-time ODE solvers.

## Explicitly Claimed Contributions (with evidence)
1. **Curved State-Space Formulation:** Proving that the square-root Born mapping guarantees exact simplex constraints natively (Section 3.2).
2. **Closed-Form Geodesic Updates (Slerp):** Deriving a computationally efficient, Rodrigues-like operator that performs spherical interpolation (Slerp) in $\mathcal{O}(K)$ complexity (Section 3.4).
3. **Torque-Driven Adaptive Agility:** Designing a self-regulating control loop that eliminates representational lag, backed by decomposed jitter analyses showing a clear 1.8$\times$ separation between intra-task stability and inter-task agility (Section 3.5 & Section 4.4.7).
4. **Spatial-Temporal Geodesic Coupling:** Design of cross-query boundary recurrence, supported by extensive ablation studies isolating the gains of coupling from the geodesic manifold geometry (Section 3.6 & Section 4.5).
5. **Real-World and Synthetic Benchmarking:** Delivering state-of-the-art results across both synthetic and real-world NLP workloads with high statistical significance (10 and 5 seeds, standard deviations reported).
