# Summary of the Paper

This paper presents **Unitary Geodesic Routing (UGR)**, a novel geometric framework for stateful test-time model ensembling across heterogeneous, non-stationary task streams. 

## Core Problem
Dynamic test-time model ensembling blends specialized expert weights (e.g., PEFT modules like LoRA) on-the-fly for incoming sequential queries. To smooth out representation noise while remaining agile under task transitions, stateful routing is preferred. However, existing stateful routers (e.g., *Momentum-Merge*, *ChemMerge*) perform updates in unconstrained flat Euclidean spaces and then project onto the probability simplex via post-hoc Softmax normalization. The authors claim this mismatch introduces:
1. **Representational Lag (Hysteresis):** Accumulating flat-space inertia under task transitions.
2. **Geometric Distortion:** Activation scale and norm mismatches when Euclidean interpolations "wiggle off" the probability manifold.
3. **High-Frequency Jitter:** Oscillations when flat routers use small inertia or high Softmax temperatures to overcome lag.

## Proposed Solution: Unitary Geodesic Routing (UGR)
UGR models the ensembling state directly on the curved $(K-1)$-dimensional hypersphere $\mathbb{S}^{K-1} \subset \mathbb{R}^K$.
- **Born Mapping:** Maps the spherical state to the probability simplex natively via $\alpha_{k,t} = (s_{k,t})^2$, which is mathematically related to the square-root (Hellinger/Bhattacharyya) mapping in Information Geometry and Born's rule in quantum mechanics. This completely eliminates post-hoc Softmax normalization for the state representation and simplex projection.
- **Closed-Form Geodesic Updates:** Derives a Rodrigues-like spherical linear interpolation (Slerp) along the shortest great-circle path of the hypersphere, bypassing costly matrix exponentials and virtual-time numerical ODE solvers.
- **Torque-Driven Adaptive Agility:** Dynamically scales the update step size proportionally to the angular distance (representational torque) between the current state and incoming signals. Torque vanishes when the stream is stationary (suppressing jitter) and explodes under sudden transitions (accelerating updates).
- **Spatial-Temporal Geodesic Coupling:** Propagates the ensembling state from the final layer of the previous query to the initial adapted layer of the current query to ensure cross-query temporal coherence.

## Evaluation and Key Results
The authors conduct evaluations on:
1. **Synthetic Analytical Coordinate Sandbox (ICS):** A 14-layer, 192-dimensional environment evaluated across 10 synchronized seeds. UGR achieves **75.08%** Joint Mean Accuracy (outperforming ChemMerge Reset by **+5.43%**) and slashes layer-to-layer routing jitter ($L \ge 5$) by **2.10$\times$**.
2. **Real-World Text Classification (20newsgroups):** A multi-task text classification stream with 4 meta-domains and 4 specialized MLP classifiers across 5 synchronized seeds. UGR delivers a Joint Mean Accuracy of **92.25%** (+21.60% over ChemMerge) while reducing routing jitter by **1.63$\times$** compared to Coupled Momentum-Merge.
3. **Latency Benchmarking:** The standard UGR adds less than 0.07 ms per query. The fully Softmax-free target variant achieves **0.436 ms/query** and **2295.3 QPS** on an Intel Xeon CPU.

## Claimed Contributions (with Evidence)
1. *Curved State-Space Formulation:* Verified via mathematical proofs of sign symmetry and positive orthant persistence in Appendix A.2.
2. *Closed-Form Geodesic Updates:* Verified via operations count and complexity analysis showing $\mathcal{O}(KD)$ scaling.
3. *Torque-Driven Agility:* Verified via control-theoretic proofs and decomposed jitter analyses showing separation between stability and plasticity.
4. *Spatial-Temporal Coupling:* Verified via cross-query state persistence evaluation (Reset vs. Coupled baselines).
5. *Real-World Validation:* Verified on the `20newsgroups` dataset using specialized MLP classifiers.
