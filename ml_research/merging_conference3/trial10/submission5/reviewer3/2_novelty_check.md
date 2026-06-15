# Novelty and Delta Assessment

## Key Novel Aspects
The submission proposes several unique aspects:
1. **Geometric Framing of Test-Time Ensembling State:** It is the first work to represent the time-dependent expert ensembling routing state as a point on the unit hypersphere $\mathbb{S}^{K-1}$, leveraging the square-root map (Born mapping) from classical Information Geometry to project directly onto the probability simplex without using a Softmax layer.
2. **Torque-Driven Step-Size Modulation:** The concept of "representational torque" as a training-free feedback loop that dynamically scales the angular step-size based on the mismatch between the previous routing state and the bottom-up target vector is highly creative. It acts as a non-linear damping mechanism to resolve the stability-plasticity dilemma.
3. **2D Spatial-Temporal Geodesic Coupling:** Re-initializing the first adapted layer's state of the current query using the final layer's state of the previous query ($\mathbf{s}_t^{(L_{\text{frozen}})} = \mathbf{s}_{t-1}^{(L)}$). This propagates converged deep semantic priors across temporal sample boundaries.

---

## Detailed "Delta" from Prior Work

| Aspect | Prior Work (SABLE, ChemMerge, Momentum-Merge) | Proposed Work (UGR) | Delta & Significance |
| :--- | :--- | :--- | :--- |
| **State Space** | Unconstrained flat Euclidean space $\mathbb{R}^K$. | Curved unit hypersphere $\mathbb{S}^{K-1}$. | **Significant:** Prevents updates from departing the probability manifold, avoiding scale and norm distortions of intermediate activations. |
| **Simplex Projection** | Post-hoc Softmax or boundary clipping. | Coordinate-wise squared magnitudes (Born's rule). | **Significant:** Bypasses Softmax, eliminating representational boundary compression and scale-warping. Under exact mapping, it represents closed-form Fisher-Rao geodesic flow. |
| **Update Mechanism** | Linear EMA interpolation or virtual-time biochemical ODEs. | Orthonormal basis geodesic rotation (Slerp) along great-circle paths. | **Significant:** Derives a closed-form Rodrigues-like rotation that completely bypasses numerical integration or matrix exponentials. |
| **Adaptivity Control** | Fixed hyperparameters or continuous decay constants. | Torque-driven step-size adjustment (physics-inspired). | **Moderate-to-Significant:** Step-size dynamically expands on task switches and vanishes on stable streams, avoiding overshoot and kinetic momentum. |
| **State Propagation** | Stateless (SABLE), layer-wise EMA, or biochemical kinetics. | Cross-layer Spatial-Temporal Coupling ($\mathbf{s}_t^{(L_{\text{frozen}})} = \mathbf{s}_{t-1}^{(L)}$). | **Moderate:** Propagates the final (most semantically mature) state of the previous query to the initial boundary of the next query. |

---

## Characterization of Novelty
The novelty of this paper is **significant**. 

While individual components leverage established mathematical tools (e.g., Shoemake's Spherical Linear Interpolation (Slerp), the square-root mapping from Information Geometry), their integration into a unified, training-free, stateful test-time routing pipeline is highly original. 

Rather than proposing a minor tweak or hyperparameter tuning of existing Euclidean EMA (Momentum-Merge) or biochemical kinetic (ChemMerge) models, UGR rejects the flat-space paradigm altogether. It establishes a completely new geometric foundation for test-time adaptive serving. The mathematical rigor of the closed-form geodesic updates and the physics-inspired torque control loop are elegant and provide a fresh perspective to a field dominated by unconstrained optimization.
